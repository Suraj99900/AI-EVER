# models/DBSchemaExtractor.py
import mysql.connector
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from models.log_manager import LogManager
import re

logmgr = LogManager()
logger = logmgr.setup_logger("extract")

class DBSchemaExtractor:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        password: str = "",
        database: str = "",
        out_dir: Path | str = Path("data/processed"),
        sample_rows: int = 10,
        redact: bool = True,
        redact_columns: Optional[List[str]] = None,   # column names to always redact
    ):
        self._db_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database
        }
        self.sample_rows = int(sample_rows)
        self.redact = bool(redact)
        self.redact_columns = set([c.lower() for c in (redact_columns or [])])

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.db_summary_json = out_dir / "db_summary.json"
        self.db_summary_text = out_dir / "db_summary.txt"
        self.output_path = out_dir / "train_sql.jsonl"

    # -------------- logging helper ------------------
    def _emit(self, msg: str, level: str = "info", callback: Optional[Callable[[str], None]] = None):
        text = str(msg)
        # LogManager (used by UI polling)
        try:
            logmgr.append("extract", text)
        except Exception:
            pass
        # python logging
        try:
            if hasattr(logger, level):
                getattr(logger, level)(text)
            else:
                logger.info(text)
        except Exception:
            pass
        # external callback (used for streaming)
        if callback:
            try:
                callback(text)
            except Exception:
                pass

    # ------------- internal helpers -----------------
    def _sanitize_value(self, val: Any) -> str:
        if val is None:
            return "NULL"
        if isinstance(val, (int, float)):
            return str(val)
        if isinstance(val, bool):
            return "TRUE" if val else "FALSE"

        s = str(val)

        if self.redact:
            # redact emails, long strings or long numbers
            if "@" in s or re.search(r"\b\d{6,}\b", s) or len(s) > 120:
                return "'<REDACTED>'"

        # escape single quotes safely
        s_escaped = s.replace("'", "''")
        return "'" + s_escaped + "'"

    def _row_to_insert(self, table: str, columns: List[str], row: tuple) -> str:
        vals = []
        for col, val in zip(columns, row):
            if col.lower() in self.redact_columns:
                vals.append("'<REDACTED>'")
            else:
                vals.append(self._sanitize_value(val))
        cols_sql = ", ".join([f"`{c}`" for c in columns])
        vals_sql = ", ".join(vals)
        return f"INSERT INTO `{table}` ({cols_sql}) VALUES ({vals_sql});"

    # ------------- main extraction helpers --------------
    def extract_schema_info(self, cursor, db_name: str, callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        result = {
            "database": db_name,
            "extracted_at": datetime.now().isoformat(),
            "tables": {}
        }

        for table in tables:
            self._emit(f"Extracting table: {table}", callback=callback)
            # Columns
            cursor.execute(f"SHOW COLUMNS FROM `{table}`")
            columns_raw = cursor.fetchall()
            columns = [{"Field": col[0], "Type": col[1], "Null": col[2],
                        "Key": col[3], "Default": col[4], "Extra": col[5]}
                       for col in columns_raw]

            # Row count
            try:
                cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
                row_count = cursor.fetchone()[0]
            except Exception as e:
                row_count = -1
                self._emit(f"Could not get row count for {table}: {e}", level="warning", callback=callback)

            # Create statement
            try:
                cursor.execute(f"SHOW CREATE TABLE `{table}`")
                create_stmt = cursor.fetchone()[1]
            except Exception as e:
                create_stmt = ""
                self._emit(f"Could not get CREATE TABLE for {table}: {e}", level="warning", callback=callback)

            # Indexes
            try:
                cursor.execute(f"SHOW INDEX FROM `{table}`")
                indexes_raw = cursor.fetchall()
                indexes = {}
                for idx in indexes_raw:
                    key_name = idx[2]
                    if key_name not in indexes:
                        indexes[key_name] = {"unique": not bool(idx[1]), "columns": []}
                    indexes[key_name]["columns"].append(idx[4])
            except Exception:
                indexes = {}

            # Foreign keys (information_schema)
            try:
                cursor.execute("""
                    SELECT
                        constraint_name, table_name, column_name, referenced_table_name, referenced_column_name
                    FROM
                        information_schema.key_column_usage
                    WHERE
                        table_schema = %s
                        AND table_name = %s
                        AND referenced_table_name IS NOT NULL
                """, (db_name, table))
                fkeys_raw = cursor.fetchall()
                foreign_keys = [{"constraint_name": fk[0], "column_name": fk[2],
                                 "referenced_table": fk[3], "referenced_column": fk[4]} for fk in fkeys_raw]
            except Exception:
                foreign_keys = []

            # sample rows
            sample_rows, sample_inserts = [], []
            try:
                cursor.execute(f"SELECT * FROM `{table}` LIMIT %s", (self.sample_rows,))
                rows = cursor.fetchall()
                col_names = [c[0] for c in cursor.description] if cursor.description else [c["Field"] for c in columns]
                for r in rows:
                    sample_rows.append(dict(zip(col_names, [
                        ("<REDACTED>" if (cn.lower() in self.redact_columns and self.redact)
                         else (None if v is None else str(v)))
                        for cn, v in zip(col_names, r)
                    ])))
                    sample_inserts.append(self._row_to_insert(table, col_names, r))
                self._emit(f"Fetched {len(sample_rows)} sample rows for {table}", callback=callback)
            except Exception as e:
                self._emit(f"Could not fetch sample rows for {table}: {e}", level="warning", callback=callback)

            result["tables"][table] = {
                "columns": columns,
                "row_count": row_count,
                "create_statement": create_stmt,
                "indexes": indexes,
                "foreign_keys": foreign_keys,
                "sample_rows": sample_rows,
                "sample_inserts": sample_inserts
            }

        return result

    # --------- prompts (extended) ----------
    def generate_instruction_prompts_for_table(self, table_name: str, table_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Basic prompts already produced earlier (schema, column list, sample rows, inserts, fks, simple queries).
        This function will produce those and then append single-table advanced prompts.
        """
        prompts = []
        create_text = table_info.get("create_statement", "").strip()
        # Schema description
        schema_prompt = {"text": f"### Instruction:\nDescribe the schema of the `{table_name}` table including columns, indexes and foreign keys.\n\n### Response:\n{create_text}"}
        prompts.append(schema_prompt)

        # Column list summary
        column_lines = [f"- `{col['Field']}`: {col['Type']} (Nullable: {col['Null']}) Default: {col['Default']}" for col in table_info.get("columns", [])]
        summary_prompt = {"text": f"### Instruction:\nList the columns of `{table_name}` with types and nullability and suggest which columns are commonly used for joins or indexing.\n\n### Response:\n" + "\n".join(column_lines) + f"\n\nRow count: {table_info.get('row_count')}"}
        prompts.append(summary_prompt)

        # sample rows
        sample_rows = table_info.get("sample_rows", [])
        if sample_rows:
            sample_text = "\n".join([json.dumps(r, ensure_ascii=False) for r in sample_rows])
            prompts.append({"text": f"### Instruction:\nShow example rows for the `{table_name}` table and a short natural language description of what each row represents.\n\n### Response:\n{sample_text}"})

        # sample inserts
        inserts = table_info.get("sample_inserts", [])
        if inserts:
            insert_text = "\n".join(inserts[: max(1, min(len(inserts), 5))])
            prompts.append({"text": f"### Instruction:\nProvide INSERT statements example for the `{table_name}` table using real sample values.\n\n### Response:\n{insert_text}"})

        # foreign keys for this table
        fks = table_info.get("foreign_keys", [])
        if fks:
            fk_lines = [f"- `{fk['constraint_name']}`: `{fk['column_name']}` -> `{fk['referenced_table']}`(`{fk['referenced_column']}`)" for fk in fks]
            prompts.append({"text": f"### Instruction:\nExplain the foreign key relationships for the `{table_name}` table and suggest join queries.\n\n### Response:\n" + "\n".join(fk_lines)})

        # common queries
        common_queries = (
            f"SELECT * FROM `{table_name}` LIMIT 10;\n"
            f"SELECT COUNT(*) FROM `{table_name}`;\n"
            f"SELECT * FROM `{table_name}` WHERE <column>=<value> LIMIT 10;\n"
        )
        prompts.append({"text": f"### Instruction:\nGive some example queries for the `{table_name}` table useful for analytics or debugging.\n\n### Response:\n{common_queries}"})

        # single-table advanced prompts (aggregations, window functions, DML, integrity checks)
        prompts.extend(self.generate_advanced_prompts_for_table(table_name, table_info))
        return prompts

    def generate_advanced_prompts_for_table(self, table_name: str, table_info: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Create more complex single-table prompt examples: aggregations, group by, window functions,
        upsert examples, explain plan, update/delete examples with conditions, index suggestions.
        """
        prompts = []
        columns = [c["Field"] for c in table_info.get("columns", [])]

        # Aggregation example (pick numeric columns heuristically)
        numeric_cols = [c for c in table_info.get("columns", []) if re.search(r"(int|decimal|numeric|float|double|bigint|smallint)", c["Type"], re.IGNORECASE)]
        if numeric_cols:
            col = numeric_cols[0]["Field"]
            agg_q = (
                f"SELECT {col}, COUNT(*) AS cnt, AVG({col}) AS avg_{col} "
                f"FROM `{table_name}` GROUP BY {col} ORDER BY cnt DESC LIMIT 20;"
            )
            prompts.append({"text": f"### Instruction:\nProvide an aggregation query for `{table_name}` that groups by `{col}` and returns counts and averages.\n\n### Response:\n{agg_q}"})

        # Window function example
        if numeric_cols:
            col = numeric_cols[0]["Field"]
            wf = (
                f"SELECT *, ROW_NUMBER() OVER (ORDER BY {col} DESC) AS rn, "
                f"AVG({col}) OVER () AS overall_avg_{col} FROM `{table_name}` LIMIT 100;"
            )
            prompts.append({"text": f"### Instruction:\nShow an example query using window functions on `{table_name}` to rank rows by `{col}` and compute overall average.\n\n### Response:\n{wf}"})

        # Upsert / Insert-from-select
        if columns:
            cols_list = ", ".join([f"`{c}`" for c in columns[:3]])  # limit to first 3 columns
            select_example = (
                f"INSERT INTO `{table_name}` ({cols_list})\nSELECT {cols_list} FROM `{table_name}` WHERE 1=0; -- example: copy structure\n"
            )
            prompts.append({"text": f"### Instruction:\nProvide an INSERT ... SELECT example for `{table_name}` (copying rows or transforming values).\n\n### Response:\n{select_example}"})

        # Update & Delete examples
        if columns:
            pk_col = None
            for c in table_info.get("columns", []):
                if c.get("Key") == "PRI":
                    pk_col = c["Field"]
                    break
            where_hint = f"`{pk_col}` = <value>" if pk_col else "`id` = <value>"
            update_q = f"UPDATE `{table_name}` SET /* column = value */ WHERE {where_hint};"
            delete_q = f"DELETE FROM `{table_name}` WHERE {where_hint};"
            prompts.append({"text": f"### Instruction:\nShow example UPDATE and DELETE statements for `{table_name}` using a primary key or id.\n\n### Response:\n{update_q}\n\n{delete_q}"})

        # Explain plan & indexing suggestions
        if columns:
            sample_select = f"SELECT * FROM `{table_name}` LIMIT 10;"
            explain_q = f"EXPLAIN {sample_select}"
            # Suggest indexes on columns used in WHERE or foreign keys
            suggestions = []
            for fk in table_info.get("foreign_keys", []):
                suggestions.append(f"Consider adding an index on `{fk['column_name']}` as it is a foreign key referencing `{fk['referenced_table']}`.")
            if not suggestions and columns:
                suggestions.append(f"Consider indexing columns used frequently in WHERE clauses, e.g., `{columns[0]}`.")
            prompts.append({"text": f"### Instruction:\nProvide an EXPLAIN plan example and index suggestions for `{table_name}`.\n\n### Response:\n{explain_q}\n\nSuggestions:\n" + "\n".join(suggestions)})

        # Integrity checks
        integrity_checks = (
            f"-- Find rows with NULL foreign key values (if any)\n"
            f"SELECT * FROM `{table_name}` WHERE 1=0; -- replace with actual fk checks\n"
            f"-- Find duplicates by candidate unique columns\n"
            f"SELECT <col>, COUNT(*) FROM `{table_name}` GROUP BY <col> HAVING COUNT(*) > 1;\n"
        )
        prompts.append({"text": f"### Instruction:\nList SQL queries to validate data integrity for `{table_name}` (NULLs, duplicates, referential integrity checks).\n\n### Response:\n{integrity_checks}"})

        return prompts

    def generate_cross_table_prompts(self, full_schema: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Analyze foreign key relationships across tables and produce multi-table join/subquery prompts:
          - INNER JOIN / LEFT JOIN examples
          - Aggregation across joins
          - EXISTS / NOT EXISTS and IN subqueries
          - INSERT ... SELECT involving joins
          - DELETE / UPDATE patterns based on joins
        """
        prompts = []
        tables = full_schema.get("tables", {})
        # Build quick map: referenced_table -> list of (table, fk)
        ref_map = {}
        for tbl_name, info in tables.items():
            for fk in info.get("foreign_keys", []):
                ref = fk["referenced_table"]
                ref_map.setdefault(ref, []).append({
                    "from_table": tbl_name,
                    "from_column": fk["column_name"],
                    "to_column": fk["referenced_column"],
                    "constraint": fk.get("constraint_name")
                })

        # For each referenced table produce join examples
        for ref_table, fk_list in ref_map.items():
            # Example INNER JOIN: each referencing table with ref_table
            for fk in fk_list:
                jt = fk["from_table"]
                left = ref_table
                right = jt
                left_pk = fk["to_column"]
                right_fk = fk["from_column"]

                inner_join_q = (
                    f"SELECT a.*, b.* FROM `{left}` a\n"
                    f"INNER JOIN `{right}` b ON a.`{left_pk}` = b.`{right_fk}`\n"
                    f"WHERE a.`{left_pk}` IS NOT NULL LIMIT 200;"
                )
                prompts.append({"text": f"### Instruction:\nWrite an INNER JOIN query joining `{left}` and `{right}` using `{left_pk}` = `{right_fk}` and explain when to use INNER JOIN vs LEFT JOIN.\n\n### Response:\n{inner_join_q}"})

                left_join_q = (
                    f"SELECT a.*, b.* FROM `{left}` a\n"
                    f"LEFT JOIN `{right}` b ON a.`{left_pk}` = b.`{right_fk}`\n"
                    f"ORDER BY a.`{left_pk}` NULLS LAST LIMIT 200;"
                )
                prompts.append({"text": f"### Instruction:\nWrite a LEFT JOIN query for `{left}` -> `{right}` and show how missing references appear in results.\n\n### Response:\n{left_join_q}"})

                # Aggregation across join
                agg_join = (
                    f"SELECT a.`{left_pk}`, COUNT(b.`{right_fk}`) AS related_count\n"
                    f"FROM `{left}` a\n"
                    f"LEFT JOIN `{right}` b ON a.`{left_pk}` = b.`{right_fk}`\n"
                    f"GROUP BY a.`{left_pk}`\n"
                    f"ORDER BY related_count DESC LIMIT 50;"
                )
                prompts.append({"text": f"### Instruction:\nProvide an aggregation query that counts related rows from `{right}` for each row in `{left}`.\n\n### Response:\n{agg_join}"})

                # EXISTS subquery
                exists_q = (
                    f"SELECT a.* FROM `{left}` a WHERE EXISTS (\n"
                    f"  SELECT 1 FROM `{right}` b WHERE b.`{right_fk}` = a.`{left_pk}`\n"
                    f") LIMIT 200;"
                )
                prompts.append({"text": f"### Instruction:\nShow an EXISTS subquery example to find rows in `{left}` that have related rows in `{right}`.\n\n### Response:\n{exists_q}"})

                # INSERT ... SELECT example joining data (copy with join)
                insert_select = (
                    f"INSERT INTO `{right}` (/* columns */)\n"
                    f"SELECT /* transformed columns */ FROM `{left}` a\n"
                    f"JOIN `{right}` b ON a.`{left_pk}` = b.`{right_fk}`\n"
                    f"WHERE 1=0; -- adjust columns and conditions"
                )
                prompts.append({"text": f"### Instruction:\nProvide an INSERT ... SELECT example involving a join between `{left}` and `{right}`.\n\n### Response:\n{insert_select}"})

                # Delete by join pattern
                delete_join = (
                    f"DELETE t FROM `{right}` t\n"
                    f"JOIN `{left}` a ON a.`{left_pk}` = t.`{right_fk}`\n"
                    f"WHERE a.`{left_pk}` IS NULL; -- example: delete orphaned rows"
                )
                prompts.append({"text": f"### Instruction:\nShow a DELETE statement that removes orphaned rows in `{right}` using a JOIN with `{left}`.\n\n### Response:\n{delete_join}"})

        # Additionally: multi-way joins & analytical query example
        # Find simple candidate multi-way edges if a table references a ref_table that itself references another table
        # Build a few triplet examples
        for table_a, info_a in tables.items():
            for fk_a in info_a.get("foreign_keys", []):
                mid = fk_a["referenced_table"]
                # find tables that reference mid
                next_refs = ref_map.get(mid, [])
                for fk_b in next_refs:
                    table_b = fk_b["from_table"]
                    # pattern: table_a -> mid -> table_b (A references MID; B references MID); produce a 3-way example
                    q3 = (
                        f"SELECT a.*, m.*, b.* FROM `{table_a}` a\n"
                        f"JOIN `{mid}` m ON a.`{fk_a['column_name']}` = m.`{fk_a['referenced_column']}`\n"
                        f"LEFT JOIN `{table_b}` b ON m.`{fk_b['to_column'] if 'to_column' in fk_b else fk_b['to_column']}` = b.`{fk_b['from_column']}`\n"
                        f"LIMIT 200;"
                    )
                    prompts.append({"text": f"### Instruction:\nGive a 3-way join example connecting `{table_a}` -> `{mid}` -> `{table_b}` and explain the join ordering/performance considerations.\n\n### Response:\n{q3}"})

        # If no foreign keys found, provide generic cross-table examples
        if not ref_map:
            # pick two sample tables
            table_names = list(tables.keys())[:3]
            if len(table_names) >= 2:
                t1, t2 = table_names[0], table_names[1]
                generic_join = f"SELECT a.*, b.* FROM `{t1}` a LEFT JOIN `{t2}` b ON a.id = b.{t1}_id LIMIT 200;"
                prompts.append({"text": f"### Instruction:\nCreate an example LEFT JOIN between `{t1}` and `{t2}` with an explanation of when LEFT JOIN is useful.\n\n### Response:\n{generic_join}"})

        return prompts

    def save_output(self, data: Dict[str, Any]) -> None:
        # save json summary
        with open(self.db_summary_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # save text summary
        with open(self.db_summary_text, "w", encoding="utf-8") as f:
            f.write(f"Database: {data.get('database')}\nExtracted at: {data.get('extracted_at')}\n\n")
            for table, info in data.get("tables", {}).items():
                f.write(f"--- TABLE: {table} ---\nRow count: {info.get('row_count')}\nColumns:\n")
                for col in info.get("columns", []):
                    key_info = f"[{col['Key']}]" if col.get('Key') else ""
                    f.write(f" - {col['Field']} ({col['Type']}) Nullable: {col['Null']} Default: {col['Default']} {key_info}\n")
                f.write("\n")

        # write prompts to jsonl: per-table plus cross-table prompts
        with open(self.output_path, "w", encoding="utf-8") as out:
            # per-table prompts
            for table, info in data.get("tables", {}).items():
                for p in self.generate_instruction_prompts_for_table(table, info):
                    out.write(json.dumps(p, ensure_ascii=False) + "\n")

            # cross-table prompts using whole schema
            cross_prompts = self.generate_cross_table_prompts(data)
            for p in cross_prompts:
                out.write(json.dumps(p, ensure_ascii=False) + "\n")

    # --------------- public runner --------------------
    def run(self, callback: Optional[Callable[[str], None]] = None) -> None:
        self._emit(f"Connecting to DB: {self._db_config.get('database')} at {self._db_config.get('host')}:{self._db_config.get('port')}", callback=callback)
        conn = None
        try:
            conn = mysql.connector.connect(**self._db_config)
            cursor = conn.cursor()
            self._emit("Connection established.", callback=callback)
            data = self.extract_schema_info(cursor, self._db_config["database"], callback=callback)
            self._emit("Schema extracted, saving outputs...", callback=callback)
            self.save_output(data)
            self._emit(f"Saved summary JSON: {self.db_summary_json.resolve()}", callback=callback)
            self._emit(f"Saved summary TXT: {self.db_summary_text.resolve()}", callback=callback)
            self._emit(f"Saved train JSONL (prompts): {self.output_path.resolve()}", callback=callback)
            cursor.close()
        except mysql.connector.Error as err:
            self._emit(f"MySQL error: {err}", level="error", callback=callback)
            raise
        except Exception as e:
            self._emit(f"Unexpected error: {e}", level="error", callback=callback)
            raise
        finally:
            if conn:
                try: conn.close()
                except Exception: pass
            self._emit("DB extraction finished.", callback=callback)
