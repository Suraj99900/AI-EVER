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

            # Foreign keys
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

    # --------- prompts (unchanged) ----------
    def generate_instruction_prompts_for_table(self, table_name: str, table_info: Dict[str, Any]) -> List[Dict[str, str]]:
        prompts = []
        create_text = table_info.get("create_statement", "").strip()
        schema_prompt = {"text": f"### Instruction:\nDescribe the schema of the `{table_name}` table including columns, indexes and foreign keys.\n\n### Response:\n{create_text}"}
        prompts.append(schema_prompt)
        column_lines = [f"- `{col['Field']}`: {col['Type']} (Nullable: {col['Null']}) Default: {col['Default']}" for col in table_info.get("columns", [])]
        summary_prompt = {"text": f"### Instruction:\nList the columns of `{table_name}` with types and nullability and suggest which columns are commonly used for joins or indexing.\n\n### Response:\n" + "\n".join(column_lines) + f"\n\nRow count: {table_info.get('row_count')}"}
        prompts.append(summary_prompt)
        sample_rows = table_info.get("sample_rows", [])
        if sample_rows:
            sample_text = "\n".join([json.dumps(r, ensure_ascii=False) for r in sample_rows])
            prompts.append({"text": f"### Instruction:\nShow example rows for the `{table_name}` table and a short natural language description of what each row represents.\n\n### Response:\n{sample_text}"})
        inserts = table_info.get("sample_inserts", [])
        if inserts:
            insert_text = "\n".join(inserts[: max(1, min(len(inserts), 5))])
            prompts.append({"text": f"### Instruction:\nProvide INSERT statements example for the `{table_name}` table using real sample values.\n\n### Response:\n{insert_text}"})
        fks = table_info.get("foreign_keys", [])
        if fks:
            fk_lines = [f"- `{fk['constraint_name']}`: `{fk['column_name']}` -> `{fk['referenced_table']}`(`{fk['referenced_column']}`)" for fk in fks]
            prompts.append({"text": f"### Instruction:\nExplain the foreign key relationships for the `{table_name}` table and suggest join queries.\n\n### Response:\n" + "\n".join(fk_lines)})
        common_queries = f"SELECT * FROM `{table_name}` LIMIT 10;\nSELECT COUNT(*) FROM `{table_name}`;\nSELECT * FROM `{table_name}` WHERE <column>=<value> LIMIT 10;\n"
        prompts.append({"text": f"### Instruction:\nGive some example queries for the `{table_name}` table useful for analytics or debugging.\n\n### Response:\n{common_queries}"})
        return prompts

    def save_output(self, data: Dict[str, Any]) -> None:
        with open(self.db_summary_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        with open(self.db_summary_text, "w", encoding="utf-8") as f:
            f.write(f"Database: {data.get('database')}\nExtracted at: {data.get('extracted_at')}\n\n")
            for table, info in data.get("tables", {}).items():
                f.write(f"--- TABLE: {table} ---\nRow count: {info.get('row_count')}\nColumns:\n")
                for col in info.get("columns", []):
                    key_info = f"[{col['Key']}]" if col.get('Key') else ""
                    f.write(f" - {col['Field']} ({col['Type']}) Nullable: {col['Null']} Default: {col['Default']} {key_info}\n")
                f.write("\n")
        with open(self.output_path, "w", encoding="utf-8") as out:
            for table, info in data.get("tables", {}).items():
                for p in self.generate_instruction_prompts_for_table(table, info):
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
