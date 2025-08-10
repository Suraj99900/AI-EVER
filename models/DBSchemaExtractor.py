import mysql.connector
import json
from pathlib import Path
from datetime import datetime

class DBSchemaExtractor:
    def __init__(self, host="localhost", port=3306, user="root", password="", database=""):
        self._db_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database
        }

        self.db_summary_json = Path("data/processed/db_summary.json")
        self.db_summary_text = Path("data/processed/db_summary.txt")
        self.output_path = Path("data/processed/train_sql.jsonl")

        self.db_summary_json.parent.mkdir(parents=True, exist_ok=True)

    def extract_schema_info(self, cursor, db_name):
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        result = {
            "database": db_name,
            "extracted_at": datetime.now().isoformat(),
            "tables": {}
        }

        for table in tables:
            print(f"📦 Extracting: {table}")

            # Columns
            cursor.execute(f"SHOW COLUMNS FROM `{table}`")
            columns = cursor.fetchall()

            # Row count
            cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
            row_count = cursor.fetchone()[0]

            # Table creation statement
            cursor.execute(f"SHOW CREATE TABLE `{table}`")
            create_stmt = cursor.fetchone()[1]

            # Indexes
            cursor.execute(f"SHOW INDEX FROM `{table}`")
            indexes_raw = cursor.fetchall()
            indexes = {}
            for idx in indexes_raw:
                key_name = idx[2]
                if key_name not in indexes:
                    indexes[key_name] = {
                        "unique": not bool(idx[1]),
                        "columns": []
                    }
                indexes[key_name]["columns"].append(idx[4])

            # Foreign keys (from information_schema)
            cursor.execute(f"""
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
            foreign_keys = []
            for fk in fkeys_raw:
                foreign_keys.append({
                    "constraint_name": fk[0],
                    "column_name": fk[2],
                    "referenced_table": fk[3],
                    "referenced_column": fk[4],
                })

            # Compose table info
            result["tables"][table] = {
                "columns": [
                    {
                        "Field": col[0],
                        "Type": col[1],
                        "Null": col[2],
                        "Key": col[3],
                        "Default": col[4],
                        "Extra": col[5],
                    } for col in columns
                ],
                "row_count": row_count,
                "create_statement": create_stmt,
                "indexes": indexes,
                "foreign_keys": foreign_keys
            }

        return result

    def generate_instruction_response(self, table_name, table_info):
        # Instruction + create statement prompt
        prompt_create = {
            "text": (
                f"### Instruction:\nDescribe the schema of the `{table_name}` table including columns, indexes, and foreign keys.\n\n"
                f"### Response:\n{table_info['create_statement']}\n\n"
                f"Indexes:\n" + "\n".join(
                    [f"- `{idx}`: columns {', '.join(details['columns'])}, unique: {details['unique']}" for idx, details in table_info['indexes'].items()]
                ) + "\n\n" +
                f"Foreign Keys:\n" + "\n".join(
                    [f"- `{fk['constraint_name']}`: column `{fk['column_name']}` references `{fk['referenced_table']}`(`{fk['referenced_column']}`)" for fk in table_info['foreign_keys']]
                )
            )
        }

        # Instruction + columns summary prompt
        prompt_summary = {
            "text": (
                f"### Instruction:\nList the columns in the `{table_name}` table with types and nullability.\n\n"
                f"### Response:\n" +
                "\n".join([
                    f"- `{col['Field']}`: {col['Type']} (Nullable: {col['Null']}) Default: {col['Default']}"
                    for col in table_info['columns']
                ]) +
                f"\n\nRow count: {table_info['row_count']}"
            )
        }

        return [prompt_create, prompt_summary]

    def save_output(self, data):
        # Save as JSON
        with open(self.db_summary_json, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save readable text
        with open(self.db_summary_text, "w", encoding='utf-8') as f:
            f.write(f"Database: {data['database']}\nExtracted at: {data['extracted_at']}\n\n")
            for table, info in data["tables"].items():
                f.write(f"--- TABLE: {table} ---\n")
                f.write(f"Row count: {info['row_count']}\n")
                f.write("Columns:\n")
                for col in info["columns"]:
                    key_info = f"[{col['Key']}]" if col['Key'] else ""
                    f.write(f"  - {col['Field']} ({col['Type']}) Nullable: {col['Null']} Default: {col['Default']} {key_info}\n")
                f.write("Indexes:\n")
                for idx_name, idx_info in info.get("indexes", {}).items():
                    f.write(f"  - {idx_name}: columns {', '.join(idx_info['columns'])}, unique: {idx_info['unique']}\n")
                f.write("Foreign Keys:\n")
                for fk in info.get("foreign_keys", []):
                    f.write(f"  - {fk['constraint_name']}: column {fk['column_name']} references {fk['referenced_table']}({fk['referenced_column']})\n")
                f.write("\nCREATE TABLE statement:\n")
                f.write(info["create_statement"] + "\n\n")

    def run(self):
        try:
            print(f"🔌 Connecting to DB: {self._db_config['database']} ...")
            conn = mysql.connector.connect(**self._db_config)
            cursor = conn.cursor()

            print("🔍 Extracting schema and metadata...")
            data = self.extract_schema_info(cursor, self._db_config["database"])
            self.save_output(data)

            print("✅ Extraction complete. Files saved:")
            print(f" - {self.db_summary_json.resolve()}")
            print(f" - {self.db_summary_text.resolve()}")

            # Generate JSONL train data
            out_lines = []
            for table, info in data.get("tables", {}).items():
                prompts = self.generate_instruction_response(table, info)
                out_lines.extend(prompts)

            with open(self.output_path, "w", encoding='utf-8') as out_f:
                for entry in out_lines:
                    out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"✅ Transformed {len(out_lines)} entries into {self.output_path.resolve()}")

            cursor.close()
            conn.close()

        except mysql.connector.Error as err:
            print(f"❌ MySQL error: {err}")
            raise
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise
