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

        # File paths
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
            cursor.execute(f"SHOW COLUMNS FROM `{table}`")
            columns = cursor.fetchall()

            cursor.execute(f"SELECT COUNT(*) FROM `{table}`")
            row_count = cursor.fetchone()[0]

            cursor.execute(f"SHOW CREATE TABLE `{table}`")
            create_stmt = cursor.fetchone()[1]

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
                "create_statement": create_stmt
            }

        return result

    def generate_instruction_response(self, table_name, table_info):
        prompt_create = {
            "text": f"### Instruction:\nDescribe the schema of the `{table_name}` table\n\n### Response:\n{table_info['create_statement']}"
        }

        prompt_summary = {
            "text": f"### Instruction:\nList the columns in the `{table_name}` table with types\n\n### Response:\n" +
                    "\n".join([f"- `{col['Field']}`: {col['Type']}" for col in table_info['columns']])
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
                    f.write(f"  - {col['Field']} ({col['Type']}) {key_info}\n")
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
