import mysql.connector
import json
from pathlib import Path
from datetime import datetime

# CONFIGURE YOUR DB CONNECTION HERE
DB_CONFIG = {
    "host": "localhost",
    "user": "root",              # Replace with your username
    "password": "99900",         # Replace with your password
    "database": "app_sm_loan"     # Replace with your DB name
}

# OUTPUT PATHS
db_summary_json = Path("../data/processed/db_summary.json")
db_summary_text = Path("../data/processed/db_summary.txt")
db_summary_json.parent.mkdir(parents=True, exist_ok=True)

INPUT_PATH = "../data/processed/db_summary.json"
OUTPUT_PATH = "../data/processed/train_sql_data.jsonl"

def extract_schema_info(cursor, db_name):
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


def generate_instruction_response(table_name, table_info):
    prompt_create = {
        "text": f"### Instruction:\nDescribe the schema of the `{table_name}` table\n\n### Response:\n{table_info['create_statement']}"
    }

    prompt_summary = {
        "text": f"### Instruction:\nList the columns in the `{table_name}` table with types\n\n### Response:\n" +
                "\n".join([f"- `{col['Field']}`: {col['Type']}" for col in table_info['columns']])
    }

    return [prompt_create, prompt_summary]

def save_output(data):
    # Save as JSON
    with open(db_summary_json, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Save as readable text
    with open(db_summary_text, "w", encoding='utf-8') as f:
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

def main():
    try:
        print(f"🔌 Connecting to DB: {DB_CONFIG['database']} ...")
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("🔍 Extracting schema and metadata...")
        data = extract_schema_info(cursor, DB_CONFIG["database"])
        save_output(data)

        print("✅ Extraction complete. Files saved:")
        print(f" - {db_summary_json.resolve()}")
        print(f" - {db_summary_text.resolve()}")

        cursor.close()
        conn.close()

        with open(INPUT_PATH, "r", encoding='utf-8') as f:
            db_data = json.load(f)

        out_lines = []
        for table, info in db_data.get("tables", {}).items():
            prompts = generate_instruction_response(table, info)
            out_lines.extend(prompts)

        with open(OUTPUT_PATH, "w", encoding='utf-8') as out_f:
            for entry in out_lines:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"✅ Transformed {len(out_lines)} entries into {OUTPUT_PATH}")

    except mysql.connector.Error as err:
        print(f"❌ MySQL error: {err}")

if __name__ == "__main__":
    main()
