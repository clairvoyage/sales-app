import os
import duckdb
import openai
import pandas as pd
from dotenv import load_dotenv

# Initialize OpenAI client
load_dotenv()
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# 2. Data Ingestion & SQL Preparation
def process_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Clean, Standardize, Dedupe
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()
    df['date'] = pd.to_datetime(df['date'].str.strip())

    # Handle Store E03 Cost Error (Scale by 1000)
    df.loc[df['store_id'] == 'E03', 'cost'] = df.loc[df['store_id'] == 'E03', 'cost'] * 1000

    # Create DuckDB connection
    con = duckdb.connect(database=":memory:")
    con.register("sales_data", df)
    
    return con, df

# 3. Generate SQL from Plain English Question using OpenAI
def generate_sql(user_question, df_columns):
    system_prompt = f"""You are an expert SQL generator for DuckDB. 
Table name: sales_data
Columns and types: {df_columns}

Rule: Return ONLY a valid executable SQL query. Do not wrap in markdown code blocks like ```sql or add explanations."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a SQL query to answer this question: {user_question}"}
        ],
        temperature=0.0
    )
    
    sql_query = response.choices[0].message.content.strip()
    return sql_query.replace("```sql", "").replace("```", "").strip()

# 4. Generate Natural Language Summary from Query Results
def summarize_results(user_question, df_result):
    data_str = df_result.to_string(index=False)
    
    system_prompt = "You are a concise business analyst. Explain the query results clearly in plain English, highlighting key insights for management."

    user_prompt = f"""User Question: {user_question}

Data Table Result:
{data_str}

Provide a clear executive answer and key takeaways based ONLY on the data above."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )
    
    return response.choices[0].message.content.strip()