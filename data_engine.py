import duckdb
import openai
import pandas as pd
import streamlit as st

# Initialize OpenAI client
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Data Ingestion & SQL Preparation
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

# Generate SQL from Plain English Question using OpenAI
def generate_sql(user_question, df_columns):
    system_prompt = f"""You are an expert SQL generator for DuckDB. 
Table name: sales_data
Columns and types: {df_columns}

Rule: Return ONLY a valid executable SQL query. Do not wrap in markdown code blocks like ```sql or add explanations."""

    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a SQL query to answer this question: {user_question}"}
        ],
    )
    
    sql_query = response.choices[0].message.content.strip()
    return sql_query.replace("```sql", "").replace("```", "").strip()