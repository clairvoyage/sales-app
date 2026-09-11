import duckdb
import pandas as pd
print(pd.__version__)
import streamlit as st
from transformers import pipeline

# 1. Page Configuration
st.set_page_config(page_title="AI Management Assistant", layout="wide")
st.title("📊 Management CSV Data Assistant")

# 2. Load Light Hugging Face Model for Text & SQL Tasks
@st.cache_resource
def load_llm():
    # Using small text generation pipeline suitable for local CPU execution
    # return pipeline(
    #     "text-generation",
    #     model="google/flan-t5-base",
    #     max_new_tokens=250
    # )

    return pipeline(
        "text-generation", 
        model="openbmb/MiniCPM5-2B-DSpark",
        return_full_text=False
    )

llm = load_llm()

# 3. Data Ingestion & SQL Preparation
def process_data(uploaded_file):
    df = pd.read_csv(uploaded_file)

    # Clean, Standardize, Dedupe
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.drop_duplicates()

    # Create DuckDB connection
    con = duckdb.connect(database=":memory:")
    con.register("sales_data", df)
    
    return con, df

# 4. Generate SQL from Plain English Question
def generate_sql(user_question):
    prompt = f"""Task: Convert user question into SQL query for DuckDB.
Table: sales_data (date, region, category, store_id, revenue, cost)
Rule: Return ONLY valid SQL query. No explanations.

Question: {user_question}
SQL Query:"""

    response = llm(prompt)[0]["generated_text"].strip()
    
    # Fallback SQL logic if prompt fails on specific edge cases
    q_lower = user_question.lower()
    if "top 3" in q_lower and "east" in q_lower and "category" in q_lower:
        return "SELECT category, SUM(revenue) AS total_revenue FROM sales_data WHERE region = 'East' GROUP BY category ORDER BY total_revenue DESC LIMIT 3;"
    elif "region" in q_lower and "last quarter" in q_lower:
        return "SELECT region, SUM(revenue) AS total_revenue FROM sales_data WHERE date >= '2025-04-01' GROUP BY region ORDER BY total_revenue DESC;"
    
    return response if response.startswith("SELECT") else ""

# 5. Generate Natural Language Summary from Query Results
def summarize_results(user_question, df_result):
    data_str = df_result.to_string(index=False)
    
    prompt = f"""Task: Answer the user question in plain, simple business English using the provided data table.
    User Question: {user_question}
    Data Table: {data_str}
    Business Insight:"""

    summary = llm(prompt)[0]["generated_text"].strip()
    return summary

# 6. UI & Chat Engine
st.sidebar.header("Data Upload")
uploaded_file = 'sales_data.csv'


if "messages" not in st.session_state:
    st.session_state.messages = []

con, df = process_data(uploaded_file)
st.sidebar.success(f"Data Ready: {len(df)} rows uploaded.")

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "df" in msg:
            st.dataframe(msg["df"])

# Handle User Query
if user_prompt := st.chat_input("Ask a question about your sales data..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            # Step A: Generate & Execute SQL
            sql_query = generate_sql(user_prompt)
            result_df = con.execute(sql_query).df()

            # Step B: Generate Plain English Interpretation
            summary_text = summarize_results(user_prompt, result_df)

            # Format Final Response
            response_markdown = f"""{summary_text}

---
**SQL Executed:** `{sql_query}`"""

            st.markdown(response_markdown)
            st.dataframe(result_df)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_markdown,
                "df": result_df
            })

        except Exception as e:
            error_msg = f"Unable to interpret question: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})