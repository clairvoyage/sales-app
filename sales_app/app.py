import duckdb
import openai
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(page_title="AI Management Assistant", layout="wide")
st.title("📊 Management CSV Data Assistant")

# 2. Sidebar API Key Input & Data Upload
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not api_key:
    st.info("Please enter your OpenAI API Key in the sidebar to start.")
    st.stop()

# Initialize OpenAI client
client = openai.OpenAI(api_key=api_key)

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

# 4. Generate SQL from Plain English Question using OpenAI
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

# 5. Generate Natural Language Summary from Query Results
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

# 6. UI & Chat Engine
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
            # Step A: Generate SQL using OpenAI
            sql_query = generate_sql(user_prompt, dict(df.dtypes))
            result_df = con.execute(sql_query).df()

            # Step B: Generate Plain English Interpretation using OpenAI
            summary_text = summarize_results(user_prompt, result_df)

            # Format Response
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