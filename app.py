import streamlit as st
from data_engine import process_data, generate_sql

# Page Configuration
st.set_page_config(page_title="AI Management Assistant", layout="centered")
st.title("AI Management Assistant")

# UI & Chat Engine
uploaded_file = 'sales_data.csv'

if "messages" not in st.session_state:
    st.session_state.messages = []

con, df = process_data(uploaded_file)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "df" in msg:
            st.dataframe(msg["df"])

# Handle User Query
if user_prompt := st.chat_input("Ask a question about your sales data..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        try:
            # Generate SQL using OpenAI
            sql_query = generate_sql(user_prompt, dict(df.dtypes))
            result_df = con.execute(sql_query).df()

            # Format Response
            response_markdown = f"""

---
**SQL Executed:** `{sql_query}`"""

            st.markdown(response_markdown)
            st.dataframe(result_df)

            st.session_state.messages.append({
                "role": "assistant",
                "df": result_df
            })

        except Exception as e:
            error_msg = f"Unable to interpret question: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})