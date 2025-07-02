import streamlit as st

st.set_page_config(page_title="Financial Chatbot", page_icon="📈")
st.title("Financial News Chatbot")
st.write("Ask questions about current financial markets and news")

with st.spinner("Initializing AI models..."):
    # import the AI agent modules
    from ai_agent_source_code import base_llm
    from ai_agent_source_code import base_rags
    from ai_agent_source_code import ai_agent_1
    from ai_agent_source_code import ai_agent_2
    from ai_agent_source_code import ai_agent_3

# dictionary mapping display names directly to functions
model_mapping = {
    "Basic LLM Output": base_llm,
    "RAGs Output": base_rags,
    "Base Advanced Agentic RAG": ai_agent_1,
    "Advanced Agentic RAG + Summarizer of the generated answer": ai_agent_2,
    "Advanced Agentic RAG + Summarizer of the retrieved documents": ai_agent_3
}

# create a dropdown with display names
display_name = st.selectbox(
    "Select AI Model:",
    options=list(model_mapping.keys()),
    index=0
)

# create input for user question
user_question = st.text_input("Enter your question related to finance news:")

# generate answer button
if st.button("Generate Answer"):
    if user_question.strip():
        with st.spinner(f"Processing with {display_name}..."):
            # get the function directly from the mapping and call it
            result = model_mapping[display_name](user_question)
            
            # display the result
            st.success(f"Answer from {display_name}:")
            st.write(result)

    else:
        st.warning("Please enter a question before generating an answer")