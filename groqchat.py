import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the Mixtral ChatGroq API
chat = ChatGroq(temperature=0, groq_api_key="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", model_name="mixtral-8x7b-32768")

# Define the system message and create the ChatPromptTemplate
system = "You are a very smart AI assistant that has vast knowledge in science and you answer only medical / science related questions."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

# Streamlit UI
def main():
    st.title("💊 MediBot")

    # Initialize messages if not present in session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you with your medical queries?"}]

    # Write Message History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
        else:
            st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

    # Get user input
    user_input = st.chat_input("Enter your message:")

    # Process user input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Invoke the chain with the user's input
        chain = prompt | chat


        # Stream the response
        response = chain.stream({"text": user_input})

        # Join the streamed response chunks
        response_text = "".join(chunk.content for chunk in response)


        st.session_state.messages.append({"role": "assistant", "content": response_text})

        # Display the response
        st.chat_message("assistant", avatar="🤖").write(response_text)


if __name__ == "__main__":
    main()
