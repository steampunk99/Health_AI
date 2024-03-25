import streamlit as st
import ollama

st.title("💬 MediBot👩🏿‍⚕️")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, how may I assist you?"}]

### Write Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

## Generator for Streaming Tokens
def generate_response():
    response = ollama.chat(model='llama2', stream=True, messages=st.session_state.messages)
    assistant_name = "MediBot"  # Replace with your desired name

    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token

        # Check if the user is asking for the assistant's name
        user_input = st.session_state.messages[-1]["content"].lower()
        if "name" in user_input and ("your" in user_input or "you" in user_input):
            st.session_state["full_message"] += f"\n\nMy name is {assistant_name}."

        yield token

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)
    st.session_state["full_message"] = ""
    st.chat_message("assistant", avatar="🤖").write_stream(generate_response)
    st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})