import streamlit as st
import ollama
import pypdf2
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader

st.title("💬 MediBot👩🏿‍⚕️")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi, how may I assist you?"}]

### Write Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

# Add a file uploader for PDF files
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Store the uploaded file data in the session state
if uploaded_file is not None:
    st.session_state["pdf_data"] = uploaded_file.getvalue()
else:
    st.session_state["pdf_data"] = None

def parse_pdf_and_create_index(pdf_data):
    # Extract text from the PDF file
    pdf_reader = pypdf2.PdfReader(pdf_data)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    # Create a vector index from the PDF text
    documents = SimpleDirectoryReader('pdf_text').load_data([text])
    index = GPTVectorStoreIndex.from_documents(documents)

    return index

# Check if a PDF file is uploaded
if "pdf_data" in st.session_state and st.session_state["pdf_data"] is not None:
    # Parse the PDF file and create a vector index
    pdf_index = parse_pdf_and_create_index(st.session_state["pdf_data"])
else:
    pdf_index = None

## Generator for Streaming Tokens
def generate_response():
    response = ollama.chat(model='llama2', stream=True, messages=st.session_state.messages)
    assistant_name = "MediBot"

    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token

        # Check if the user is asking for the assistant's name
        user_input = st.session_state.messages[-1]["content"].lower()
        if "name" in user_input and ("your" in user_input or "you" in user_input):
            st.session_state["full_message"] += f"\n\nMy name is {assistant_name}."

        # Check if the user's query is related to the PDF
        if pdf_index is not None and ("pdf" in user_input or "document" in user_input):
            query_result = pdf_index.query(user_input)
            st.session_state["full_message"] += f"\n\nRelevant information from the PDF:\n{query_result}"

        yield token

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)
    st.session_state["full_message"] = ""
    st.chat_message("assistant", avatar="🤖").write_stream(generate_response)
    st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})