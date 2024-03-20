import os
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Determine the path to the model in the current directory
current_directory = os.getcwd()
model_path = os.path.join(current_directory, "llama/llama2")

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)


# Function to generate response
@st.cache(allow_output_mutation=True)
def generate_response(input_text):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    response_ids = model.generate(input_ids, max_length=50, num_return_sequences=1, temperature=0.9)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response_text

# Streamlit UI
st.title("Chat with the Chatbot!")

user_input = st.text_input("You:", "")
if st.button("Send"):
    if user_input.strip() != "":
        bot_response = generate_response(user_input)
        st.text("Bot: " + bot_response)
