from flask import Flask, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os


app = Flask(__name__)

#Initialize mistral chat api
API_KEY = os.environ.get("GROQAPI_KEY")
chat = ChatGroq(temperature=0, groq_api_key=API_KEY, model_name="mixtral-8x7b-32768")

# Define the system message and create the chat prompt template
system = "You are a helpful assistant that only answers medical related questions, if asked anything other than a medical question, politely decline and ask them if they have a medical question"
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

@app.route('/chat', methods=["POST"])
def chat_endpoint():
    try:
        #get user input from the body
        user_input = request.json.get("message")

        if not user_input:
            return jsonify({"error":"No message provided"}), 400
        
        # invoke the chain with the user's input
        chain = prompt | chat

        # stream the response
        response = chain.stream({"text": user_input})

        # join the streamed response chunks
        response_text = "".join(chunk.content for chunk in response)

        return jsonify({"response":response_text})

    except Exception as e:
        return jsonify({"error":str(e)}), 500
    
if __name__ == "__main__":
    # Specify the port you want to bind to (e.g., 5000)
    port = int(os.environ.get("PORT", 5000))
    # Run the Flask application, binding it to 0.0.0.0 (all available interfaces) and the specified port
    app.run(host='0.0.0.0', port=port, debug=True)

    