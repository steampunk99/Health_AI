from flask import Flask, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import pandas as pd
import os
import json

app = Flask(__name__)

# Initialize mistral API
API_KEY = os.environ.get("GROQAPI_KEY")
chat = ChatGroq(temperature=0, groq_api_key=API_KEY, model_name="mixtral-8x7b-32768")

# Load medicine data from JSON
with open('medicine_data.json', 'r') as file:
    medicine_data = json.load(file)

# Load doctor data
with open('professionals.json', 'r') as file:
    doctors_data = json.load(file)

# Load publications data
with open('articles.json', 'r') as file:
    publications_data = json.load(file)

# Combine search index for medicine, doctors, and publications
search_index = []
for doctor in doctors_data:
    search_index.append(doctor['name'] + ' ' + doctor['title'] + ' ' + ' '.join(doctor['specialties']))
search_index.extend([publication['topic'] + ' ' + publication['body'] for publication in publications_data])
search_index.extend([medicine['genericName'] + ' ' + medicine['brandName'] + ' ' + medicine['strength'] + ' ' + medicine['category']  for medicine in medicine_data])


# nlp = spacy.load('/model/encore/en_core_web_sm')
# nlp.to_disk(r'C:\Users\HP ZBOOK G5\Desktop\health-ai\model\encore')


# model = SentenceTransformer('/model/sentence/all-MiniLM-L6-v2')
# model.save(r'C:\Users\HP ZBOOK G5\Desktop\health-ai\model\sentence')

# Load the sentence transformer model
model_path = './model/all-MiniLM-L6-v2'
model = SentenceTransformer(model_path)

# Load the spaCy model
spacy_model_path = './model/encore'
nlp = spacy.load(spacy_model_path)

@app.route('/search', methods=["POST"])
def search_endpoint():
    try:
        # Get the user's query from the request body
        query = request.json.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Lowercase and preprocess the query using spaCy
        query = nlp(query.lower())
        query_tokens = [token.text for token in query if not token.is_stop and not token.is_punct]
        query_embed = model.encode([' '.join(query_tokens)])

        # Compute the cosine similarity between the query and the search index
        search_index_embed = model.encode(search_index)
        sims = [cosine_similarity(query_embed, [doc_embed])[0][0] for doc_embed in search_index_embed]

        # Find the top 5 results
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:5]

        # Get the top results
        top_results = []
        for idx in top_indices:
            if idx < len(doctors_data):
                top_results.append({"type": "doctor", "data": doctors_data[idx]})
            elif idx < len(doctors_data) + len(publications_data):
                top_results.append({"type": "publication", "data": publications_data[idx - len(doctors_data)]})
            else:
                top_results.append({"type": "medicine", "data": medicine_data[idx - len(doctors_data) - len(publications_data)]})

        # Return the search results
        return jsonify({"results": top_results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Dynamic recommendation system 
@app.route('/recommendations', methods=["POST"])
def recommendations_endpoint():
    try:
        # Get the recommendation type from the request body
        rec_type = request.json.get("type")
        if not rec_type or rec_type != "medicines":  # Only handle medicines now
            return jsonify({"error": "Invalid recommendation type"}), 400

        # Get the top 5 medicine recommendations (will probably consider factors like price, effectiveness)
        top_medicine_indices = sorted(range(len(medicine_data)), key=lambda i: medicine_data.iloc[i]['MedicinePrice'])[:5]  # Example: prioritize lower priced meds for now
        top_medicines = medicine_data.iloc[top_medicine_indices].to_dict('records')

        return jsonify({"recommendations": top_medicines})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Define the system message and create the chat prompt template
system = "You are a helpful assistant that only answers medical related questions, if asked anything other than a medical question, politely decline and ask them if they have a medical question"
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

@app.route('/chat', methods=["POST"])
def chat_endpoint():
    try:
        # Get user input from the body
        user_input = request.json.get("message")

        if not user_input:
            return jsonify({"error":"No message provided"}), 400
        
        # Invoke the chain with the user's input
        chain = prompt | chat

        # Stream the response
        response = chain.stream({"text": user_input})

        # Join the streamed response chunks
        response_text = "".join(chunk.content for chunk in response)

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    # Specify the port you want to bind to (e.g., 5000)
    port = int(os.environ.get("PORT", 5000))
    # Run the Flask application, binding it to 0.0.0.0 (all available interfaces) and the specified port
    app.run(host='0.0.0.0', port=port, debug=True)
