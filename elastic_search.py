from elasticsearch import Elasticsearch
import os, json
from txtai.embeddings import Embeddings
import streamlit as st

with open('articles.json', 'r') as f:
    articles = json.load(f)

with open('professionals.json', 'r') as f:
    professionals = json.load(f)

with open('medicine_data.json', 'r') as f:
    medicines = json.load(f)


es = Elasticsearch()
es.indices.create(index='combined_index', ignore=400)

for doc in medicines:
    es.index(index='combined_index', id=doc['genericName'], body=doc)
for doc in professionals:
    es.index(index='combined_index', id=doc['name'], body=doc)
for doc in articles:
    es.index(index='combined_index', id=doc['topic'], body=doc)

embeddings = Embeddings(method="transformers", model="sentence-transformers/all-mpnet-base-v2")

def search_documents(query, top_k=10):
    # Generate query embedding
    query_embedding = embeddings.embed([query])[0]

    # Perform vector similarity search in Elasticsearch
    search_body = {
        "query": {
            "knn": {
                "document_embedding": {
                    "vector": query_embedding.tolist(),
                    "k": top_k
                }
            }
        }
    }

    response = es.search(index="combined_index", body=search_body)
    return [hit['_source'] for hit in response['hits']['hits']]

def main():
    st.title("Semantic Search with Elasticsearch and txtai")

    query = st.text_input("Enter your search query:")

    if st.button("Search"):
        results = search_documents(query)
        st.write("Search results:")
        for result in results:
            if 'genericName' in result:
                st.write(f"Medicine: {result['genericName']} ({result['brandName']})")
                st.write(f"Strength: {result['strength']} {result['units']}")
                st.write(f"Price: ${result['price']/100:.2f}")
                st.write(f"Category: {result['category']}")
                st.write("---")
            elif 'name' in result:
                st.write(f"Professional: {result['name']}")
                st.write(f"Title: {result['title']}")
                st.write(f"Association: {result['association']}")
                st.write(f"Consultation Fee: ${result['consultationFee']}")
                st.write("---")
            elif 'topic' in result:
                st.write(f"Article: {result['topic']}")
                st.write(f"Published: {result['publishedDate']}")
                st.write(f"Read Time: {result['readTime']}")
                st.write(f"Author: {result['author']}")
                st.write("---")

if __name__ == "__main__":
    main()

