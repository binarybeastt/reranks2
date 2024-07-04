import streamlit as st
from pinecone import Pinecone
import re
import pandas as pd
from openai import OpenAI
from pinecone_text.sparse import BM25Encoder

# Read the data
df = pd.read_csv('keyword_included.csv')

# Initialize BM25Encoder
bm25 = BM25Encoder()
bm25.fit(df['text'])

# Initialize Pinecone
pc = Pinecone(api_key="3661dc2a-3710-4669-a187-51faaa0cc557")
index = pc.Index("semantic-rag-chunking4")

import cohere
co = cohere.Client('HW8yZAl7aYmOzWKd3LVWi87bXrwomxKec1cLBL3k')

# Define OpenAI client
client = None  # Initialize to None until user provides API key

# Define the Streamlit application
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    hsparse = {'indices': sparse['indices'], 'values': [v * (1 - alpha) for v in sparse['values']]}
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def main():
    st.title("Pinecone Search Application")
    
    search_text = st.text_input("Enter search text:")
    search_intent = st.text_input("Enter your search intent")
    
    openai_api_key = st.text_input("Enter OpenAI API Key:", type="password")
    global client
    client = OpenAI(api_key=openai_api_key)

    if st.button("Search"):
        if search_text:          
            dense = get_embedding(search_intent)
            sparse = bm25.encode_queries(search_text)
            hdense, hsparse = hybrid_scale(dense, sparse, alpha=0)

            query_result = index.query(
                top_k=5,
                vector=hdense,
                sparse_vector=hsparse,
                include_metadata=True
            )
            docs = {x["metadata"]['text']: i for i, x in enumerate(query_result["matches"])}
            rerank_docs = co.rerank(query=search_text, documents=docs.keys(), top_n=10, model="rerank-english-v2.0")
            reranked_docs = []
            for i, doc in enumerate(rerank_docs):
                rerank_i = docs[doc.document["text"]]
                print(str(i)+"\t->\t"+str(rerank_i))
                if i != rerank_i:
                    reranked_docs.append(f"[{rerank_i}]\n"+doc.document["text"])
            results = []
            for text in reranked_docs:
                id = int(text.split(']')[0].strip('['))  
                original_data = query_result["matches"][id]  
                results.append({
                    "text": text.split(']')[1].strip(),  
                    "id": original_data["metadata"]["id"] 
                })
            ids = [d['id'] for d in results]
            filtered_df = df[df['id'].isin(ids)]
            filtered_df['id'] = pd.Categorical(filtered_df['id'], categories=ids, ordered=True)
            filtered_df = filtered_df.sort_values('id')

            match_ids =[m['metadata']['id'] for m in query_result['matches']]
            filtered_df2 = df[df['id'].isin(match_ids)]
            attributes_to_extract = ['text', 'url', 'id', 'title', 'keyword']
            extracted_df = filtered_df2[attributes_to_extract]

            st.text('Unranked Results')
            st.table(pd.DataFrame(filtered_df))
            st.text('Reranked Results')
            st.table(pd.DataFrame(extracted_df))

if __name__ == "__main__":
    main()
