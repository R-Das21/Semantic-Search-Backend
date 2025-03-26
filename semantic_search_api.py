from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
import os
from openai import OpenAI
import time
from dotenv import load_dotenv
import json
# Load environment variables
load_dotenv()

app = Flask(__name__)
# Modify the CORS configuration like this:
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",  # Local React development
            "https://fastidious-chaja-d056a8.netlify.app/",  # Your production frontend URL
            "*"  # Be cautious with this - only use during initial testing
        ]
    }
})
# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class AcademicSearchEngine:
    def __init__(self, embeddings_path, papers_path):
        """Initialize the academic search engine."""
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is missing. Please set the OPENAI_API_KEY environment variable.")
            
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "gpt-4o"
        self.embedding_model = "text-embedding-ada-002"
        
        # Load data
        print("Loading embeddings and papers...")
        self.embeddings = np.load(embeddings_path)
        self.papers_df = pd.read_csv(papers_path)
        
        # System prompt for academic focus
        self.system_prompt = """You are an academic research assistant. Your role is to help users find relevant academic papers 
        and research materials."""

    def get_embedding(self, text):
        """Get embedding for a text using OpenAI's API."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def find_similar_papers_cosine(self, query_embedding, top_k=1):
        """Find most similar papers using cosine similarity."""
        similarities = cosine_similarity(query_embedding.reshape(1, -1), self.embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(idx, similarities[idx]) for idx in top_indices]

    def find_similar_papers_jensen_shannon(self, query_embedding, top_k=1):
        """Find most similar papers using Jensen-Shannon divergence."""
        # Ensure vectors are non-negative and normalized
        query_norm = np.maximum(query_embedding, 0)
        if np.sum(query_norm) > 0:
            query_norm = query_norm / np.sum(query_norm)
        
        # Calculate JS divergence for each paper embedding
        js_divergences = []
        for paper_embedding in self.embeddings:
            paper_norm = np.maximum(paper_embedding, 0)
            if np.sum(paper_norm) > 0:
                paper_norm = paper_norm / np.sum(paper_norm)
            
            # Calculate Jensen-Shannon divergence
            js_div = jensenshannon(query_norm, paper_norm)
            # Convert to similarity score
            js_similarity = 1.0 - js_div if not np.isnan(js_div) else 0.0
            js_divergences.append(js_similarity)
        
        js_divergences = np.array(js_divergences)
        top_indices = js_divergences.argsort()[-top_k:][::-1]
        return [(idx, js_divergences[idx]) for idx in top_indices]

    def find_similar_papers(self, query_embedding, top_k=3):
        """Find similar papers using both similarity methods."""
        cosine_results = self.find_similar_papers_cosine(query_embedding, top_k)
        js_results = self.find_similar_papers_jensen_shannon(query_embedding, top_k)
        
        return {
            'cosine': cosine_results,
            'jensen_shannon': js_results
        }

    def process_query(self, user_query, filter_option="general"):
        """Process user query and return formatted results."""
        try:
            # Get embedding for the query
            query_embedding = self.get_embedding(user_query)
            if query_embedding is None:
                return {"error": "Sorry, there was an error processing your query. Please try again."}

            # Find similar papers
            similar_papers = self.find_similar_papers(query_embedding)
            
            # Format results for frontend
            papers = []
            
            # Get unique indices from both methods
            indices = set()
            for method_results in similar_papers.values():
                for idx, _ in method_results:
                    indices.add(idx)
            
            # Create a mapping of indices to similarity scores
            similarity_map = {}
            for idx in indices:
                similarity_map[idx] = {
                    'cosine': next((score for i, score in similar_papers['cosine'] if i == idx), None),
                    'jensen_shannon': next((score for i, score in similar_papers['jensen_shannon'] if i == idx), None)
                }
            
            # Format results for each unique paper
            for idx in indices:
                paper = self.papers_df.iloc[idx]
                
                # Calculate combined relevance score (0-100)
                cosine_score = similarity_map[idx]['cosine'] if similarity_map[idx]['cosine'] is not None else 0
                js_score = similarity_map[idx]['jensen_shannon'] if similarity_map[idx]['jensen_shannon'] is not None else 0
                relevance_score = int(((cosine_score + js_score) / 2) * 100)
                
                # Get paper details
                paper_info = {
                    "Title": paper.get('title', f"Paper ID: {idx}"),
                    "Author": paper.get('author', 'Unknown'),
                    "Year": paper.get('year', 'N/A'),
                    "Abstract": paper.get('text', 'N/A')[:500] + "..." if len(paper.get('text', 'N/A')) > 500 else paper.get('text', 'N/A'),
                    "RelevanceScore": relevance_score,
                    "CosineScore": round(cosine_score * 100, 2),
                    "JensenShannonScore": round(js_score * 100, 2)
                }
                papers.append(paper_info)
            
            # Generate analysis using GPT-4
            system_prompt = self.get_system_prompt(filter_option)
            
            # Convert papers to string for GPT
            papers_text = "\n\n".join([
                f"Title: {p['Title']}\nAuthor: {p['Author']}\nYear: {p['Year']}\nAbstract: {p['Abstract']}\nRelevance: {p['RelevanceScore']}/100\nCosine: {p['CosineScore']}\nJensen-Shannon: {p['JensenShannonScore']}"
                for p in papers
            ])
            
            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""Based on the following query and search results, provide a scholarly analysis 
                    of how these papers relate to the query. Be specific about why each paper is relevant.
                    Also compare the effectiveness of Cosine similarity versus Jensen-Shannon similarity for this query.
                    
                    Query: {user_query}
                    
                    Search Results: {papers_text}"""}
                ]
            )
            
            analysis = chat_response.choices[0].message.content
            
            return {
                "papers": papers,
                "analysis": analysis,
                "query": user_query
            }

        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}
    



    def get_system_prompt(self, filter_option):
        """Get system prompt based on filter option."""
        if filter_option == "general":
            return self.system_prompt
        elif filter_option == "recent":
            return self.system_prompt + " Only include academic papers published in the last 5 years."
        elif filter_option == "highly-cited":
            return self.system_prompt + " Focus on highly cited and influential academic papers."
        elif filter_option == "methodology":
            return self.system_prompt + " Focus on research methodology aspects in academic papers."
        elif filter_option == "review":
            return self.system_prompt + " Focus on literature review papers and meta-analyses."
        elif filter_option == "practical":
            return self.system_prompt + " Focus on academic papers with practical applications and implementations."
        else:
            return self.system_prompt + f" Focus on papers in the field of {filter_option}."


# Use relative paths or environment variables for file paths
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "./paper_embeddings.npy")
PAPERS_PATH = os.getenv("PAPERS_PATH", "./papers.csv")

# Initialize search engine when app starts
search_engine = None

@app.route('/api/search', methods=['POST'])
def search():
    global search_engine
    
    # Initialize if not already done
    if search_engine is None:
        try:
            search_engine = AcademicSearchEngine(EMBEDDINGS_PATH, PAPERS_PATH)
            print("Academic Search Engine initialized successfully.")
        except Exception as e:
            error_msg = f"Error initializing search engine: {str(e)}"
            print(error_msg)
            return jsonify({"error": error_msg}), 500
    
    data = request.json
    query = data.get('prompt', '')
    filter_option = data.get('filter', 'general')
    
    if not query or query.strip() == '':
        return jsonify({'error': 'You have not entered any keywords'}), 400
    
    try:
        print(f"Processing query: {query} with filter: {filter_option}")
        results = search_engine.process_query(query, filter_option)
        return jsonify(results)
    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, port=3001)