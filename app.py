from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import google.generativeai as genai
import numpy as np
import nltk
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.DEBUG)

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
semantic_model.to(device)

with open("csvjson.json", "r", encoding='utf-8') as f:
    faculty_data = json.load(f)

faculty_embeddings1 = {}

for faculty in faculty_data:
    faculty_keywords = faculty["Keywords"]
    embedding = semantic_model.encode([faculty_keywords])[0]  # Encode as a list
    faculty_embeddings1[faculty["Name"]] = embedding

torch.save(faculty_embeddings1, 'faculty_embeddings.pt')

try:
    faculty_embeddings = torch.load('faculty_embeddings.pt')
    logging.debug("Loaded faculty embeddings from faculty_embeddings.pt")
except FileNotFoundError as e:
    logging.error("faculty_embeddings.pt not found. Please make sure the file exists.")
    raise e

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    logging.error("GEMINI_API_KEY is not set in the environment variables.")
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel('gemini-1.0-pro')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/available_tags', methods=['GET'])
def available_tags():
    try:
        tags = set()
        for faculty in faculty_data:
            keywords = faculty["Keywords"]
            clean_keywords = [kw.strip().strip("'[]") for kw in keywords.split(',')]
            tags.update(clean_keywords)
        return jsonify(list(tags))
    except Exception as e:
        logging.error("Error in available_tags: %s", str(e))
        return jsonify({'error': str(e)}), 500



@app.route('/expand_description', methods=['POST'])
def find_faculties():
    try:
        description = request.form['description']
        project_description = expand_project_description(description)
        matching_faculties = find_matching_faculties(project_description, faculty_data, faculty_embeddings)
        return jsonify(matching_faculties)
    except Exception as e:
        logging.error("Error in find_faculties: %s", str(e))
        return jsonify({'error': str(e)}), 500

@app.route('/search_by_keywords', methods=['POST'])
def search_by_keywords():
    try:
        keywords = request.form['keywords']
        # Directly encode keywords and find matching faculties
        query_embedding = semantic_model.encode([keywords])[0]
        matching_faculties = find_matching_faculties_from_keywords(query_embedding, faculty_data, faculty_embeddings)
        return jsonify(matching_faculties)
    except Exception as e:
        logging.error("Error in search_by_keywords: %s", str(e))
        return jsonify({'error': str(e)}), 500

def expand_project_description(description):
    prompt = f"'{description}'\n generate as much as possible very in-depth research topics/tags, research keywords for this research project related to the core research area."
    response = gemini_model.generate_content(prompt)
    return response.text

def find_matching_faculties(prompt, faculty_data, faculty_embeddings):
    try:
        logging.debug("Encoding query prompt")
        query_embedding = semantic_model.encode([prompt])[0]
        faculty_matches = []
        for faculty_name, faculty_embedding in faculty_embeddings.items():
            similarity_score = cosine_similarity([query_embedding], [faculty_embedding])[0][0]
            faculty_info = next(faculty for faculty in faculty_data if faculty["Name"] == faculty_name)
            faculty_matches.append({
                "Name": faculty_info["Name"],
                "SimilarityScore": float(similarity_score),
                "Department": faculty_info["Department"],
                "Profile Picture": faculty_info["Profile Picture"],
                "Profile link": faculty_info["Profile link"],
                "Office location": faculty_info["Office location"],
                "Email ID": faculty_info["Email ID"]
            })
        logging.debug("Sorting faculty matches")
        faculty_matches.sort(key=lambda x: x["SimilarityScore"], reverse=True)
        return faculty_matches[:10]
    except Exception as e:
        logging.error("Error in find_matching_faculties: %s", str(e))
        raise

def find_matching_faculties_from_keywords(query_embedding, faculty_data, faculty_embeddings):
    try:
        faculty_matches = []
        for faculty_name, faculty_embedding in faculty_embeddings.items():
            similarity_score = cosine_similarity([query_embedding], [faculty_embedding])[0][0]
            faculty_info = next(faculty for faculty in faculty_data if faculty["Name"] == faculty_name)
            faculty_matches.append({
                "Name": faculty_info["Name"],
                "SimilarityScore": float(similarity_score),
                "Department": faculty_info["Department"],
                "Profile Picture": faculty_info["Profile Picture"],
                "Profile link": faculty_info["Profile link"],
                "Office location": faculty_info["Office location"],
                "Email ID": faculty_info["Email ID"]
            })
        logging.debug("Sorting faculty matches")
        faculty_matches.sort(key=lambda x: x["SimilarityScore"], reverse=True)
        return faculty_matches[:10]
    except Exception as e:
        logging.error("Error in find_matching_faculties_from_keywords: %s", str(e))
        raise

if __name__ == '__main__':
    app.run(debug=True)
