from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import re
from openai import OpenAI
import logging
from collections import Counter
import sqlite3
import json
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure OpenAI client
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure database
DATABASE = 'code_database.sqlite'

# Define keywords for different focus areas
FOCUS_AREAS = {
    'Web Development': ['django', 'flask', 'fastapi', 'html', 'css', 'javascript', 'react', 'vue', 'angular'],
    'Data Analysis': ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 'data', 'analysis', 'visualization'],
    'Machine Learning': ['sklearn', 'tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'ai'],
    'Automation': ['selenium', 'beautifulsoup', 'requests', 'scrapy', 'automation', 'bot', 'script'],
    'GUI Development': ['tkinter', 'pyqt', 'kivy', 'wx', 'pyside', 'gui', 'interface'],
    'API Development': ['rest', 'graphql', 'api', 'endpoint', 'microservice'],
    'Database': ['sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'sqlite', 'orm'],
    'Testing': ['pytest', 'unittest', 'mock', 'tdd', 'bdd', 'testing'],
    'DevOps': ['docker', 'kubernetes', 'ci/cd', 'jenkins', 'gitlab', 'github actions'],
    'Security': ['encryption', 'hashing', 'authentication', 'authorization', 'security']
}

def get_embedding(text, model="nomic-ai/nomic-embed-text-v1.5-GGUF"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS code_files
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  content TEXT,
                  embedding TEXT)''')
    conn.commit()
    conn.close()

def store_file(filename, content):
    embedding = get_embedding(content)
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("INSERT INTO code_files (filename, content, embedding) VALUES (?, ?, ?)",
              (filename, content, json.dumps(embedding)))
    conn.commit()
    conn.close()

def search_similar_files(query, top_k=5):
    query_embedding = get_embedding(query)
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT filename, content, embedding FROM code_files")
    results = c.fetchall()
    conn.close()

    if not results:
        return []

    similarities = []
    for filename, content, embedding_str in results:
        embedding = json.loads(embedding_str)
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities.append((filename, content, similarity))

    similarities.sort(key=lambda x: x[2], reverse=True)
    return similarities[:top_k]

def analyze_code(content):
    word_count = Counter(re.findall(r'\b\w+\b', content.lower()))
    focus_areas = {area: sum(word_count[keyword] for keyword in keywords)
                   for area, keywords in FOCUS_AREAS.items()}
    return focus_areas

def get_llm_insights(file_contents, similar_files):
    try:
        similar_files_content = "\n\n".join([f"Similar file: {filename}\n{content[:500]}..." 
                                             for filename, content, _ in similar_files])
        prompt = f"""Analyze this Python project and provide key insights and improvement suggestions:

{file_contents}

Here are some similar files from the project history:

{similar_files_content}

Based on the current files and the similar files from history, provide insights and suggestions."""

        completion = client.chat.completions.create(
            model="lmstudio-community/Phi-3.1-mini-4k-instruct-GGUF",
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in analyzing Python code for ADHD developers. Provide concise, focused insights and suggestions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=750  # Increased token limit to accommodate the additional context
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error in getting LLM insights: {str(e)}")
        return "An error occurred while analyzing the code. Please try again later."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'files' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return jsonify({"error": "No selected file"}), 400

    overall_focus_areas = Counter()
    all_file_contents = ""
    processed_files = []

    try:
        for file in files:
            if file and file.filename.endswith('.py'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                processed_files.append(filename)

                with open(filepath, 'r') as f:
                    content = f.read()
                    all_file_contents += f"# File: {filename}\n{content}\n\n"
                    file_focus_areas = analyze_code(content)
                    overall_focus_areas.update(file_focus_areas)

                # Store file in the database
                store_file(filename, content)

                os.remove(filepath)  # Remove the file after analysis

        if not processed_files:
            return jsonify({"error": "No valid Python files were uploaded"}), 400

        # Normalize focus areas
        total = sum(overall_focus_areas.values())
        normalized_focus_areas = {k: v/total for k, v in overall_focus_areas.items() if v > 0}

        # Search for similar files
        similar_files = search_similar_files(all_file_contents)

        llm_insights = get_llm_insights(all_file_contents, similar_files)

        return jsonify({
            "focus_areas": normalized_focus_areas,
            "llm_insights": llm_insights,
            "processed_files": processed_files,
            "similar_files": [{"filename": filename, "similarity": similarity} for filename, _, similarity in similar_files]
        })

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return jsonify({"error": "An error occurred during analysis"}), 500

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
