from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy

app = Flask(__name__)

# Load dataset
df = pd.read_csv(r"C:\Users\jithi\Desktop\DataNeuron_DataScience_Task1\DataNeuron_DataScience_Task1\DataNeuron_Text_Similarity.csv")

# Initialize SpaCy
nlp = spacy.load("en_core_web_lg")

# Function to preprocess text
def text_processing(sentence):
    """
    Lemmatize, lowercase, remove numbers and stop words
    """
    sentence = [token.lemma_.lower()
                for token in nlp(sentence) 
                if token.is_alpha and not token.is_stop]
    return ' '.join(sentence)

# Function to calculate cosine similarity
def cos_sim(sentence1_emb, sentence2_emb):
    cos_sim = cosine_similarity(sentence1_emb, sentence2_emb)
    return np.diag(cos_sim)

# Initialize TF-IDF Vectorizer
model = TfidfVectorizer(lowercase=True, stop_words='english')

# Train the model
X_train = pd.concat([df['text1'], df['text2']]).unique()
model.fit(X_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity', methods=['POST'])
def similarity():
    # Get data from POST request
    data = request.form
    
    # Preprocess text
    text1_processed = text_processing(data['text1'])
    text2_processed = text_processing(data['text2'])
    
    # Transform text into embeddings
    sentence1_emb = model.transform([text1_processed])
    sentence2_emb = model.transform([text2_processed])
    
    # Calculate similarity score
    similarity_score = cos_sim(sentence1_emb, sentence2_emb)[0]
    
    return render_template('result.html', similarity_score=similarity_score)

if __name__ == '__main__':
    app.run(debug=True)
