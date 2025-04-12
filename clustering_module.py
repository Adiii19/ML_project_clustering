# clustering_module.py

import fitz  # PyMuPDF
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import defaultdict

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def extract_questions_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()

    questions = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"DBMS Question Paper\s*-\s*\d{4}", line, re.IGNORECASE):
            continue
        questions.append(line)
    return questions

def preprocess_questions(questions):
    processed = []
    for q in questions:
        q = q.lower()
        q = re.sub(r'\W+', ' ', q)
        q = " ".join([word for word in q.split() if word not in stop_words])
        processed.append(q)
    return processed

def cluster_questions_kmeans(processed_questions, original_questions, num_clusters=5):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(processed_questions)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(tfidf_matrix)

    clustered = defaultdict(list)
    for idx, label in enumerate(labels):
        clustered[label].append(original_questions[idx])
    
    return list(clustered.values())

def label_clusters_by_rank(clustered_questions):
    # Sort the clusters by the number of questions (size) in descending order
    sorted_clusters = sorted(clustered_questions, key=lambda x: len(x), reverse=True)
    
    # Dynamically assign meaningful names based on cluster ranking
    cluster_names = ["Most Frequently Asked", "Frequently Asked", "Less Frequently Asked", "Rarely Asked", "Asked Only Once"]
    
    labeled = []
    for i, group in enumerate(sorted_clusters):
        # If we have more than the 5 cluster names, we will just name them with "Cluster n"
        if i < len(cluster_names):
            label = cluster_names[i]
        else:
            label = f"Cluster {i+1}"  # Default name if there are more than 5 clusters
        
        # Create a frequency map of questions in this group
        freq_map = {}
        for q in group:
            freq_map[q] = freq_map.get(q, 0) + 1
        
        labeled.append((label, freq_map))
    return labeled

def group_clusters_by_label(labeled_clusters):
    grouped = defaultdict(dict)

    for label, freq_map in labeled_clusters:
        for question, count in freq_map.items():
            if question not in grouped[label]:
                grouped[label][question] = count
            else:
                grouped[label][question] += count

    return grouped
