# clustering_module.py

import fitz  # PyMuPDF
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

def cluster_similar_questions(processed_questions, original_questions, similarity_threshold=0.8):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(processed_questions)
    cosine_sim = cosine_similarity(tfidf_matrix)

    clustered = []
    used = set()

    for i in range(len(cosine_sim)):
        if i in used:
            continue
        group = [original_questions[i]]
        used.add(i)
        for j in range(i + 1, len(cosine_sim)):
            if cosine_sim[i][j] > similarity_threshold and j not in used:
                group.append(original_questions[j])
                used.add(j)
        clustered.append(group)
    return clustered

def label_clusters(clustered_questions):
    labeled = []
    for group in clustered_questions:
        count = len(group)
        if count >= 5:
            label = "Most Frequently Asked"
        elif count >= 4:
            label = "Frequently Asked"
        elif count == 3:
            label = "Less Frequently Asked"
        elif count == 2:
            label = "Rarely Asked"
        else:
            label = "Asked Only Once"
        # Deduplicate inside the group
        labeled.append((label, list(dict.fromkeys(group))))
    return labeled

def group_by_labels(labeled_clusters):
    grouped = defaultdict(list)
    for label, group in labeled_clusters:
        grouped[label].extend(group)
    # Remove duplicates again across clusters
    for key in grouped:
        grouped[key] = list(dict.fromkeys(grouped[key]))
    return grouped
 