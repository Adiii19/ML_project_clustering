import fitz
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import pandas as pd  # ✅ Added for CSV export
from sentence_transformers import SentenceTransformer
# Use a pipeline as a high-level helper
from transformers import pipeline


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def extract_questions_from_pdf(pdf_file):
    

    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    doc.close()

    # Normalize whitespace
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    lines = text.split('\n')
    merged_lines = []
    buffer = ""

    # Adjusted pattern: recognizes question start with digit + dot + space
    question_start_pattern = re.compile(r'^\s*\d{1,3}\.\s')

    for line in lines:
        line = line.strip()
        if question_start_pattern.match(line):
            if buffer:
                merged_lines.append(buffer.strip())
            buffer = line
        else:
            # Append continuation lines regardless of content
            buffer += ' ' + line

    if buffer:
        merged_lines.append(buffer.strip())

    # Final cleanup to remove headers
    questions = []
    for line in merged_lines:
        if re.search(r"(question\s*paper|dbms|past\s*year|university|exam|semester)", line, re.IGNORECASE):
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

def group_similar_questions(original_questions, similarity_threshold=0.8):
    # Encode original questions directly (without stopword removal or lowercasing)
    embeddings = model.encode(original_questions, convert_to_tensor=True)

    clusters = []
    used = set()

    for idx, emb in enumerate(embeddings):
        if idx in used:
            continue

        cluster = [idx]
        used.add(idx)

        for jdx in range(idx + 1, len(embeddings)):
            if jdx in used:
                continue

            sim = util.pytorch_cos_sim(emb, embeddings[jdx]).item()
            if sim >= similarity_threshold:
                cluster.append(jdx)
                used.add(jdx)

        clusters.append(cluster)

    grouped_questions = []
    for cluster in clusters:
        grouped = [original_questions[i] for i in cluster]
        grouped_questions.append(grouped)

    return grouped_questions

           
def label_clusters_by_frequency(grouped_questions):
    labeled = []
    for group in grouped_questions:
        rep_question = group[0]
        frequency = len(group)

        if frequency >= 4:
            label = "🔴 Frequently Asked"
        elif frequency >= 2:
            label = "🟠 Occasionally Asked"
        else:
            label = "🟢 Rarely Asked"

        labeled.append((label, rep_question, frequency, group))
    return labeled


def group_clusters_by_label(labeled_clusters):
    grouped = defaultdict(list)

    for label, question, count, group in labeled_clusters:
        grouped[label].append((question, count, group))

    # ✅ Export to CSV
    rows = []
    for label, questions in grouped.items():
        for question, count, group in questions:
            rows.append({
                "Cluster": label,
                "Question": question,
                "Count": count
            })

    df = pd.DataFrame(rows)
    df.to_csv("clustered_questions2.csv", index=False)

    return grouped

def generate_pca_plot(original_questions, grouped_questions):
    labels = []
    texts = []
    for idx, group in enumerate(grouped_questions):
        for q in group:
            texts.append(q)
            labels.append(idx)

    embeddings = model.encode(texts, convert_to_tensor=True)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings.cpu().numpy())

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title("PCA of Question Embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    return plt