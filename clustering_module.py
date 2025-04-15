import fitz  # PyMuPDF
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# Setup
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load sentence embedding model once
model = SentenceTransformer('all-MiniLM-L6-v2')

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

        # Skip common titles or headers
        if re.search(r"(question\s*paper|dbms|past\s*year|university|exam|semester)", line, re.IGNORECASE):
            continue

        # Only include actual questions (those ending with a question mark)
        if not line.endswith("?"):
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
    # Compute sentence embeddings
    embeddings = model.encode(original_questions, convert_to_tensor=True)

    # Cluster based on cosine similarity
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

    # Format clusters as list of question groups
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

        labeled.append((label, rep_question, frequency))
    return labeled

def group_clusters_by_label(labeled_clusters):
    grouped = defaultdict(list)

    for label, question, count in labeled_clusters:
        grouped[label].append((question, count))

    return grouped
