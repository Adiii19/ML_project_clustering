import streamlit as st
from clustering_module import (
    extract_questions_from_pdf,
    preprocess_questions,
    group_similar_questions,
    label_clusters_by_frequency,
    group_clusters_by_label
)
import re

st.set_page_config(page_title="Question Frequency Analyzer", layout="wide")

st.title("PYQs Frequency Analyzer 📚")
st.markdown("Upload one or more **PDFs** of DBMS question papers to identify frequently asked questions based on **semantic similarity and frequency**.")

uploaded_files = st.file_uploader("Upload PDF(s) here", type="pdf", accept_multiple_files=True)

# Similarity threshold slider
st.sidebar.subheader("Semantic Grouping Parameters")
similarity_threshold = st.sidebar.slider("Similarity Threshold", min_value=0.5, max_value=0.95, value=0.8, step=0.01)


# Optional: light cleaner for frontend display
def clean_for_display(q):
    q = re.sub(r'\s+', ' ', q).strip()
    return q


if uploaded_files:
    all_questions = []

    for uploaded_file in uploaded_files:
        questions = extract_questions_from_pdf(uploaded_file)
        all_questions.extend(questions)

    if all_questions:
        st.success(f"Extracted {len(all_questions)} questions from uploaded PDFs.")

        # Use original questions for semantic grouping to preserve full content
        groups = group_similar_questions(all_questions, similarity_threshold=similarity_threshold)
        labeled_clusters = label_clusters_by_frequency(groups)
        grouped_results = group_clusters_by_label(labeled_clusters)

        # Display grouped results
        for label, questions in grouped_results.items():
            with st.expander(f"{label} ({len(questions)} questions)"):
                for q, count in questions:
                    st.markdown(f"- **[{count} times]** {clean_for_display(q)}")
