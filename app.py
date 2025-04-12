# app.py

import streamlit as st
from clustering_module import extract_questions_from_pdf, preprocess_questions, cluster_similar_questions, label_clusters, group_by_labels

st.set_page_config(page_title="Question Frequency Analyzer", layout="wide")

st.title("📚 Question Frequency Analyzer (Clustering-based)")
st.markdown("Upload one or more **PDFs** of DBMS question papers to identify and group questions based on how often they appear.")

uploaded_files = st.file_uploader("Upload PDF(s) here", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_questions = []

    for uploaded_file in uploaded_files:
        questions = extract_questions_from_pdf(uploaded_file)
        all_questions.extend(questions)

    if all_questions:
        processed = preprocess_questions(all_questions)
        clusters = cluster_similar_questions(processed, all_questions)
        labeled_clusters = label_clusters(clusters)
        grouped_results = group_by_labels(labeled_clusters)

        st.success(f"Processed {len(all_questions)} questions across {len(uploaded_files)} file(s).")
        
        label_order = [
            "Most Frequently Asked",
            "Frequently Asked",
            "Less Frequently Asked",
            "Rarely Asked",
            "Asked Only Once"
        ]

        for label in label_order:
            if label in grouped_results:
                with st.expander(f"🔷 {label} ({len(grouped_results[label])} questions)"):
                    for q in grouped_results[label]:
                        st.markdown(f"- {q}")
