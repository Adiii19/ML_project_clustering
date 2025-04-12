# app.py

import streamlit as st
from clustering_module import (
    extract_questions_from_pdf,
    preprocess_questions,
    cluster_questions_kmeans,  # updated import
    label_clusters_by_rank,  # updated import to use the new method
    group_clusters_by_label  # updated import to use the new method
)

st.set_page_config(page_title="Question Frequency Analyzer", layout="wide")

st.title("PYQs Analyzer📚")
st.markdown("Upload one or more **PDFs** of DBMS question papers to identify and group questions based on how often they appear using **K-Means clustering**.")

uploaded_files = st.file_uploader("Upload PDF(s) here", type="pdf", accept_multiple_files=True)

# Optional: Let user choose number of clusters
num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=5, value=5)

if uploaded_files:
    all_questions = []

    for uploaded_file in uploaded_files:
        questions = extract_questions_from_pdf(uploaded_file)
        all_questions.extend(questions)

    if all_questions:
        processed = preprocess_questions(all_questions)

        # Use KMeans clustering now
        clusters = cluster_questions_kmeans(processed, all_questions, num_clusters=num_clusters)
        labeled_clusters = label_clusters_by_rank(clusters)  # Get dynamic cluster labels by rank
        grouped_results = group_clusters_by_label(labeled_clusters)  # Group clusters by their labels
        
        st.success(f"Processed {len(all_questions)} questions across {len(uploaded_files)} file(s).")
        
        # Dynamically display the clusters
        for label in grouped_results:
            with st.expander(f"🔷 {label} ({len(grouped_results[label])} questions)"):
                for q in grouped_results[label]:
                    st.markdown(f"- {q}")
