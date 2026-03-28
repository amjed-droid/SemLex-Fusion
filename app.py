import streamlit as st
import pandas as pd
import numpy as np
import urllib.request
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(page_title="SemLex-Fusion", layout="wide")
st.title(" SemLex-Fusion")
st.markdown("About An intelligent hybrid search engine that combines BM25 and Transformers with an adaptive weighting mechanism for real-time academic paper retrieval from ArXiv.")


@st.cache_data(show_spinner=False)
def fetch_real_papers(topic, max_results=50):
    """Fetches the latest research papers from Cornell's ArXiv API based on a query."""
    query = topic.replace(' ', '+')
    url = f'http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={max_results}'
    
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        papers = []
        
        for entry in root.findall('atom:entry', ns):
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
            papers.append({"title": title, "abstract": summary})
            
        return pd.DataFrame(papers)
    except Exception as e:
        st.error(f"Error fetching data from ArXiv: {e}")
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()


st.sidebar.header("1️ Step 1: Build Knowledge Base")
topic = st.sidebar.text_input("Enter a scientific domain (e.g., Quantum Computing):", value="Data Science")

if st.sidebar.button(" Fetch Latest 50 Papers"):
    with st.spinner('Connecting to ArXiv and fetching papers...'):
        st.session_state['df'] = fetch_real_papers(topic)
        st.sidebar.success("Papers fetched successfully!")


if 'df' in st.session_state and not st.session_state['df'].empty:
    df = st.session_state['df']
    
    with st.expander(" View Recently Fetched Papers"):
        st.dataframe(df['title'])

    @st.cache_resource(show_spinner=False)
    def prepare_engines(dataframe):
        dense_embeddings = model.encode(dataframe['abstract'].tolist())
        tokenized_corpus = [abstract.lower().split() for abstract in dataframe['abstract']]
        bm25 = BM25Okapi(tokenized_corpus)
        return dense_embeddings, bm25

    dense_embeddings, bm25 = prepare_engines(df)

    st.write("---")
    st.header("2️ Step 2: Smart Semantic Search")
    
    alpha = st.slider(
        "Hybrid Weight (Alpha α)", 
        min_value=0.0, max_value=1.0, value=0.7, step=0.1,
        help="1.0 = Semantic Meaning Only (AI) | 0.0 = Exact Keyword Match Only (BM25)"
    )
    
    search_query = st.text_input("💡 Search for a specific concept:", placeholder="e.g., handling missing values in neural networks")

   
    if search_query:
        with st.spinner('Performing semantic analysis...'):
            # 1. Dense Score (Cosine Similarity)
            query_embedding = model.encode([search_query])
            dense_scores = cosine_similarity(query_embedding, dense_embeddings)[0]
            
            # 2. Sparse Score (BM25)
            tokenized_query = search_query.lower().split()
            sparse_scores = bm25.get_scores(tokenized_query)
            
            # 3. Normalization (Min-Max Scaling)
            scaler = MinMaxScaler()
            sparse_scores_scaled = scaler.fit_transform(sparse_scores.reshape(-1, 1)).flatten()
            dense_scores_scaled = scaler.fit_transform(dense_scores.reshape(-1, 1)).flatten()
            
            # 4. Final Hybrid Score
            hybrid_scores = (alpha * dense_scores_scaled) + ((1 - alpha) * sparse_scores_scaled)
            
            # 5. Retrieve Top 3 indices
            top_indices = np.argsort(hybrid_scores)[::-1][:3]
            
            st.subheader(" Top Matches:")
            for idx in top_indices:
                score = hybrid_scores[idx] * 100
                st.markdown(f"**Title:** {df.iloc[idx]['title']}")
                # Display only the first 500 characters of the abstract for a cleaner UI
                st.info(f"**Abstract:** {df.iloc[idx]['abstract'][:500]}...") 
                st.caption(f"Match Score: **{score:.1f}%** | (Semantic: {dense_scores_scaled[idx]:.2f} - Keyword: {sparse_scores_scaled[idx]:.2f})")
                st.divider()