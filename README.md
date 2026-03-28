# AHR-Search: Adaptive Hybrid Retrieval Engine for Academic Papers 🔍

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg.svg)](YOUR_DEPLOYMENT_LINK)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AHR-Search** is a high-performance, intelligent search engine designed to bridge the gap between keyword-based search and semantic understanding in academic repositories. By integrating **BM25 (Lexical)** and **Transformer-based (Semantic)** retrieval, it provides a dynamic weighting mechanism to improve search precision.

---

## 🚀 Key Features
- **Live ArXiv Integration:** Fetches the latest research papers directly from Cornell University's repository.
- **Adaptive Hybrid Scoring:** Dynamically balances lexical and semantic weights ($\alpha$-adaptive) based on query information density.
- **Transformers-powered:** Uses `all-MiniLM-L6-v2` for deep semantic context understanding.
- **Statistical Normalization:** Implements `RobustScaler` to ensure score alignment between disparate retrieval models.

---

## 🧬 Mathematical Core
The core innovation of this project is the **Adaptive Alpha Fusion**. The final relevance score $S$ is calculated as:

$$Score_{Final} = \alpha_q \cdot \hat{S}_{lex} + (1 - \alpha_q) \cdot \hat{S}_{sem}$$

Where $\alpha_q$ is a sigmoid-gated function of the query's **Average Inverse Document Frequency (AIDF)**, ensuring that rare technical terms are prioritized via lexical matching while general concepts are handled semantically.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/AHR-Search.git](https://github.com/YOUR_USERNAME/AHR-Search.git)
   cd AHR-Search