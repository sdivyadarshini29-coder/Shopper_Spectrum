# Shopper Spectrum: Retail Customer Analytics & AI Recommendation Engine

An enterprise-grade data science workflow combining unsupervised machine learning (K-Means Clustering) for RFM behavioral segmentation and vector-space similarity mapping (Cosine Similarity) for item-based collaborative product recommendations.

## 📊 Systems Architecture
* **Phase 1: Statistical EDA Pipeline** - Tracking underlying distribution characteristics of purchase datasets.
* **Phase 2: RFM Feature Matrix Engineering** - Transforming transactional rows into Recency, Frequency, and Monetary mathematical scores.
* **Phase 3: Machine Learning Segmentation** - Calibrating 4 explicit shopper persona clusters via KMeans using Elbow and Silhouette constraints.
* **Phase 4: Collaborative Recommendations Engine** - Deploying dense vector matching using Cosine Similarity matrices over product stock pairs.

## 🛠️ Technology Stack
* **Core Language:** Python 3.12
* **Analytics Layer:** Pandas, NumPy, Scikit-Learn
* **Application Framework:** Streamlit (Native Dashboard Environment)

## 🚀 Execution & Local Launch Parameters
1. Initialize virtual environment bounds: `.\venv\Scripts\activate`
2. Launch the user interface instance via terminal core:
   ```bash
   streamlit run streamlit_app/app.py