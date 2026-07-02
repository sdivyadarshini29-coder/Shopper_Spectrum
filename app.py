import streamlit as st
import pandas as pd
import os

# Set up clean professional project dashboard configurations
st.set_page_config(
    page_title="Shopper Spectrum - Analytics Architecture",
    layout="wide",
    page_icon="🛍️"
)

# Header Module
st.title("🛍️ SHOPPER SPECTRUM ENGINE")
st.caption("Enterprise Customer Segmentation & Collaborative Product Recommendations Platform")
st.markdown("---")

# Data File Paths Linker
if os.path.exists("data/customer_segments_data.csv"):
    segments_path = "data/customer_segments_data.csv"
    similarity_path = "data/product_similarity_matrix.csv"
else:
    segments_path = "notebooks/data/customer_segments_data.csv"
    similarity_path = "notebooks/data/product_similarity_matrix.csv"

if os.path.exists(segments_path) and os.path.exists(similarity_path):
    rfm_df = pd.read_csv(segments_path)
    similarity_df = pd.read_csv(similarity_path, index_col=0)
    
    # Navigation Matrix Control Link Panel
    st.sidebar.title("🛰️ CORE TERMINAL")
    app_mode = st.sidebar.selectbox(
        "Navigate Project Lifecycle Panels:", 
        [
            "📈 Phase 1: Exploratory Data Analysis (EDA)", 
            "👥 Phase 2: ML Customer Segmentation", 
            "🎯 Phase 3: AI Product Recommendation System"
        ]
    )
    st.sidebar.markdown("---")
    st.sidebar.success("System Link Status: ONLINE")
    st.sidebar.metric(label="Total Analyzed Profiles", value=f"{len(rfm_df):,}")

    # ==================== PANEL 1: EXPLORATORY DATA ANALYSIS (EDA) ====================
    if "EDA" in app_mode:
        st.header("📈 Exploratory Data Analysis (EDA) Insights Log")
        st.write("Reviewing underlying behavioral distribution insights from statistical data layers.")
        
        # Section A: Statistical Descriptive Logs
        st.subheader("📋 Preprocessed RFM Statistical Summary")
        st.write("Summary logs outlining Recency (days since purchase), Frequency (visit counts), and Monetary (total spend value).")
        st.dataframe(rfm_df[['Recency', 'Frequency', 'Monetary']].describe(), use_container_width=True)
        
        st.markdown("---")
        
        # Section B: Layout breakdown for key feature metrics distributions
        st.subheader("📊 Feature Scale Matrix View")
        col_rec, col_freq, col_mon = st.columns(3)
        
        with col_rec:
            st.markdown("**Recency (Days) Metrics Summary**")
            st.write(f"🔹 **Average Recency:** {rfm_df['Recency'].mean():.1f} Days")
            st.write(f"🔹 **Maximum Inactive Period:** {rfm_df['Recency'].max()} Days")
            st.bar_chart(rfm_df['Recency'].head(150), height=200)
            
        with col_freq:
            st.markdown("**Frequency (Purchase Counts) Summary**")
            st.write(f"🔹 **Average Purchase Hits:** {rfm_df['Frequency'].mean():.1f} Times")
            st.write(f"🔹 **Highest Customer Orders:** {rfm_df['Frequency'].max()} Orders")
            st.bar_chart(rfm_df['Frequency'].head(150), height=200)
            
        with col_mon:
            st.markdown("**Monetary (Revenue Spend) Summary**")
            st.write(f"🔹 **Average Transaction Basket:** £{rfm_df['Monetary'].mean():.2f}")
            st.write(f"🔹 **Peak Enterprise Customer Spend:** £{rfm_df['Monetary'].max():,.2f}")
            st.bar_chart(rfm_df['Monetary'].head(150), height=200)

    # ==================== PANEL 2: CUSTOMER SEGMENTATION ====================
    elif "Customer Segmentation" in app_mode:
        st.header("👥 Machine Learning Cluster Personas (KMeans)")
        st.write("Segment profiles calibrated using standard metric scaling optimizations.")
        
        # Metrics summary cards row
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="Configured Engine Clusters", value="4 Segments", delta="KMeans Optimized")
        with m_col2:
            st.metric(label="Evaluated Feature Matrix Range", value="RFM Standardized", delta="100% Variance Balanced")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        pane_left, pane_right = st.columns([2, 3])
        with pane_left:
            st.subheader("📊 Cluster Volume Distribution")
            segment_counts = rfm_df['Customer_Segment'].value_counts().reset_index()
            segment_counts.columns = ['Customer Persona Group', 'Active Volume Count']
            st.dataframe(segment_counts, use_container_width=True, hide_index=True)
            
        with pane_right:
            st.subheader("🔍 Filter Segment Profiles")
            selected_segment = st.selectbox("Select Target Cluster to Analyze:", rfm_df['Customer_Segment'].unique())
            filtered_data = rfm_df[rfm_df['Customer_Segment'] == selected_segment]
            st.dataframe(
                filtered_data[['CustomerID', 'Recency', 'Frequency', 'Monetary']].head(100), 
                use_container_width=True, 
                hide_index=True
            )

    # ==================== PANEL 3: PRODUCT RECOMMENDATION SYSTEM ====================
    elif "Recommendation System" in app_mode:
        st.header("🎯 AI Product Recommendation Core")
        st.write("Dynamic product association indexing via dense vector space Cosine Similarities.")
        
        # Central selector box
        available_products = similarity_df.index.tolist()
        selected_product = st.selectbox("⚡ Search System Inventory Database (Type item name):", available_products)
        
        if selected_product:
            st.markdown(f"#### 🚀 Top 5 Correlation Vectors for: **`{selected_product}`**")
            
            recommendations = similarity_df[selected_product].sort_values(ascending=False).iloc[1:6]
            rec_df = pd.DataFrame({
                'System Target Recommended Item': recommendations.index,
                'Mathematical Match Score (Cosine)': recommendations.values.round(4)
            })
            
            st.dataframe(rec_df, use_container_width=True, hide_index=True)

else:
    st.error("Matrix Configuration Error: Required component datasets are missing from directory channels.")