import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load or Train Model
MODEL_PATH = "clustering_model.pkl"
SCALER_PATH = "scaler.pkl"

def train_and_save_model(df, selected_features, k):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[selected_features])
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    
    # Save model & scaler
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(kmeans, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    
    return kmeans, scaler

def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            kmeans = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return kmeans, scaler
    except FileNotFoundError:
        return None, None

# Streamlit App Title
st.title("Customer Segmentation App")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data")
    st.dataframe(df.head())
    
    # Select Features
    selected_features = st.multiselect("Select features for clustering", df.columns, default=df.columns.tolist())
    
    if selected_features:
        k = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)
        
        # Load or Train Model
        kmeans, scaler = load_model()
        if kmeans is None or scaler is None:
            kmeans, scaler = train_and_save_model(df, selected_features, k)
            st.write("Trained and saved a new model.")
        
        df_scaled = scaler.transform(df[selected_features])
        df["Cluster"] = kmeans.predict(df_scaled)
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        df_pca = pca.fit_transform(df_scaled)
        df["PCA1"] = df_pca[:, 0]
        df["PCA2"] = df_pca[:, 1]
        
        # Show Clustered Data
        st.write("### Clustered Data")
        st.dataframe(df)
        
        # Plot Clusters
        fig = px.scatter(df, x="PCA1", y="PCA2", color=df["Cluster"].astype(str), title="Customer Segments")
        st.plotly_chart(fig)
        
        # Download Results
        st.download_button("Download Segmentation Results", df.to_csv(index=False).encode("utf-8"), "segmentation_results.csv", "text/csv")n recs.to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
