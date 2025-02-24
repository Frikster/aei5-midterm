import sys
from pathlib import Path
import streamlit as st
from qdrant_client import QdrantClient
import os

# Add the parent directory to Python path so we can import from rag_service
sys.path.append(str(Path(__file__).parent.parent))

from rag_service.rag_pipeline import GrantSummaryPipeline

def main():
    st.title("EA Funds Grant Summary Generator")
    
    # Modify vector store path for Hugging Face deployment
    vector_store_path = Path(__file__).parent.parent / "vector_store"
    
    # Add a check for HF_SPACE environment variable
    if 'HF_SPACE' in os.environ:
        vector_store_path = Path("/data/vector_store")  # HF Spaces persistent storage
    
    # Use st.session_state to maintain pipeline instance across reruns
    if 'pipeline' not in st.session_state:
        try:
            st.session_state.pipeline = GrantSummaryPipeline(
                persist_dir=str(vector_store_path),
                force_disable_check_same_thread=True  # Allow multiple threads to access Qdrant
            )
        except Exception as e:
            st.error(f"Error initializing pipeline: {str(e)}")
            return
    
    # Input for grant ID
    grant_id = st.text_input("Enter Grant ID")
    
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                summary = st.session_state.pipeline.generate_summary(grant_id)
                st.write("### Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

if __name__ == "__main__":
    main()
