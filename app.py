from pathlib import Path
import streamlit as st
import os

from rag_service.rag_pipeline import GrantSummaryPipeline

def main():
    st.title("EA Funds Grant Summary Generator")

    vector_store_path = Path(__file__).parent / "vector_store"
    if 'HF_SPACE' in os.environ:
        vector_store_path = Path("/data/vector_store")  # HF Spaces persistent storage
    
    # Use st.session_state to maintain pipeline instance across reruns
    if 'pipeline' not in st.session_state:
        try:
            pipeline = GrantSummaryPipeline(
                persist_dir=str(vector_store_path),
                force_disable_check_same_thread=True
            )
            
            # Check if collection is empty and load grants if needed
            collection_info = pipeline.vectorstore.client.get_collection("grants")
            if collection_info.points_count == 0:
                st.info("Vector store is empty. Loading grants... This may take a few minutes.")
                
                # Create a placeholder for progress updates
                progress_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                # Override print to also write to Streamlit
                original_print = print
                def custom_print(*args, **kwargs):
                    # Call the original print
                    original_print(*args, **kwargs)
                    # Write to Streamlit
                    progress_placeholder.write(" ".join(map(str, args)))
                
                # Custom tqdm class for Streamlit
                class StreamlitTqdm:
                    def __init__(self, iterable, desc=None):
                        self.iterable = iterable
                        self.total = len(iterable)
                        self.desc = desc
                        self.current = 0
                        
                    def __iter__(self):
                        for obj in self.iterable:
                            yield obj
                            self.current += 1
                            if self.desc:
                                progress_placeholder.write(f"{self.desc}: {self.current}/{self.total}")
                            progress_bar.progress(self.current / self.total)
                
                # Temporarily replace print and tqdm
                import builtins
                builtins.print = custom_print
                import rag_service.rag_pipeline
                rag_service.rag_pipeline.tqdm = StreamlitTqdm
                
                try:
                    pipeline.load_grants()
                finally:
                    # Restore original print function and tqdm
                    builtins.print = original_print
                    rag_service.rag_pipeline.tqdm = __import__('tqdm').tqdm
                    progress_bar.empty()
                
                st.success(f"Loaded grants into vector store.")
            else:
                st.info(f"Vector store contains {collection_info.points_count} points")
            
            st.session_state.pipeline = pipeline
            
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
