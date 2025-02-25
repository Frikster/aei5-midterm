from pathlib import Path
import streamlit as st
import os
import json

from rag_service.rag_pipeline import GrantSummaryPipeline
from rag_service.data_loader import load_grants_from_airtable
from rag_service.synthetic_data import GrantEvaluationGenerator
from rag_service.evaluate import evaluate_ragas

@st.cache_resource
def get_pipeline(vector_store_path) -> GrantSummaryPipeline:
    """Create and cache the pipeline instance. 
    This will only run once per session and handle cleanup automatically."""
    pipeline = GrantSummaryPipeline(
        persist_dir=str(vector_store_path),
        force_disable_check_same_thread=True
    )
    return pipeline

def main():
    st.title("EA Funds Grant Summary Generator")

    vector_store_path = Path(__file__).parent / "vector_store"
    if 'HF_SPACE' in os.environ:
        vector_store_path = Path("/data/vector_store")  # HF Spaces persistent storage
    
    try:
        pipeline = get_pipeline(vector_store_path)
        
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
            
            st.success("Loaded grants into vector store.")
        else:
            st.info(f"Vector store contains {collection_info.points_count} points")
        
        st.session_state.pipeline = pipeline
        
        # Add evaluation section
        st.write("## Evaluation")
        eval_col1, eval_col2 = st.columns(2)
        
        with eval_col1:
            if st.button("Generate Synthetic Dataset"):
                with st.spinner("Generating synthetic dataset..."):
                    try:
                        # Load grants directly from Airtable
                        grants = load_grants_from_airtable()
                        
                        generator = GrantEvaluationGenerator(rag_pipeline=pipeline)
                        evaluation_examples = generator.create_evaluation_dataset(grants)
                        
                        # Save dataset
                        synthetic_output_path = Path(__file__).parent / "evaluation_dataset.json"
                        with synthetic_output_path.open("w") as f:
                            json.dump(
                                [vars(example) for example in evaluation_examples],
                                f,
                                indent=2
                            )
                        st.success(f"Generated and saved {len(evaluation_examples)} evaluation examples!")
                    except Exception as e:
                        st.error(f"Error generating dataset: {str(e)}")
        
        with eval_col2:
            if st.button("Run Evaluation"):
                with st.spinner("Running evaluation..."):
                    try:
                        # Try to load from file
                        dataset_path = Path(__file__).parent / "evaluation_dataset.json"
                        if not dataset_path.exists():
                            st.error("No evaluation dataset found. Generate one first!")
                            return
                        
                        # Run RAGAS evaluations with and without privacy check
                        ragas_results = evaluate_ragas(rag_pipeline=pipeline)
                        ragas_results_without_privacy = evaluate_ragas(rag_pipeline=pipeline, skip_privacy=True)
                        
                        # Convert results to pandas DataFrames
                        results_df = ragas_results.to_pandas()
                        results_no_privacy_df = ragas_results_without_privacy.to_pandas()
                        
                        ragas_eval_output_path = Path(__file__).parent / "ragas_eval_output_path.json"
                        results_df.to_json(ragas_eval_output_path, orient="records", indent=2)
                        print(f"\nSaved detailed results to {ragas_eval_output_path}")
                        
                        ragas_eval_no_privacy_output_path = Path(__file__).parent / "ragas_eval_no_privacy_output_path.json"
                        results_no_privacy_df.to_json(ragas_eval_no_privacy_output_path, orient="records", indent=2)
                        print(f"\nSaved detailed results to {ragas_eval_no_privacy_output_path}")
                        
                        # Display results by fund
                        st.write("### RAGAS Evaluation Results")
                        detailed_tabs = st.tabs(["With Privacy", "Without Privacy"])
                                
                        with detailed_tabs[0]:
                            for metric_name, metric_value in ragas_results._repr_dict.items():
                                st.metric(metric_name, f"{metric_value:.3f}")
                        
                        with detailed_tabs[1]:
                            for metric_name, metric_value in ragas_results_without_privacy._repr_dict.items():
                                st.metric(metric_name, f"{metric_value:.3f}")
                        
                    except Exception as e:
                        st.error(f"Error running evaluation: {str(e)}")

    except Exception as e:
        st.error(f"Error initializing pipeline: {str(e)}")
        return
    
    # Input for grant ID
    grant_id = st.text_input("Enter Grant ID")
    
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                # summary = st.session_state.pipeline.generate_summary(grant_id=grant_id)
                summary = st.session_state.pipeline.run_agent(grant_id=grant_id)
                st.write("### Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error generating summary: {str(e)}")

if __name__ == "__main__":
    main()
