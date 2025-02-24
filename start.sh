#!/bin/bash
set -e  # Exit on any error

# Run initialization scripts
echo "Running environment tests..."
uv run python rag_service/test_env.py

echo "Initializing RAG pipeline..."
uv run python rag_service/rag_pipeline.py

echo "Generating synthetic data..."
uv run python rag_service/synthetic_data.py

# Start Streamlit app
echo "Starting Streamlit app..."
uv run streamlit run streamlit/app.py --server.address 0.0.0.0 --server.port 7860 