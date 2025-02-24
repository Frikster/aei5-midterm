# Get a distribution that has uv already installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Add user - this is the user that will run the app
# If you do not set user, the app will run as root (undesirable)
RUN useradd -m -u 1000 user
USER user

# Set the home directory and path
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH        

ENV UVICORN_WS_PROTOCOL=websockets


# Set the working directory
WORKDIR $HOME/app

# Copy the app to the container
COPY --chown=user . $HOME/app

# Install the dependencies
# RUN uv sync --frozen
RUN uv sync

# Add current directory to PYTHONPATH before running scripts
ENV PYTHONPATH=$HOME/app:$PYTHONPATH

RUN uv run python rag_service/test_env.py
RUN uv run python rag_service/rag_pipeline.py
RUN uv run python rag_service/synthetic_data.py

# Expose the port
EXPOSE 7860

# Run the app
CMD ["uv", "run", "streamlit", "run", "streamlit/app.py", "--server.address", "0.0.0.0", "--server.port", "7860"]