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
RUN . .venv/bin/activate && uv pip install streamlit

# Add current directory to PYTHONPATH before running scripts
ENV PYTHONPATH=$HOME/app:$PYTHONPATH

# Make start script executable
RUN chmod +x start.sh

# Expose the port
EXPOSE 7860

# Use the start script as the entry point
# CMD ["./start.sh"]
# # Run the app
CMD ["uv", "run" ,"streamlit", "run", "streamlit/app.py", "--server.address", "0.0.0.0", "--server.port", "7860"]