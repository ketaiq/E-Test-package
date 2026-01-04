FROM python:3.10-slim

# Arguments
# You can override this with -e OLLAMA_MODEL="..." in docker run
ENV OLLAMA_MODEL="llama3.2:1b"

# Set environment variables to prevent Python from buffering stdout
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install curl tar -y

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# Create a Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
# Update PATH so we use the venv by default
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY AutonomicTester AutonomicTester
COPY DataAnalysis DataAnalysis
# COPY ollama_models /root/.ollama/
COPY extract_archives.sh .

# Setup the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
# Open a bash shell to run the specific Python commands manually
CMD ["/bin/bash"]