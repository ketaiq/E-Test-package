# Stage 1: Java Source
FROM adoptopenjdk/openjdk8:jre AS java-source

# Stage 2: E-Test Image
FROM python:3.10-slim

# Build Arguments
ENV OLLAMA_MODEL="llama3.2:1b"
# This is required only when using other LLMs
ENV HUGGING_FACE_API_KEY=""

# Configure Python environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TRANSFORMERS_VERBOSITY="error"

# Install system dependencies required for Defects4J and Perl modules
RUN apt-get update && apt-get install -y \
    curl unzip pciutils vim subversion \
    maven cpanminus \
    git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Java 8 from source stage
COPY --from=java-source /opt/java/openjdk /opt/java/openjdk
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install Ollama binary
RUN curl -fsSL https://ollama.com/install.sh | sh
COPY ollama_models /root/.ollama/

# Install and initialize Defects4J
RUN git clone https://github.com/rjust/defects4j.git /defects4j
WORKDIR /defects4j
RUN git checkout tags/v2.1.0
RUN cpanm --installdeps .
RUN ./init.sh
RUN rm -rf .git
ENV PATH="/defects4j/framework/bin:${PATH}"

WORKDIR /app

# Create and activate Python virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Configure entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]