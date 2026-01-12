#!/bin/bash

model=${OLLAMA_MODEL}
bash /app/extract_archives.sh

# Start Ollama in the background
echo "Starting Ollama server..."
nohup ollama serve > ollama.log 2>&1 &

# Wait for the Ollama server to wake up
echo "Waiting for Ollama to be ready..."
until curl -s http://localhost:11434/api/version > /dev/null; do
    sleep 2
    echo "..."
done

# Pull the LLM
# This checks if the model exists to save time on restarts if using volumes
if ! ollama list | grep -q "$model"; then
    echo "Pulling $model model (this may take a while)..."
    ollama pull "$model"
else
    echo "$model model already present."
fi


# Launch Jupyter Lab
nohup jupyter lab --notebook-dir=/app/DataAnalysis --ip=0.0.0.0 --allow-root \
    --no-browser --NotebookApp.token='' > /app/jupyter.log 2>&1 &

# Print the Access Info
echo "-----------------------------------------------------"
echo "Jupyter is running at: http://localhost:20268/lab"
echo "-----------------------------------------------------"

# Execute the command passed to the docker container
exec "$@"