# E-Test-package

This replication package can be used to fully replicate the results of our `E-TEST` paper.

## Repository Structure
- **AutonomicTester:** A Python application designed to implement advanced techniques for E-TEST.
- **DataAnalysis:** A set of Jupyter notebooks to analyze results and compute evaluation metrics.
- **Archives:** A set of tar archives of datasets of prompts and responses from LLMs.

## Environment Setup
### Prerequisites
- Linux or macOS with at least 8 GB of RAM
- [Python 3.10+](https://www.python.org/downloads/)
- [Ollama](https://ollama.com/)

Run following commands from project root directory:

### Install Python Dependencies
```sh
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r ./requirements.txt
```

### Extract Archives
```sh
bash extract_archives.sh
```

## Getting Started

### Data Analysis
To reproduce evaluation results shown in the paper, please run notebooks directly with the created `.venv` as the kernel.
- `dataset_stats.ipynb` computes statistics about the dataset of prompts, which corresponds to **Section 4.2 Dataset** and **Table 2** in the paper.
- `evaluation.ipynb`
    - computes precision, recall, F1-score for each scenario, the overall accuracy and the average F1-score of E-TEST, which corresponds to **Table 3** in the paper.
    - calculates average F1-scores for different combinations of queries, which corresponds to **Table 5** in the paper.
    - summarizes time efficiency of different LLMs, which corresponds to **Table 6** in the paper.
- `benchmark.ipynb` compares E-TEST with Gazzola et al.'s approach, which corresponds to **Table 4** in the paper.

### Autonomic Tester
To start an experiment from scratch, please run *AutonomicTester* with appropriate arguments.
Here we explain how to reproduce results of Llama3 8B, which is feasible to run on a local machine with at least 8 GB of RAM. For other LLMs mentioned in the paper, please check the help message via `python AutonomicTester/main.py -h`.

#### Step 1: Install and Start Ollama

For Linux
```sh
# Install Ollama
sudo curl -L https://github.com/ollama/ollama/releases/download/v0.1.48/ollama-linux-amd64 -o /usr/bin/ollama
sudo chmod +x /usr/bin/ollama
# Start Ollama Server
nohup ollama serve > ollma.log 2>&1 &
```

For macOS, please download Ollama directly from https://ollama.com/ and start it from Launchpad.

#### Step 2: Pull Llama3 8B
```sh
ollama pull llama3:8b
```

#### Step 3: Prompt Llama3 8B
Before prompting, please export your [Hugging Face](https://huggingface.co/) user access token as an environment variable
```sh
export HUGGING_FACE_API_KEY=YOUR_USER_ACCESS_TOKEN
```

```sh
# Test Llama3 8B with prompts generated from error-prone scenarios in Defects4J
python AutonomicTester/main.py prompt -v 4 -d Defects4J -m LLama3_8B -s BUGGY
# Test Llama3 8B with prompts generated from safe-not-yet-tested scenarios in Defects4J
python AutonomicTester/main.py prompt -v 4 -d Defects4J -m LLama3_8B -s FIXED
# Test Llama3 8B with prompts generated from already-tested scenarios in Defects4J
python AutonomicTester/main.py prompt -v 4 -d Defects4J -m LLama3_8B -s SIMILAR

# Test Llama3 8B with prompts generated from error-prone scenarios in the mined dataset from GitHub
python AutonomicTester/main.py prompt -v 4 -d Defects4AT -m LLama3_8B -s BUGGY
# Test Llama3 8B with prompts generated from safe-not-yet-tested scenarios in the mined dataset from GitHub
python AutonomicTester/main.py prompt -v 4 -d Defects4AT -m LLama3_8B -s FIXED
# Test Llama3 8B with prompts generated from already-tested scenarios in the mined dataset from GitHub
python AutonomicTester/main.py prompt -v 4 -d Defects4AT -m LLama3_8B -s SIMILAR
```

#### Step 4: Summarize Results
Replace `EXPERIMENT_FOLDER_NAME` with the one generated in `AutonomicTester/experiment_results`
```sh
python AutonomicTester/main.py summarize -v 4 -e {EXPERIMENT_FOLDER_NAME}
```
Check following files in the experiment folder for data analysis:
- `summary.json` contains answers of 5 queries from the LLM for each prompt.
- `scenario_votes.csv` contains the predicted scenario for each prompt. The columns *bug id* and *project* indicate the source of the prompt. The columns *buggy*, *fixed* and *similar* store the vote for each scenario. The column *max* stores the maximum vote.
- `statistics.csv` contains the time costs of the LLM's response and the prompt size in terms of characters and tokens.