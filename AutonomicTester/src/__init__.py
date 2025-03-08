# This is a placeholder for any initialization code.
# You can define any variables, functions, or classes that should be available when this module is imported.
import os

LLAMA3_CONTEXT_LIMIT = 8 * 1024

BUGS_PATH = os.path.join("Defect4AutonomicTesting", "bugs")
EXPERIMENT_RESULTS_PATH = os.path.join("AutonomicTester", "experiment_results")
PROMPT_TEMPLATE_PATH = "AutonomicTester/src/prompt/template"
CONFIG_PATH = os.path.join("AutonomicTester", "config.json")
DEFACTS4J_PATH = "Defects4jDataset"
DEFACTS4J_PROMPT_PATH = os.path.join(DEFACTS4J_PATH, "prompts")
FEW_SHOTS_PATH = "AutonomicTester/src/prompt/fewshots"
PROMPT_DATASET_PATH = "PromptDataset"
FINE_TUNE_LLM_VALIDATION_PATH = os.path.join("FineTuneLLM", "validation_paths_v4_q3diff5p.json")