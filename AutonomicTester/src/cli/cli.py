import argparse

from src.rag.index_format import IndexFormat
from src.cli.command import fine_tune, generate_prompts, prompt_llm, query_llm_with_rag, summarize_answers
from src.llm.llm_kind import LLMKind
from src.prompt.prompt_kind import PromptKind
from src import EXPERIMENT_RESULTS_PATH

# Shared parser varilables
version_parser = argparse.ArgumentParser(add_help=False)
version_parser.add_argument(
    "-v",
    "--version",
    choices=["0", "4"],
    help="the version of prompting template, including the corresponding questions",
    required=True,
)

dataset_parser = argparse.ArgumentParser(add_help=False)
dataset_parser.add_argument(
    "-d",
    "--dataset",
    choices=["Defects4J", "Defects4AT", "Validation", "SpringBoot"],
    help="a target Java defect dataset",
    required=True,
)

project_parser = argparse.ArgumentParser(add_help=False)
project_parser.add_argument(
    "-p",
    "--projects",
    nargs="*",
    default=["spring-boot", "shardingsphere", "dolphinscheduler", "micrometer"],
    help="a list of target projects separated by space (e.g., spring-boot shardingsphere dolphinscheduler)",
)

query_parser = argparse.ArgumentParser(add_help=False)
query_parser.add_argument(
    "-q",
    "--queries",
    nargs="+",
    default=["Q1", "Q2", "Q3", "Q4", "Q5"],
    help="the selection of queries"
)

main_args_parser = argparse.ArgumentParser(
    prog="Autonomic Tester",
    description="This program generates prompting texts from extracted testing components, prompts Large Language Models (LLMs) with predefined questions, and summarize answers from LLMs.",
)
subparsers = main_args_parser.add_subparsers()

# Functionality for generating prompts
parser_generate = subparsers.add_parser(
    "generate",
    parents=[version_parser, project_parser, query_parser],
    help="generate prompting texts from extracted testing components and save to the prompts folder",
)
parser_generate.set_defaults(func=generate_prompts)

# Functionality for fine-tuning

parser_finetune = subparsers.add_parser(
    "finetune",
    parents=[version_parser, project_parser],
    help="fine-tune GPT-3.5 Turbo",
)
parser_finetune.add_argument(
    "--create-dataset",
    action="store_true",
    help=f"create a JSON fine-tuning dataset using 10% from both Defects4J and Defects4AT",
)
parser_finetune.add_argument(
    "--submit-job",
    action="store_true",
    help=f"submit fine-tuning job",
)
parser_finetune.add_argument(
    "--check-status",
    action="store_true",
    help=f"check status of fine-tuning job",
)
parser_finetune.add_argument(
    "-s",
    "--source-datasets",
    default=["Defects4J"],
    help=f"source dataset for splitting",
)

parser_finetune.set_defaults(func=fine_tune)

# Functionality for prompting LLMs
parser_prompt = subparsers.add_parser(
    "prompt",
    parents=[version_parser, dataset_parser, project_parser, query_parser],
    help="prompt LLMs with predefined questions",
)
parser_prompt.add_argument(
    "-m",
    "--model",
    choices=[llm.name for llm in LLMKind],
    help=f"an LLM chosen to prompt. {LLMKind.generate_help_msg()}",
    required=True,
)
parser_prompt.add_argument(
    "-s",
    "--scenario",
    choices=[scenario.name for scenario in PromptKind],
    help=f"a testing scenario to prompt. {PromptKind.generate_help_msg()}",
    required=True,
)
parser_prompt.add_argument(
    "--few-shots",
    help=f"an integer number that specifies few-shots learning, default using zero-shot",
    default=0
)
parser_prompt.add_argument(
    "-t",
    "--temperature",
    help=f"a float number that influences the LLM's output (higher is more creative, lower is more coherent)",
    default=0.75,
)
parser_prompt.add_argument(
    "--format",
    choices=["jsonline", "txt"],
    default="txt",
    help=f"output format of LLM responses",
)
parser_prompt.set_defaults(func=prompt_llm)

# Functionality for querying LLMs with RAG
parser_rag_query = subparsers.add_parser(
    "ragquery",
    parents=[version_parser, dataset_parser, project_parser, query_parser],
    help="query LLMs with RAG",
)
parser_rag_query.add_argument(
    "-m",
    "--model",
    choices=[llm.name for llm in LLMKind],
    help=f"an LLM chosen to prompt. {LLMKind.generate_help_msg()}",
    required=True,
)
parser_rag_query.add_argument(
    "-s",
    "--scenario",
    choices=[scenario.name for scenario in PromptKind],
    help=f"a testing scenario to prompt. {PromptKind.generate_help_msg()}",
    required=True,
)
parser_rag_query.add_argument(
    "-f",
    "--index-format",
    choices=[index_format.name for index_format in IndexFormat],
    help="an index format for RAG.",
    required=True,
)
parser_rag_query.add_argument(
    "-t",
    "--temperature",
    help=f"a float number that influences the LLM's output (higher is more creative, lower is more coherent)",
    default=0.75,
)
parser_rag_query.set_defaults(func=query_llm_with_rag)

# Functionality for summarizing answers from LLMs
parser_summarize = subparsers.add_parser(
    "summarize",
    parents=[version_parser, query_parser],
    help="summarize answers from the LLM",
)
parser_summarize.add_argument(
    "-e",
    "--experiment",
    help=f"a target experiment folder in AutonomicTester/experiment_results/ to summarize results",
    default="",
)
parser_summarize.add_argument(
    "--validation",
    help=f"consider only prompts for validation",
    action="store_true",
)
parser_summarize.add_argument(
    "-p",
    "--path",
    help=f"a path to the target experiment folder",
    default=EXPERIMENT_RESULTS_PATH,
)
parser_summarize.set_defaults(func=summarize_answers)
