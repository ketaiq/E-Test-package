import json
import os
import re
import time
import ollama
import tiktoken
import subprocess
from pathlib import Path
import shutil
import huggingface_hub
import pandas as pd
from transformers import AutoTokenizer
from src.rag.rag_query_handler import RagQueryHandler
from src.cli.prompt_llm_handler import PromptLlmHandler
from src import PROMPT_DATASET_PATH, PROMPT_TEMPLATE_PATH
from src.prompt.dataset import create_fine_tuning_dataset
from src.prompt.fewshots import generate_few_shots_msg
from src.stats.stats import (
    analyze_answers_from_summary,
    analyze_answers_from_summary_in_binary_classification,
    summarize_prompt_statistics_for_defects4j,
)
from src.llm.chatgpt.chatgpt_api import check_job_status, fine_tune_gpt, prompt_gpt
from src.output.output import (
    create_experiment_folder,
    extract_and_save_results,
    summarize_results,
    write_arguments,
)
from src.prompt.prompt import PromptBuilder


def generate_prompts(args):
    """
    Generates prompts in TXT format with the specified components.
    """
    args.queries.sort(key=lambda x: int(x[1:]))
    pb = PromptBuilder(int(args.version), args.queries)
    pb.generate_prompts_for_defects4at()
    pb.generate_prompts_for_defects4j()


def fine_tune(args):
    """
    Fine-tunes a specified LLM with the curated datatset.
    """
    if args.create_dataset:
        create_fine_tuning_dataset(args.version, args.projects, args.source_datasets)
    if args.submit_job:
        fine_tune_gpt()
    if args.check_status:
        check_job_status()


def prompt_llm(args):
    """
    Prompts an LLM with the generated prompting texts.
    """
    prompt_llm_handler = PromptLlmHandler(args)
    prompt_llm_handler.start_prompting()


def query_llm_with_rag(args):
    """
    Queries an LLM with RAG.
    """
    rag_query_handler = RagQueryHandler(args)
    rag_query_handler.analyze_test_suites()
    rag_query_handler.run_experiments()


def summarize_answers(args):
    version = args.version
    queries = args.queries
    path = args.path
    is_validation = args.validation
    experiment_folder = args.experiment

    if experiment_folder != "":
        # summarize 1 experiment in the path
        summarize_results(experiment_folder, version, path)
        analyze_answers_from_summary(experiment_folder, version, queries, path)
    else:
        # summarize all experiments in the path
        for experiment in os.listdir(path):
            if not os.path.isdir(os.path.join(path, experiment)):
                continue
            print(f"Summarize experiment {experiment}")
            summarize_results(experiment, version, path, is_validation)
            analyze_answers_from_summary(experiment, version, queries, path)
