import json
import os
import re
from src import DEFACTS4J_PATH, PROMPT_TEMPLATE_PATH
from src.prompt.prompt import (
    search_prompts_from_defects4at,
    search_prompts_from_defects4j,
)
from src.prompt.prompt_kind import PromptKind
import tiktoken
import numpy as np
from collections import defaultdict
import jsonlines
import pandas as pd

MAX_TOKEN_SIZE = 16000
FINE_TUNE_PERCENTAGE = 0.05
NUM_FINE_TUNE_SAMPLE_PER_SCENARIO = 20


def organize_prompts_by_project(prompt_paths):
    prompt_paths_by_project = {}
    for prompt_path in prompt_paths:
        matched_prompt = re.search(
            r"prompt_.*?_\d+_(.*?)_v\d+\.txt", os.path.basename(prompt_path)
        )
        if matched_prompt:
            project_id = matched_prompt.group(1)
            if project_id in prompt_paths_by_project:
                prompt_paths_by_project[project_id].append(prompt_path)
            else:
                prompt_paths_by_project[project_id] = [prompt_path]
        else:
            print(f"Fail to find project ID in {prompt_path}!")
    return prompt_paths_by_project


def generate_messages(
    fine_tuning_dataset,
    validation_paths,
    answers,
    system_msg,
    enc,
    num_fine_tuning_samples_per_scenario,
    prompt_paths,
):
    prompt_paths_by_project = organize_prompts_by_project(prompt_paths)
    num_samples = 0
    project_indices = {project_id: 0 for project_id in prompt_paths_by_project}
    # count number of fine-tuning samples from each project
    stats = {project_id: 0 for project_id in prompt_paths_by_project}
    while True:
        for project_id in prompt_paths_by_project:
            current_index = project_indices[project_id]
            if current_index >= len(prompt_paths_by_project[project_id]):
                continue
            project_indices[project_id] += 1
            prompt_path = prompt_paths_by_project[project_id][current_index]
            with open(prompt_path) as prompt_file:
                prompt = prompt_file.read()
            num_tokens = len(enc.encode(prompt))
            if (
                num_tokens < MAX_TOKEN_SIZE
                and num_samples < num_fine_tuning_samples_per_scenario
            ):
                messages = {
                    "messages": [
                        system_msg,
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": json.dumps(answers)},
                    ]
                }
                fine_tuning_dataset.append(messages)
                stats[project_id] += 1
                num_samples += 1
            else:
                validation_paths.append(prompt_path)
                continue
        if all(
            [
                index == len(prompt_paths_by_project[project_id])
                for project_id, index in project_indices.items()
            ]
        ):
            break
    return stats


def create_fine_tuning_dataset(version, projects, source_datasets):
    # Read correct answers
    with open(
        os.path.join(PROMPT_TEMPLATE_PATH, f"answers_v{version}.json")
    ) as answer_file:
        all_answers = json.load(answer_file)
    # Read system message
    with open(os.path.join(PROMPT_TEMPLATE_PATH, "system_message.json")) as f:
        system_msg = json.load(f)
    # Fetch prompts for fine-tuning
    prompt_paths = {}
    for dataset in source_datasets:
        prompt_paths[dataset] = {}
        for prompt_kind in list(PromptKind):
            if dataset == "Defects4J":
                prompt_paths[dataset][prompt_kind] = search_prompts_from_defects4j(
                    prompt_kind, version
                )
            elif dataset == "Defects4AT":
                prompt_paths[dataset][prompt_kind] = search_prompts_from_defects4at(
                    prompt_kind, projects, version
                )
    # Create fine-tuning dataset
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    fine_tuning_dataset = []
    validation_paths = []
    fine_tune_stats = []
    for dataset in prompt_paths:
        # num_fine_tuning_samples_per_scenario = int(
        #     min(
        #         len(prompt_paths[dataset][prompt_kind])
        #         for prompt_kind in list(PromptKind)
        #     )
        #     * FINE_TUNE_PERCENTAGE
        # )
        # print(
        #     f"{num_fine_tuning_samples_per_scenario} samples per scenario from {dataset} for fine-tuning.",
        # )
        for prompt_kind in list(PromptKind):
            stats = generate_messages(
                fine_tuning_dataset,
                validation_paths,
                all_answers[prompt_kind.name],
                system_msg,
                enc,
                NUM_FINE_TUNE_SAMPLE_PER_SCENARIO,
                prompt_paths[dataset][prompt_kind],
            )
            stats["scenario"] = prompt_kind.name.lower()
            stats["dataset"] = dataset
            fine_tune_stats.append(stats)
    pd.DataFrame(fine_tune_stats).to_csv("fine_tuning_stats.csv", index=False)
    check_fine_tuning_dataset(fine_tuning_dataset)
    # Store fine-tuning dataset as JSONL
    with jsonlines.open(f"fine_tuning_dataset_v{version}.jsonl", "w") as writer:
        writer.write_all(fine_tuning_dataset)
    # Store validation dataset
    with open(f"validation_paths_v{version}.json", "w") as f:
        json.dump(validation_paths, f, indent=4)


def check_fine_tuning_dataset(dataset):
    # Copied from https://cookbook.openai.com/examples/chat_finetuning_data_prep
    print("Num examples:", len(dataset))
    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(
                k not in ("role", "content", "name", "function_call", "weight")
                for k in message
            ):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in (
                "system",
                "user",
                "assistant",
                "function",
            ):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

    encoding = tiktoken.get_encoding("cl100k_base")

    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")
        print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > 16385 for l in convo_lens)
    print(
        f"\n{n_too_long} examples may be over the 16,385 token limit, they will be truncated during fine-tuning"
    )

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 16385

    TARGET_EPOCHS = 3
    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    MIN_DEFAULT_EPOCHS = 1
    MAX_DEFAULT_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(
        min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens
    )
    print(
        f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training"
    )
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(
        f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens"
    )

    estimated_costs = n_billing_tokens_in_dataset / 100_000 * 2.4
    print(f"Estimated costs: ${estimated_costs:.2f} with 3 epochs.")
