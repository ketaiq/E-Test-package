"""
This file contains the main function to save the output and plot it.
"""

import json
import logging
import re
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from src.llm.llm_kind import LLMKind
from src.prompt.prompt_kind import PromptKind
from src import EXPERIMENT_RESULTS_PATH, PROMPT_TEMPLATE_PATH, FINE_TUNE_LLM_VALIDATION_PATH


def create_experiment_folder(model: LLMKind, scenario, dataset, timestamp, use_rag: bool = False):
    """
    Create a folder to store the experiment results.

    Parameters
    ----------
    - model : LLMKind
    - scenario : PromptKind
    """
    if use_rag:
        folder_name = f"{timestamp}_RAG_{model.value}_{dataset}_{scenario.name.lower()}"
    else:
        folder_name = f"{timestamp}_{model.value}_{dataset}_{scenario.name.lower()}"
    folder_path = os.path.join(
        EXPERIMENT_RESULTS_PATH, folder_name
    )
    # Create the directory
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def find_complete_json_strings(long_string):
    json_strings = []
    start = long_string.find("{")
    while start != -1:
        stack = []
        end = start
        for i, char in enumerate(long_string[start:]):
            if char == "{":
                stack.append("{")
            elif char == "}":
                if not stack:
                    break
                stack.pop()
                if not stack:
                    end = start + i + 1
                    break
        if end > start:
            json_strings.append(long_string[start:end])
        start = long_string.find("{", end)
    if json_strings:
        return json_strings[0]
    else:
        return None


def extract_and_save_json(text, file_path):
    """
    Extracts JSON data from a text string and saves it to a file.
    """
    json_string = find_complete_json_strings(text)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(json.loads(json_string), file, indent=4)
    print("JSON data has been written to", file_path)


def extract_and_save_results(text, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        if text:
            file.write(text)
        else:
            file.write("")


def plot_results(results, directory):
    """
    Plot the results of the analysis.
    """
    # Prepare subplots
    _, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    questions = [
        "Same Input Characteristics (NO --> buggy)",
        "Cover new code fragment (YES --> buggy)",
        "Runtime Error (YES --> buggy)",
        "Same Result (NO --> buggy)",
    ]

    # Plot each result category
    for i, key in enumerate(results):
        responses = results[key]
        response_counts = {
            response: responses.count(response) for response in set(responses)
        }
        axes[i].bar(response_counts.keys(), response_counts.values(), color="skyblue")
        axes[i].set_title(questions[i])
        axes[i].set_ylabel("Counts")
        axes[i].set_xlabel("Responses")

    # Save the plot as a PDF file in the same directory
    pdf_path = os.path.join(directory, "results_plot.pdf")
    plt.savefig(pdf_path, format="pdf")

    plt.tight_layout()
    plt.show(block=False)


def plot_all_results(prompt_version):
    for experiment in os.listdir(EXPERIMENT_RESULTS_PATH):
        if experiment == ".DS_Store":
            continue
        plot_results_from_summary(experiment, prompt_version)


def plot_results_from_summary(experiment_folder, prompt_version, plot_table):
    """
    Plot the results of the analysis.
    """

    if experiment_folder.endswith("buggy"):
        prompt_kind = PromptKind.BUGGY
    elif experiment_folder.endswith("similar"):
        prompt_kind = PromptKind.SIMILAR
    elif experiment_folder.endswith("fixed"):
        prompt_kind = PromptKind.FIXED
    else:
        raise ValueError(
            f"No information about prompt kind from directory name: {experiment_folder}!"
        )

    question_fn = f"questions_v{prompt_version}.json"
    answer_fn = f"answers_v{prompt_version}.json"

    with open(os.path.join(PROMPT_TEMPLATE_PATH, question_fn)) as question_file:
        questions = json.load(question_file)
    with open(os.path.join(PROMPT_TEMPLATE_PATH, answer_fn)) as answer_file:
        answers = json.load(answer_file)[prompt_kind.name]
    with open(
        os.path.join(EXPERIMENT_RESULTS_PATH, experiment_folder, "summary.json")
    ) as f:
        results = json.load(f)

    # use 1 to indicate correct result and 0 to indicate wrong result
    for i in range(len(results)):
        for question_id, answer in answers.items():
            if question_id in results[i] and results[i][question_id] == answer:
                results[i][question_id] = 1
            else:
                results[i][question_id] = 0

    df_results = pd.DataFrame(results)
    info_columns = ["project", "bug id", "#correct"]
    df_results["project"] = df_results["id"].str.extract(
        r"prompt_(?:buggy|fixed|similar)_\d+_([-A-Za-z]+)_v\d+_result.txt"
    )
    df_results["bug id"] = df_results["id"].str.extract(
        r"prompt_(?:buggy|fixed|similar)_(\d+)_[-A-Za-z]+_v\d+_result.txt"
    )
    df_results.sort_values(by=["project", "bug id"], inplace=True)
    df_results.drop("id", axis=1, inplace=True)

    df_results.to_csv(
        os.path.join(
            EXPERIMENT_RESULTS_PATH, experiment_folder, f"{experiment_folder}.csv"
        ),
        index=False,
    )
    question_columns = [col for col in df_results.columns if col.startswith("Q")]
    df_results["#correct"] = df_results[question_columns].sum(axis=1)
    accuracy = (df_results["#correct"] > 2).sum() / len(df_results)
    print(f"The accuracy of {experiment_folder} is {accuracy:.4f}")

    if plot_table:
        # Get cell colors to indicate correctness
        cell_colours = []
        for _, result in df_results.iterrows():
            row_colours = []
            for question_id, answer in answers.items():
                if result[question_id] == answer:
                    row_colours.append("tab:blue")
                else:
                    row_colours.append("tab:orange")
            cell_colours.append(row_colours + ["w"] * len(info_columns))

        fig, ax = plt.subplots(figsize=(10, 12))
        # fig.patch.set_visible(False)
        ax.axis("off")
        ax.axis("tight")
        ax.table(
            cellText=df_results.values,
            cellColours=cell_colours,
            colLabels=df_results.columns,
            alpha=0.7,
            loc="center",
        )
        fig.tight_layout()
        plt.savefig(
            os.path.join(
                EXPERIMENT_RESULTS_PATH, experiment_folder, f"{experiment_folder}.pdf"
            ),
            format="pdf",
        )

    # Prepare subplots

    # _, axes = plt.subplots(2, 2, figsize=(10, 10))
    # axes = axes.flatten()

    # for i, question in enumerate(questions):
    #     answer_count = {"YES": 0, "NO": 0}
    #     for result in results:
    #         answer_count[result[question]] += 1
    #     if correct_answers[i] == "YES":
    #         axes[i].bar("YES (Correct)", answer_count["YES"], color='tab:blue')
    #         axes[i].bar("NO", answer_count["NO"], color='tab:orange')
    #     else:
    #         axes[i].bar("YES", answer_count["YES"], color='tab:orange')
    #         axes[i].bar("NO (Correct)", answer_count["NO"], color='tab:blue')
    #     axes[i].set_title(titles[i])
    #     axes[i].set_ylabel('Counts')
    #     axes[i].set_xlabel('Responses')

    # # Save the plot as a PDF file in the same directory
    # pdf_path = os.path.join(EXPERIMENT_RESULTS_PATH, directory, 'results_plot.pdf')
    # plt.savefig(pdf_path, format='pdf')

    # plt.tight_layout()
    # plt.show(block=False)


def summarize_results(directory, version, path, is_validation = False):
    results = []
    # read prompts for validation
    with open(FINE_TUNE_LLM_VALIDATION_PATH) as f:
        validation_paths = json.load(f)
        validation_prompts = [os.path.basename(p)[:-4] for p in validation_paths]

    for fn in os.listdir(os.path.join(path, directory)):
        if "result.txt" not in fn:
            continue
        if is_validation and fn.removesuffix("_result.txt") not in validation_prompts:
            continue
        with open(os.path.join(path, directory, fn)) as f:
            if version == "4":
                result = f.read()
                pattern = r"{(\s*\"Q\d\":\s*\".*?\",?\s*)+}"
                match = re.search(pattern, result)
                if match:
                    result = json.loads(match.group(0))
                    result["id"] = fn
                    results.append(result)
            elif version in ["2", "3"]:
                pattern = r"\[ANSWER\][^\[]*(YES|NO)[^\[]*\[\/ANSWER\]"
                matches = re.findall(pattern, result)
                # Extract the content if a match is found
                if matches:
                    result = {"id": fn}
                    for i, answer in enumerate(matches, 1):
                        question_id = f"Q{i}"
                        result[question_id] = answer
                    results.append(result)
                else:
                    logging.warning(
                        f"Ignore {fn} because fail to match answers between [ANSWER] and [/ANSWER]! Please fix them and run summarize again to include more results."
                    )
            else:
                raise ValueError(f"Illegal prompt template version {version}!")
    with open(
        os.path.join(path, directory, "summary.json"), "w"
    ) as f:
        f.write(json.dumps(results, sort_keys=True, indent=4))


def write_arguments(experiment_folder_path, args, model_name):
    arguments = vars(args)
    arguments.pop("func", None)
    arguments["model"] = model_name
    with open(os.path.join(experiment_folder_path, "arguments.json"), "w") as f:
        json.dump(arguments, f, sort_keys=True, indent=4)
