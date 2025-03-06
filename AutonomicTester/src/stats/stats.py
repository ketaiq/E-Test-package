import json
import os
import re
import pandas as pd

from src.prompt.prompt_kind import PromptKind
from src import DEFACTS4J_PROMPT_PATH, EXPERIMENT_RESULTS_PATH, PROMPT_TEMPLATE_PATH


def summarize_prompt_statistics_for_defects4j(version):
    prompts = []
    for project_id in os.listdir(DEFACTS4J_PROMPT_PATH):
        project_path = os.path.join(DEFACTS4J_PROMPT_PATH, project_id)
        if not os.path.isdir(project_path):
            continue
        for bug_id in os.listdir(project_path):
            bug_path = os.path.join(project_path, bug_id)
            if not os.path.isdir(bug_path):
                continue
            for prompt_kind in list(PromptKind):
                prompt_name = f"prompt_{prompt_kind.name.lower()}_{bug_id}_{project_id}_v{version}.txt"
                prompt_path = os.path.join(bug_path, "prompt", prompt_name)
                if os.path.exists(prompt_path):
                    prompts.append(
                        {
                            "project_id": project_id,
                            "bug_id": int(bug_id),
                            "prompt": prompt_kind.name,
                        }
                    )
    df_stats = pd.DataFrame(prompts)
    df_stats.sort_values(["project_id", "bug_id"]).to_csv(
        os.path.join(DEFACTS4J_PROMPT_PATH, "prompt_stats.csv"), index=False
    )
    num_buggy_prompts = (df_stats["prompt"] == PromptKind.BUGGY.name).sum()
    num_similar_prompts = (df_stats["prompt"] == PromptKind.SIMILAR.name).sum()
    num_fixed_prompts = (df_stats["prompt"] == PromptKind.FIXED.name).sum()
    print(
        f"Prompt statistics for Defects4J: {num_buggy_prompts} buggy prompts, {num_similar_prompts} similar prompts, {num_fixed_prompts} fixed prompts"
    )


def analyze_answers_from_summary(experiment_folder, prompt_version, queries, path):
    true_scenario = experiment_folder.split("_")[-1]
    answer_fn = f"answers_v{prompt_version}.json"
    with open(os.path.join(PROMPT_TEMPLATE_PATH, answer_fn)) as answer_file:
        answers = json.load(answer_file)
    with open(
        os.path.join(path, experiment_folder, "summary.json")
    ) as f:
        results = json.load(f)

    # Vote for category using the number of correct results for 3 scenarios
    votes = []
    encoded_answers = []
    for result in results:
        result_id = result["id"]
        matched_prompt = re.search(
            r"prompt_(?:buggy|fixed|similar)_(\d+)_([-A-Za-z]+)_v\d+_result.txt",
            result_id,
        )
        if not matched_prompt:
            print(f"Fail to match result file name: {result_id}!")
            continue
        bug_id = int(matched_prompt.group(1))
        project = matched_prompt.group(2)
        # Encode answers with 0 meaning incorrect and 1 meaning correct
        encoded_answer = {"bug id": bug_id, "project": project}
        true_answers = {q: answers[true_scenario.upper()][q] for q in queries}
        for question_id, answer in true_answers.items():
            if question_id in result and result[question_id] == answer:
                encoded_answer[question_id] = 1
            else:
                encoded_answer[question_id] = 0
        encoded_answers.append(encoded_answer)

        # Voting
        vote = {"bug id": bug_id, "project": project}
        # Compute the number of correct results for each scenario
        for scenario in [p.name.lower() for p in PromptKind]:
            vote[scenario] = 0
            scenario_answers = {q: answers[scenario.upper()][q] for q in queries}
            for question_id, answer in scenario_answers.items():
                # correct answer
                if question_id in result and result[question_id] == answer:
                    vote[scenario] += 1
        # Vote for majority
        votes.append(vote)

    df_encoded_answers = pd.DataFrame(encoded_answers)
    df_encoded_answers.sort_values(by=["project", "bug id"], inplace=True)
    df_encoded_answers.to_csv(
        os.path.join(path, experiment_folder, "encoded_answers.csv"),
        index=False,
    )

    df_votes = pd.DataFrame(votes)
    df_votes.sort_values(by=["project", "bug id"], inplace=True)
    df_votes["max"] = df_votes[["buggy", "fixed", "similar"]].max(axis=1)
    df_votes["scenario"] = df_votes[["buggy", "fixed", "similar"]].idxmax(axis=1)
    equal_vote_rows = df_votes[
        (df_votes["buggy"] == df_votes["fixed"])
        & (df_votes["buggy"] == df_votes["max"])
        | (df_votes["buggy"] == df_votes["similar"])
        & (df_votes["buggy"] == df_votes["max"])
        | (df_votes["fixed"] == df_votes["similar"])
        & (df_votes["fixed"] == df_votes["max"])
    ]
    if not equal_vote_rows.empty:
        print(f"{len(equal_vote_rows)} rows with ambiguous vote")
        # print(equal_vote_rows)
    df_votes.to_csv(
        os.path.join(path, experiment_folder, "scenario_votes.csv"),
        index=False,
    )
    num_correct = (df_votes["scenario"] == true_scenario).sum()
    accuracy = num_correct / len(df_votes)
    print(f"The accuracy of {experiment_folder} is {accuracy:.4f}")


def analyze_answers_from_summary_in_binary_classification(
    experiment_folder, prompt_version
):
    true_scenario = experiment_folder.split("_")[-1]
    answer_fn = f"answers_v{prompt_version}.json"
    with open(os.path.join(PROMPT_TEMPLATE_PATH, answer_fn)) as answer_file:
        answers = json.load(answer_file)
    with open(
        os.path.join(EXPERIMENT_RESULTS_PATH, experiment_folder, "summary.json")
    ) as f:
        results = json.load(f)
    with open(
        os.path.join(PROMPT_TEMPLATE_PATH, f"template_v{prompt_version}.json")
    ) as prompt_file:
        min_num_match = json.load(prompt_file)["min_num_correct"]

    classification = []
    for result in results:
        result_id = result["id"]
        matched_prompt = re.search(
            r"prompt_(?:buggy|fixed|similar)_(\d+)_([-A-Za-z]+)_v\d+_result.txt",
            result_id,
        )
        if not matched_prompt:
            print(f"Fail to match result file name: {result_id}!")
            continue
        category = {
            "bug id": matched_prompt.group(1),
            "project": matched_prompt.group(2),
        }
        count = 0
        for question_id, answer in answers["SIMILAR"].items():
            # count matched answer in similar category
            if question_id in result and result[question_id] == answer:
                count += 1
        if count >= min_num_match:
            category["scenario"] = "similar"
        else:
            category["scenario"] = "dissimilar"
        classification.append(category)

    df_classifier = pd.DataFrame(classification)
    df_classifier.sort_values(by=["project", "bug id"], inplace=True)
    df_classifier.to_csv(
        os.path.join(
            EXPERIMENT_RESULTS_PATH,
            experiment_folder,
            "scenario_binary_classification.csv",
        ),
        index=False,
    )

    if true_scenario == "similar":
        num_correct = (df_classifier["scenario"] == true_scenario).sum()
        accuracy = num_correct / len(df_classifier)
    else:
        num_correct = (df_classifier["scenario"] == "dissimilar").sum()
        accuracy = num_correct / len(df_classifier)
    print(f"The accuracy of {experiment_folder} is {accuracy:.4f}")
