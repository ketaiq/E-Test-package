import os
import pandas as pd
import shutil

from src import DEFACTS4J_PROMPT_PATH

DEFACTS4J_ROLOS_DATASET_PATH = os.path.join("Defects4jDataset", "rolos_dataset")


def find_unique_prompts():
    df_local = pd.read_csv("Defects4jDataset/dataset/prompt_stats.csv")
    df_local["source"] = "local"
    df_rolos = pd.read_csv("Defects4jDataset/dataset/prompt_stats_rolos.csv")
    df_rolos["source"] = "rolos"
    concat_df = pd.concat([df_local, df_rolos])
    unique_prompts = concat_df.drop_duplicates(
        ["project_id", "bug_id", "prompt"], keep=False
    )
    unique_prompts.to_csv(
        os.path.join(DEFACTS4J_PROMPT_PATH, "unique_prompt_stats.csv"), index=False
    )
    return unique_prompts


def merge_dataset():
    df_unique_prompts = find_unique_prompts()
    unique_prompts_from_rolos = df_unique_prompts[
        df_unique_prompts["source"] == "rolos"
    ]
    for _, row in unique_prompts_from_rolos.iterrows():
        source_prompt_path = os.path.join(
            DEFACTS4J_ROLOS_DATASET_PATH,
            row["project_id"],
            str(row["bug_id"]),
            "prompt",
        )
        target_prompt_path = os.path.join(
            DEFACTS4J_PROMPT_PATH, row["project_id"], str(row["bug_id"]), "prompt"
        )
        shutil.copytree(source_prompt_path, target_prompt_path, dirs_exist_ok=True)
