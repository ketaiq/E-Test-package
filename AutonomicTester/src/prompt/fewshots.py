import json
import os

from src import FEW_SHOTS_PATH, PROMPT_DATASET_PATH, PROMPT_TEMPLATE_PATH

CANDIDATE_PROMPTS = {
    "buggy": [
        "prompt_buggy_1_Chart_v4.txt",
        "prompt_buggy_55_Lang_v4.txt",
        "prompt_buggy_86_Closure_v4.txt",
    ],
    "fixed": [
        "prompt_fixed_1_Chart_v4.txt",
        "prompt_fixed_55_Lang_v4.txt",
        "prompt_fixed_86_Closure_v4.txt",
    ],
    "similar": [
        "prompt_similar_1_Chart_v4.txt",
        "prompt_similar_55_Lang_v4.txt",
        "prompt_similar_86_Closure_v4.txt",
    ],
}

def generate_few_shots_msg(num_shots, prompt_paths):
    messages = []
    with open(os.path.join(PROMPT_TEMPLATE_PATH, "answers_v4.json")) as fanswers:
        answers = json.load(fanswers)
    for i in range(num_shots // 3):
        for scenario in CANDIDATE_PROMPTS.keys():
            prompt_path = os.path.join(PROMPT_DATASET_PATH, CANDIDATE_PROMPTS[scenario][i])
            with open(prompt_path) as f:
                prompt = f.read()
            messages += [
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": json.dumps(answers[scenario.upper()]),
                },
            ]
            # Remove prompts for few-shots learning when prompting
            if prompt_path in prompt_paths:
                prompt_paths.remove(prompt_path)
    return messages