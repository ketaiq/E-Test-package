"""
File to call the ChatGPT APIs.
"""

import os
from openai import OpenAI

from src import DEFACTS4J_PATH

FINE_TUNE_SUFFIX = "defects4j-20"
FINE_TUNE_NUM_EPOCHS = 3


def prompt_gpt(model, messages, temperature):
    """
    Function to call the ChatGPT API and return the response.
    """
    client = OpenAI(timeout=300)
    response = client.chat.completions.create(
        model=model,
        seed=27,
        messages=messages,
        temperature=temperature,
        max_tokens=800,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
    )
    return response.choices[0].message


def fine_tune_gpt():
    client = OpenAI()
    response = client.files.create(
        file=open(
            "fine_tuning_dataset_v4.jsonl",
            "rb",
        ),
        purpose="fine-tune",
    )
    print(response)
    response = client.fine_tuning.jobs.create(
        training_file=response.id,
        suffix=FINE_TUNE_SUFFIX,
        model="gpt-3.5-turbo-0125",
        hyperparameters={"n_epochs": FINE_TUNE_NUM_EPOCHS},
    )
    print(response.id)


def list_files():
    client = OpenAI()
    client.files.list()


def check_job_status():
    client = OpenAI()

    # List 10 fine-tuning jobs
    response = client.fine_tuning.jobs.list(limit=10)
    print("Available fine-tuned LLMs")
    for data in response.data:
        print(data.fine_tuned_model)
    print("Last fine-tuning job status:", response.data[0].status)
    print(response.data[0].fine_tuned_model)
    # Retrieve the state of a fine-tune
    # client.fine_tuning.jobs.retrieve("ftjob-abc123")
    # List up to 10 events from a fine-tuning job
    # client.fine_tuning.jobs.list_events(fine_tuning_job_id="ftjob-abc123", limit=10)
