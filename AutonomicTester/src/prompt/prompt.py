"""
This file contains the main function to create prompts.
"""

import logging
import json
from pathlib import Path
import re
import os
from src.prompt.prompt_kind import PromptKind
from src import (
    DEFACTS4J_PATH,
    LLAMA3_CONTEXT_LIMIT,
    BUGS_PATH,
    DEFACTS4J_PROMPT_PATH,
    PROMPT_DATASET_PATH,
    PROMPT_TEMPLATE_PATH,
    FINE_TUNE_LLM_VALIDATION_PATH,
)
from src.prompt.prompt_component import (
    BuggyUnitPromptComponent,
    FixedUnitPromptComponent,
    SimilarUnitPromptComponent,
)


class Prompt:

    def __init__(
        self,
        project: str,
        bug: str,
        template_version: int,
        scenario_kind: PromptKind,
        queries: list[str],
    ):
        self.project = project
        self.bug = bug
        self.template_version = template_version
        self.scenario_kind = scenario_kind
        self.queries = queries

    def generate(self, components_path: str, prompts_path: str):
        prompt_name = f"prompt_{self.scenario_kind.name.lower()}_{self.bug}_{self.project}_v{self.template_version}.txt"
        prompt_path = os.path.join(prompts_path, prompt_name)
        prompt = Prompt.create_prompt(
            self.template_version,
            self.queries,
            *Prompt.extract_prompt_components(
                self.scenario_kind, self.project, self.bug, components_path
            ),
        )
        # check if prompt is valid
        if prompt is None:
            return
        if len(prompt) > LLAMA3_CONTEXT_LIMIT:
            logging.warning(
                f"{prompt_name} has too many characters ({len(prompt)} > 8K)"
            )
        # store generated prompt
        with open(prompt_path, "w") as f:
            f.write(prompt.strip())

    @staticmethod
    def create_prompt(
        version, queries, function_candidate, existing_test_cases, new_scenario
    ):
        """
        Generate a structured prompt with variable sections for autonomic testing.

        Parameters
        ----------
        - version : str
        - queries : list
            the selection of queries to create the prompting text
        - function_candidate : str
        - existing_test_cases : str
        - new_scenario : str

        Returns
        -------
        str : a structured prompt for analysis or testing
        """
        # check if prompt components are successfully extracted
        if (
            function_candidate is None
            or existing_test_cases is None
            or new_scenario is None
        ):
            return None

        with open(os.path.join(PROMPT_TEMPLATE_PATH, f"template_v{version}.json")) as f:
            template_json = json.load(f)
        task = template_json["task"]
        prompt = f"""
FUNCTION CANDIDATE:
{function_candidate}

EXISTING TEST CASES:
{existing_test_cases}

MONITORED SCENARIO:
{new_scenario}

TASK:
{task}
"""
        if "questions" in template_json:
            questions = template_json["questions"]
            if not set(queries).issubset(set(questions)):
                raise ValueError(
                    "The selection of queries is not a subset of the available queries!"
                )
            questions = "\n".join(
                [f"Q{index}: {questions[k]}" for index, k in enumerate(queries, 1)]
            )
            prompt += f"\n\nQUESTIONS:\n{questions}"
        return prompt

    @staticmethod
    def extract_prompt_components(prompt_kind, project, bug_id, component_folder_path):
        """
        Extract the components for each field prompt.

        Parameters
        ----------
        - prompt_kind : PromptKind
        - project : str
        - bug_id : str

        Returns
        -------
        tuple(str, str, str)
        """
        components = {}
        if prompt_kind is PromptKind.BUGGY:
            prompt_component = BuggyUnitPromptComponent
        elif prompt_kind is PromptKind.FIXED:
            prompt_component = FixedUnitPromptComponent
        elif prompt_kind is PromptKind.SIMILAR:
            prompt_component = SimilarUnitPromptComponent
        else:
            raise ValueError(f"PromptKind {prompt_kind} is not supported!")
        # Read relevant components
        for component in prompt_component:
            component_path = os.path.join(
                component_folder_path, component.name + ".txt"
            )
            if os.path.exists(component_path):
                with open(component_path) as f:
                    components[component.value] = f.read()
            else:
                logging.warning(
                    f"Skip {prompt_kind.name} prompt of project {project} bug {bug_id} because component path {component_path} does not exist."
                )
                return None, None, None
        return (
            components["function_candidate"],
            components["existing_test_cases"],
            components["new_input"],
        )


class PromptBuilder:
    Defects4J_COMPONENTS_PATH = os.path.join(
        DEFACTS4J_PATH, "prompts"
    )  # path to store components for building Defects4J prompts
    Defects4AT_COMPONENTS_PATH = os.path.join(
        "Defect4AutonomicTesting", "bugs"
    )  # path to store components for building Defects4AT prompts (collected from open-source Java projects on GitHub)
    PROMPT_DATASET_PATH = (
        "PromptDataset"  # path to store testing scenario prompts for LLM
    )
    Defects4AT_PROJECTS = ["spring-boot", "shardingsphere", "dolphinscheduler", "micrometer"]

    def __init__(self, template_version: int, queries: list[str]):
        self.template_version = template_version
        self.queries = queries
        # Create path to prompts folder
        prompts_path = Path(
            PromptBuilder.PROMPT_DATASET_PATH, f"v{template_version}", "".join(queries)
        )
        prompts_path.mkdir(parents=True, exist_ok=True)
        self.prompts_path = str(prompts_path)

    def generate_prompts_for_defects4at(self):
        for project_id in PromptBuilder.Defects4AT_PROJECTS:
            project_bugs_path = os.path.join(
                PromptBuilder.Defects4AT_COMPONENTS_PATH, project_id
            )
            bug_ids = filter(str.isnumeric, os.listdir(project_bugs_path))
            for bug_id in bug_ids:
                components_path = os.path.join(
                    PromptBuilder.Defects4AT_COMPONENTS_PATH,
                    project_id,
                    bug_id,
                    "prompt",
                )
                for scenario_kind in list(PromptKind):
                    prompt = Prompt(
                        project_id,
                        bug_id,
                        self.template_version,
                        scenario_kind,
                        self.queries,
                    )
                    prompt.generate(components_path, self.prompts_path)

    def generate_prompts_for_defects4j(self):
        for project_id in os.listdir(PromptBuilder.Defects4J_COMPONENTS_PATH):
            project_path = os.path.join(
                PromptBuilder.Defects4J_COMPONENTS_PATH, project_id
            )
            if not os.path.isdir(project_path):
                continue
            for bug_id in os.listdir(project_path):
                bug_path = os.path.join(project_path, bug_id)
                if not os.path.isdir(bug_path):
                    continue
                components_path = os.path.join(bug_path, "prompt")
                for scenario_kind in list(PromptKind):
                    prompt = Prompt(
                        project_id,
                        bug_id,
                        self.template_version,
                        scenario_kind,
                        self.queries,
                    )
                    prompt.generate(components_path, self.prompts_path)


def extract_prompt_paths(dataset, chosen_scenario, version, projects, queries) -> list:
    prompt_paths = []
    prompt_folder_path = os.path.join(
        PROMPT_DATASET_PATH, f"v{version}", "".join(queries)
    )
    if dataset == "Defects4AT":
        for prompt_file in os.listdir(prompt_folder_path):
            # Skip files not matching the pattern
            matched_prompt = re.search(
                chosen_scenario.get_prompt_filename_pattern(version),
                prompt_file,
            )
            if not matched_prompt:
                continue
            project_id = matched_prompt.group(1)
            if project_id in projects:
                prompt_paths.append(os.path.join(prompt_folder_path, prompt_file))
    elif dataset == "Defects4J":
        for prompt_file in os.listdir(prompt_folder_path):
            # Skip files not matching the pattern
            matched_prompt = re.search(
                chosen_scenario.get_prompt_filename_pattern(version),
                prompt_file,
            )
            if not matched_prompt:
                continue
            project_id = matched_prompt.group(1)
            if project_id not in projects:
                prompt_paths.append(os.path.join(prompt_folder_path, prompt_file))
    elif dataset == "Validation":
        with open(FINE_TUNE_LLM_VALIDATION_PATH) as f:
            validation_paths = json.load(f)

        # Filter paths by the specified scenario
        for validation_path in validation_paths:
            prompt_path = os.path.join(
                prompt_folder_path, os.path.basename(validation_path)
            )
            if chosen_scenario.name.lower() in prompt_path:
                prompt_paths.append(prompt_path)
    return prompt_paths


def search_prompts_from_defects4at(prompt_kind: PromptKind, projects, version):
    """
    Search for prompts in Defect4AutonomicTesting.

    Parameters
    ----------
    - prompt_kind : PromptKind
    - projects : list of projects to search
    - version : int

    Returns:
    - list: A list of prompt files.
    """
    # List to hold all matching file paths
    matching_files = []
    # Regex pattern to match files like "prompt_abc.txt"
    pattern = prompt_kind.get_prompt_filename_pattern(version)

    # Search prompts only in the project
    for project in projects:
        directory = os.path.join(BUGS_PATH, project)
        # Walk through the directory
        for root, _, files in os.walk(directory):
            for file in files:
                # Check if file matches the pattern
                if pattern.match(file):
                    # Add the full path of the file to the list
                    full_path = os.path.join(root, file)
                    matching_files.append(full_path)
    return matching_files


def search_prompts_from_defects4j(prompt_kind: PromptKind, version):
    """
    Search prompts from Defects4J dataset.
    """
    matching_files = []
    pattern = prompt_kind.get_prompt_filename_pattern(version)
    # Walk through the directory
    for root, _, files in os.walk(DEFACTS4J_PROMPT_PATH):
        for file in files:
            # Check if file matches the pattern
            if pattern.match(file):
                # Add the full path of the file to the list
                full_path = os.path.join(root, file)
                matching_files.append(full_path)
    return matching_files


def extract_filename(path):
    """Regex to match the part of the path between the last "/" and ".json" """
    match = re.search(r"/([^/]+)\.json$", path)
    if match:
        return match.group(
            1
        )  # Returns the captured group which is the file name without ".json"
    else:
        return path.replace("/", "_").replace(
            ".json", ""
        )  # Return None if no match is found
