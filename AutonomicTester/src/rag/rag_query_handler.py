from datetime import datetime
import json
import os
import time
import huggingface_hub
import javalang
import re
import pandas as pd
from javalang.tree import MethodDeclaration
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, PromptTemplate
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken
from transformers import AutoTokenizer
from src import DEFACTS4J_PATH, PROMPT_TEMPLATE_PATH
from src.output.output import create_experiment_folder, write_arguments
from src.rag.etest_query_engine import EtestQueryEngine
from src.utils.defects4j_util import get_src_class_path, get_src_tests_path
from src.prompt.prompt_kind import PromptKind
from src.rag.index_format import IndexFormat
from src.llm.llm_kind import LLMKind
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama


class RagQueryHandler:
    Defects4J_PROMPT_DATASET_PATH = os.path.join(DEFACTS4J_PATH, "dataset")
    Defects4J_DATASET_PATH = os.path.join(
        DEFACTS4J_PATH, "rag_dataset"
    )  # preprocessed Defects4J dataset without trigger tests
    COMPONENTS_FILE = "components.jsonl"
    PROGRESS_FILE = "progress.jsonl"
    SUMMARY_FILE = "summary.jsonl"
    FILTERED_SCENARIOS_FILE = "filtered_scenarios.csv"

    def __init__(self, args):
        self.chosen_llm = LLMKind[args.model]
        self.chosen_scenario = PromptKind[args.scenario]
        self.version = args.version
        self.dataset = args.dataset
        self.queries = args.queries
        self.temperature = float(args.temperature)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.index_format = IndexFormat[args.index_format]
        self.results_path = create_experiment_folder(
            self.chosen_llm,
            self.chosen_scenario,
            self.dataset,
            self.timestamp,
            use_rag=True,
        )

        self.df_filtered_scenarios = pd.read_csv(os.path.join(RagQueryHandler.Defects4J_DATASET_PATH, RagQueryHandler.FILTERED_SCENARIOS_FILE))

        # save complete arguments to a file
        write_arguments(
            self.results_path,
            args,
            self.chosen_llm.get_intenal_model_name(),
        )
        self._get_llm_caller()
        self._read_qa_template()
        self.total_llm_token_count = 0
        self.prompt_llm_token_count = 0

    def _get_llm_caller(self):
        if self.chosen_llm.is_gpt_model():
            llm_name = self.chosen_llm.get_intenal_model_name()
            # Set up OpenAI token counter call back
            self.token_counter = TokenCountingHandler(
                tokenizer=tiktoken.encoding_for_model(llm_name).encode
            )
            callback_manager = CallbackManager([self.token_counter])
            self.llm_caller = OpenAI(
                model=llm_name,
                temperature=self.temperature,
                callback_manager=callback_manager,
            )
        elif self.chosen_llm.is_llama_model():
            # Set up Llama3 token counter call back
            if os.environ["HUGGING_FACE_API_KEY"]:
                huggingface_hub.login(os.environ["HUGGING_FACE_API_KEY"])
                llama3_tokenizer = AutoTokenizer.from_pretrained(
                    self.chosen_llm.get_hf_model_name()
                )
            else:
                llama3_tokenizer = AutoTokenizer.from_pretrained(
                    "./meta-llama-Llama-3.2-1B"
                )
            self.token_counter = TokenCountingHandler(
                tokenizer=llama3_tokenizer.tokenize
            )
            callback_manager = CallbackManager([self.token_counter])
            self.llm_caller = Ollama(
                model=self.chosen_llm.get_intenal_model_name(),
                temperature=self.temperature,
                callback_manager=callback_manager,
                request_timeout=300.0
            )
        else:
            raise ValueError(f"Illegal LLM name: {self.chosen_llm}!")

    def _read_qa_template(self):
        with open("AutonomicTester/src/rag/qa_template.json") as f:
            self.qa_template = json.load(f)

    @staticmethod
    def read_progress() -> list:
        # Read progress JSON lines
        progress = []
        with open(
            os.path.join(
                RagQueryHandler.Defects4J_DATASET_PATH,
                RagQueryHandler.PROGRESS_FILE,
            )
        ) as f:
            for line in f:
                progress_item = json.loads(line.strip())
                # Skip invalid scenarios
                if progress_item["num_extracted_scenarios"] == 0:
                    continue
                progress.append(progress_item)
        return progress

    def _extract_experiments(self, target_project: str = None) -> list:
        experiments = []
        if self.dataset == "Defects4J":
            progress = RagQueryHandler.read_progress()
            for progress_item in progress:
                project_id = progress_item["project_id"]
                if target_project is not None and project_id != target_project:
                    # Filter only target project if specified
                    continue
                bug_id = progress_item["bug_id"]
                project_path = os.path.join(
                    RagQueryHandler.Defects4J_DATASET_PATH, project_id
                )
                if self.chosen_scenario in [PromptKind.SIMILAR, PromptKind.BUGGY]:
                    version_path = os.path.join(project_path, bug_id + "b")
                elif self.chosen_scenario is PromptKind.FIXED:
                    version_path = os.path.join(project_path, bug_id + "f")
                else:
                    print(f"Invalid scenario {self.chosen_scenario}!")
                    return
                experiments.append(
                    {"project": project_id, "bug": bug_id, "path": version_path}
                )

        return experiments

    def _build_query_engine(self, exp_path: str):
        """
        Builds QueryEngine object from indexes.
        """
        print(
            f"Building RAG query engine with indexing of all .java files in folder {exp_path} ..."
        )
        if self.index_format is IndexFormat.RAW:
            # Index all .java files in the project
            src_class_docs = SimpleDirectoryReader(
                input_dir=get_src_class_path(exp_path),
                recursive=True,
                required_exts=[".java"],
            ).load_data()
            src_tests_docs = SimpleDirectoryReader(
                input_dir=get_src_tests_path(exp_path),
                recursive=True,
                required_exts=[".java"],
            ).load_data()
            documents = src_class_docs + src_tests_docs
        elif self.index_format is IndexFormat.JSON:
            # TODO
            pass

        # Measure indexing time
        index_time_start = time.time_ns()
        index = VectorStoreIndex.from_documents(documents)
        elapsed_nanoseconds = time.time_ns() - index_time_start
        retriever = index.as_retriever()
        query_engine = EtestQueryEngine(
            retriever=retriever,
            llm=self.llm_caller,
            qa_prompt=PromptTemplate(self.qa_template["template"]),
        )

        return query_engine, elapsed_nanoseconds

    def _read_project_components(self, exp_path: str, project: str, bug: str) -> list:
        """
        Read a list of project components, including monitored_scenario, class_name and method_name.
        """
        components_list = []
        similar_scenario = self._extract_similar_scenario(project, bug)
        # Skip empty similar scenario if chosen
        if self.chosen_scenario is PromptKind.SIMILAR and similar_scenario is None:
            return []
        with open(os.path.join(exp_path, RagQueryHandler.COMPONENTS_FILE)) as f:
            for line in f:
                component_json = json.loads(line.strip())
                # Replace with similar scenario if chosen
                if self.chosen_scenario is PromptKind.SIMILAR:
                    component_json["scenario"] = similar_scenario
                components_list.append(component_json)
        return components_list

    def _generate_metadata_list(self, components_list: list):
        metadata_list = []
        for components in components_list:
            metadata = {
                "buggy_class_name": components["buggy_class"],
                "buggy_method_name": components["buggy_method"],
                "monitored_scenario": components["scenario"],
                "test_suite": components["test_suite"],
            }
            metadata_list.append(metadata)
        return metadata_list

    def _extract_similar_scenario(self, project: str, bug: str) -> str | None:
        """
        Extracts similar scenario from generated dataset.
        """
        if self.dataset == "Defects4J":
            similar_scenario_path = os.path.join(
                RagQueryHandler.Defects4J_PROMPT_DATASET_PATH,
                project,
                bug,
                "prompt",
                "SimilarScenario.txt",
            )
            if os.path.exists(similar_scenario_path):
                with open(similar_scenario_path) as f:
                    similar_scenario = (
                        f.read().replace("public void monitoredScenario()", "").strip()
                    )
                    return similar_scenario
            else:
                return None

    def analyze_test_suites(self):
        """
        Count the number of test cases, characters, tokens in test suites of buggy classes.
        """
        test_suite_stats_path = os.path.join(
            RagQueryHandler.Defects4J_DATASET_PATH,
            "test_suite_stats.jsonl",
        )
        if os.path.exists(test_suite_stats_path):
            return
        # huggingface_hub.login(os.environ["HUGGING_FACE_API_KEY"])
        experiments = self._extract_experiments()
        num_exps = len(experiments)
        # Iterate over Java projects
        for i, exp in enumerate(experiments):
            project = exp["project"]
            bug = exp["bug"]
            exp_path = exp["path"]
            print(f"[{i + 1}/{num_exps}] - Processing project {project} bug {bug} ...")
            components_list = self._read_project_components(
                exp_path.replace(f"{bug}f", f"{bug}b"), project, bug
            )
            num_scenarios = len(components_list)
            prev_test_suite_name = ""
            # Iterate over scenarios
            for index, components in enumerate(components_list):
                print(f"\t[{index+1}/{num_scenarios}] Checking scenario ...")
                test_suite_name = components["test_suite"]
                # Reuse processed test suite
                if test_suite_name != prev_test_suite_name:
                    test_suite_path = os.path.join(
                        get_src_tests_path(exp_path),
                        test_suite_name.replace(".", "/") + ".java",
                    )
                    with open(test_suite_path) as f_test_suite:
                        test_suite = f_test_suite.read()
                    tree = javalang.parse.parse(test_suite)
                    test_case_count = sum(
                        1
                        for _, node in tree.filter(MethodDeclaration)
                        if "test" in node.name.lower() or "@Test" in node.annotations
                    )
                    num_chars = len(test_suite)
                    llama3_tokenizer = AutoTokenizer.from_pretrained("./meta-llama-Llama-3.2-1B")
                    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
                    num_gpt_tokens = len(gpt_tokenizer.encode(test_suite))
                    num_llama_tokens = len(llama3_tokenizer.tokenize(test_suite))
                prev_test_suite_name = test_suite_name
                test_suite_stats = {
                    "project": project,
                    "bug": bug,
                    "scenario_index": index,
                    "buggy_class": components["buggy_class"],
                    "buggy_method": components["buggy_method"],
                    "test_suite_name": test_suite_name,
                    "test_case_count": test_case_count,
                    "num_chars": num_chars,
                    "num_gpt_tokens": num_gpt_tokens,
                    "num_llama_tokens": num_llama_tokens,
                }
                with open(
                    test_suite_stats_path,
                    "a",
                ) as f:
                    f.write(json.dumps(test_suite_stats) + "\n")

    def run_experiments(self):
        """
        Entry point for CLI to run querying experiments over all projects.
        """
        experiments = self._extract_experiments()
        num_exps = len(experiments)
        # Iterate over Java projects
        for i, exp in enumerate(experiments):
            project = exp["project"]
            bug = exp["bug"]
            exp_path = exp["path"]
            print(f"[{i + 1}/{num_exps}] - Processing project {project} bug {bug} ...")
            self.query(project, bug, exp_path)

    def query(self, project: str, bug: str, exp_path: str):
        """
        Queries LLM with context from RAG.
        """
        query_engine, index_nanoseconds = self._build_query_engine(exp_path)
        components_list = self._read_project_components(
            exp_path.replace(f"{bug}f", f"{bug}b"), project, bug
        )
        if not components_list:
            # Stop if scenarios do not exist
            return
        metadata_list = self._generate_metadata_list(components_list)
        prompt_path = os.path.join(self.results_path, f"prompt_{project}_{bug}.jsonl")
        summary_path = os.path.join(self.results_path, RagQueryHandler.SUMMARY_FILE)
        answers = {
            "project": project,
            "bug": bug,
            "dataset_path": exp_path,
            "scenarios": [],
            "index_nanoseconds": index_nanoseconds,
        }
        if self.chosen_scenario is PromptKind.SIMILAR:
            # Read the only similar scenario
            scenario_index = 0
        else:
            # Read filtered scenario
            scenario_row = self.df_filtered_scenarios.query(f"(project == '{project}') & (bug == {bug})")
            if scenario_row.empty:
                return
            scenario_index = int(scenario_row.iloc[0]["scenario_index"])
        metadata = metadata_list[scenario_index]
        print(f"Querying scenario {scenario_index} in project {project} bug {bug} ...")
        scenario = {"scenario_index": scenario_index}  # saved to summary JSON lines
        prompt_details = dict(metadata)  # saved to each bug prompt JSON lines
        prompt_details["scenario_index"] = scenario_index
        prompt_details["queries"] = []
        try:
            self._query_scenario(metadata, query_engine, prompt_details, scenario)
        except Exception as e:
            # Skip scenario with query exception
            return
        # Vote for scenario classification
        scenario["classified_scenario"] = self.vote_scenario(scenario)
        answers["scenarios"].append(scenario)
        # Append prompt details to JSON lines
        with open(prompt_path, "a") as f:
            f.write(json.dumps(prompt_details) + "\n")
        # Append answers to JSON lines
        with open(summary_path, "a") as f:
            f.write(json.dumps(answers) + "\n")

    def _query_scenario(self, metadata: dict, query_engine: EtestQueryEngine, prompt_details: dict, scenario: dict):
        # Query with scenario described by metadata
        for query in self.queries:
            prompt = self.qa_template["questions"][query].format(
                monitored_scenario=metadata["monitored_scenario"],
                test_suite=metadata["test_suite"],
                class_name=metadata["buggy_class_name"],
                method_name=metadata["buggy_method_name"],
            )
            query_time_start = time.time_ns()
            query_answer = query_engine.query(prompt)
            elapsed_nanoseconds = time.time_ns() - query_time_start
            # Save query details
            query_detail = dict(query_engine.prompt_dict)
            query_detail["query"] = query
            query_detail["answer"] = str(query_answer)
            query_detail["elapsed_nanoseconds"] = elapsed_nanoseconds
            # Count consumed tokens
            query_detail["total_llm_token_count"] = (
                self.token_counter.total_llm_token_count
                - self.total_llm_token_count
            )
            self.total_llm_token_count = self.token_counter.total_llm_token_count
            query_detail["prompt_llm_token_count"] = (
                self.token_counter.prompt_llm_token_count
                - self.prompt_llm_token_count
            )
            self.prompt_llm_token_count = self.token_counter.prompt_llm_token_count
            prompt_details["queries"].append(query_detail)
            # Save answer
            if self.chosen_llm is LLMKind.Deepseek_R1_70B:
                query_answer = re.sub(r"<think>.*?</think>", "", query_answer.response, flags=re.DOTALL).strip()
            scenario[query] = str(query_answer)

    def vote_scenario(self, answers: dict):
        scenario_vote = {}
        # Read correct answers for queries
        with open(
            os.path.join(PROMPT_TEMPLATE_PATH, f"answers_v{self.version}.json")
        ) as f:
            correct_answers = json.load(f)
        for scenario in PromptKind:
            scenario_str = scenario.name.lower()
            scenario_vote[scenario_str] = 0
            for question_id in self.queries:
                # compare with correct answer
                if correct_answers[scenario.name][question_id] in answers[question_id].upper():
                    scenario_vote[scenario_str] += 1
        # Fix order in case of ties
        order = {"buggy": 2, "fixed": 1, "similar": 0}
        return max(scenario_vote, key=lambda k: (scenario_vote[k], order[k]))
