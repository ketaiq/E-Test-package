from datetime import datetime
import json
import os
import re
import time
import ollama
import pandas as pd
import huggingface_hub
import tiktoken
from transformers import AutoTokenizer
from javalang.parser import JavaSyntaxError

from src.testexe.iohelper import parse_generated_test_case
from src.testexe.defects4j_driver import Defects4jDriver
from src.llm.chatgpt.chatgpt_api import prompt_gpt
from src import PROMPT_TEMPLATE_PATH
from src.prompt.fewshots import generate_few_shots_msg
from src.prompt.prompt import extract_prompt_paths
from src.output.output import (
    create_experiment_folder,
    extract_and_save_results,
    summarize_results,
    write_arguments,
)

from src.llm.llm_kind import LLMKind
from src.prompt.prompt_kind import PromptKind


class PromptLlmHandler:
    LOG_FNAME = "prompt.log"
    STATS_FNAME = "statistics.csv"
    SYSTEM_MSG_FNAME = "system_message.json"
    TCG_MSG_FNAME = "tcg_message.json"
    RESULTS_FNAME = "results.jsonl"

    def __init__(self, args):
        self.chosen_llm = LLMKind[args.model]
        self.chosen_scenario = PromptKind[args.scenario]
        self.version = args.version
        self.dataset = args.dataset
        self.queries = args.queries
        self.projects = args.projects
        self.temperature = float(args.temperature)
        self.num_shots = int(args.few_shots)
        self.response_output_format = args.format
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract prompt paths
        self.prompt_paths = extract_prompt_paths(
            self.dataset,
            self.chosen_scenario,
            self.version,
            self.projects,
            self.queries,
        )
        self._initialize_paths(args)
        self._initialize_few_shots()
        self._initialize_messages()
        self._initialize_tokenizer()

    def _initialize_paths(self, args):
        self.experiment_results_folder_path = create_experiment_folder(
            self.chosen_llm, self.chosen_scenario, self.dataset, self.timestamp
        )
        self.prompt_log_path = os.path.join(
            self.experiment_results_folder_path, PromptLlmHandler.LOG_FNAME
        )
        # save complete arguments to a file
        write_arguments(
            self.experiment_results_folder_path,
            args,
            self.chosen_llm.get_intenal_model_name(),
        )
        # Initialize CSV file of statistics
        self.statistics_path = os.path.join(
            self.experiment_results_folder_path, PromptLlmHandler.STATS_FNAME
        )
        pd.DataFrame(
            columns=[
                "project_id",
                "bug_id",
                "miss_location",
                "#syntax_fix_times",
                "has_valid_syntax",
                "#compilation_fix_times",
                "can_compile",
                "#assertion_fix_times",
                "#failing_tests",
                "elapsed_nanoseconds",
                "#characters",
                "#tokens",
            ]
        ).to_csv(self.statistics_path, index=False)

    def _initialize_few_shots(self):
        # Extract few shots if enabled
        self.few_shots = None
        if self.num_shots > 0:
            self.few_shots = generate_few_shots_msg(self.num_shots, self.prompt_paths)

    def _initialize_messages(self):
        # Extract additional messages for LLM conversation
        # read system message
        with open(
            os.path.join(PROMPT_TEMPLATE_PATH, PromptLlmHandler.SYSTEM_MSG_FNAME)
        ) as f:
            self.system_msg = json.load(f)
        # read test case generation message
        with open(
            os.path.join(PROMPT_TEMPLATE_PATH, PromptLlmHandler.TCG_MSG_FNAME)
        ) as f:
            self.tcg_msg = json.load(f)

    def _initialize_tokenizer(self):
        # Get Llama tokenizer
        if self._is_llama_model():
            # huggingface_hub.login(os.environ["HUGGING_FACE_API_KEY"])
            llama3_tokenizer = AutoTokenizer.from_pretrained(
                self.chosen_llm.get_hf_model_name()
            )
            self.tokenizer_encode = llama3_tokenizer.tokenize
        elif self.chosen_llm is LLMKind.GPT4o:
            self.tokenizer_encode = tiktoken.encoding_for_model("gpt-4o").encode
        elif (
            self.chosen_llm is LLMKind.GPT3FT
            or self.chosen_llm is LLMKind.GPT3turbo
            or self.chosen_llm is LLMKind.GPT4
            or self.chosen_llm is LLMKind.GPT4turbo
        ):
            self.tokenizer_encode = tiktoken.get_encoding("cl100k_base").encode
        else:
            self.tokenizer_encode = None

    def _is_llama_model(self):
        return (
            self.chosen_llm is LLMKind.LLama3_8B
            or self.chosen_llm is LLMKind.LLama3_70B
            or self.chosen_llm is LLMKind.LLama3_1_8B
        )

    def _is_gpt_model(self):
        return (
            self.chosen_llm is LLMKind.GPT3FT
            or self.chosen_llm is LLMKind.GPT3turbo
            or self.chosen_llm is LLMKind.GPT4
            or self.chosen_llm is LLMKind.GPT4o
            or self.chosen_llm is LLMKind.GPT4turbo
        )

    def _prompt_llm(self, chat_msgs: list) -> str:
        """
        Wraps LLM prompting API in a single function
        """
        if self._is_llama_model():
            response = ollama.chat(
                model=self.chosen_llm.get_intenal_model_name(),
                messages=chat_msgs,
                options={
                    "seed": 27,
                    "temperature": self.temperature,
                    "num_ctx": self.chosen_llm.get_context_limit(),
                },
                stream=False,
            )
            return response["message"]["content"]
        elif self._is_gpt_model():
            response = prompt_gpt(
                self.chosen_llm.get_intenal_model_name(),
                chat_msgs,
                self.temperature,
            )
            return response.content

    def _prompt_llama_model(
        self, messages, bug_id, project_id, prompt_stats: dict, enable_tcg: bool
    ):
        scenario_response = ollama.chat(
            model=self.chosen_llm.get_intenal_model_name(),
            messages=messages,
            options={
                "seed": 27,
                "temperature": self.temperature,
                "num_ctx": self.chosen_llm.get_context_limit(),
            },
            stream=False,
        )
        # store response metrics
        prompt_stats["elapsed_nanoseconds"] = scenario_response["total_duration"]
        num_tokens = scenario_response["prompt_eval_count"]
        response = scenario_response["message"]["content"]
        # continue conversation with test case generation
        if enable_tcg:
            chat_msgs = messages + [scenario_response["message"], self.tcg_msg]
            generated_test_case = self._check_validity(
                chat_msgs, bug_id, project_id, prompt_stats
            )

            return response, generated_test_case, num_tokens
        else:
            return response, None, num_tokens

    def _prompt_gpt_model(
        self, messages, bug_id, project_id, prompt_stats: dict, enable_tcg: bool
    ):
        # Measure the time taken to process each prompt
        t_init = time.time_ns()
        response_msg = prompt_gpt(
            self.chosen_llm.get_intenal_model_name(),
            messages,
            self.temperature,
        )
        response = response_msg.content
        elapsed_nanoseconds = time.time_ns() - t_init
        prompt_stats["elapsed_nanoseconds"] = elapsed_nanoseconds
        generated_test_case = None
        # continue conversation with test case generation
        if enable_tcg:
            chat_msgs = messages + [
                {"role": "assistant", "content": response_msg.content},
                self.tcg_msg,
            ]
            generated_test_case = self._check_validity(
                chat_msgs, bug_id, project_id, prompt_stats
            )
        return response, generated_test_case

    def _check_validity(
        self, chat_msgs: list, bug_id: str, project_id: str, prompt_stats: dict
    ) -> tuple:
        """
        Checks the syntax and compilation of the code generated by LLM.

        Parameters
        ----------
        chat_msgs
            a list of messages for LLM conversation
        """
        MAX_RETRY = 5  # attempt at most 3 times
        generated_test_case = None
        has_valid_syntax = True
        syntax_fix_times = 0
        miss_location = False  # record if the trigger test case is found or not
        can_compile = True
        compilation_fix_times = 0
        assertion_fix_times = 0
        num_failing_tests = 0

        while (
            syntax_fix_times < MAX_RETRY
            and compilation_fix_times < MAX_RETRY
            and assertion_fix_times < MAX_RETRY
        ):
            tcg_response = self._prompt_llm(chat_msgs)
            try:
                generated_test_case = parse_generated_test_case(tcg_response)
                if generated_test_case is None:
                    err_msg = "Please generate only 1 test case within a Java code block in Markdown format."
                    print(
                        f"Retrying {syntax_fix_times+1} time(s) test case generation due to syntax error ..."
                    )
                    has_valid_syntax = False
                    syntax_fix_times += 1
                else:
                    has_valid_syntax = True
                    # check if compile
                    driver = Defects4jDriver(bug_id, project_id, self.timestamp)
                    aug_state = driver.augment_test_suite_with_generated_test_case(
                        generated_test_case
                    )
                    if aug_state is None:
                        # fail to add test case to test suite because fail to locate the trigger test case
                        miss_location = True
                        break
                    exe_result = driver.evaluate_test_execution()
                    if isinstance(exe_result, str):
                        # compilation error
                        exe_result = self._compress_compilation_msg(exe_result)
                        can_compile = False
                        print(
                            f"Retrying {compilation_fix_times+1} time(s) test case generation due to compilation error ..."
                        )
                        err_msg = f"Fix following compilation errors in your test case\n{exe_result}"
                        compilation_fix_times += 1
                    else:
                        # no compilation errors
                        can_compile = True
                        num_failing_tests = exe_result
                        if num_failing_tests == 0:
                            print(
                                f"Retrying {assertion_fix_times+1} time(s) test case generation due to inaccurate assertions ..."
                            )
                            err_msg = "Please change the assertions in your test case to make it fail, but you should not use raise keyword."
                            assertion_fix_times += 1
                        else:
                            break
            except JavaSyntaxError as e:
                err_msg = f"Your code has a JavaSyntaxError: {e.description} after {e.at}. Fix it with valid method code without syntax errors."
                print(
                    f"Retrying {syntax_fix_times+1} time(s) test case generation due to syntax error ..."
                )
                has_valid_syntax = False
                syntax_fix_times += 1
            # retry until generated test case is valid or beyond retry limit
            chat_msgs += [
                {"role": "assistant", "content": tcg_response},
                {
                    "role": "user",
                    "content": err_msg,
                },
            ]
        # Update statistics
        prompt_stats["miss_location"] = miss_location
        prompt_stats["#syntax_fix_times"] = syntax_fix_times
        prompt_stats["has_valid_syntax"] = has_valid_syntax
        prompt_stats["#compilation_fix_times"] = compilation_fix_times
        prompt_stats["can_compile"] = can_compile
        prompt_stats["#assertion_fix_times"] = assertion_fix_times
        prompt_stats["#failing_tests"] = num_failing_tests
        return generated_test_case

    def start_prompting(self):
        """
        Entry point for prompting experiments.
        """
        # Iterate over prompts and query LLM
        for i, prompt_path in enumerate(self.prompt_paths):
            print(f"{i + 1} - {prompt_path}")
            # Read prompt texts
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt = f.read()
            # Check token limit
            num_tokens = -1
            if self.tokenizer_encode:
                num_tokens = len(self.tokenizer_encode(prompt))
                if num_tokens > self.chosen_llm.get_context_limit():
                    with open(self.prompt_log_path, "a") as f:
                        f.write(f"Ignore {prompt_path} due to context limit!\n")
                    continue
            # Extract project metadata
            result_name, tcg_name, bug_id, project_id = self._extract_prompt_metadata(
                prompt_path
            )

            # Construct messages for LLM prompting
            messages = [self.system_msg]
            if self.few_shots is not None:
                messages += self.few_shots
            messages += [{"role": "user", "content": prompt}]
            print(f"Waiting for response from {self.chosen_llm.value}...")
            prompt_stats = {
                "project_id": project_id,
                "bug_id": bug_id,
                "miss_location": None,
                "#syntax_fix_times": None,
                "has_valid_syntax": None,
                "#compilation_fix_times": None,
                "can_compile": None,
                "#assertion_fix_times": None,
                "#failing_tests": None,
            }
            # Send requests with messages to prompt LLM
            try:
                if self._is_gpt_model():
                    response, tcg_response = self._prompt_gpt_model(
                        messages, bug_id, project_id, prompt_stats, False
                    )
                elif self._is_llama_model():
                    response, tcg_response, num_tokens = self._prompt_llama_model(
                        messages, bug_id, project_id, prompt_stats, False
                    )
            except Exception as e:
                print(e)
                continue
            # Save responses and statistics of prompting
            self._save_results(
                project_id,
                bug_id,
                response,
                result_name,
                tcg_response,
                tcg_name,
                prompt,
                num_tokens,
                prompt_stats,
            )
        print("Results are saved to " + self.experiment_results_folder_path)

    def _extract_prompt_metadata(self, prompt_path):
        prompt_name = os.path.basename(prompt_path)
        result_name = prompt_name[:-4] + "_result.txt"
        tcg_name = prompt_name[:-4] + "_testcase.txt"
        matched_name = re.search(
            rf"^prompt_[a-z]+_(\d+)_([-A-Za-z]+)_v{self.version}_result\.txt$",
            result_name,
        )
        bug_id = matched_name.group(1)
        project_id = matched_name.group(2)
        return result_name, tcg_name, bug_id, project_id

    def _save_results(
        self,
        project_id,
        bug_id,
        response,
        result_name,
        tcg_response,
        tcg_name,
        prompt,
        num_tokens,
        prompt_stats,
    ):
        # Save responses from LLM and statistics
        # store answers
        if self.response_output_format == "txt":
            extract_and_save_results(
                response, os.path.join(self.experiment_results_folder_path, result_name)
            )
        elif self.response_output_format == "jsonline":
            with open(
                os.path.join(self.experiment_results_folder_path, PromptLlmHandler.RESULTS_FNAME),
                "a",
            ) as f:
                f.write(
                    json.dumps(
                        {"project": project_id, "bug": bug_id, "response": response}
                    )
                    + "\n"
                )
        # store generate test case
        if tcg_response:
            extract_and_save_results(
                tcg_response,
                os.path.join(self.experiment_results_folder_path, tcg_name),
            )
        # record statistics
        prompt_stats["#characters"] = sum(len(word) for word in prompt.split())
        prompt_stats["#tokens"] = num_tokens
        pd.DataFrame([prompt_stats]).to_csv(
            self.statistics_path,
            mode="a",
            index=False,
            header=False,
        )

    def _compress_compilation_msg(self, compile_msg):
        """
        Compresses the compilation message to include only error-related messages to prompt LLM for bug fixing.
        """
        line_msg = compile_msg.split("\n")
        compressed_msg = ""
        for i in range(len(line_msg)):
            if "error:" in line_msg[i]:
                for j in range(i, len(line_msg)):
                    line = line_msg[j]
                    if "[javac]" not in line or len(compressed_msg) + len(line) > 10000:
                        break
                    compressed_msg += line + "\n"
                return compressed_msg
        return None
