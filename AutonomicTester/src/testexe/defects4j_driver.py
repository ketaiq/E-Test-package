import datetime
import os
import re
import shutil
import subprocess
import logging
from src.testexe.iohelper import add_generated_test_case, get_generated_test_name
import pandas as pd
from io import StringIO


class Defects4jDriver:
    def __init__(self, bug_id, project_id, timestamp):
        self.bug_id = bug_id
        self.project_id = project_id
        self.timestamp = timestamp
        self.dataset_path = os.path.join(".",
            "Defects4jDataset",
            f"{self.timestamp}_llm_test_cases")
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path, exist_ok=True)

    def _get_checkout_path(self, version):
        # version is either character b or f.
        return os.path.join(
            self.dataset_path,
            self.project_id,
            f"{self.bug_id}{version}",
        )
    
    def finish_parsing(self):
        return os.path.exists(self._get_checkout_path("b")) and os.path.exists(self._get_checkout_path("f"))

    def augment_test_suite_with_generated_test_case(
        self, generated_test_case
    ):
        # add test case to buggy version
        checkout_path_buggy = self._get_checkout_path("b")
        if os.path.exists(checkout_path_buggy):
            # delete folder to check again
            shutil.rmtree(checkout_path_buggy)
        self._checkout_project_version("b", checkout_path_buggy)
        trigger_test_path, _, test_method_name = self._extract_trigger_test(
            checkout_path_buggy
        )
        add_state = add_generated_test_case(
            trigger_test_path, test_method_name, generated_test_case
        )
        if add_state is None:
            return None
        # add test case to fixed version
        checkout_path_fixed = self._get_checkout_path("f")
        if os.path.exists(checkout_path_fixed):
            # delete folder to check again
            shutil.rmtree(checkout_path_fixed)
        self._checkout_project_version("f", checkout_path_fixed)
        trigger_test_path, _, test_method_name = self._extract_trigger_test(
            checkout_path_fixed
        )
        add_state = add_generated_test_case(
            trigger_test_path, test_method_name, generated_test_case
        )
        if add_state is None:
            return None
        return True

    def evaluate_test_execution(self):
        # Compare execution of test in the buggy version
        checkout_path_buggy = self._get_checkout_path("b")
        # skip unchecked project
        if not os.path.exists(checkout_path_buggy):
            return
        trigger_test_path, trigger_test_name, test_method_name = (
            self._extract_trigger_test(checkout_path_buggy)
        )
        generated_test_name = get_generated_test_name(trigger_test_name)
        result = self._execute_test_case(
            checkout_path_buggy, generated_test_name
        )
        
        if result.returncode == 0:
            # return the number of failing tests if the test executes
            return int(
                re.findall(r"Failing tests: (\d+)", result.stdout)[0]
            )
        else:
            # the test fails to execute
            return result.stderr

    def evaluate_test_coverage(self, experiment_path):
        coverage_path = os.path.join(experiment_path, "coverage")
        if not os.path.exists(coverage_path):
            os.mkdir(coverage_path)
        stats = {"project_id": self.project_id, "bug_id": self.bug_id}
        # Compare coverage of test in the fixed version
        checkout_path_fixed = self._get_checkout_path("f")
        # skip unchecked project
        if not os.path.exists(checkout_path_fixed):
            return

        trigger_test_path, trigger_test_name, test_method_name = (
            self._extract_trigger_test(checkout_path_fixed)
        )
        original_coverage = self._evaluate_coverage(
            checkout_path_fixed, trigger_test_name
        )
        original_coverage.to_csv(os.path.join(coverage_path, f"trigger_tc_{self.project_id}_{self.bug_id}.csv"), index=False)
        
        generated_test_name = get_generated_test_name(trigger_test_name)
        generated_test_coverage = self._evaluate_coverage(
            checkout_path_fixed, generated_test_name
        )
        generated_test_coverage.to_csv(os.path.join(coverage_path, f"llm_tc_{self.project_id}_{self.bug_id}.csv"), index=False)

        if original_coverage.equals(generated_test_coverage):
            stats["comparison"] = "identical"
        else:
            original_cov = original_coverage.set_index("Metric").loc["Condition coverage"].str.replace('%', '').astype(float)["Value"]
            llm_cov = generated_test_coverage.set_index("Metric").loc["Condition coverage"].str.replace('%', '').astype(float)["Value"]
            if original_cov > llm_cov:
                stats["comparison"] = "lower condition coverage"
            elif original_cov < llm_cov:
                stats["comparison"] = "higher condition coverage"
            else:
                stats["comparison"] = "same condition coverage"
        return stats

    def _checkout_project_version(self, version, checkout_path):
        version_map = {"b": "buggy", "f": "fixed"}
        logging.info(f"Checkout complete repository of {version_map[version]} project.")
        command = f"defects4j checkout -p {self.project_id} -v {self.bug_id}{version} -w {checkout_path}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Output results
        if result.stdout and result.returncode != 0:
            logging.error("\n" + result.stdout)
        if result.stderr and result.returncode != 0:
            logging.error("\n" + result.stderr)

    def _extract_trigger_test(self, checkout_path):
        command = f"defects4j info -p {self.project_id} -b {self.bug_id}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Regular expression to match both the full test name and the specific test method
        pattern = r"Root cause in triggering tests:\n(?:\s+-\s+([\w\.\:]+::(\w+)))"
        # Find all matches
        matches = re.findall(pattern, result.stdout)
        # Print extracted test names and specific test methods
        trigger_test_name = ""
        test_method_name = ""
        test_path = ""
        for full_name, method in matches:
            trigger_test_name = full_name
            test_path = full_name
            test_method_name = method
        # Output results
        if result.stdout and result.returncode != 0:
            logging.error("\n" + result.stdout)
        if result.stderr and result.returncode != 0:
            logging.error("\n" + result.stderr)

        # get path to source test
        command = f"defects4j export -p dir.src.tests"
        result = subprocess.run(
            command, cwd=checkout_path, shell=True, capture_output=True, text=True
        )
        trigger_test_path = os.path.join(
            checkout_path,
            result.stdout,
            test_path.split("::")[0].replace(".", "/") + ".java",
        )
        logging.info(f"Extract complete path to source test file: {trigger_test_path}")
        logging.info(f"Extract full name of trigger test: {trigger_test_name}")
        # Output results
        if result.stdout and result.returncode != 0:
            logging.error("\n" + result.stdout)
        if result.stderr and result.returncode != 0:
            logging.error("\n" + result.stderr)

        return trigger_test_path, trigger_test_name, test_method_name

    def _execute_test_case(self, checkout_path, trigger_test_name):
        logging.info("Execute trigger test case.")
        command = f"defects4j test -w {checkout_path} -t {trigger_test_name}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result

    def _evaluate_coverage(self, checkout_path, trigger_test_name):
        logging.info("Evaluate coverage of trigger test case.")
        command = f"defects4j coverage -w {checkout_path} -t {trigger_test_name}"
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # Output results
        if result.stdout:
            df_coverage_metrics = pd.read_table(
                StringIO(result.stdout), sep=":", names=["Metric", "Value"]
            )
            df_coverage_metrics["Metric"] = df_coverage_metrics["Metric"].str.strip()
            df_coverage_metrics["Value"] = df_coverage_metrics["Value"].str.strip()
            logging.debug("\n" + result.stdout)
            return df_coverage_metrics
        if result.stderr:
            logging.debug("\n" + result.stderr)
