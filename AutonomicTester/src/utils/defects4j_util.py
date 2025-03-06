import os
import subprocess


def get_src_class_path(defects4j_project_path: str):
    command = "defects4j export -p dir.src.classes"
    result = subprocess.run(
        command, shell=True, cwd=defects4j_project_path, capture_output=True, text=True
    )
    return os.path.join(defects4j_project_path, result.stdout)


def get_src_tests_path(defects4j_project_path: str):
    command = "defects4j export -p dir.src.tests"
    result = subprocess.run(
        command, shell=True, cwd=defects4j_project_path, capture_output=True, text=True
    )
    return os.path.join(defects4j_project_path, result.stdout)