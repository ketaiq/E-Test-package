import fastr
import os
import csv
import re
from datetime import datetime

def main():
    dim = 10
    B = 3
    DATASET_PATH = "../FastDataset"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_PATH = f"fastr_classification_{timestamp}.csv"
    # Write CSV header
    header = ["project", "bug", "classified_scenario", "target_scenario"]
    with open(RESULTS_PATH, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header) 
    # FAST-R classification
    for filename in os.listdir(DATASET_PATH):
        # Parse bug name
        pattern = rf"([-A-Za-z]+)_(\d+)\.txt"
        matched_name = re.search(
            pattern,
            filename,
        )
        project = matched_name.group(1)
        bug = matched_name.group(2)
        # Classify scenario
        file_path = os.path.join(DATASET_PATH, filename)
        _, _, sel = fastr.fastPlusPlus(file_path, dim=dim, B=B)
        truth = ["buggy", "fixed", "similar"]
        classification = ["", "", ""]
        classification[sel[0] - 1] = "similar"
        classification[sel[1] - 1] = "fixed"
        classification[sel[2] - 1] = "buggy"
        with open(RESULTS_PATH, mode="a", newline="") as f:
            writer = csv.writer(f)
            for i in range(3):
                writer.writerow([project, bug, classification[i], truth[i]])
        print(f"Classify project {project} bug {bug}.")

if __name__ == "__main__":
    main()