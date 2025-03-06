#!/bin/bash
RESULTS_DIR="AutonomicTester/experiment_results"
mkdir -p $RESULTS_DIR
ARCHIVES_DIR="Archives"

# Extract dataset
(
    echo "Extracting dataset ..."
    cd "${ARCHIVES_DIR}/dataset"
    for file in *.tar.gz; do
        tar -xzf "$file" -C ../../
    done
)

# Extract results
(
    echo "Extracting results ..."
    cd "${ARCHIVES_DIR}/results"
    for result_folder in *; do
        if [ -d "$result_folder" ]; then
            for file in "${result_folder}"/*.tar.gz; do
                echo "Extracting $file ..."
                mkdir -p "../../${RESULTS_DIR}/${result_folder}"
                tar -xzf "$file" -C "../../${RESULTS_DIR}/${result_folder}"
            done
        fi
    done
)

# Extract plot data
(
    echo "Extracting plot data ..."
    cd "DataAnalysis"
    tar -xzf "plotdata.tar.gz"
)