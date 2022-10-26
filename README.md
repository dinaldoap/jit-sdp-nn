# Structure

* jitsdp is the entrypoint used to run the project after installation. Run "jitsdp --help" to read the help.
* data is a folder that has the preprocessed data stored in pickle format. If jitsdp doesn't find that data, it will download the raw data from https://github.com/dinaldoap/jit-sdp-data and process again.
* mlruns is a folder that will be created by jitsdp to store the results.
* logs is a folder that will be created by jitsdp to store the logs.

# Usage
* Create and update the environment
    * create:
        ```bash
        git clone https://github.com/dinaldoap/jit-sdp-nn.git && \
        cd jit-sdp-nn && \
        conda env create --name pytorch --file conda.yml && \
        conda run --name pytorch pip install -r requirements.lock
        ```

    * update (before executing, go to jit-sdp-nn folder):
        ```bash
        conda activate pytorch && \
        conda env update --file conda.yml --prune && \
        pip install -r requirements.lock
        ```

* Hyperparameter tuning
    * Generate script in jit-sdp-nn/tuning0-128.sh (before executing, go to jit-sdp-nn folder):
        ```bash
        jitsdp tuning --start 0 --end 128 --cross-project 0 1 --validation-end 5000 1000 --filename tuning0-128.sh
        ```
    * The script generated will run 128 random configs with 3 distinct seeds for each combination of training data (WP/CP), dataset and model. The first 5000 commits will be used for validation in all runs.
    * If necessary, you can generate incrementally with more commands:
        ```bash
        jitsdp tuning --start 0 --end 64 --cross-project 0 1 --validation-end 5000 1000 --filename tuning0-64.sh
        jitsdp tuning --start 64 --end 128 --cross-project 0 1 --validation-end 5000 1000 --filename tuning64-128.sh
        ```
    * Use jitsdp and tuning0-128.sh as you need to execute on your machines.    
    
* Testing
    * Merge all mlruns folders from all of your machines to a single mlruns folder inside your jit-sdp-nn folder
    * Generate script jit-sdp-nn/testing0-128.sh (before executing, go to jit-sdp-nn folder):
        ```bash
        jitsdp testing --start 0 --end 128 --cross-project 0 1 --testing-start 0 --filename testing0-128.sh
        ```
    * Use the same procedure used to execute the Hyperparameter tuning

* Report
    * Merge all mlruns folders from all of your machines to a single mlruns folder inside your jit-sdp-nn folder
    * Generate plots and record statistical tests results in jit-sdp-nn (before executing, go to jit-sdp-nn folder):
        ```bash
        jitsdp report --start 0 --end 128 --cross-project 0 1 --filename mlruns-report
        ```
* Export
    * Export the datasets, the average metrics of the tuning, the average metrics of the testing and the prequential metrics of each testing run:
        ```bash
        jitsdp export --filename mlruns-export
        ```

## Alternative entrypoint

* Generate bundle in jit-sdp-nn/jitsdp/dist/jitsdp (before executing, go to jit-sdp-nn folder):
    ```bash
    conda activate pytorch && \
    bash release.sh
    ```
# Changelog

## 2022-10-26

* Disabled adaptive threshold by default.
* Added requirements.lock to lock indirect dependency versions and improve reproducibility.