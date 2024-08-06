# paper-OGODCKC
This repository contains code and access to the dataset used for our paper titled `Ontology-Guided On-Device Conversational Knowledge Capture with Large Language Models`. This document is intended for researchers, developers and those who would like to build, run, and experiment with paper-PTODSKC.

## Prerequisites and Dependencies

* requires M series Apple silicon 
* requires macOS >= 13.5
* requires native Python >= 3.8. It's recommended to use a virtual environment to manage dependencies.
In our work, we created virtual environment directory named `venv` under the repository root folder and used it run Python from code as `venv/bin/python3`.  It's important to note that if you opt to use a different version of Python, you will need to update the Python path in `runBenchmarks.py` and `fineTunedShot.py`.

## Installation

`mlx-lm` is available on [PyPI]. Please refer to the official [MLX documentation] and  [MLX examples] for more details on the MLX platform.  
To install the Python API with all the requirements, run:

```bash
python3 -m pip install -r requirements.txt
```

## How To Use

### Fine-tuning datasets
Datasets needed to generate the KC QLoRA adapter exists under the `data` directory.

* Please note that test and validation sets holds the same prompts to be able to ease the control the quality of the QLoRA adapter generation process. 

### Generating ground-truth file
`generateGroundTruth.py` script processes the `data/completeSet/test.jsonl` file line-by-line and writes the expected prompt response for each user input to a separate file under `results/test_ground_truth_output/`. The output of each line is written to a different file in the followign format, first line's output is written to 1.ttl file etc... The generated ground-truth file will be used in performance evaluations.

To generate the `results/test_ground_truth.jsonl`file, run the following command:
```bash
python3 scripts/generateGroundTruth.py 
```

### Model file
In our work, we utilize the 4-bit quantized and mlx-converted version of the Mistral-7B-Instruct-v0.2 model. All model files must be placed under the `model/Mistral-7B-Instruct-v0.2-4bit-mlx` folder located in the main directory of our repository. To replicate our test results accurately, please download the mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx file from the mlx-community on Hugging Face and ensure it is placed in the specified path.

### Adapters
The generated QLoRA adapters are saved under the `adapters/` directory. The adapters that produced the results obtained in our study are saved under the `adapters\` directory.

### Fine-tuning
QLoRA adapter creation can be done with the following command. Please feel free to update the QLoRA training parameters:

```bash
python3 -m mlx_lm.lora --train --model model/Mistral-7B-Instruct-v0.2-4bit-mlx --iters 800 --data ./data/completeSet --batch-size 4 --lora-layers 16 --adapter-path adapters/adapters_b4_l16_i800_ts177_mistral_I_v02_4b_completeSet.npz
```

### Running the benchmarks
To be able to evaluate the generated adapter with the model please use the `runBenchmarks.py` script with `--model-path`, `--adapter-path`, and `--testset-path` arguments. This script calls `fineTunedShot.py`, reading input from `testset-path` and writing the results to the `results/test_evaluation_output/` directory.

```bash
python3 scripts/runBenchmarks.py --model-path model/Mistral-7B-Instruct-v0.2-4bit-mlx --adapter-path adapters/adapters_b4_l16_i800_ts177_epoch18_mistral_I_v02_4b_completeSet.npz --testset-path data/completeSet/test.jsonl
```

### Evaluation
`calculateF1Score.py` script compares each method's result file with the ground-truth file and calculates precision, recall and f1-score. All results are written to the `evaluation_results.txt` file under `results` directory.
```bash
python3 scripts/calculateF1Score.py
```

[PyPI]: https://pypi.org/project/mlx-lm/
[MLX documentation]: https://ml-explore.github.io/mlx/build/html/install.html
[MLX examples]: https://github.com/ml-explore/mlx-examples
[mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx]: https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx/tree/main

### Cite

```bash

```