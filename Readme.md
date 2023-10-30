## Text-to-SQL Generation Using Fine-tuned LLMs on Intel GPUs(XPUs) using QLoRA.

This repository contains code for fine-tuning a Language Model for text-to-SQL generation tasks and then using the fine-tuned model for SQL query generation on Intel GPUs using QLoRA.

### Prerequisites

    - Python 3.x
    - PyTorch
    - Transformers library
    - Datasets library
    - Intel Extension for PyTorch (IPEX)
    - BigDL-LLM[XPU]

### Installation

1. Clone this repo.

```bash
git clone https://github.com/your_username/your_repository.git
```

2. Install required python packages

```bash
pip install -r requirements
```

3. Install Intel BigDL llm package

```bash
pip install --pre --upgrade bigdl-llm[xpu] -f https://developer.intel.com/ipex-whl-stable-xpu
```

### Default Configurations

#### Model

    - Default base model for fine-tuning: **openlm-research/open_llama_3b**
    - Model path for saving the fine-tuned LoRA adaptor (incase of interruptions): `./saved_model`
    - Path for saving task based (here it is text to sql) LoRA adaptors: `./lora_models`

#### Dataset

    - Default dataset for fine-tuning: **b-mc2/sql-create-context**

### File Descriptions

    - **finetune.py** : Contains code for fine-tuning a pre-trained Language Model on text-to-SQL tasks.
    - **generate.py** : Contains code for generating SQL queries using a fine-tuned model.

### Fine-Tuning a Model (finetune.py)

To finetune a model, run the `finetune.py` script

```bash
python finetune.py
```

#### Key Features:

    - Downloads a pre-trained model based on the given base model ID.
    - Tokenizes the input questions, context, and answers.
    - Fine-tunes the model using the tokenized data and qLoRA.
    - Saves the fine-tuned model.

#### Configuration:

    - BASE_MODEL: The pre-trained model to use for fine-tuning.
    - MODEL_PATH: Path to save the fine-tuned model.
    - DEVICE: Device to run the model on.

### SQL Query Generation (generate.py)

To generate SQL queries using the fine-tuned model, run the generate.py script.

#### Key Features:

    - Uses either the base model or a fine-tuned model for SQL query generation.
    - Loads sample data and generates SQL queries for each sample.

#### Configuration:

    - BASE_MODEL: The base model to use for inference.
    - MODEL_PATH: Path to the fine-tuned model.
    - LORA_CHECKPOINT: Latest checkpoint for the fine-tuned model.
    - TEST_DATA: Path to the test data file.

### Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


