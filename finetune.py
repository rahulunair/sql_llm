import logging
import os
from math import ceil
from typing import Optional, Tuple
import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="intel_extension_for_pytorch"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torchvision.io.image", lineno=13
)
logging.getLogger("transformers").setLevel(logging.ERROR)


import torch
import intel_extension_for_pytorch as ipex
from datasets import load_dataset
from datasets import Dataset
from bigdl.llm.transformers import AutoModelForCausalLM
from bigdl.llm.transformers.qlora import (
    get_peft_model,
    prepare_model_for_kbit_training as prepare_model,
)
import wandb
from peft import LoraConfig
from transformers import (
    DataCollatorForSeq2Seq,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
)


# TODO(rahul): Move these to a config file later
ENABLE_WANDB = True
BASE_MODEL = "openlm-research/open_llama_7b_v2"
DATA_PATH = "b-mc2/sql-create-context"
MODEL_PATH = "./final_model"
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


if ENABLE_WANDB:
    wandb.init(project="text-to-sql-sc")


def generate_prompt_sql(input_question, context, output=""):
    """
    Generates a prompt for fine-tuning the LLM model for text-to-SQL tasks.

    Parameters:
        input_question (str): The input text or question to be converted to SQL.
        context (str): The schema or context in which the SQL query operates.
        output (str, optional): The expected SQL query as the output.

    Returns:
        str: A formatted string serving as the prompt for the fine-tuning task.
    """
    return f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output the SQL query that answers the question.

### Input:
{input_question}

### Context:
{context}

### Response:
{output}"""


class FineTuner:
    """A class to handle the fine-tuning of LLM models."""

    def __init__(self, base_model_id: str, model_path: str, device: torch.device):
        """
        Initialize the FineTuner with base model, model path, and device.

        Parameters:
            base_model_id (str): Id of pre-trained model to use for fine-tuning.
            model_path (str): Path to save the fine-tuned model.
            device (torch.device): Device to run the model on.
        """
        self.base_model_id = base_model_id
        self.model_path = model_path
        self.device = device

    def find_sublist(self, lst, sublist):
        n = len(lst)
        m = len(sublist)
        for i in range(n - m + 1):
            if lst[i : i + m] == sublist:
                return i
        return -1

    def setup_models(self):
        """Downloads the pre-trained model and tokenizer based on the given base model ID."""
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                load_in_low_bit="nf4",
                optimize_model=False,
                torch_dtype=torch.float16,
                modules_to_not_convert=["lm_head"],
            )
            self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model_id)
            self.tokenizer.pad_token_id = 0
            self.tokenizer.padding_side = "left"

        except Exception as e:
            logging.error(f"Error in downloading models: {e}")

    def tokenize_batch(
        self, data_points, add_eos_token=True, train_on_inputs=False, cutoff_len=512
    ) -> dict:
        """
        Tokenizes a batch of SQL related data points consisting of questions, context, and answers.

        Parameters:
            data_points (dict): A batch from the dataset containing 'question', 'context', and 'answer'.
            add_eos_token (bool): Whether to add an EOS token at the end of each tokenized sequence.
            cutoff_len (int): The maximum length for each tokenized sequence.

        Returns:
            dict: A dictionary containing tokenized 'input_ids', 'attention_mask', and 'labels'.
        """
        try:
            question = data_points["question"]
            context = data_points["context"]
            answer = data_points["answer"]
            if train_on_inputs:
                user_prompt = generate_prompt_sql(question, context)
                tokenized_user_prompt = self.tokenizer(
                    user_prompt,
                    truncation=True,
                    max_length=cutoff_len,
                    padding=False,
                    return_tensors=None,
                )
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                if add_eos_token:
                    user_prompt_len -= 1

            combined_text = generate_prompt_sql(question, context, answer)
            tokenized = self.tokenizer(
                combined_text,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                tokenized["input_ids"][-1] != self.tokenizer.eos_token_id
                and add_eos_token
                and len(tokenized["input_ids"]) < cutoff_len
            ):
                tokenized["input_ids"].append(self.tokenizer.eos_token_id)
                tokenized["attention_mask"].append(1)
            tokenized["labels"] = tokenized["input_ids"].copy()
            if train_on_inputs:
                tokenized["labels"] = [-100] * user_prompt_len + tokenized["labels"][
                    user_prompt_len:
                ]

            return tokenized
        except Exception as e:
            logging.error(
                f"Error in batch tokenization: {e}, Line: {e.__traceback__.tb_lineno}"
            )
            raise e

    def prepare_data(self, data, val_set_size=100) -> Dataset:
        """Prepare training and validation datasets."""
        try:
            train_val_split = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = train_val_split["train"].shuffle().map(self.tokenize_batch)
            val_data = train_val_split["test"].shuffle().map(self.tokenize_batch)
            return train_data, val_data
        except Exception as e:
            logging.error(
                f"Error in preparing data: {e}, Line: {e.__traceback__.tb_lineno}"
            )
            raise e

    def train_model(self, train_data, val_data, training_args):
        """
        Fine-tune the model with the given training and validation data.

        Parameters:
            train_data (Dataset): Training data.
            val_data (Optional[Dataset]): Validation data.
            training_args (TrainingArguments): Training configuration.
        """
        try:
            self.model = self.model.to(DEVICE)
            self.model = prepare_model(self.model)
            self.model = get_peft_model(self.model, LORA_CONFIG)
            trainer = Trainer(
                model=self.model,
                train_dataset=train_data,
                eval_dataset=val_data,
                args=training_args,
                data_collator=DataCollatorForSeq2Seq(
                    self.tokenizer,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                    padding=True,
                ),
            )
            self.model.config.use_cache = False
            trainer.train()
            self.model.save_pretrained(self.model_path)
        except Exception as e:
            logging.error(f"Error in model training: {e}")

    def finetune(self, data_path, training_args):
        """
        Execute the fine-tuning pipeline.

        Parameters:
            data_path (str): Path to the data for fine-tuning.
            training_args (TrainingArguments): Training configuration.
        """
        try:
            self.setup_models()
            data = load_dataset(data_path)
            train_data, val_data = self.prepare_data(data)
            self.train_model(train_data, val_data, training_args)
        except KeyboardInterrupt:
            print("Interrupt received, saving model...")
            self.model.save_pretrained(f"{self.model_path}_interrupted")
            print(f"Model saved to {self.model_path}_interrupted")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Error in fintuning: {e}")


if __name__ == "__main__":
    print(f"Finetuning on device: {ipex.xpu.get_device_name()}")
    try:
        finetuner = FineTuner(
            base_model_id=BASE_MODEL, model_path=MODEL_PATH, device=DEVICE
        )
        training_args = TrainingArguments(
            per_device_train_batch_size=32,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            save_steps=50,
            save_strategy="steps",
            eval_steps=50,
            evaluation_strategy="steps",
            max_steps=5000,
            learning_rate=3e-4,
            # num_train_epochs=2,
            max_grad_norm=0.3,
            bf16=True,
            lr_scheduler_type="cosine",
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            save_total_limit=3,
            logging_steps=20,
            optim="adamw_hf",
            output_dir="./lora_adapters",
            logging_dir="./logs",
            report_to="wandb" if ENABLE_WANDB else [],
        )
        finetuner.finetune(DATA_PATH, training_args)
    except Exception as e:
        logging.error(f"Error occurred: {e}")
