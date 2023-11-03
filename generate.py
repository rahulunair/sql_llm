import gc
import time
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

import torch
import intel_extension_for_pytorch as ipex
from datasets import load_dataset
from datasets import Dataset
from bigdl.llm.transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer
from bigdl.llm.transformers.qlora import PeftModel

logging.basicConfig(level=logging.INFO)


# TODO(rahul): Move these to a config file later
BASE_MODELS = {
    "1": "openlm-research/open_llama_7b_v2",
    "2": "mistralai/Mistral-7B-Instruct-v0.1",
    "3": "NousResearch/Nous-Hermes-Llama2-13b",
    "4": "NousResearch/Llama-2-7b-chat-hf",
    "5": "mistralai/Mistral-7B-v0.1",
}

print("Available base models:")
for key, model in BASE_MODELS.items():
    print(f"{key}: {model}")
user_choice = input(
    "Select a model number (or press Enter to use the default): "
).strip()
BASE_MODEL = BASE_MODELS.get(user_choice, "openlm-research/open_llama_7b_v2")
MODEL_PATH = "./model"
LORA_CHECKPOINT = "./lora_adapters/checkpoint-20"  # !!! update the latest checkpoint
TEST_DATA = "./test_data/sample_test_data.json"
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")


def generate_prompt_sql(input_question, context, output=""):
    """
    Generates the SQL prompt in the required format.
    Parameters:
        input_question (str): The SQL question.
        context (str): The SQL context.
        output (str, optional): The SQL output. Defaults to an empty string.
    Returns:
        str: The formatted SQL prompt.
    """
    return f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. 

You must output the SQL query that answers the question.

### Input:
{input_question}

### Context:
{context}

### Response:
{output}"""


class InferenceModel:
    """Handles SQL query generation for a given text prompt."""

    def __init__(self, use_lora=False):
        """
        Initialize the InferenceModel class.
        Parameters:
            use_lora (bool, optional): Whether to use LoRA model. Defaults to False.
        """
        try:
            self.base_model_id = BASE_MODEL
            # Choose the appropriate tokenizer based on the model name
            if "llama" in self.base_model_id.lower():
                self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model_id)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                low_cpu_mem_usage=True,
                load_in_4bit=True,
                optimize_model=False,
                use_cache=True,
            )
            if use_lora:
                self.model = PeftModel.from_pretrained(self.model, LORA_CHECKPOINT)
        except Exception as e:
            logging.error(f"Exception occurred during model initialization: {e}")
            raise

        self.model.to(DEVICE)
        self.max_length = 512
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

    def generate(self, prompt, **kwargs):
        """Generates an SQL query based on the given prompt.
        Parameters:
            prompt (str): The SQL prompt.
        Returns:
            str: The generated SQL query.
        """
        try:
            encoded_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt",
            ).input_ids.to(DEVICE)
            with torch.no_grad():
                with torch.xpu.amp.autocast():
                    outputs = self.model.generate(
                        input_ids=encoded_prompt,
                        do_sample=False,
                        max_length=self.max_length,
                        temperature=0.3,
                        num_beams=5,
                        repetition_penalty=1.2,
                    )
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated
        except Exception as e:
            logging.error(f"Exception occurred during query generation: {e}")
            raise


def main():
    try:
        base_model = InferenceModel()
        finetuned_model = InferenceModel(use_lora=True)
        sample_data = load_dataset("json", data_files=TEST_DATA)["train"]
        for row in sample_data:
            try:
                prompt = generate_prompt_sql(row["question"], context=row["context"])
                print("Using base model...")
                output = base_model.generate(prompt)
                print(f"\n\tbot response: {output}\n")

                print("Using finetuned model...")
                output = finetuned_model.generate(prompt)
                print(f"\n\tbot response: {output}\n")
            except Exception as e:
                logging.error(f"Exception occurred during sample processing: {e}")
    except Exception as e:
        logging.error(f"Error during main execution: {e}")


if __name__ == "__main__":
    main()
