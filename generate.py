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
from transformers import LlamaTokenizer
from bigdl.llm.transformers.qlora import PeftModel

logging.basicConfig(level=logging.INFO)


# TODO: Move these to a config file later
BASE_MODEL = "openlm-research/open_llama_3b"
MODEL_PATH = "./model"
LORA_CHECKPOINT = "./lora_output/checkpoint-300"  # update the latest checkpoint
TEST_DATA = "./test_data/sample_test_data.json"
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")


def generate_prompt(text, context, output=None):
    """prompt template"""
    return f"""You are an expert text-to-SQL converter.
    Given a question and the Context generate the SQL query to answer the question based on the context.

## Question:
{text}

## Context:
{context}

## SQL Query:
{output}"""


class InferenceModel:
    """Class for handling SQL query generation."""

    def __init__(self, use_lora=False):
        self.tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            low_cpu_mem_usage=True,
            load_in_low_bit="nf4",
            optimize_model=False,
        )
        if use_lora:
            self.model = PeftModel.from_pretrained(self.model, LORA_CHECKPOINT)
        self.model.to(DEVICE)
        self.max_length = 256
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

    def generate(self, prompt, **kwargs):
        """Generate SQL query based on the provided prompt."""
        encoded_prompt = self.tokenizer.encode(prompt, add_special_tokens=False)
        inputs = torch.tensor([encoded_prompt], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            with torch.xpu.amp.autocast():
                outputs = self.model.generate(
                    input_ids=inputs,
                    do_sample=True,
                    max_length=self.max_length,
                    temperature=0.6,
                    num_beams=5,
                    repetition_penalty=1.2,
                )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated


def main():
    base_model = InferenceModel()
    finetuned_model = InferenceModel(use_lora=True)

    sample_data = load_dataset("json", data_files=TEST_DATA)["train"]
    for row in sample_data:
        prompt = generate_prompt(row["question"], context=row["context"])

        print("Using base model...")
        output = model1.generate(prompt)
        print(f"\n\tbot response: {output}\n")

        print("Using finetuned model...")
        output = model2.generate(prompt)
        print(f"\n\tbot response: {output}\n")


if __name__ == "__main__":
    main()
