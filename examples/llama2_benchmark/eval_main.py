import os
import time
import sys
import argparse

import torch

from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
from hqq.core.quantize import BaseQuantizeConfig
from awq import AutoAWQForCausalLM

import lm_eval
import logging

#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data
logger = None

def eval_task(model_path:str) -> tuple[dict, dict]:
    
    task_fewshot_map = {
        "arc_easy": 25,
        "arc_challenge": 25,
        "hellaswag": 10,
        "mmlu": 5,
        "truthfulqa": 0,
        "winogrande": 5,
        "gsm8k": 5,
    }

    for _task, _num_fewshot in task_fewshot_map.items():
        start = time.time()

        result = lm_eval.simple_evaluate(
            model="hf",
            model_args={"pretrained":model_path},
            tasks = [_task],
            num_fewshot = _num_fewshot,
            batch_size = 4,
            log_samples= False,
            device = f"cuda" if torch.cuda.is_available() else "cpu",
        )
        end = time.time()
        eval_time = end - start
        
        logger.info(f"{_task} eval time = {eval_time}")
        logger.info(f"{_task} accuracy = {result}")


def setup_logger(filename: str) -> logging.logger:
    
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def main() -> int:
    parser = argparse.ArgumentParser(description="hqq quantized model evaluation for multiple tasks")
    parser.add_argument("--model_id", type=str, default="", help="huggingface model id")
    parser.add_argument("--quant_alg", type=str, default="hqq", help="quantization algorithm (hqq or awq)")
    parser.add_argument("--nbits", type=int, default=4, help="weight bits")
    parser.add_argument("--group_size", type=int, default=64, help="group size")
    parser.add_argument("--axis", type=int, default=1, help="axis of per group quantization")
    args = parser.parse_args()
    
    model_name = f"{args.model_id}-{args.quant_alg}-n{args.nbits}-g{args.group_size}-a{args.axis}"
    model_path = os.path.join("./output_model", model_name)

    global logger
    logger = setup_logger(f"{model_path}/eval.log")
    
    eval_task(model_path)

if __name__ == "__main__":
    sys.exit(main())
    