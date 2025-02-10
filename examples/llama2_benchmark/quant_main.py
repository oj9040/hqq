import os
import gc, time
import sys
import argparse

import torch

from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
from hqq.core.quantize import BaseQuantizeConfig
from awq import AutoAWQForCausalLM

import logging

#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data
logger = None
        
        
def quant_hqq(model_id: str, save_path: str, nbits: int=4, group_size: int=64, axis: int=1):
    global logger
 
    model     = HQQModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=axis)

    start = time.time()
    model.quantize_model(quant_config=quant_config)
    end = time.time()
    quant_time = end - start
    logger.info(f"quantization time = {quant_time:.2f} sec")
    
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"model saved to {save_path}")


def quant_awq(model_id, save_path, nbits: int=4, group_size: int=64, axis: int=1):
    global logger
    
    model     = AutoAWQForCausalLM.from_pretrained(model_id) 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    quant_config = {"w_bit": nbits, "q_group_size": group_size, "zero_point": True, 'version':'GEMM'}
    
    start = time.time()
    model.quantize(tokenizer, quant_config = quant_config)
    end = time.time()
    quant_time = end - start
    logger.info(f"quantization time = {quant_time:.2f} sec")
        
    torch.cuda.empty_cache()
    gc.collect()
    
    model.save_quantized(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"model saved to {save_path}")


def setup_logger(filename: str) -> logging.Logger:
    
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
    
    save_name = f"{args.model_id}-{args.quant_alg}-n{args.nbits}-g{args.group_size}-a{args.axis}"
    save_path = os.path.join("./output_model", save_name)

    global logger
    logger = setup_logger(f"{save_path}/quant.log")
    
    globals()[f"quant_{args.quant_alg}"](args.model_id, save_path, args.nbits, args.group_size, args.axis)


if __name__ == "__main__":
    sys.exit(main())
    