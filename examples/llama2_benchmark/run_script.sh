#!/bin/bash

## quantization
# hqq
CUDA_VISIBLE_DEVICES=0 python3 quant_main.py --model_id meta-llama/Llama-2-7b-hf --quant_alg hqq --nbits 4 --group_size 64 --axis 1
# awq
CUDA_VISIBLE_DEVICES=0 python3 quant_main.py --model_id meta-llama/Llama-2-7b-hf --quant_alg awq --nbits 4 --group_size 64 --axis 1

# lm-eval harness
# hqq
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./accelerate_config.yaml \
    -m lm_eval --model hf \
    --model_args pretrained=./output_model/meta-llama/Llama-2-7b-hf-hqq-n4-g64-a1 \
    --tasks arc_easy,arc_challenge,hellaswag,mmlu,truthfulqa,winogrande,gsm8k \
    --num_fewshot 0 \
    --batch_size 4 \
    --trust_remote_code \
    2>&1 | tee ./output_model/meta-llama/Llama-2-7b-hf-hqq-n4-g64-a1/eval.log

# awq
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file ./accelerate_config.yaml \
    -m lm_eval --model hf \
    --model_args pretrained=./output_model/meta-llama/Llama-2-7b-hf-awq-n4-g64-a1 \
    --tasks arc_easy,arc_challenge,hellaswag,mmlu,truthfulqa,winogrande,gsm8k \
    --num_fewshot 0 \
    --batch_size 4 \
    --trust_remote_code \
    2>&1 | tee ./output_model/meta-llama/Llama-2-7b-hf-awq-n4-g64-a1/eval.log