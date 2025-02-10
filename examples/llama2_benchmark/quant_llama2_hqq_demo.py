import os
import time

#Settings
######################################################################################
hf_auth    = None #HuggingFace token
cache_path = ''   #cache directory to store data

#Chose a model
model_id  = "meta-llama/Llama-2-7b-hf" 
#model_id  = "meta-llama/Llama-2-13b-hf" 
#model_id  = "meta-llama/Llama-2-70b-hf" 

#Load model on the CPU
######################################################################################
from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
#model     = HQQModelForCausalLM.from_pretrained(model_id, use_auth_token=hf_auth, cache_dir=cache_path)
#tokenizer = AutoTokenizer.from_pretrained(model_id,       use_auth_token=hf_auth, cache_dir=cache_path) 
model     = HQQModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id) 

#Quantize the model
######################################################################################
from hqq.core.quantize import *
nbits = 4
group_size = 64
axis = 1
#quant_config = BaseQuantizeConfig(nbits=8, group_size=128, axis=0)
#quant_config = BaseQuantizeConfig(nbits=4, group_size=64, axis=0)
quant_config = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=axis)
#quant_config = BaseQuantizeConfig(nbits=3, group_size=64, axis=0)
#quant_config = BaseQuantizeConfig(nbits=2, group_size=16, axis=0)
#quant_config = BaseQuantizeConfig(nbits=2, group_size=16, quant_scale=True, axis=0) #scale is quantized to 8-bit/g=128

model.quantize_model(quant_config=quant_config)

save_path = os.path.join("./output_model", model_id + f"-hqq-n{nbits}-g{group_size}-a{axis}")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("model saved!")
#Evaluate the quantized model 
######################################################################################
from eval_model import eval_wikitext2
#eval_wikitext2(model, tokenizer, verbose=True)

import lm_eval
# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args={"pretrained":save_path},
#     tasks = ["arc", "hellaswag", "mmlu"],
#     num_fewshot = 0,
#     batch_size = 4,
#     device = "cuda:0",
# )

task_fewshot_map = {
    "arc_easy": 25,
    "arc_challenge": 25,
    "hellaswag": 10,
    "mmlu": 5,
    "truthfulqa": 0,
    "winogrande": 5,
    "gsm8k": 5,
}

results = {}
times = {}
for _task, _num_fewshot in task_fewshot_map.items():
    start = time.time()
    results[_task] = lm_eval.simple_evaluate(
        model="hf",
        model_args={"pretrained":save_path},
        tasks = [_task],
        num_fewshot = _num_fewshot,
        batch_size = 4,
        device = "cuda:0"
    )
    end = time.time()
    times[_task] = end - start

print(results)
print(times)
import pdb; pdb.set_trace()
