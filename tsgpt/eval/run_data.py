import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import re
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event
import time

# Set the start method to 'spawn' for CUDA compatibility
if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)


from tsgpt.conversation import conv_templates, SeparatorStyle
from tsgpt.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from tsgpt.model import *
from tsgpt.model.STQwen2 import STQwen2ForCausalLM, STQwen2Config
from tsgpt.model.utils import KeywordsStoppingCriteria
import json
from peft import PeftModel, LoraConfig, get_peft_model

from tqdm import tqdm
import json
import os.path as osp
import pickle

from multiprocessing import Pool, current_process
import multiprocessing

DEFAULT_STHIS_TOKEN = "<ST_HIS>"
DEFAULT_STPRE_TOKEN = "<ST_PRE>"
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str, mask=None):
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x, mask)
        elif mode == 'denorm':
            x = self._denormalize(x, mask)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        dim2reduce = tuple(range(1, x.ndim-1))
        if mask is not None:
            # Create masked tensor for statistics calculation
            masked_x = torch.where(mask, x, torch.zeros_like(x))
            valid_counts = mask.sum(dim=dim2reduce, keepdim=True)
            if self.subtract_last:
                self.last = masked_x[:,-1,:].unsqueeze(1)
            else:
                self.mean = (masked_x.sum(dim=dim2reduce, keepdim=True) / (valid_counts + self.eps)).detach()
            # Calculate variance using masked values
            centered = masked_x - self.mean
            squared_diff = torch.where(mask, centered * centered, torch.zeros_like(centered))
            self.stdev = torch.sqrt(squared_diff.sum(dim=dim2reduce, keepdim=True) / (valid_counts + self.eps) + self.eps).detach()
        else:
            if self.subtract_last:
                self.last = x[:,-1,:].unsqueeze(1)
            else:
                self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x, mask=None):
        if mask is not None:
            if self.subtract_last:
                x = torch.where(mask, x - self.last, x)
            else:
                x = torch.where(mask, x - self.mean, x)
            x = torch.where(mask, x / (self.stdev + self.eps), x)
        else:
            if self.subtract_last:
                x = x - self.last
            else:
                x = x - self.mean
            x = x / (self.stdev + self.eps)

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, mask=None):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        
        if mask is not None:
            x = torch.where(mask, x * self.stdev, x)
            if self.subtract_last:
                x = torch.where(mask, x + self.last, x)
            else:
                x = torch.where(mask, x + self.mean, x)
        else:
            x = x * self.stdev
            if self.subtract_last:
                x = x + self.last
            else:
                x = x + self.mean
        return x

revin_layer = RevIN(1, affine=0, subtract_last=0)
padding_patch_layer = nn.ReplicationPad1d((0, 10))

def translate_conversation(input_text):
    import re

    # Find all patterns between < and >
    pattern = r'<(.*?)>'
    matches = re.findall(pattern, input_text)

    # Store original matches for return
    original_vars = matches.copy()

    # Replace each match with <timeseries>
    translated_text = input_text
    for match in matches:
        translated_text = translated_text.replace(f'<{match}>', '<timeseries>')

    return translated_text, original_vars

def load_st(idx, instruct_item, st_data_all, patch_len, stride):

    query = instruct_item['question']

    df_id = instruct_item['df_id']
    timeseries_operation = instruct_item['timeseries_operation']
    if 'start' in instruct_item.keys():
        start_time, end_time = instruct_item['start'], instruct_item['end']
        X = st_data_all[df_id][start_time:end_time].copy()
    else:
        X = st_data_all[df_id].copy()
    st_data = eval(timeseries_operation)
    st_data = st_data.dropna()
    # Drop non-numeric columns
    st_data = st_data.select_dtypes(include=[np.number])
    # normalize the data    
    data_values = torch.from_numpy(st_data.values.T[...,np.newaxis])
    nan_mask = ~torch.isnan(data_values)
    data_values = revin_layer(data_values, mask=nan_mask, mode='norm')
    data_values = data_values.permute(0,2,1)
    # padding the data
    data_values = padding_patch_layer(data_values)
    # unfold the data
    data_values = data_values.unfold(dimension=-1, size=patch_len, step=stride)
    st_len = data_values.shape[-2]
    data_values = data_values.reshape(-1, patch_len)
    nan_mask = ~torch.isnan(data_values)
    nan_mask = ~torch.any(~nan_mask, dim=-1)  # If any value in a patch is False, the whole patch is False

    time_str = ''
    for col in st_data.columns:
        time_str += '\n' + col + ":" + "<ST_patch>" * st_len
        time_str += f' ST_statistic: MIN:{st_data[col].min():.3f}, MAX:{st_data[col].max():.3f}, MEAN:{st_data[col].mean():.3f}, STD:{st_data[col].std():.3f}'
    query += '\n' + time_str

    st_data = torch.nan_to_num(data_values, nan=0.0)

    return {
        'st_data': st_data,   
        'st_mask': nan_mask,
        'query': query
    }

def load_prompting_file(file_path, data_tag):

    data = []
    with open(file_path + data_tag + '.jsonl', 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def load_model(args, gpu_id):
    # Set CUDA_VISIBLE_DEVICES for this GPU
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side='left')
    
    # Load model on CPU first
    model = STQwen2ForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16, 
        use_cache=True,
        low_cpu_mem_usage=True,
        device_map=None,  # Disable device_map to prevent expandable_segments
    )
    
    model.get_model().initialize_st_tower(
        args.patch_len, model.config.hidden_size
    )
    
    model.initialize_st_tokenizer(use_st_start_end=True, tokenizer=tokenizer, 
                                device=torch.device('cpu'))
    
    adapter_model_path = args.model_name
    adapter_config_path = 'adapter_config.json'
    
    st_tower_path = os.path.join(adapter_model_path, 'st_tower.pth')
    st_tower = torch.load(st_tower_path, map_location=torch.device('cpu'))
    model.get_model().st_tower.load_state_dict(st_tower)
    
    if args.lora:
        adapter_path = os.path.join(adapter_model_path, 'adapter_model.safetensors')
        with open(os.path.join(adapter_model_path, adapter_config_path), 'r') as f:
            config_data = json.load(f)
        lora_config = LoraConfig(**config_data)
        model = get_peft_model(model, lora_config)
        from safetensors.torch import load_file
        loaded_state_dict = load_file(adapter_path)
        model_state_dict = model.state_dict()
        for name, param in loaded_state_dict.items():
            new_name = name.replace('.weight', '.default.weight')
            model_state_dict[new_name].copy_(param)
    
    # Convert to bfloat16 and move to GPU
    model = model.to(torch.bfloat16)
    model = model.cuda(gpu_id)
    
    # Ensure model is in eval mode
    model.eval()
    
    return model, tokenizer

def worker_process(gpu_id, model, tokenizer, input_queue, output_queue, stop_event, batch_size=4):
    # Create output directory for this GPU
    gpu_output_dir = os.path.join(args.output_res_path, f'gpu_{gpu_id}')
    os.makedirs(gpu_output_dir, exist_ok=True)
    
    # Initialize results list for this GPU
    gpu_results = []
    
    while not stop_event.is_set():
        try:
            # Collect batch_size items from queue
            batch_data = []
            for _ in range(batch_size):
                try:
                    input_data = input_queue.get()
                    if input_data is None:
                        break
                    batch_data.append(input_data)
                except:
                    break
            
            if not batch_data:
                continue
                
            # Process batch
            batch_st_data = []
            batch_st_mask = []
            batch_input_ids = []
            batch_prompts = []
            batch_indices = []
            batch_instruct_items = []
            
            for idx, instruct_item, st_dict in batch_data:
                st_data = st_dict['st_data']
                qs = st_dict["query"]
                
                conv_mode = "mpt"
                conv = conv_templates[conv_mode].copy()
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                inputs = tokenizer([prompt], padding=True, return_tensors="pt")
                
                batch_st_data.append(st_data)
                batch_st_mask.append(st_dict["st_mask"])
                batch_input_ids.append(inputs.input_ids[0])  # Remove batch dimension
                batch_prompts.append(prompt)
                batch_indices.append(idx)
                batch_instruct_items.append(instruct_item)
            
            # Reverse sequences before padding
            reversed_batch_input_ids = [torch.flip(ids, [0]) for ids in batch_input_ids]
            
            batch_input_ids = torch.nn.utils.rnn.pad_sequence(
                reversed_batch_input_ids,
                batch_first=True,
                padding_value=tokenizer.pad_token_id).cuda(device=gpu_id)
            
            # Reverse back after padding
            batch_input_ids = torch.flip(batch_input_ids, [1])
            
            batch_attention_mask=batch_input_ids.ne(tokenizer.pad_token_id).cuda(device=gpu_id)
            batch_st_data = [item.to(torch.bfloat16).cuda(device=gpu_id) for item in batch_st_data]
            batch_st_mask = [item.cuda(device=gpu_id) for item in batch_st_mask]
                       
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, batch_input_ids)
            
            with torch.inference_mode():
                output_ids = model.generate(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    st_data=batch_st_data,
                    st_mask=batch_st_mask,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])
            
            # Process each output
            for i, (idx, instruct_item) in enumerate(zip(batch_indices, batch_instruct_items)):
                input_token_len = batch_input_ids[i].shape[0]
                # Get the actual output tokens for this sample
                sample_output_ids = output_ids[i, input_token_len:]
                # Decode only the generated tokens for this sample
                outputs = tokenizer.decode(sample_output_ids, skip_special_tokens=False)
                
                # Find the position of stop_str and truncate the output
                if stop_str in outputs:
                    outputs = outputs.split(stop_str)[0]
                outputs = outputs.strip()
                
                # Extract reasoning steps
                rationales = []
                if "Reasoning:" in outputs:
                    reasoning_text = outputs.split("Reasoning:")[1].split("Answer:")[0].strip()
                    steps = [step.strip() for step in reasoning_text.split("Step") if step.strip()]
                    rationales = [f"Step{step}" for step in steps]
                
                parts = outputs.split('Answer:')
                if len(parts) > 1:
                    answer = parts[1].strip()
                else:
                    answer = outputs.strip()
                
                result = {
                    "id": idx,
                    "question": instruct_item['question'],
                    "role": instruct_item['role'],
                    "timeseries_operation": instruct_item['timeseries_operation'],
                    "df_id": instruct_item['df_id'],
                    "rationales": rationales,
                    "prediction": answer,
                    "groundtruth": instruct_item['answer']
                }
                
                # Add result to GPU's results list
                gpu_results.append(result)
                
                # Save results periodically (every 10 results)
                if len(gpu_results) % 10 == 0:
                    with open(os.path.join(gpu_output_dir, 'results.json'), 'w') as f:
                        json.dump(gpu_results, f, indent=4)
            
        except Exception as e:
            print(f"Error in worker process {gpu_id}: {str(e)}")
            continue
    
    # Save final results for this GPU
    with open(os.path.join(gpu_output_dir, 'results.json'), 'w') as f:
        json.dump(gpu_results, f, indent=4)

def data_loader_process(prompt_file, st_data_all, patch_len, stride, input_queue, stop_event):
    try:
        for idx, instruct_item in enumerate(prompt_file):
            if stop_event.is_set():
                break
            st_dict = load_st(instruct_item['df_id'], instruct_item, st_data_all, patch_len, stride)
            input_queue.put((idx, instruct_item, st_dict))
    except Exception as e:
        print(f"Error in data loader process: {str(e)}")
    finally:
        # Signal that all data has been loaded
        input_queue.put(None)

def worker_fn(rank, world_size, args, input_queue, output_queue, stop_event):
    model, tokenizer = load_model(args, rank)
    worker_process(rank, model, tokenizer, input_queue, output_queue, stop_event, args.batch_size)

def run_eval(args):
    os.makedirs(args.output_res_path, exist_ok=True)
    
    # Load prompting file
    prompt_file = load_prompting_file(args.prompting_file, args.data_tag)[-100:]
    
    # Load ST data
    with open(args.st_data_path + 'df.pkl', 'rb') as file:
        st_data_all = pickle.load(file)
    
    # Create input queue and stop event
    input_queue = Queue(maxsize=args.num_gpus * 4)  # Queue size for single process per GPU
    stop_event = Event()
    
    # Start data loader process
    loader_process = Process(
        target=data_loader_process,
        args=(prompt_file, st_data_all, args.patch_len, args.stride, input_queue, stop_event)
    )
    loader_process.start()
    
    # Start worker processes using spawn
    worker_processes = []
    for rank in range(args.num_gpus):
        p = Process(
            target=worker_fn,
            args=(rank, args.num_gpus, args, input_queue, None, stop_event)
        )
        p.start()
        worker_processes.append(p)
    
    # Wait for data loader to finish
    loader_process.join()
    
    # Wait for input queue to be empty
    while not input_queue.empty():
        time.sleep(1)
    
    # Set stop event to signal workers to finish
    stop_event.set()
    
    # Wait for all worker processes to finish
    for p in worker_processes:
        p.join()
    
    print("All processes completed. Results are saved in GPU-specific directories under", args.output_res_path)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="./checkpoints/Qwen2-7B-Instruct")
    parser.add_argument("--model-name", type=str, default="./checkpoints/reason/")
    parser.add_argument("--prompting_file", type=str, default="ST_data/transport/")
    parser.add_argument("--st_data_path", type=str, default='ST_data/transport/')
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--output_res_path", type=str, default='outputs1')
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)  # Batch size for processing
    parser.add_argument("--lora", type=bool, default=True)
    parser.add_argument("--patch_len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--data_tag", type=str, default='test')

    args = parser.parse_args()

    run_eval(args)
