import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import pickle
from datetime import datetime
import logging
from tqdm import tqdm
import torch.multiprocessing as mp

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(curPath)[0])[0]
print(curPath, rootPath)
sys.path.append(rootPath)

from tsgpt.conversation import conv_templates, SeparatorStyle
from tsgpt.utils import disable_torch_init
from tsgpt.model.STQwen2 import STQwen2ForCausalLM
from tsgpt.model.utils import KeywordsStoppingCriteria
from peft import PeftModel, LoraConfig, get_peft_model

# Initialize tokenizer in main process
def init_tokenizer(base_model):
    return AutoTokenizer.from_pretrained(
        base_model,
        padding_side='left',
        use_fast=False  # Disable fast tokenizer to avoid multiprocessing issues
    )

# RevIN and padding layers
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
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
        else: 
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask=None):
        dim2reduce = tuple(range(1, x.ndim-1))
        if mask is not None:
            masked_x = torch.where(mask, x, torch.zeros_like(x))
            valid_counts = mask.sum(dim=dim2reduce, keepdim=True)
            if self.subtract_last:
                self.last = masked_x[:,-1,:].unsqueeze(1)
            else:
                self.mean = (masked_x.sum(dim=dim2reduce, keepdim=True) / (valid_counts + self.eps)).detach()
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
    st_data = st_data.select_dtypes(include=[np.number])
    
    data_values = torch.from_numpy(st_data.values.T[...,np.newaxis])
    nan_mask = ~torch.isnan(data_values)
    data_values = revin_layer(data_values, mask=nan_mask, mode='norm')
    data_values = data_values.permute(0,2,1)
    data_values = padding_patch_layer(data_values)
    data_values = data_values.unfold(dimension=-1, size=patch_len, step=stride)
    st_len = data_values.shape[-2]
    data_values = data_values.reshape(-1, patch_len)
    nan_mask = ~torch.isnan(data_values)
    nan_mask = ~torch.any(~nan_mask, dim=-1)

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

def load_model(args, gpu_id, tokenizer):
    try:
        # Set device for this process
        torch.cuda.set_device(gpu_id)
        
        disable_torch_init()
        
        model = STQwen2ForCausalLM.from_pretrained(
            args.base_model, 
            torch_dtype=torch.bfloat16, 
            use_cache=True,
            low_cpu_mem_usage=True,
            device_map=None,
        )
        
        model.get_model().initialize_st_tower(args.patch_len, model.config.hidden_size)
        model.initialize_st_tokenizer(use_st_start_end=True, tokenizer=tokenizer, device=torch.device('cpu'))
        
        st_tower_path = os.path.join(args.model_name, 'st_tower.pth')
        st_tower = torch.load(st_tower_path, map_location=torch.device('cpu'))
        model.get_model().st_tower.load_state_dict(st_tower)
        
        if args.lora:
            adapter_path = os.path.join(args.model_name, 'adapter_model.safetensors')
            with open(os.path.join(args.model_name, 'adapter_config.json'), 'r') as f:
                config_data = json.load(f)
            lora_config = LoraConfig(**config_data)
            model = get_peft_model(model, lora_config)
            from safetensors.torch import load_file
            loaded_state_dict = load_file(adapter_path)
            model_state_dict = model.state_dict()
            for name, param in loaded_state_dict.items():
                new_name = name.replace('.weight', '.default.weight')
                model_state_dict[new_name].copy_(param)
        
        model = model.to(torch.bfloat16)
        model = model.cuda(gpu_id)
        model.eval()
        
        return model
    except Exception as e:
        print(f"GPU {gpu_id}: Error loading model: {str(e)}")
        raise

def process_batch(model, tokenizer, batch_data, gpu_id):
    batch_st_data = []
    batch_st_mask = []
    batch_input_ids = []
    batch_indices = []
    batch_instruct_items = []
    
    for idx, instruct_item, st_dict in batch_data:
        st_data = st_dict['st_data']
        qs = st_dict["query"]
        
        conv = conv_templates["mpt"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt], padding=True, return_tensors="pt")
        
        batch_st_data.append(st_data)
        batch_st_mask.append(st_dict["st_mask"])
        batch_input_ids.append(inputs.input_ids[0])
        batch_indices.append(idx)
        batch_instruct_items.append(instruct_item)
    
    reversed_batch_input_ids = [torch.flip(ids, [0]) for ids in batch_input_ids]
    batch_input_ids = torch.nn.utils.rnn.pad_sequence(
        reversed_batch_input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id).cuda(device=gpu_id)
    batch_input_ids = torch.flip(batch_input_ids, [1])
    
    batch_attention_mask = batch_input_ids.ne(tokenizer.pad_token_id).cuda(device=gpu_id)
    batch_st_data = [item.to(torch.bfloat16).cuda(device=gpu_id) for item in batch_st_data]
    batch_st_mask = [item.cuda(device=gpu_id) for item in batch_st_mask]
    
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, batch_input_ids)
    
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
    
    results = []
    for i, (idx, instruct_item) in enumerate(zip(batch_indices, batch_instruct_items)):
        input_token_len = batch_input_ids[i].shape[0]
        sample_output_ids = output_ids[i, input_token_len:]
        outputs = tokenizer.decode(sample_output_ids, skip_special_tokens=False)
        
        if stop_str in outputs:
            outputs = outputs.split(stop_str)[0]
        outputs = outputs.strip()
        
        rationales = []
        if "Reasoning:" in outputs:
            reasoning_text = outputs.split("Reasoning:")[1].split("Answer:")[0].strip()
            steps = [step.strip() for step in reasoning_text.split("Step") if step.strip()]
            rationales = [f"Step{step}" for step in steps]
        
        parts = outputs.split('Answer:')
        answer = parts[1].strip() if len(parts) > 1 else outputs.strip()
        
        results.append({
            "id": idx,
            "question": instruct_item['question'],
            "role": instruct_item['role'],
            "timeseries_operation": instruct_item['timeseries_operation'],
            "df_id": instruct_item['df_id'],
            "rationales": rationales,
            "prediction": answer,
            "groundtruth": instruct_item['answer']
        })
    
    return results

def process_worker(rank, world_size, args, all_data, tokenizer):
    try:
        # Set device for this process
        torch.cuda.set_device(rank)
        
        # Load model
        model = load_model(args, rank, tokenizer)
        
        # Calculate data split for this GPU
        per_gpu_data = len(all_data) // world_size
        start_idx = rank * per_gpu_data
        end_idx = start_idx + per_gpu_data if rank < world_size - 1 else len(all_data)
        gpu_data = all_data[start_idx:end_idx]
        
        # Process data in batches
        all_results = []
        for i in tqdm(range(0, len(gpu_data), args.batch_size), desc=f'{rank}'):
            batch = gpu_data[i:i + args.batch_size]
            results = process_batch(model, tokenizer, batch, rank)
            all_results.extend(results)
            
            # Save intermediate results
            if (i + args.batch_size) % (args.batch_size * 10) == 0:
                with open(os.path.join(args.output_res_path, f'results_gpu{rank}.json'), 'w') as f:
                    json.dump(all_results, f, indent=4)
        
        # Save final results for this GPU
        with open(os.path.join(args.output_res_path, f'results_gpu{rank}.json'), 'w') as f:
            json.dump(all_results, f, indent=4)
            
    except Exception as e:
        print(f"GPU {rank}: Error in worker: {str(e)}")
    # finally:
    #     torch.cuda.empty_cache()

def run_eval(args):
    os.makedirs(args.output_res_path, exist_ok=True)
    
    try:
        
        # Initialize tokenizer in main process
        tokenizer = init_tokenizer(args.base_model)
        
        # Load data
        with open(args.prompting_file + args.data_tag + '.jsonl', 'r') as file:
            prompt_file = [json.loads(line) for line in file]
        
        with open(args.st_data_path + 'df.pkl', 'rb') as file:
            st_data_all = pickle.load(file)
        
        # Prepare data
        all_data = []
        for idx, instruct_item in tqdm(enumerate(prompt_file), total=len(prompt_file), desc='loading data'):
            st_dict = load_st(idx, instruct_item, st_data_all, args.patch_len, args.stride)
            all_data.append((idx, instruct_item, st_dict))
        
        # Set start method for multiprocessing
        mp.set_start_method('spawn', force=True)
        
        # Start worker processes
        mp.spawn(
            process_worker,
            args=(args.num_gpus, args, all_data, tokenizer),
            nprocs=args.num_gpus,
            join=True
        )
        
        # # Combine results from all GPUs
        # all_results = []
        # for gpu_id in range(args.num_gpus):
        #     result_file = os.path.join(args.output_res_path, f'results_gpu{gpu_id}.json')
        #     if os.path.exists(result_file):
        #         with open(result_file, 'r') as f:
        #             gpu_results = json.load(f)
        #             all_results.extend(gpu_results)
        
        # # Save combined results
        # with open(os.path.join(args.output_res_path, 'results.json'), 'w') as f:
        #     json.dump(all_results, f, indent=4)
        
        print("Evaluation completed successfully")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="./checkpoints/Qwen2-7B-Instruct")
    parser.add_argument("--model-name", type=str, default="./checkpoints/reason/")
    parser.add_argument("--prompting_file", type=str, default="ST_data/transport/")
    parser.add_argument("--st_data_path", type=str, default='ST_data/transport/')
    parser.add_argument("--output_res_path", type=str, default='outputs1')
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lora", type=bool, default=True)
    parser.add_argument("--patch_len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--data_tag", type=str, default='test')

    args = parser.parse_args()
    run_eval(args)
