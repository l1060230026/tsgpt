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
import logging
from tqdm import tqdm
import traceback
from datetime import datetime

# Set up logging
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

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
    logger = logging.getLogger(__name__)
    try:
        logger.info(f"GPU {gpu_id}: Starting model loading...")
        disable_torch_init()
        logger.info(f"GPU {gpu_id}: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side='left')
        
        # Load model on CPU first
        logger.info(f"GPU {gpu_id}: Loading base model...")
        model = STQwen2ForCausalLM.from_pretrained(
            args.base_model, 
            torch_dtype=torch.bfloat16, 
            use_cache=True,
            low_cpu_mem_usage=True,
            device_map=None,
        )
        
        logger.info(f"GPU {gpu_id}: Initializing ST tower...")
        model.get_model().initialize_st_tower(
            args.patch_len, model.config.hidden_size
        )
        
        logger.info(f"GPU {gpu_id}: Initializing ST tokenizer...")
        model.initialize_st_tokenizer(use_st_start_end=True, tokenizer=tokenizer, 
                                    device=torch.device('cpu'))
        
        adapter_model_path = args.model_name
        adapter_config_path = 'adapter_config.json'
        
        logger.info(f"GPU {gpu_id}: Loading ST tower weights...")
        st_tower_path = os.path.join(adapter_model_path, 'st_tower.pth')
        st_tower = torch.load(st_tower_path, map_location=torch.device('cpu'))
        model.get_model().st_tower.load_state_dict(st_tower)
        
        if args.lora:
            logger.info(f"GPU {gpu_id}: Loading LoRA adapter...")
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
        logger.info(f"GPU {gpu_id}: Moving model to GPU...")
        model = model.to(torch.bfloat16)
        model = model.cuda(gpu_id)
        
        # Ensure model is in eval mode
        model.eval()
        logger.info(f"GPU {gpu_id}: Model loaded successfully")
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"GPU {gpu_id}: Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def process_fn(rank, world_size, args, input_queue, stop_event):
    """Main processing function for each GPU worker"""
    logger = logging.getLogger(__name__)
    logger.info(f"GPU {rank}: Starting worker process")
    
    try:
        # Load model and tokenizer
        logger.info(f"GPU {rank}: Loading model and tokenizer...")
        model, tokenizer = load_model(args, rank)
        logger.info(f"GPU {rank}: Model and tokenizer loaded successfully")
        
        # Initialize results list for this GPU
        gpu_results = []
        processed_count = 0
        empty_queue_count = 0  # Counter for consecutive empty queue checks
        
        logger.info(f"GPU {rank}: Starting main processing loop")
        while not stop_event.is_set():
            try:
                # Collect batch_size items from queue with timeout
                batch_data = []
                logger.info(f"GPU {rank}: Waiting for data from queue...")
                for _ in range(args.batch_size):
                    try:
                        input_data = input_queue.get(timeout=5)  # 5 second timeout
                        if input_data is None:
                            logger.info(f"GPU {rank}: Received None signal, finishing up...")
                            break
                        batch_data.append(input_data)
                        logger.info(f"GPU {rank}: Received data item {len(batch_data)}")
                    except:
                        break
                
                if not batch_data:
                    empty_queue_count += 1
                    logger.info(f"GPU {rank}: Empty queue count: {empty_queue_count}")
                    if empty_queue_count >= 3:  # If queue is empty for 3 consecutive times
                        logger.info(f"GPU {rank}: Detected empty queue multiple times, checking if should exit...")
                        if input_queue.empty() and stop_event.is_set():
                            logger.info(f"GPU {rank}: Exiting due to empty queue and stop event")
                            break
                    continue
                
                empty_queue_count = 0  # Reset counter if we got data
                logger.info(f"GPU {rank}: Processing batch of size {len(batch_data)}")
                
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
                    batch_input_ids.append(inputs.input_ids[0])
                    batch_prompts.append(prompt)
                    batch_indices.append(idx)
                    batch_instruct_items.append(instruct_item)
                
                logger.info(f"GPU {rank}: Preparing batch for model...")
                # Reverse sequences before padding
                reversed_batch_input_ids = [torch.flip(ids, [0]) for ids in batch_input_ids]
                
                batch_input_ids = torch.nn.utils.rnn.pad_sequence(
                    reversed_batch_input_ids,
                    batch_first=True,
                    padding_value=tokenizer.pad_token_id).cuda(device=rank)
                
                # Reverse back after padding
                batch_input_ids = torch.flip(batch_input_ids, [1])
                
                batch_attention_mask = batch_input_ids.ne(tokenizer.pad_token_id).cuda(device=rank)
                batch_st_data = [item.to(torch.bfloat16).cuda(device=rank) for item in batch_st_data]
                batch_st_mask = [item.cuda(device=rank) for item in batch_st_mask]
                           
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, batch_input_ids)
                
                logger.info(f"GPU {rank}: Generating outputs...")
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
                
                logger.info(f"GPU {rank}: Processing generated outputs...")
                # Process each output
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
                    
                    gpu_results.append(result)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        logger.info(f"GPU {rank}: Processed {processed_count} samples")
                        with open(os.path.join(args.output_res_path, f'results_{rank}.json'), 'w') as f:
                            json.dump(gpu_results, f, indent=4)
                
            except Exception as e:
                logger.error(f"GPU {rank}: Error in processing loop: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        # Save final results for this GPU
        logger.info(f"GPU {rank}: Finished processing {processed_count} samples")
        with open(os.path.join(args.output_res_path, f'results_{rank}.json'), 'w') as f:
            json.dump(gpu_results, f, indent=4)
            
    except Exception as e:
        logger.error(f"GPU {rank}: Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Clean up CUDA memory
        torch.cuda.empty_cache()
        logger.info(f"GPU {rank}: Process cleaned up and exiting")

def data_loader_process(prompt_file, st_data_all, patch_len, stride, input_queue, stop_event):
    logger = logging.getLogger(__name__)
    logger.info("Starting data loader process")
    
    try:
        total_samples = len(prompt_file)
        logger.info(f"Total samples to process: {total_samples}")
        with tqdm(total=total_samples, desc="Loading data") as pbar:
            for idx, instruct_item in enumerate(prompt_file):
                if stop_event.is_set():
                    logger.info("Data loader received stop signal")
                    break
                try:
                    logger.info(f"Processing sample {idx}")
                    st_dict = load_st(instruct_item['df_id'], instruct_item, st_data_all, patch_len, stride)
                    input_queue.put((idx, instruct_item, st_dict))
                    logger.info(f"Put sample {idx} in queue")
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing sample {idx}: {str(e)}")
                    continue
    except Exception as e:
        logger.error(f"Error in data loader process: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        # Signal that all data has been loaded
        logger.info("Data loader finished, sending None signal to all workers")
        for _ in range(args.num_gpus):  # Send None signal to each worker
            input_queue.put(None)
            logger.info("Sent None signal to a worker")
        logger.info("Data loader process finished")

def run_eval(args):
    logger = setup_logging(args.output_res_path)
    logger.info("Starting evaluation")
    
    try:
        os.makedirs(args.output_res_path, exist_ok=True)
        
        # Load prompting file
        logger.info("Loading prompting file")
        prompt_file = load_prompting_file(args.prompting_file, args.data_tag)[-100:]
        logger.info(f"Loaded {len(prompt_file)} samples from prompting file")
        
        # Load ST data
        logger.info("Loading ST data")
        with open(args.st_data_path + 'df.pkl', 'rb') as file:
            st_data_all = pickle.load(file)
        logger.info("ST data loaded successfully")
        
        # Create input queue and stop event
        input_queue = Queue(maxsize=args.num_gpus * 4)
        stop_event = Event()
        
        # Start data loader process
        logger.info("Starting data loader process")
        loader_process = Process(
            target=data_loader_process,
            args=(prompt_file, st_data_all, args.patch_len, args.stride, input_queue, stop_event)
        )
        loader_process.start()
        
        # Use mp.spawn to start worker processes
        logger.info(f"Starting {args.num_gpus} worker processes")
        mp.spawn(
            process_fn,
            args=(args.num_gpus, args, input_queue, stop_event),
            nprocs=args.num_gpus,
            join=True
        )
        
        # Wait for data loader to finish
        logger.info("Waiting for data loader to finish")
        loader_process.join(timeout=30)  # Add timeout to prevent hanging
        
        if loader_process.is_alive():
            logger.warning("Data loader process did not finish in time, terminating...")
            loader_process.terminate()
        
        # Wait for input queue to be empty
        logger.info("Waiting for input queue to be empty")
        while not input_queue.empty():
            time.sleep(1)
        
        # Set stop event to signal workers to finish
        logger.info("Setting stop event")
        stop_event.set()
        
        # Wait a bit for processes to finish
        time.sleep(5)
        
        logger.info("All processes completed. Results are saved in", args.output_res_path)
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # Clean up
        if 'loader_process' in locals() and loader_process.is_alive():
            logger.info("Terminating loader process")
            loader_process.terminate()
        torch.cuda.empty_cache()
        logger.info("Cleanup completed")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=str, default="./checkpoints/Qwen2-7B-Instruct")
    parser.add_argument("--model-name", type=str, default="./checkpoints/reason/")
    parser.add_argument("--prompting_file", type=str, default="ST_data/transport/")
    parser.add_argument("--st_data_path", type=str, default='ST_data/transport/')
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--output_res_path", type=str, default='outputs1')
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lora", type=bool, default=True)
    parser.add_argument("--patch_len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--data_tag", type=str, default='test')

    args = parser.parse_args()
    run_eval(args)
