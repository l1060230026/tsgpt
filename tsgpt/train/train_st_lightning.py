# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
import copy
import numpy as np
from dataclasses import dataclass, field
import json
import logging
import pathlib
import pickle
from typing import Dict, Optional, Sequence, List
from lightning.pytorch.strategies import DeepSpeedStrategy
from tsgpt.model.STQwen2_pl import STQwen2Lightning
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
import lightning as pl
import random

import torch
import torch.nn as nn
import copy
import pandas as pd
import os.path as osp
import deepspeed
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

import transformers
from torch.utils.data import Dataset
from lightning.pytorch import LightningModule, Trainer, seed_everything
import lightning

import tsgpt.conversation as conversation_lib
import re


# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

DEFAULT_STHIS_TOKEN = "<ST_HIS>"
DEFAULT_STPRE_TOKEN = "<ST_PRE>"
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"
DEFAULT_ST_START_TOKEN = "<ST_start>"
DEFAULT_ST_END_TOKEN = "<ST_end>"

class PrintProgressCallback(lightning.Callback):
    def __init__(self, total_epochs):
        super(PrintProgressCallback, self).__init__()
        self.total_epochs = total_epochs

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        current_epoch = trainer.current_epoch
        total_batches = len(trainer.train_dataloader)
        progress = (batch_idx + 1) / total_batches * 100
        sys.stdout.write(f"\rEpoch [{current_epoch + 1}/{self.total_epochs}] - Batch [{batch_idx + 1}/{total_batches}] - {progress:.2f}% complete")
        sys.stdout.flush()
    
    def on_train_epoch_end(self, trainer, pl_module):
        sys.stdout.write("\n")  # Move to the next line after each epoch

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    llm_version: Optional[str] = field(default="v0")
    st_version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mlp_adapter: bool = field(default=False)
    st_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    use_st_start_end: bool = field(default=False)
    model_save_name: Optional[str] = field(default="model_{epoch}-{val_loss:.2f}")
    

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_st: bool = False
    sep_st_conv_front: bool = False
    st_path: Optional[str] = field(default='./checkpoints/align/st_tower.pth') 
    patch_len: int = 20
    stride: int = 10


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_st_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    disable_tqdm: bool = False
    strategy: str = "auto"
    precision: str = "32"
    num_workers: int = 8
    num_nodes: int = 1

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_ST(
        sources: Sequence[str],
        st_cfg: dict,
        cur_token_len: int,
) -> Dict:
    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_ST_PATCH_TOKEN * cur_token_len
            if st_cfg['use_st_start_end']:
                replace_token = DEFAULT_ST_START_TOKEN + replace_token + DEFAULT_ST_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_STHIS_TOKEN, replace_token)
            sentence["value"] = sentence["value"].replace(DEFAULT_STPRE_TOKEN, replace_token)
    return sources


def preprocess_v1(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # mask多余长度
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []

    if roles[sources[0]["from"]] != conv.roles[0]:
        # Skip the first one if it is not from human
        sources = sources[1:]

    conv.messages = []
    for j, sentence in enumerate(sources):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{i}"
        conv.append_message(role, sentence["value"])
    conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids)
            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_ts: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer,)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_ts=has_ts)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_ts=has_ts)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_ts=has_ts)
    # Add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # Tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_ts_token(prompt, tokenizer)) for prompt in prompts]

    if has_ts:
        input_ids = [tokenizer_ts_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_ts:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)
    
class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 st_cfg: dict,
                 **kwargs, ):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        self.patch_len = kwargs.get('patch_len', 20)
        self.stride = kwargs.get('stride', 10)
        self.revin_layer = RevIN(1, affine=0, subtract_last=0)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 

        # Load and analyze data distribution
        list_data_dict = []
        data_path += 'align.jsonl'
        with open(data_path, "r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                list_data_dict.append(record)

        logging.warning("Formatting inputs...")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.st_cfg = st_cfg

    def __len__(self):
        # return 100
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw_data = self.list_data_dict[i]
        query = raw_data['question']
        time_series_str = raw_data['time_series']
        time_series_match = re.search(r'\[([\d\.,\s-]+)\]', time_series_str)
        time_series_str = time_series_match.group(1)
        time_series = np.array([float(x.strip()) for x in time_series_str.split(',')])

        # normalize the data
        data_values = torch.from_numpy(time_series[np.newaxis,:,np.newaxis])
        nan_mask = ~torch.isnan(data_values)
        data_values = self.revin_layer(data_values, mask=nan_mask, mode='norm')
        data_values = data_values.permute(0,2,1)
        # padding the data
        data_values = self.padding_patch_layer(data_values)
        # unfold the data

        data_values = data_values.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        st_len = data_values.shape[-2]
        data_values = data_values.reshape(-1, self.patch_len)
        nan_mask = ~torch.isnan(data_values)
        nan_mask = ~torch.any(~nan_mask, dim=-1)  # If any value in a patch is False, the whole patch is False

        time_str = ''
        time_str += "time_series" + ":" + "<ST_patch>" * st_len
        time_str += f' ST_statistic: MIN:{time_series.min():.3f}, MAX:{time_series.max():.3f}, MEAN:{time_series.mean():.3f}, STD:{time_series.std():.3f}'
        query += '\n' + time_str

        st_data = torch.nan_to_num(data_values, nan=0.0)

        response = raw_data['answer']

        sources = [{'from': 'human', 'value': query}, {'from': 'gpt', 'value': response}]

        data_dict = preprocess(
            sources,
            self.tokenizer)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        
        data_dict['st_data'] = st_data
        data_dict['nan_mask'] = nan_mask  # Add the NaN mask to the returned dictionary

        return data_dict

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

class LazySupervisedDataset_ST(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 st_cfg: dict,
                 **kwargs, ):
        super(LazySupervisedDataset_ST, self).__init__()
        logging.warning("Loading data...")

        data_path = kwargs.get('st_data_path')
        self.patch_len = kwargs.get('patch_len', 20)
        self.stride = kwargs.get('stride', 10)
        self.revin_layer = RevIN(1, affine=0, subtract_last=0)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 

        # Load and analyze data distribution
        list_data_dict = []
        json_path = data_path + 'train.jsonl'
        with open(json_path, "r", encoding="utf-8") as file:
            for line in file:
                record = json.loads(line)
                list_data_dict.append(record)

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.st_cfg = st_cfg
        pkl_path = data_path + 'df.pkl'
        with open(pkl_path, 'rb') as file:
            self.st_data_all = pickle.load(file)

    def __len__(self):
        # return 50
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Use weighted sampling to select an index

        raw_data = self.list_data_dict[i]
        query = raw_data['question']
        df_id = raw_data['df_id']
        timeseries_operation = raw_data['timeseries_operation']
        if 'start' in raw_data.keys():
            start_time, end_time = raw_data['start'], raw_data['end']
            X = self.st_data_all[df_id][start_time:end_time].copy()
        else:
            X = self.st_data_all[df_id].copy()
        st_data = eval(timeseries_operation)
        st_data = st_data.dropna() 
        # Drop non-numeric columns
        st_data = st_data.select_dtypes(include=[np.number])
        # normalize the data    \
        data_values = torch.from_numpy(st_data.values.T[...,np.newaxis])
        nan_mask = ~torch.isnan(data_values)
        data_values = self.revin_layer(data_values, mask=nan_mask, mode='norm')
        data_values = data_values.permute(0,2,1)
        # padding the data
        data_values = self.padding_patch_layer(data_values)
        # unfold the data
        data_values = data_values.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        st_len = data_values.shape[-2]
        data_values = data_values.reshape(-1, self.patch_len)
        nan_mask = ~torch.isnan(data_values)
        nan_mask = ~torch.any(~nan_mask, dim=-1)  # If any value in a patch is False, the whole patch is False
        
        time_str = ''
        for col in st_data.columns:
            time_str += '\n' + col + ":" + "<ST_patch>" * st_len
            time_str += f' ST_statistic: MIN:{st_data[col].min():.3f}, MAX:{st_data[col].max():.3f}, MEAN:{st_data[col].mean():.3f}, STD:{st_data[col].std():.3f}'
        query += '\n' + time_str

        st_data = torch.nan_to_num(data_values, nan=0.0)

        answer = raw_data['answer']
        if 'rationales' in raw_data.keys():
            rationales = raw_data['rationales']
            response = 'Reasoning:\n'
            for rationale in rationales:
                response += f'{rationale}\n'
            response += f'Answer: {answer}'
        else:
            solutions = raw_data['solutions']
            query += '\n\n' + solutions

            response = f'Answer: {answer}'

        sources = [{'from': 'human', 'value': query}, {'from': 'gpt', 'value': response}]

        data_dict = preprocess(
            sources,
            self.tokenizer)
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        
        data_dict['st_data'] = st_data
        data_dict['nan_mask'] = nan_mask  # Add the NaN mask to the returned dictionary

        return data_dict

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        
        # Reverse sequences before padding
        reversed_input_ids = [torch.flip(ids, [0]) for ids in input_ids]

        reversed_input_ids = torch.nn.utils.rnn.pad_sequence(
            reversed_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        # Reverse back after padding
        input_ids = torch.flip(reversed_input_ids, [1])

        # Reverse sequences before padding
        reversed_labels = [torch.flip(ids, [0]) for ids in labels]

        reversed_labels = torch.nn.utils.rnn.pad_sequence(
            reversed_labels,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        
        # Reverse back after padding
        labels = torch.flip(reversed_labels, [1])

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        st_data_batch = [instance['st_data'].to(torch.bfloat16) for instance in instances]
        nan_mask_batch = [instance['nan_mask'] for instance in instances]

        batch['st_data'] = st_data_batch
        batch['st_mask'] = nan_mask_batch
        
        return batch

def make_supervised_stdata_module(tokenizer: transformers.PreTrainedTokenizer,
                                  data_args, training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    print('lazy_preprocess', data_args.lazy_preprocess)
    dataset_cls = (LazySupervisedDataset_ST
                   if data_args.lazy_preprocess else SupervisedDataset)
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                st_cfg=dict(
                                    is_st=data_args.is_st,
                                    sep_st_conv_front=data_args.sep_st_conv_front,
                                    use_st_start_end=getattr(data_args, 'use_st_start_end', False),
                                ),
                                st_data_path=data_args.data_path,
                                patch_len=data_args.patch_len,
                                stride=data_args.stride)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=training_args.per_device_train_batch_size,
                                  num_workers=training_args.num_workers,
                                  collate_fn=data_collator,
                                  prefetch_factor=4,
                                  shuffle=True,
                                  pin_memory=True)
    
    return {"train_dataloader":train_dataloader}

#ST-LLM模型训练函数
def train():

    #参数解析
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    #语言tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    print('model_args.version: ', model_args.llm_version)
    if model_args.llm_version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        # else:
            # tokenizer.pad_token = "<unk>"

        if model_args.llm_version == "mpt":
            conversation_lib.default_conversation = conversation_lib.conv_templates["mpt"]
        elif model_args.llm_version == "mpt_text":
            conversation_lib.default_conversation = conversation_lib.conv_templates["mpt_text"]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    #ST-LLM训练模型加载（STQwen2_pl）
    model = STQwen2Lightning(training_args, model_args, data_args, tokenizer)

    #ST-LLM训练数据处理
    data_module = make_supervised_stdata_module(tokenizer=tokenizer, data_args=data_args, training_args=training_args)

    class CustomCheckpointCallback(pl.Callback):
        def __init__(self, save_path, monitor='val_loss', mode='min', lora=False, tune_mlp_adapter=False):
            super().__init__()
            self.save_path = save_path
            self.monitor = monitor
            self.mode = mode
            self.best_score = None
            self.lora = lora
            self.tune_mlp_adapter = tune_mlp_adapter
        def _is_master_process(self):
            # Check if the current process is the master process
            # Adjust this check according to your configuration
            return deepspeed.comm.get_rank() == 0

        def on_train_epoch_end(self, trainer, pl_module):
            def save_model():
                # Gather and save the model weights
                if self._is_master_process():
                    if self.tune_mlp_adapter:
                        os.makedirs(f"{self.save_path}", exist_ok=True)
                        st_tower = pl_module.model.get_model().st_tower.state_dict()
                        torch.save(st_tower, os.path.join(f"{self.save_path}", 'st_tower.pth'))
                    if self.lora:
                        os.makedirs(f"{self.save_path}", exist_ok=True)
                        pl_module.model.save_pretrained(f"{self.save_path}", save_embedding_layers=False)

            save_model()

        def _is_master_process(self):
            # Check if distributed training is initialized and if the current process is the main process
            if torch.distributed.is_initialized():
                return torch.distributed.get_rank() == 0
            return True
        
    adapter_checkpoint_callback = CustomCheckpointCallback(training_args.output_dir, lora=training_args.lora_enable, tune_mlp_adapter=model_args.tune_mlp_adapter)
    
    if training_args.strategy == "deepspeed_stage_3_offload":
        strategy=DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            allgather_bucket_size=5e6,
            reduce_bucket_size=5e6,
        )

    elif training_args.strategy == "deepspeed_stage_2_offload":
        
        strategy=DeepSpeedStrategy(
            stage=2,
            offload_optimizer=False,
            offload_parameters=True,
            allgather_bucket_size=5e6,
            reduce_bucket_size=5e6,
        )

    else:
        strategy = training_args.strategy

    tf_dir = pathlib.Path(training_args.output_dir) / "tf_logs"
    tf_dir.mkdir(exist_ok=True, parents=True)
    tb_logger = TensorBoardLogger(
        save_dir=tf_dir.resolve(),
        name="STQwen2",
        version=model_args.st_version
    )
    
    csv_logger = CSVLogger(tf_dir.resolve(), 
                           name="csv",
                           version=model_args.st_version)

    trainer = Trainer(
        default_root_dir=training_args.output_dir,
        max_epochs=int(training_args.num_train_epochs),
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        accelerator="gpu",
        strategy=strategy,
        precision=training_args.precision,
        logger=[csv_logger],
        callbacks=[adapter_checkpoint_callback],
        num_nodes=training_args.num_nodes
    )

    #ST-LLM模型执行训练
    trainer.fit(model, train_dataloaders=data_module['train_dataloader'],)

if __name__ == "__main__":
    train()
