from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    Qwen2Config, Qwen2Model, Qwen2ForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.configuration_utils import PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import json
import os.path as osp
import glob


IGNORE_INDEX = -100
DEFAULT_STHIS_TOKEN = "<ST_HIS>"
DEFAULT_STPRE_TOKEN = "<ST_PRE>"
DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"
DEFAULT_ST_START_TOKEN = "<ST_start>"
DEFAULT_ST_END_TOKEN = "<ST_end>"


def MAE_torch(pred, true, mask_value=None):
    if mask_value != None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    mae_loss = torch.abs(true - pred)
    return torch.mean(mae_loss)

def scaler_mae_loss(scaler=None, mask_value=None):
    def loss(preds, labels, mask=None):
        if scaler is not None:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        if mask is not None:
            preds = preds * mask
            labels = labels * mask
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

class STQwen2Config(Qwen2Config):
    model_type = "STQwen2"

class STPretrainConfig:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def load_model_pretrained(model_name, pretrain_model_path):
    # load conig json
    print("************************", pretrain_model_path)
    assert osp.exists(osp.join(pretrain_model_path, 'config.json')), 'config.json missing'
    with open(osp.join(pretrain_model_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)
    args = STPretrainConfig(config_dict)
    model = model_name(args)
    pkl_files = glob.glob(osp.join(pretrain_model_path, '*.pkl'))
    state_dict = torch.load(pkl_files[0])
    # print(state_dict.keys())
    if 'logit_scale' in state_dict.keys():
        state_dict.pop('logit_scale')
    print('loading ST pre train model')
    model.load_state_dict(state_dict)

    return model, args

class TimeMoeInputEmbedding(nn.Module):
    """
    Use a mlp layer to embedding the time-series.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.config = PretrainedConfig()
        self.input_size =  input_size # default 1
        self.hidden_size = hidden_size
        self.emb_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.gate_layer = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN['silu']

    def forward(self, x):
        emb = self.act_fn(self.gate_layer(x)) * self.emb_layer(x)
        return emb


class STQwen2Model(Qwen2Model):
    config_class = STQwen2Config

    def __init__(self, config: Qwen2Model):
        super(STQwen2Model, self).__init__(config)
        self.st_start_id0 = -100000
        self.st_start_id1 = -100000
        self.st_start_id2 = -100000
        self.st_end_id1 = -100000
        self.st_end_id2 = -100000
        self.pre_STE = None
        self.gradient_checkpointing = None
        self._gradient_checkpointing_func = None

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            st_data: Optional[list] = None,
            st_mask: Optional[list] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        st_tower = self.st_tower
        # wt_tower = self.get_wt_tower()
        if st_tower is not None and (input_ids.shape[1] != 1 or self.training) and st_data is not None:

            new_input_embeds = []
            # new_stpre_embeds = []
            cur_st_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                
                cur_st_data = st_data[cur_st_idx]
                cur_st_mask = st_mask[cur_st_idx]
                cur_st_features = st_tower(cur_st_data)
                # pre_WTE, WTE_out = wt_tower(wt_data_x[..., :-2], wt_data_x[..., -2].to(torch.long))

                num_patches = cur_st_features.shape[0]

                st_start_tokens = torch.where(cur_input_ids == self.st_tower.config.st_patch_token)[0]

                # Replace all positions specified by st_start_tokens with cur_st_features
                for i in range(len(st_start_tokens)):
                    st_start_token_pos = st_start_tokens[i]
                    attention_mask[cur_st_idx, st_start_token_pos] = cur_st_mask[i]
                    cur_new_input_embeds = torch.cat((
                        cur_input_embeds[:st_start_token_pos].detach(),
                        cur_st_features[i:i+1],
                        cur_input_embeds[st_start_token_pos + 1:].detach(),
                    ), dim=0)
                    cur_input_embeds = cur_new_input_embeds

                cur_st_idx += 1
                new_input_embeds.append(cur_input_embeds)

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        if self.training and self.gradient_checkpointing:
            inputs_embeds.requires_grad_(True)

        return super(STQwen2Model, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    def initialize_st_tower(self, enc_dim, hidden_size, st_path=None):
        self.st_tower = TimeMoeInputEmbedding(enc_dim, hidden_size)
        if st_path is not None:
            self.st_tower.load_state_dict(torch.load(st_path))



class STQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = STQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = STQwen2Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_st_pre_res(self):
        return self.st_pre_res

    def reset_st_pre_res(self):
        self.st_pre_res = []

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            st_data: Optional[list] = None,
            st_mask: Optional[list] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            st_data=st_data,
            st_mask=st_mask,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()


            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            loss = loss_fct(shift_logits, shift_labels)
            
        if not return_dict:
            # print('not return dict')
            output = (logits,) + outputs[1:]
            print(loss.shape)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # print(sss)
        if past_key_values:
            # print('past_key_values', input_ids.shape)
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                # "st_data": [kwargs.get("st_data", None)],
                "st_data": kwargs.get("st_data", None),
                "st_mask": kwargs.get("st_mask", None),
                # "edge_index_reps": kwargs.get("edge_index_reps", None),
            }
        )
        # print('model_inputs.update')
        return model_inputs

    def reset_lm_head(self):
        self.get_input_embeddings().weight.data[-3:, :] = self.lm_head_add.weight.data

    def initialize_st_tokenizer(self, use_st_start_end, tokenizer, device):
        vision_config = self.get_model().st_tower.config
        vision_config.use_st_start_end = use_st_start_end
        tokenizer.add_tokens([DEFAULT_ST_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        vision_config.st_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_ST_PATCH_TOKEN])[0]
        
AutoConfig.register("STQwen2", STQwen2Config)
AutoModelForCausalLM.register(STQwen2Config, STQwen2ForCausalLM)    
