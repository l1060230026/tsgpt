import os
import random
from typing import Any, Optional, Dict, List
import logging
import torch
from lightning.pytorch import LightningModule
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
from tsgpt.model.STQwen2 import STQwen2ForCausalLM
import tsgpt.conversation as conversation_lib
import transformers
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

class STQwen2Lightning(LightningModule):
    def __init__(self,
        training_args, model_args, data_args, tokenizer,
        **kwargs,
    ):
        super().__init__()
        self.training_args = training_args
        self.model_args = model_args
        self.data_args = data_args
        compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

        bnb_model_from_pretrained_args = {'use_flash_attention_2':True,
        'torch_dtype':torch.bfloat16}

        ## load 4 8 bit
        if training_args.bits in [4, 8]:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_int8_training
            bnb_model_from_pretrained_args.update(dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
                )
            ))

        self.model = STQwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )

        self.model.config.use_cache = False
        if model_args.freeze_backbone:
            self.model.requires_grad_(False)


        if training_args.bits in [4, 8]:
            print('training_args.bits in [4, 8]')
            self.model.config.torch_dtype = (
                torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
            self.model = prepare_model_for_int8_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

        if training_args.gradient_checkpointing:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            self.model.model.gradient_checkpointing = True
            self.model.model._gradient_checkpointing_func = torch.utils.checkpoint.checkpoint

        #是否Lora训练（Step2）
        if training_args.lora_enable:
            print('lora_enable:', training_args.lora_enable)
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=training_args.lora_r,
                lora_alpha=training_args.lora_alpha,
                target_modules=find_all_linear_names(self.model),
                lora_dropout=training_args.lora_dropout,
                bias=training_args.lora_bias,
                task_type="CAUSAL_LM",
                inference_mode=False,
            )
            
            if training_args.bits == 16:
                if training_args.bf16:
                    self.model.to(torch.bfloat16)
                if training_args.fp16:
                    self.model.to(torch.float16)
            logging.warning("Adding LoRA adapters...")
            self.model = get_peft_model(self.model, lora_config)

            adapter_path = os.path.join(training_args.output_dir, 'adapter_model.safetensors')
            if os.path.exists(adapter_path):
                from safetensors.torch import load_file
                loaded_state_dict = load_file(adapter_path)
                model_state_dict = self.model.state_dict()
                for name, param in loaded_state_dict.items():
                    new_name = name.replace('.weight', '.default.weight')
                    model_state_dict[new_name].copy_(param)

        #时空tokenizer
        if os.path.exists(training_args.output_dir + 'st_tower.pth'):
            self.model.get_model().initialize_st_tower(data_args.patch_len, self.model.config.hidden_size, training_args.output_dir + 'st_tower.pth')
        elif os.path.exists(data_args.st_path):
            self.model.get_model().initialize_st_tower(data_args.patch_len, self.model.config.hidden_size, data_args.st_path)
        else:
            self.model.get_model().initialize_st_tower(data_args.patch_len, self.model.config.hidden_size)
        data_args.is_st = True
        
        #是否时空tokenizer训练（Step1）
        if model_args.tune_mlp_adapter:
            for p in self.model.get_model().st_tower.parameters():
                p.requires_grad = True

        else:
            for p in self.model.get_model().st_tower.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            self.model.get_model().st_projector.to(dtype=compute_dtype)
            self.model.st_pred_linear_1.to(dtype=compute_dtype)
            self.model.st_pred_linear_2.to(dtype=compute_dtype)
            self.model.st_pred_linear_3.to(dtype=compute_dtype)

        self.model.config.use_st_start_end = data_args.use_st_start_end = model_args.use_st_start_end
        training_args.use_st_start_end = model_args.use_st_start_end
        self.model.config.sep_st_conv_front = data_args.sep_st_conv_front
        print('use_st_start_end', training_args.use_st_start_end, 'sep_st_conv_front', self.model.config.sep_st_conv_front)
        self.model.initialize_st_tokenizer(use_st_start_end=model_args.use_st_start_end, tokenizer=tokenizer, device=torch.device('cpu'))

        params_no_grad = [n for n, p in self.model.named_parameters() if not p.requires_grad]
        if len(params_no_grad) > 0:
            if training_args.fsdp is not None and len(training_args.fsdp) > 0:
                if len(params_no_grad) < 10:
                    print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'.format(
                        len(params_no_grad), params_no_grad))
                else:
                    print(
                        '[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'.format(
                            len(params_no_grad), ', '.join(params_no_grad[:10])))
                print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
                print(
                    "[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

                from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
                def patch_FSDP_use_orig_params(func):
                    def wrap_func(*args, **kwargs):
                        use_orig_params = kwargs.pop('use_orig_params', True)
                        return func(*args, **kwargs, use_orig_params=use_orig_params)

                    return wrap_func

                FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)

        if training_args.bits in [4, 8]:
            from peft.tuners.lora import LoraLayer
            for name, module in self.model.named_modules():
                if isinstance(module, LoraLayer):
                    if training_args.bf16:
                        module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        if training_args.bf16 and module.weight.dtype == torch.float32:
                            module = module.to(torch.bfloat16)

        self.model.train()

    def training_step(self, batch, batch_idx):
        bs = len(batch["input_ids"])
        loss = self.model(**batch)[0]
        log_dict = {f'train_loss': loss.item()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        return loss
        
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()], "lr_scale": [1e-5, 1e-4]
            }
        ]

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay,
        )
        
        num_training_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * self.training_args.warmup_ratio),
            num_training_steps=num_training_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
