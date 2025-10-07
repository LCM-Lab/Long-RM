from typing import Literal
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
def init_tokenizer(model_name: str) -> PreTrainedTokenizer:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name,
                                         trust_remote_code = True)

def init_model(module_name: str,
               enable_gradient_checkpoint:bool = True,
               attn_implementation = "flash_attention_2",
               model_type :Literal["causal", "classifier"] = "causal", 
               torch_dtype = torch.bfloat16,
               **hf_kwargs) -> PreTrainedModel:
    if model_type == "causal":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            module_name,
            attn_implementation = attn_implementation,
            torch_dtype = torch_dtype,
            trust_remote_code = True,
            ** hf_kwargs
        )
    elif model_type =="classifier":
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(
            module_name, 
            attn_implementation = attn_implementation,
            torch_dtype = torch_dtype,
            trust_remote_code = True, 
            ** hf_kwargs
        )
        
        model._load_path = module_name
    if enable_gradient_checkpoint:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    return model