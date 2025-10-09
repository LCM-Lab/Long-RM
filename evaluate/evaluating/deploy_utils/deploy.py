from typing import List, Dict, Literal 
import os, torch
from transformers import (
    AutoConfig, 
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    PreTrainedTokenizer
)
from evaluating.base_utils.integration import Selector


def get_cuda_visible_devices():
    return os.environ["CUDA_VISIBLE_DEVICES"]

def get_tokenizer_special_len(tokenizer):
    normal = tokenizer.eos_token_id
    template = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': tokenizer.eos_token}],
        add_generation_prompt=True,
        return_tensors = 'pt'
    ).flatten()
    # return template,normal
    out = torch.nonzero(template==normal)
    prefix = out[-2].item()# the second last, include <begin_of_text>
    suffix = -(template.size(0) - (out[-1].item() + 1))  # the last
    if suffix == 0: suffix = int(1e18) 
    user_token_id = tokenizer("user",add_special_tokens=False,
                              return_tensors = 'pt')['input_ids'].flatten()
    default_prefix = (template == user_token_id).nonzero()[0].item() - 1

    return prefix, suffix, default_prefix


def role_template(content:str, role: Literal["system", "user", "assistant"]):
    return [dict(role = role, content = content)]


def apply_chat_template(tokenizer: AutoTokenizer, 
                        messages: List[Dict[str, str]], 
                        add_generation_prompt:bool = True,
                        tokenize: bool = True):

    return tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt = add_generation_prompt,
        tokenize = tokenize
    )


def path_from_config(config, dtype:Literal["model", "tokenizer"] = "model"):
    return getattr(config, f"{dtype}_path",
                   getattr(config, f"{dtype}_name", ""))

def tokenizer_path_from_config(config):
    tokenier_name = path_from_config(config, "tokenizer")
    if tokenier_name: return tokenier_name
    return path_from_config(config)


class _NoneTokenizer:
    chat_template = "openai-api"
    def apply_chat_template(messages, *args, **kwargs):
        return messages
    def __call__(self, *args, **kwargs):
        return dict(input_ids = None)
    
def setup_tokenizer(config) -> PreTrainedTokenizer:
    try:
        if not isinstance(config, str):
            config = tokenizer_path_from_config(config)
        tokenizer = AutoTokenizer.from_pretrained(config)
        try:
            (tokenizer.repeat_prefix_len,
            tokenizer.repeat_suffix_len,
            tokenizer.default_prefix_len) = get_tokenizer_special_len(tokenizer)
        except Exception as e: pass
        return tokenizer

    except Exception as e:
        return _NoneTokenizer()

def setup_model(model_name: str):
    if not isinstance(model_name, str):
       model_name = path_from_config(model_name)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code = True)
    return AutoModelForCausalLM.from_pretrained(model_name, 
                                                config=config, 
                                                attn_implementation="flash_attention_2",
                                                device_map = "auto",
                                                trust_remote_code=True,
                                                torch_dtype=torch.float16
                                                ).half().eval()

class NormalGPUDeploy:
    @classmethod
    def deploy(cls, model_config):
        model_name = model_config.model_name \
            if hasattr(model_config,"model_name") \
                else model_config.model_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path_from_config(model_config),
            trust_remote_code = True
        )
        
        config = AutoConfig.from_pretrained(model_name, trust_remote_code = True)
        model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                    config=config, 
                                                    attn_implementation="flash_attention_2",
                                                    device_map = "auto",
                                                    trust_remote_code = True,
                                                    torch_dtype=torch.float16
                                                    ).half().eval()
        return model, tokenizer


class DisRMGPUDeploy:
    @classmethod
    def deploy(cls, model_config):
        model_name = model_config.model_name \
            if hasattr(model_config,"model_name") \
                else model_config.model_path
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path_from_config(model_config),
            trust_remote_code = True
        )
        
        config = AutoConfig.from_pretrained(model_name, trust_remote_code = True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            config=config, 
            attn_implementation="flash_attention_2",
            device_map = "auto", 
            trust_remote_code = True, 
            torch_dtype = torch.bfloat16, 
            # num_labels = 1
            ).half().eval()
        return model, tokenizer

class VllmGPUDeploy:
    @classmethod
    def deploy(cls, model_config):
        import vllm
        return vllm.LLM(
            model = getattr(model_config, "model_path",
                            getattr(model_config, "model_name","")),
            tensor_parallel_size = len(get_cuda_visible_devices().split(",")),
            trust_remote_code=True,
            gpu_memory_utilization=getattr(
                model_config, "gpu_memory_utilization", 0.98
                ),
            swap_space = 4,
            enforce_eager=True
            )


class LLMDeploySelector(Selector):
    mapping = {
        'normal_gpu'    : NormalGPUDeploy,
        'vllm_gpu'      : VllmGPUDeploy,
        'vllm_gpu_batch': VllmGPUDeploy,
    }
