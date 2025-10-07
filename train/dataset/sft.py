import torch
import torch.nn.functional as F
from typing import Callable
from train.dataset.base import CollateDataset, role_template
from train.utils.sequence import pad_sequences
from transformers import PreTrainedTokenizer

import datasets

class SFTDataset(CollateDataset):
    def __init__(
            self,
            dataset: datasets.Dataset, 
            prompt_key: str,
            response_key: str,
            tokenizer: PreTrainedTokenizer,
            max_length: int, 
            num_processors: int = 8,
            ring_attn_size: int = None, 
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ring_attn_size = ring_attn_size

        self.prompt_key = prompt_key
        self.response_key = response_key
        # self.dataset = dataset#.filter(lambda x: x['ctx_length'] in ('128k',))

        
        dataset = dataset.filter(self.filter_data, 
                                 num_proc = 1)
        
        processed_dataset = dataset.map(self.process_data,
                                        remove_columns = dataset.column_names,
                                        num_proc = num_processors)

        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.input_ids_lens = processed_dataset["input_ids_len"]


    def filter_data(self, data:dict):
        if self.max_length < 128000:
            prompt_template = data[self.prompt_key]
            response_template = role_template(data[self.response_key], "assistant")
            tokenized = self.tokenizer.apply_chat_template(
                prompt_template + response_template, tokenize = True, 
                max_length = 128000, padding = False,
                truncation = True, return_tensors = 'pt'
            )
            if tokenized.numel() > self.max_length: return False
        return True

    def process_data(self, data):
        prompt_template = data[self.prompt_key]
        response_text = data[self.response_key]
        if isinstance(response_text, str):
            response_text = [{"role":"assistant", "content": response_text}]
        prompt = self.tokenizer.apply_chat_template(
            prompt_template, tokenize = False, add_generation_prompt = True
        )
        response = self.tokenizer.apply_chat_template(
            prompt_template + response_text, tokenize = False
        )[len(prompt): ]


        prompt_token = self.tokenizer(prompt, max_length = self.max_length,
                                      padding = False,
                                      truncation = True,
                                      return_tensors = 'pt',
                                      add_special_tokens = False)
        response_token = self.tokenizer(response, max_length = self.max_length,
                                        padding = False,
                                        truncation = True,
                                        return_tensors = 'pt',
                                        add_special_tokens = False)
        
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
        input_ids_len = prompt_ids_len + response_token["attention_mask"].int().sum().item()

        return dict(
            prompt = prompt,
            response = response,
            prompt_ids_len = prompt_ids_len,
            input_ids_len = input_ids_len,
            response_ranges = None # not multiturn
        )

    def __len__(self): return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]
        prompt_ids_len = self.prompt_ids_lens[idx]

        text = (prompt + response).rstrip("\n")
        if not text.endswith(self.tokenizer.eos_token):
            text += " " + self.tokenizer.eos_token
        
        input_token = self.tokenizer(
            text, max_length = self.max_length,
            padding = False,
            truncation = True,
            return_tensors = "pt",
            add_special_tokens = False
        )

        loss_mask = self.get_loss_mask(input_token["input_ids"], idx)

        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True


        return input_token["input_ids"], input_token["attention_mask"], loss_mask


    def get_loss_mask(self, input_ids, idx):
        loss_mask = torch.zeros_like(input_ids, dtype = torch.bool)
        prompt_ids_len = self.prompt_ids_lens[idx]
        # TODO: multi-turn
        loss_mask[0, prompt_ids_len: ] = True

        return loss_mask


    def collate_fn(self, item_list):
        packed_input_ids = []
        packed_attention_masks = []
        packed_loss_masks = []
        seq_lens = []
        for index, (input_id, attention_mask, loss_mask) in enumerate(item_list):
            packed_input_ids += [input_id.flatten()]
            packed_attention_masks += [torch.full_like(input_id.flatten(), index + 1)]
            packed_loss_masks += [loss_mask.flatten()]
            seq_lens += [attention_mask.int().sum().item()]

        packed_input_ids = torch.concat(packed_input_ids).unsqueeze(0)
        packed_attention_masks = torch.concat(packed_attention_masks).unsqueeze(0)
        packed_loss_masks = torch.concat(packed_loss_masks).unsqueeze(0)

        if packed_input_ids.numel() % self.ring_attn_size:
            padding_len = self.ring_attn_size - (packed_input_ids.numel() % self.ring_attn_size)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value = self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value = 0)
            packed_loss_masks = F.pad(packed_loss_masks, (0, padding_len), value = 0)

        
        return packed_input_ids, packed_attention_masks, packed_loss_masks, seq_lens
        





class SFTDataset_release(CollateDataset):
    def __init__(
            self,
            dataset: datasets.Dataset, # raw_dataset
            chat_template_key: str = None,
            chat_template_builder: Callable = None,
            tokenizer: PreTrainedTokenizer = None,
            max_length: int = 128000, # max_length for tokenizer
            num_processors: int = 8,
            ring_attn_size: int = None, 
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ring_attn_size = ring_attn_size

        self.chat_template_key = chat_template_key
        self.chat_template_builder = chat_template_builder
        
        self.dataset = dataset#.filter(lambda x: x['ctx_length'] in ('128k',))
        processed_dataset = dataset.map(self.process_data,
                                        remove_columns = dataset.column_names,
                                        num_proc = num_processors)

        self.input_texts = processed_dataset["input_text"]
        self.response_rang_lists = processed_dataset["response_range_list"]
        self.input_ids_lens = processed_dataset["input_ids_len"]
    
    def process_data(self, data):
        if self.chat_template_builder is not None: # means that dataset has no key 'chat_template'
            data[self.chat_template_key] = self.chat_template_builder(data)
        
        chat_template = data[self.chat_template_key]
        
        response_range_list = []
        assistent_ids = [-1] + [i for i, k in enumerate(chat_template) \
                         if k["role"] == "assistant"]
        assert assistent_ids[-1] + 1 == len(chat_template), \
            f"You pass an message after assistant but no reply: {chat_template}"

        for j in assistent_ids:
            prompt_template = chat_template[: j]
            assistent_template = chat_template[j]
            prompt = self.tokenizer.apply_chat_template(
                prompt_template, tokenize = False, add_generation_prompt = True
            )
            response = self.tokenizer.apply_chat_template(
                prompt_template + assistent_template, tokenize = False
            )[len(prompt): ]

            prompt_token = self.tokenizer(prompt, max_length = self.max_length,
                                          padding = False, 
                                          truncation = True,
                                          return_tensors = "pt",
                                          add_special_tokens = False)
            response_token = self.tokenizer(response, max_length = self.max_length,
                                            padding = False,
                                            truncation = True,
                                            return_tensors = "pt",
                                            add_special_tokens = False)
            
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            full_ids_len = prompt_ids_len + response_token["attention_mask"].int().sum().item()
            
            response_range_list += [(prompt_ids_len, full_ids_len)]
        
        input_ids_len = response_range_list[-1][-1]

        return dict(
            input_text = prompt + response,
            input_ids_len = input_ids_len,
            response_range_list = response_range_list
        )

    def __len__(self): return len(self.input_texts)

    def __getitem__(self, idx):
        text = self.input_texts[idx]
        if not text.endswith(self.tokenizer.eos_token):
            text += " " + self.tokenizer.eos_token
        
        input_token = self.tokenizer(
            text, max_length = self.max_length,
            padding = False,
            truncation = True,
            return_tensors = "pt",
            add_special_tokens = False
        )

        loss_mask = self.get_loss_mask(input_token["input_ids"], idx)

        input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        input_token["attention_mask"][0][-1] = True


        return input_token["input_ids"], input_token["attention_mask"], loss_mask


    def get_loss_mask(self, input_ids, idx):
        loss_mask = torch.zeros_like(input_ids, dtype = torch.bool)
        for l, r in self.response_rang_lists[idx]:
            loss_mask[0, l: r] = True
        return loss_mask


    def collate_fn(self, item_list):
        packed_input_ids = []
        packed_attention_masks = []
        packed_loss_masks = []
        seq_lens = []
        for index, (input_id, attention_mask, loss_mask) in enumerate(item_list):
            packed_input_ids += [input_id.flatten()]
            packed_attention_masks += [torch.full_like(input_id.flatten(), index + 1)]
            packed_loss_masks += [loss_mask.flatten()]
            seq_lens += [attention_mask.int().sum().item()]

        packed_input_ids = torch.concat(packed_input_ids).unsqueeze(0)
        packed_attention_masks = torch.concat(packed_attention_masks).unsqueeze(0)
        packed_loss_masks = torch.concat(packed_loss_masks).unsqueeze(0)

        if packed_input_ids.numel() % self.ring_attn_size:
            padding_len = self.ring_attn_size - (packed_input_ids.numel() % self.ring_attn_size)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value = self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value = 0)
            packed_loss_masks = F.pad(packed_loss_masks, (0, padding_len), value = 0)

        
        return packed_input_ids, packed_attention_masks, packed_loss_masks, seq_lens