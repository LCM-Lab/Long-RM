from typing import Callable
import torch, random
import torch.nn.functional as F
from train.dataset.base import CollateDataset, role_template
from train.utils.sequence import pad_sequences
from transformers import PreTrainedTokenizer

import datasets


class PreferenceDataset(CollateDataset):
    '''
    A preference dataset for preference-based reward training.
    
    Used for DPOTrainer and SimPOTrainer, when in SimPOTrainer, 
    the dataset's rejected sequences may be multiple.
    '''
    def __init__(
            self,
            dataset: datasets.Dataset,
            prompt_key: str,
            chosen_key: str,
            rejected_key: str,
            tokenizer: PreTrainedTokenizer,
            max_length: int,
            num_rejects: int = 1,
            sample_rejects: Callable = None,
            num_processors: int = 8,
            ring_attn_size: int = 1, 
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ring_attn_size = ring_attn_size

        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        self.num_rejects = num_rejects
        self.sample_rejects = sample_rejects
        
        dataset = dataset.filter(self.filter_data, 
                                 num_proc = num_processors)

        processed_dataset = dataset.map(self.process_data, 
                                        remove_columns = dataset.column_names, 
                                        num_proc = num_processors)
        
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.input_ids_lens = processed_dataset["input_ids_len"]
    

    def filter_data(self, data:dict):
        if self.max_length < 128000:
            prompt_template = data[self.prompt_key]
            chosen_template = role_template(data[self.chosen_key], "assistant")
            tokenized = self.tokenizer.apply_chat_template(
                prompt_template + chosen_template, tokenize = True, 
                max_length = 128000,padding = False,
                truncation = True
            )
            if len(tokenized) > self.max_length: return False
        reject = data[self.rejected_key]
        return (1 if isinstance(reject, str) else len(reject)) >= self.num_rejects

    def process_data(self, data: dict):
        rejects = [data[self.rejected_key]] \
            if isinstance(data[self.rejected_key], str)\
            else data[self.rejected_key]
        
        if len(rejects) > self.num_rejects:
            rejects = random.sample(rejects, self.num_rejects) \
                if self.sample_rejects is None else self.sample_rejects(rejects, self.num_rejects)

        prompt_template = data[self.prompt_key]
        if isinstance(prompt_template, str):
            prompt_template = role_template(prompt_template, "user")
        chosen_template = role_template(data[self.chosen_key], "assistant")
        rejected_templates = [role_template(r, "assistant") for r in rejects]

        prompt = self.tokenizer.apply_chat_template(
            prompt_template, tokenize = False, add_generation_prompt = True
        )
        chosen = self.tokenizer.apply_chat_template(
            prompt_template + chosen_template, tokenize = False
        )[len(prompt): ]
        rejecteds = [self.tokenizer.apply_chat_template(
            prompt_template + r_template, tokenize = False
        )[len(prompt): ] for r_template in rejected_templates]
        
        margin = data.get("margin", 0) 

        prompt_token = self.tokenizer(prompt,
                                      max_length = self.max_length,
                                      padding = False,
                                      truncation = True,
                                      return_tensors = 'pt',
                                      add_special_tokens = False)
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

        reponses = [self.tokenizer(r, max_length = self.max_length,
                                   padding = False,truncation = True,
                                   return_tensors = 'pt', add_special_tokens = False)\
                                    for r in [chosen] + rejecteds]
        responses_ids_len = [r['attention_mask'].int().sum().item() for r in reponses]

        return dict(
            prompt = prompt,
            chosen = chosen,
            reject = rejecteds,
            prompt_ids_len = prompt_ids_len,
            input_ids_len = prompt_ids_len + max(responses_ids_len)
        )

    def __len__(self): return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        chosen = self.chosens[idx]
        rejects = self.rejects[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenizer.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen, max_length = self.max_length,
            padding = False,
            truncation = True,
            return_tensors = "pt",
            add_special_tokens = False
        )
        chosen_loss_mask = self.get_loss_mask(chosen_token["input_ids"], idx)
        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True

        rejects_input_ids, rejects_attn_mask, rejects_loss_mask = [], [], []
        for reject in rejects:
            reject = (prompt + reject).rstrip("\n")
            if not reject.endswith(self.tokenizer.eos_token):
                reject += " " + self.tokenizer.eos_token
            reject_token = self.tokenizer(
                reject, max_length = self.max_length,
                padding = False,
                truncation=  True,
                return_tensors = "pt",
                add_special_tokens = False
            )
            reject_loss_mask = self.get_loss_mask(reject_token["input_ids"], idx)
            reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            reject_token["attention_mask"][0][-1] = True

            rejects_input_ids += [reject_token['input_ids']]
            rejects_attn_mask += [reject_token['attention_mask']]
            rejects_loss_mask += [reject_loss_mask]

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            chosen_loss_mask,
            rejects_input_ids,
            rejects_attn_mask,
            rejects_loss_mask
        )

    def get_loss_mask(self, input_ids:torch.Tensor, idx:int):
        loss_mask = torch.zeros_like(input_ids, dtype = torch.bool)
        prompt_ids_len = self.prompt_ids_lens[idx]
        loss_mask[0, prompt_ids_len: ] = True
        loss_mask[0, -1] = True

        return loss_mask


    def collate_fn(self, item_list):

        packed_input_ids_list = [[] for _ in range(self.num_rejects + 1)]
        packed_attention_masks_list = [[] for _ in range(self.num_rejects + 1)]
        packed_loss_masks_list = [[] for _ in range(self.num_rejects + 1)]
        seq_lens_list = [[] for _ in range(self.num_rejects + 1)]

        merged_seq_lens = [0 for _ in range(self.num_rejects + 2)]

        for index, (chosen_id, chosen_attention_mask, chosen_loss_mask,
                    rejects_id, rejects_attention_mask, rejects_loss_mask) in enumerate(item_list):
            packed_input_ids_list[0] += [chosen_id.flatten()]
            packed_attention_masks_list[0] += [torch.full_like(chosen_id.flatten(), index + 1)]
            packed_loss_masks_list[0] += [chosen_loss_mask.flatten()]
            seq_lens_list[0] += [len(chosen_id.flatten())]

            for r_index, (reject_id, reject_attention_mask, reject_loss_mask) in enumerate(zip(
                rejects_id[: self.num_rejects], rejects_attention_mask[: self.num_rejects], rejects_loss_mask[: self.num_rejects]
            )):

                packed_input_ids_list[r_index + 1] += [reject_id.flatten()]
                packed_attention_masks_list[r_index + 1] += [torch.full_like(reject_id.flatten(), 
                                                                             index + 1 + (r_index + 1) * len(item_list))]
                packed_loss_masks_list[r_index + 1] += [reject_loss_mask.flatten()]
                seq_lens_list[r_index + 1] += [len(reject_id.flatten())]
        

        for i_index in range(self.num_rejects + 1):
            packed_input_ids_list[i_index] = torch.concat(packed_input_ids_list[i_index]).unsqueeze(0)
            packed_attention_masks_list[i_index] = torch.concat(packed_attention_masks_list[i_index]).unsqueeze(0)
            packed_loss_masks_list[i_index] = torch.concat(packed_loss_masks_list[i_index]).unsqueeze(0)
            merged_seq_lens[i_index + 1] = sum(seq_lens_list[i_index]) + merged_seq_lens[i_index]
        
        packed_input_ids = torch.concat(packed_input_ids_list, dim = -1)
        packed_attention_masks = torch.concat(packed_attention_masks_list, dim = -1)
        packed_loss_masks = torch.concat(packed_loss_masks_list, dim = -1)
        packed_seq_lens = [sl for seq_lens in seq_lens_list for sl in seq_lens]

        if packed_input_ids.numel() % self.ring_attn_size:
            padding_len = self.ring_attn_size - (packed_input_ids.numel() % self.ring_attn_size)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value = self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value = 0)
            packed_loss_masks = F.pad(packed_loss_masks, (0, padding_len), value = 0)
            
        


        return (packed_input_ids, packed_attention_masks, packed_loss_masks, 
                seq_lens_list, packed_seq_lens, merged_seq_lens)








class PreferenceDataset_legacy(CollateDataset):
    '''
    curruently support DPO only
    '''
    def __init__(
            self,
            dataset: datasets.Dataset, # raw_dataset
            prompt_key: str,
            chosen_key: str,
            rejected_key: str,
            tokenizer: PreTrainedTokenizer,
            max_length: int, # max_length for tokenizer
            num_processors: int = 8,
            ring_attn_size: int = 1, 
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ring_attn_size = ring_attn_size

        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key

        processed_dataset = dataset.map(self.process_data, 
                                        remove_columns = dataset.columns, 
                                        num_proc = num_processors)
        
        self.prompts = processed_dataset["prompt"]
        self.chosens = processed_dataset["chosen"]
        self.rejects = processed_dataset["reject"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]

    def process_data(self, data: dict):
        prompt_template = data[self.prompt_key]
        chosen_template = role_template(data[self.chosen_key], "assistant")
        rejected_template = role_template(data[self.rejected_key], "assistant")

        prompt = self.tokenizer.apply_chat_template(
            prompt_template, tokenize = False, add_generation_prompt = True
        )
        chosen = self.tokenizer.apply_chat_template(
            prompt_template + chosen_template, tokenize = False
        )[len(prompt): ]
        rejected = self.tokenizer.apply_chat_template(
            prompt_template + rejected_template, tokenize = False
        )[len(prompt): ]
        
        margin = data.get("margin", 0) 

        prompt_token = self.tokenizer(prompt,
                                      max_length = self.max_length,
                                      padding = False,
                                      truncation = True,
                                      return_tensors = 'pt',
                                      add_special_tokens = False)
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()

        return dict(
            prompt = prompt,
            chosen = chosen,
            reject = rejected,
            prompt_ids_len = prompt_ids_len
        )

    def __len__(self): return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        chosen = self.chosens[idx]
        reject = self.rejects[idx]

        chosen = (prompt + chosen).rstrip("\n")
        if not chosen.endswith(self.tokenier.eos_token):
            chosen += " " + self.tokenizer.eos_token
        chosen_token = self.tokenizer(
            chosen, max_length = self.max_length,
            padding = False,
            truncation = True,
            return_tensors = "pt",
            add_special_tokens = False
        )
        chosen_loss_mask = self.get_loss_mask(chosen_token["input_ids"], idx)

        reject = (prompt + reject).rstrip("\n")
        if not reject.endswith(self.tokenizer.eos_token):
            reject += " " + self.tokenizer.eos_token
        reject_token = self.tokenizer(
            reject, max_length = self.max_length,
            padding = False,
            truncation=  True,
            return_tensors = "pt",
            add_special_tokens = False
        )
        reject_loss_mask = self.get_loss_mask(reject_token["input_ids"], idx)


        chosen_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        reject_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
        chosen_token["attention_mask"][0][-1] = True
        reject_token["attention_mask"][0][-1] = True

        return (
            chosen_token["input_ids"],
            chosen_token["attention_mask"],
            chosen_loss_mask,
            reject_token["input_ids"],
            reject_token["attention_mask"],
            reject_loss_mask
        )

    def get_loss_mask(self, input_ids:torch.Tensor, idx:int):
        loss_mask = torch.zeros_like(input_ids, dtype = torch.bool)
        prompt_ids_len = self.prompt_ids_lens[idx]
        loss_mask[0, prompt_ids_len: ] = True
        loss_mask[0, -1] = True

        return loss_mask


    def collate_fn(self, item_list):
        packed_chosen_ids = []
        packed_chosen_attention_masks = []
        packed_chosen_loss_masks = []
        chosen_seq_lens = []

        packed_reject_ids = []
        packed_reject_attention_masks = []
        packed_reject_loss_masks = []
        reject_seq_lens = []


        for index, (chosen_id, chosen_attention_mask, chosen_loss_mask,
                    reject_id, reject_attention_mask, reject_loss_mask) in enumerate(item_list):
            packed_chosen_ids += [chosen_id.flatten()]
            packed_chosen_attention_masks += [torch.full_like(chosen_id.flatten(), index + 1)]
            packed_chosen_loss_masks += [chosen_loss_mask.flatten()]
            chosen_seq_lens += [len(chosen_id.flatten())]

            packed_reject_ids += [reject_id.flatten()]
            packed_reject_attention_masks += [torch.full_like(reject_id.flatten(), index + 1 + len(item_list))]
            packed_reject_loss_masks += [chosen_loss_mask.flatten()]
            reject_seq_lens += [len(reject_id.flatten())]
            
        
        packed_input_ids = torch.concat(packed_chosen_ids + packed_reject_ids).unsqueeze(0)
        packed_attention_masks = torch.concat(chosen_attention_mask + reject_attention_mask).unsqueeze(0)
        packed_loss_masks = torch.concat(chosen_loss_mask + reject_loss_mask).unsqueeze(0)
        packed_seq_lens = chosen_seq_lens + reject_seq_lens

        if packed_input_ids % self.ring_attn_size:
            padding_len = self.ring_attn_size - (packed_input_ids.numel() % self.ring_attn_size)
            packed_input_ids = F.pad(packed_input_ids, (0, padding_len), value = self.tokenizer.pad_token_id)
            packed_attention_masks = F.pad(packed_attention_masks, (0, padding_len), value = 0)
            packed_loss_masks = F.pad(packed_loss_masks, (0, padding_len), value = 0)
        
        return packed_input_ids, packed_attention_masks, packed_loss_masks, packed_seq_lens