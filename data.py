import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel,AutoTokenizer,Trainer,TrainingArguments,GPT2Config
from torch.utils.data.dataloader import DataLoader, Dataset
import os
import tqdm
from random import randint
import glob

class CustomerDataset(Dataset):
    def __init__(self,tokenizer,file_path,block_size: int):
        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
#         print(f"block_size = {block_size}")
        self.examples = []
        self.mask = []
        for file in file_path:
            with open(file, encoding="utf-8") as f:
                text = f.read()
            # text --> token-->number
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            if len(tokenized_text) < block_size:
                inds = [tokenizer.convert_tokens_to_ids(tokenizer.bos_token)] + tokenized_text + [tokenizer.convert_tokens_to_ids(tokenizer.eos_token)] + \
            (block_size - len(tokenized_text)) * [tokenizer.convert_tokens_to_ids(tokenizer.pad_token)]
                mask = [x != tokenizer.convert_tokens_to_ids(tokenizer.pad_token) for x in inds]
#                 print(mask)
            else:
                inds = [tokenizer.convert_tokens_to_ids(tokenizer.bos_token)] + tokenized_text[:block_size] + [tokenizer.convert_tokens_to_ids(tokenizer.eos_token)]
                mask = [1] * len(inds)

            self.examples.append(inds) 
            self.mask.append(mask)
            
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return {
            "input_ids":torch.tensor(self.examples[i], dtype=torch.long), 
            "attention_mask":torch.tensor(self.mask[i], dtype = torch.long)
        }