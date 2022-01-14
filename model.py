import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel,AutoTokenizer,Trainer,TrainingArguments,GPT2Config
from torch.utils.data.dataloader import DataLoader, Dataset
import os
from  data import CustomerDataset
import tqdm
from random import randint
import glob
from transformers.trainer_callback import TrainerCallback
from transformers import pipeline
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling, LineByLineWithSOPTextDataset

def load_dataset(train_path, tokenizer):
    train_dataset = CustomerDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size= 256)#256
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,data_collator
def main():
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    tokenizer.add_tokens('\n')
    train_dataset,data_collator = load_dataset(files_name,tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained('danghuy1999/gpt2-viwiki')
    rand_weight = torch.rand(model.lm_head.weight.shape)
    model.lm_head.weight = torch.nn.parameter.Parameter(rand_weight)
    task_gpt2 = {"text-generation": {"do_sample": True, "max_length": 256}} 
    configuration = GPT2Config(vocab_size=64002, n_positions=260, n_ctx=260,
                            task_specific_params=task_gpt2,
                            eos_token_id = 2,
                            bos_token_id = 0,
                            pad_token_id = 1,
                            sep_token_id = 2,)
    model = GPT2LMHeadModel(configuration)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    training_args = TrainingArguments(
    output_dir="./output_dir", 
    overwrite_output_dir=True,
    num_train_epochs=40,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16, 
    save_steps=1000,
    save_total_limit = 2,
    warmup_steps=1000, 
    logging_steps=100,
    report_to="wandb"
    )

    trainer = Trainer(
    model=model, 
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    )
    trainer.train()
