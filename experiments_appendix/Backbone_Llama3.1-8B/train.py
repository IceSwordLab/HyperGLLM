import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["NCCL_TIMEOUT"] = "900"
import re
import random
import numpy as np
import json
import shutil
from functools import partial
import torch
import datasets
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import argparse
import deepspeed
from PIL import Image
from src.utils import *
from src.CustomLlamaForCausalLM import *
from transformers import LlamaTokenizerFast, LlamaConfig, LlamaForCausalLM, DataCollatorForSeq2Seq, AddedToken
from transformers import TrainingArguments, Trainer
torch.cuda.empty_cache()



parser = argparse.ArgumentParser()
parser.add_argument('--datasets_path', default="", type=str, help="")
parser.add_argument('--datasets_name', default="", type=str, help="")
parser.add_argument('--datasets_fnspath', default="", type=str, help="")
parser.add_argument('--datasets_fnsname', default="", type=str, help="")
parser.add_argument('--output_dir', default="", type=str, help="")
parser.add_argument('--model_path', default="", type=str, help="")
parser.add_argument('--model_name', default="", type=str, help="")
parser.add_argument('--deepspeed', default="", type=str, help="")
parser.add_argument('--gradient_checkpointing', default=0, type=int, help="")
parser.add_argument('--system_prompt', default="", type=str, help="")
parser.add_argument('--template_prompt', default="", type=str, help="")
parser.add_argument('--is_training', default=0, type=int, help="")
parser.add_argument('--max_time_len', default=0, type=int, help="")
parser.add_argument('--kvgraph_dim', default=0, type=int, help="")
parser.add_argument('--kvgraph_layernum', default=3, type=int, help="")
parser.add_argument('--kvgraph_drop', default=0.0, type=float, help="")
parser.add_argument('--hypergraph_dim', default=0, type=int, help="")
parser.add_argument('--hypergraph_layernum', default=0, type=int, help="")
parser.add_argument('--hypergraph_drop', default=0.0, type=float, help="")
parser.add_argument('--C_list', default=[], nargs='+', help="")
parser.add_argument('--topK', default=0, type=int, help="")
parser.add_argument('--is_ae', default=0, type=int, help="")
parser.add_argument('--is_ac', default=0, type=int, help="")
parser.add_argument('--each_p', default=0.0, type=float, help="")
parser.add_argument('--batch_size', default=0, type=int, help="")
parser.add_argument('--accumulation_steps', default=0, type=int, help="")
parser.add_argument('--max_steps', default=0, type=int, help="")
parser.add_argument('--learning_rate', default=0.0, type=float, help="")
parser.add_argument('--lr_scheduler_type', default="", type=str, help="")
parser.add_argument('--weight_decay', default=0.0, type=float, help="")
parser.add_argument('--warmup_ratio', default=0.0, type=float, help="")
parser.add_argument('--save_strategy', default="", type=str, help="")
parser.add_argument('--save_steps', default=0, type=int, help="")
parser.add_argument('--logging_steps', default=0, type=int, help="")
args = parser.parse_args()
args.logging_dir = os.path.join(args.output_dir, "logging.json")
print(args.output_dir)
os.makedirs(args.output_dir, exist_ok=True)
with open(os.path.join(args.output_dir, "args.json"), 'w') as f:
    json.dump(vars(args), f)



# data load
with open(os.path.join(args.datasets_path, args.datasets_name), "r", encoding="utf-8") as file:
    dataset = json.load(file)
    for item in dataset:
        item["family"] = "<"+item["family"]+">"
random.shuffle(dataset)
dataset = Dataset.from_list(dataset)
print(dataset)

with open(os.path.join(args.datasets_fnspath, args.datasets_fnsname), "r", encoding="utf-8") as file:
    family_tokens = json.load(file)
    family_tokens = ["<"+item+">" for item in family_tokens]
print(family_tokens)


# Initialize tokenizer and model
tokenizer = LlamaTokenizerFast.from_pretrained(pretrained_model_name_or_path=os.path.join(args.model_path, args.model_name))
tokenizer.pad_token_id = tokenizer.eos_token_id
new_tokens = family_tokens + ["<|event_patch|>"] + ["<|seldefunk|>"]
for token in new_tokens:
    if tokenizer.convert_tokens_to_ids(token)!=128256:
        raise ValueError(f"token {token} is exist!!!")
tokenizer.add_tokens([AddedToken(t, normalized=False, special=False) for t in new_tokens])
tokenizer.save_pretrained(args.output_dir)
config = LlamaConfig.from_pretrained(os.path.join(args.model_path, args.model_name))
config.use_cache = False if args.gradient_checkpointing==1 else True
config.family_tokens = family_tokens
config.list_key_dims = list_key_dims
config.kvgraph_dim = args.kvgraph_dim
config.kvgraph_layernum = args.kvgraph_layernum
config.kvgraph_drop = args.kvgraph_drop
config.hypergraph_dim = args.hypergraph_dim
config.hypergraph_layernum = args.hypergraph_layernum
config.hypergraph_drop = args.hypergraph_drop
config.C_list = [int(v) for v in args.C_list]
config.topK = args.topK
config.num_newtokens = len(family_tokens)+2
config.vocab_size += config.num_newtokens
model = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(args.model_path, args.model_name), torch_dtype="auto")
old_embeddings = model.model.embed_tokens.weight.data
old_lm_head = model.lm_head.weight.data
model = CustomLlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(args.model_path, args.model_name), attn_implementation="flash_attention_2", torch_dtype="auto", config=config, ignore_mismatched_sizes=True)
model.model.embed_tokens.weight.data[:-config.num_newtokens,:] = old_embeddings
model.lm_head.weight.data[:-config.num_newtokens,:] = old_lm_head
del old_embeddings, old_lm_head
model = init_embed_tokens(model, tokenizer, new_tokens)
model.save_pretrained(args.output_dir)



# Training
training_args = TrainingArguments(
    output_dir=args.output_dir,
    do_train=True,
    bf16=True,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.accumulation_steps,
    max_steps=args.max_steps,
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    save_strategy=args.save_strategy,
    save_steps=args.save_steps,
    logging_steps=args.logging_steps,
    label_names=["labels"],
    #dataloader_num_workers=0,
    remove_unused_columns = False,
    deepspeed=args.deepspeed if len(args.deepspeed)>0 else None,
    gradient_checkpointing = True if args.gradient_checkpointing==1 else False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=CustomDataCollator(tokenizer, args.system_prompt, args.template_prompt, args.max_time_len, args.is_training, args.is_ae, args.is_ac, args.each_p, config.family_tokens, args.datasets_path, args.datasets_name, config),
    callbacks=[JSONLoggerCallback(args.logging_dir, args.logging_steps)],
)
trainer.train()
