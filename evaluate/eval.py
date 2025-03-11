import os
os.environ['HF_HOME'] = '/gz-data/hf-cache/'
from qwen_generation_utils import decode_tokens
import torch
import datasets
from torch.utils.data import DataLoader
from metric import compute_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding


#load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct',
    # pad_token='<|extra_0|>',
    # eos_token='<|endoftext|>',
    padding_side='left',
    # cache_dir='/gz-data/hf-cache/'
    # trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct',
    # pad_token_id=tokenizer.pad_token_id,
    device_map="cuda:0",
    # trust_remote_code=True
).eval()


#set test params
test_num = 1596 #测试数据条数
batch_size = 12


#prepare data and dataloader
dataset = datasets.load_dataset('json', data_files='./data_dir/converted.json', split=f"train[:{test_num}]")
references = dataset['output'][:test_num]
hypotheses = []


def preprocess(data):
    length = list(map(len, data['instruction']))
    model_inputs = tokenizer(data['instruction'], max_length=512, truncation=True)
    labels = tokenizer(data['output'], padding=True,max_length=128, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    model_inputs['length'] = length
    return model_inputs

preprocessed_dataset = dataset.map(preprocess, batched=True, remove_columns=['instruction', 'output'])
collator = DataCollatorWithPadding(tokenizer=tokenizer,)
dataloader = DataLoader(preprocessed_dataset, batch_size=batch_size, collate_fn=collator)


for batch in dataloader:
    batch_input_ids = torch.LongTensor(batch['input_ids']).to(model.device)
    batch_labels = batch['labels']
    attention_mask = batch['attention_mask']
    length = batch['length']
    batch_out_ids = model.generate(
        batch_input_ids.to(model.device),
        return_dict_in_generate=False,
        max_new_tokens=256,
        temperature=0.1,
        pad_token_id=tokenizer.eos_token_id
    )
    padding_lens = [batch_input_ids[i].eq(tokenizer.pad_token_id).sum().item() for i in range(batch_input_ids.size(0))]
    batch_response = [
    decode_tokens(
        batch_out_ids[i][padding_lens[i]:],
        tokenizer,
        context_length=0,
        raw_text_len=length[i],
        chat_format="raw",
        verbose=False,
        errors='replace'
    ) for i in range(batch_size)
    ]
    hypotheses.extend(batch_response)


print(compute_metrics((hypotheses,references)))