import os
import json
import datetime
# os.environ['HF_HOME'] = '/gz-data/hf-cache/'
# from qwen_generation_utils import decode_tokens
import torch
import datasets
from torch.utils.data import DataLoader
from metric import compute_metrics
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm

# load model and tokenizer
model_path = 'Qwen/Qwen2.5-7B-Instruct' # 'Qwen/Qwen2.5-7B-Instruct' 'Qwen/Qwen2.5-0.5B-Instruct' 'Qwen/Qwen2.5-0.5B'
model_name = model_path.split('/')[-1]

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    padding_side='left', 
    # trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    # attn_implementation="flash_attention_2"
    # trust_remote_code=True
).eval()


# prepare data and dataloader
dataset = datasets.load_dataset('json', data_files='./data_dir/converted.json', split='train')
references = dataset['output']
hypotheses = []

# set test params
batch_size = 8


def preprocess(data):
    model_inputs = tokenizer(data['instruction'], max_length=512, truncation=True)
    # labels = tokenizer(data['output'], padding=True, max_length=128, truncation=True)
    # model_inputs['labels'] = labels['input_ids']
    return model_inputs

preprocessed_dataset = dataset.map(preprocess, remove_columns=['instruction', 'output'], num_proc=8)
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
dataloader = DataLoader(preprocessed_dataset, batch_size=batch_size, collate_fn=collator)


for batch in tqdm(dataloader):
    batch_input_ids = torch.LongTensor(batch['input_ids']).to(model.device)
    # batch_labels = batch['labels']
    attention_mask = batch['attention_mask'].to(model.device)
    batch_out_ids = model.generate(
        batch_input_ids.to(model.device),
        attention_mask=attention_mask,
        return_dict_in_generate=False,
        max_new_tokens=256,
        do_sample=False,
        # temperature=0.,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
            batch_out_ids[0, len(batch_input_ids[0]):],
            skip_special_tokens=True
        )
    batch_response = [
        tokenizer.decode(
            batch_out_ids[i, len(batch_input_ids[i]):],
            skip_special_tokens=True
        ) for i in range(len(batch_input_ids))
    ]
    hypotheses.extend(batch_response)

# Create assets directory if it doesn't exist
assets_dir = '/root/projects/ai-companion/site/assets/'
os.makedirs(assets_dir, exist_ok=True)

# Save the evaluation results to a JSON file with model name and timestamp
results = compute_metrics((hypotheses, references))
results_path = os.path.join(assets_dir, 'general_evaluation_results.json')
results_with_metadata = {
    "model_name": model_name,
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "metrics": results
}

# Read existing results if file exists
all_results = []
if os.path.exists(results_path):
    try:
        with open(results_path, 'r') as f:
            all_results = json.load(f)
    except:
        all_results = []

# If all_results is not a list, convert it to a list
if not isinstance(all_results, list):
    all_results = [all_results] if all_results else []

# Check if model name already exists
model_exists = False
existing_index = -1
for i, result in enumerate(all_results):
    if result.get("model_name") == model_name:
        model_exists = True
        existing_index = i
        break

# Handle duplicate model names
if model_exists:
    print(f"Model name '{model_name}' already exists in results.")
    while True:
        choice = input("Choose an action: [o]verwrite / [k]eep existing / [a]dd as new entry: ").lower()
        if choice == 'o':
            all_results[existing_index] = results_with_metadata
            print(f"Overwriting existing entry for model '{model_name}'.")
            break
        elif choice == 'k':
            print(f"Keeping existing entry for model '{model_name}'. New results discarded.")
            break
        elif choice == 'a':
            all_results.append(results_with_metadata)
            print(f"Adding new entry for model '{model_name}' while preserving existing entry.")
            break
        else:
            print("Invalid choice. Please enter 'o', 'k', or 'a'.")
else:
    # No duplicate, simply append
    all_results.append(results_with_metadata)

# Write back all results
with open(results_path, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Evaluation results saved for model {model_name}")
print(results)