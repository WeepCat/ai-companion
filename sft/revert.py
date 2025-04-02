import json
import jsonlines
from tqdm import tqdm   

def convert_json_to_jsonl(json_file, jsonl_file):
    with open(json_file, 'r',encoding='utf-8') as file:
        data = json.load(file)

        # 将原始数据转换为所需的格式
        converted_data = []
        for conversation_data in tqdm(data):
            history = []
            for i, conversation in enumerate(conversation_data["conversation"]):
                query = conversation["input"]
                response = conversation["output"]
                if i == 0:
                    system = conversation["system"]
                converted_data.append({
                    "system": system,
                    "query": query,
                    "response": response,
                    "history": history[:],
                })
                history.append({"query": query, "response": response})

    # 输出到JSON Lines格式的文件
    # with open('converted.jsonl', 'w', encoding='utf-8') as f:
    #     for item in converted_data:
    #         f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 输出到JSON Lines格式的文件
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
# dedup_combined_data.json
# processed_self_cognition_WeepCat.json
convert_json_to_jsonl('dedup_combined_data.json','dedup_combined_data.jsonl')