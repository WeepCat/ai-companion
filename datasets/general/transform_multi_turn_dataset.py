import json
from tqdm import tqdm

# 打开JSON文件并读取其内容

file_name = 'multi_turn_dataset_3.json' 

with open(f'../general/{file_name}', 'rt', encoding='utf-8') as file:
    data = json.load(file)

n = 0
transformed_data = []
for i in tqdm(data):
    dict_ = dict()
    try:
        dict_['conversation'] = []
        if len(i['messages']) % 2 == 0:
            for j in range(0, len(i['messages']), 2):
                dict_['conversation'].append({
                    'input': i['messages'][j]['content'],
                    'output': i['messages'][j + 1]['content']
                })
                n += 1
        transformed_data.append(dict_)
    except Exception as e:
        # print(n, i)   # 4 empty lines in data.json 425 483 742 1120 
        print(e)

print(n)
with open(f'processed_{file_name}', 'wt', encoding='utf-8') as file:
    json.dump(transformed_data, file, ensure_ascii=False, indent=4)

print(transformed_data[0])