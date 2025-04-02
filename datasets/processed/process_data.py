import json
from tqdm import tqdm

# 打开JSON文件并读取其内容

# file_name = 'multi_turn_dataset_1.json' 
# file_name = 'multi_turn_dataset_2.json' 
file_name = 'multi_turn_dataset_3.json' 
# file_name = 'data_pro.json' 
# file_name = 'data.json' 

with open(f'../general/{file_name}', 'rt', encoding='utf-8') as file:
    data = json.load(file)

n = 0
transformed_data = []
for i in tqdm(data):
    try:
        i['conversation'][0]['system'] = "你是心理健康助手WeepCat，由WeepCat打造。你旨在通过专业心理咨询，协助来访者完成心理诊断。请充分利用专业心理学知识与咨询技术，一步步帮助来访者解决心理问题。"
        transformed_data.append(i)
    except Exception as e:
        print(n, i)   # 4 empty lines in data.json 425 483 742 1120 
        print(e)
    n += 1

with open(f'processed_{file_name}', 'wt', encoding='utf-8') as file:
    json.dump(transformed_data, file, ensure_ascii=False, indent=4)

print(transformed_data[0])
