from transformers import AutoTokenizer
from typing import List
model = "weepcat/weepcat-7B-Instruct-sft"
system_prompt = "你是心理健康助手 CatLLM, 由 WeepCat 打造, 是一个研究过无数具有心理健康问题的病人与心理健康医生对话的心理专家, 在心理方面拥有广博的知识储备和丰富的研究咨询经验。你旨在通过专业心理咨询, 协助来访者完成心理诊断。请充分利用专业心理学知识与咨询技术, 一步步帮助来访者解决心理问题。"


tokenizer = AutoTokenizer.from_pretrained(model)

# 生成
def construct_conversation(prompt: str, messages: List) -> List:
    conversation = []
    conversation.append({"role": "system", "content": system_prompt})
    for message in messages:
        cur_content = message["content"]
        if message["role"] == "user":
            conversation.append({"role": "user", "content": cur_content})
        elif message["role"] == "robot":
            conversation.append({"role": "assistant", "content": cur_content})
        else:
            raise RuntimeError
    conversation.append({"role": "user", "content": prompt})
    return conversation