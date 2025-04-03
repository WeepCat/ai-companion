from transformers import AutoTokenizer
from typing import List
model = "weepcat/weepcat-7B-Instruct-sft"
system_prompt = "你是心理健康助手小咪♥, 由 WeepCat 打造, 是一个研究过无数具有心理健康问题的病人与心理健康医生对话的心理专家, 在心理方面拥有广博的知识储备和丰富的研究咨询经验。你旨在通过专业心理咨询, 协助来访者完成心理诊断。请充分利用专业心理学知识与咨询技术, 一步步帮助来访者解决心理问题。"
tokenizer = AutoTokenizer.from_pretrained(model)
system_prompt_tokens_num = len(tokenizer(system_prompt)["input_ids"])


def construct_conversation(prompt: str, history: List, max_tokens: int = 4096) -> List:
    conversation = []
    prompt_tokens_num = len(tokenizer(prompt)["input_ids"])
    total_tokens_num = system_prompt_tokens_num + prompt_tokens_num
    conversation.append({"role": "user", "content": prompt})

    for message in history[::-1]:
        cur_content = message["content"]
        cur_tokens_num = len(tokenizer(cur_content)["input_ids"])
        if total_tokens_num + cur_tokens_num > max_tokens:
            break
        if message["role"] == "user":
            conversation.append({"role": "user", "content": cur_content})
        elif message["role"] == "robot":
            conversation.append({"role": "assistant", "content": cur_content})
        else:
            raise RuntimeError
        total_tokens_num += cur_tokens_num
    conversation.append({"role": "system", "content": system_prompt})
    conversation.reverse()
    print(tokenizer.apply_chat_template(conversation, tokenize=False))
    print(len(tokenizer.apply_chat_template(conversation, tokenize=True)))
    return conversation


def construct_conversation_langchain(prompt: str, content: List, history: List, max_tokens: int = 4096) -> List:
    prompt_template = """
根据下面检索回来的信息，回答问题。
{content}
问题：{query}
"""
    prompt = prompt_template.format(query=prompt, content=content)

    conversation = []
    prompt_tokens_num = len(tokenizer(prompt)["input_ids"])
    total_tokens_num = system_prompt_tokens_num + prompt_tokens_num
    conversation.append(("human", prompt))
    for message in history[::-1]:
        cur_content = message["content"]
        cur_tokens_num = len(tokenizer(cur_content)["input_ids"])
        if total_tokens_num + cur_tokens_num > max_tokens:
            break
        if message["role"] == "user":
            conversation.append(("human", cur_content))
        elif message["role"] == "robot":
            conversation.append(("ai", cur_content))
        else:
            raise RuntimeError
        total_tokens_num += cur_tokens_num
    conversation.append(("system", system_prompt))
    conversation.reverse()
    print(conversation)
    print(total_tokens_num)
    return conversation

# def construct_conversation(prompt: str, messages: List) -> List:
#     conversation = []
#     conversation.append({"role": "system", "content": system_prompt})
#     for message in messages:
#         cur_content = message["content"]
#         if message["role"] == "user":
#             conversation.append({"role": "user", "content": cur_content})
#         elif message["role"] == "robot":
#             conversation.append({"role": "assistant", "content": cur_content})
#         else:
#             raise RuntimeError
#     conversation.append({"role": "user", "content": prompt})
#     return conversation