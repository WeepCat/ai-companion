import os
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
from openai import AsyncOpenAI, OpenAI
import torch
import streamlit as st
torch.classes.__path__ = []
os.environ['HF_HOME'] = "/gz-data/hf-cache/"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from transformers.utils import logging  
from utils.process_conversation import construct_conversation
logger = logging.get_logger(__name__)
# model = "weepcat/weepcat-7B-Instruct-sft"
# os.system(f"vllm serve {model} --dtype auto --served-model-name catllm &")
# export HF_HOME=/gz-data/hf-cache/
# export HF_ENDPOINT=https://hf-mirror.com
# vllm serve weepcat/weepcat-7B-Instruct-sft --dtype auto --served-model-name catllm


@dataclass
class GenerationConfig:
    max_completion_tokens: int = 1024
    top_p: float = 0.8
    temperature: float = 0.7
    frequency_penalty: float = 1.005


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def get_client():
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )
    return client


@st.cache_resource
def get_aclient():
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )
    return client


# 处理流式响应
@st.cache_resource
def chat(conversation: List, **kwargs):
    response = client.chat.completions.create(
        model="catllm",
        messages=conversation,
        **kwargs,
        stream=True,
    )
    cur_response = ""
    for chunk in response:
        # cur_response += chunk["choices"][0]["delta"]["content"]
        cur_response += chunk.choices[0].delta.content
        yield cur_response
    yield cur_response


def prepare_generation_config():
    with st.sidebar:
        # st.image('assets/weepcat.jpg', width=1, caption='CatLLM Logo', use_column_width=True)
        st.markdown("Our open-source [GitHub repo](https://github.com/WeepCat/ai-companion)")

        max_completion_tokens = st.slider("Max Length", min_value=8, max_value=4096, value=1024)
        top_p = st.slider("Top P", 0.0, 1.0, 0.8, step=0.01)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, step=0.01)
        st.button("Clear Chat History", on_click=on_btn_click)

    generation_config = GenerationConfig(
        max_completion_tokens=max_completion_tokens,
        top_p=top_p,
        temperature=temperature,
    )
    return generation_config


client = get_client()


def main():
    user_avator = "assets/weepcat.jpg"
    robot_avator = "assets/weepcat.jpg"
    st.title("CatLLM 心理咨询室 V1.0")
    generation_config = prepare_generation_config()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    if prompt := st.chat_input("我在这里，准备好倾听你的心声了。"):
        with st.chat_message("user", avatar=user_avator):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avator})
        conversation = construct_conversation(prompt, st.session_state.messages, max_tokens=8192)
        with st.chat_message("robot", avatar=robot_avator):
            message_placeholder = st.empty()
            for cur_response in chat(conversation=conversation, **asdict(generation_config)):
                message_placeholder.markdown(cur_response + "▌")
            message_placeholder.markdown(cur_response)  # pylint: disable=undefined-loop-variable

        st.session_state.messages.append(
            {
                "role": "robot",
                "content": cur_response,  # pylint: disable=undefined-loop-variable
                "avatar": robot_avator,
            }
        )


if __name__ == '__main__':
    main()