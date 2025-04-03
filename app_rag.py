import os
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional
from openai import AsyncOpenAI, OpenAI
import torch
from utils.rag_util import chat_rag
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
# vllm serve weepcat/weepcat-7B-Instruct-sft --dtype auto --served-model-name catllm --max_model_len 16384


def on_btn_click():
    del st.session_state.messages


def main():
    user_avator = "assets/weepcat.jpg"
    robot_avator = "assets/weepcat.jpg"
    st.title("CatLLM 心理咨询室 V2.0 (RAG)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])

    if prompt := st.chat_input("我在这里，准备好倾听你的心声了。"):
        with st.chat_message("user", avatar=user_avator):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt, "avatar": user_avator})
        with st.chat_message("robot", avatar=robot_avator):
            message_placeholder = st.empty()
            response = chat_rag(prompt, st.session_state.messages)
            message_placeholder.markdown(response)  # pylint: disable=undefined-loop-variable

        st.session_state.messages.append(
            {
                "role": "robot",
                "content": response,  # pylint: disable=undefined-loop-variable
                "avatar": robot_avator,
            }
        )


if __name__ == '__main__':
    main()