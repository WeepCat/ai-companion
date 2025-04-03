from langchain_openai import ChatOpenAI
from rag.src.pipeline import EmoLLMRAG
from typing import List
from dataclasses import asdict
from .process_conversation import construct_conversation_langchain
from .generation_config import GenerationConfig


def create_rag(model_name: str, config: GenerationConfig):
    model = ChatOpenAI(
        model_name=model_name,
        openai_api_base="http://localhost:8000/v1",
        openai_api_key="EMPTY",
        **asdict(config),
    )
    rag = EmoLLMRAG(model)
    return rag
rag = create_rag("catllm", GenerationConfig())


def chat_rag(prompt: str, history: List):
    content = rag.get_retrieval_content(prompt)
    conversation = construct_conversation_langchain(prompt, content, history, max_tokens=8192)
    response = rag.generate_answer(conversation)
    return response
