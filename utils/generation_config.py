from dataclasses import asdict, dataclass


@dataclass
class GenerationConfig:
    max_tokens: int = 1024
    top_p: float = 0.8
    temperature: float = 0.7
    frequency_penalty: float = 1.005
    streaming: bool = False