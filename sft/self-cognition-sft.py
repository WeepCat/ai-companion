# import some libraries
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['HF_HOME'] = '/gz-data/hf-cache/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from swift.llm import sft_main, TrainArguments
result = sft_main(TrainArguments(
    model='Qwen/Qwen2.5-7B-Instruct',
    train_type='lora',
    dataset='processed_self_cognition_WeepCat.jsonl',
    torch_dtype='bfloat16',
    use_hf=True,
    
    # ...
))