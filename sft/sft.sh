# Here is the command-line style training code.
# 22GB
# export HF_HOME=/gz-data/hf-cache/
# export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset sft/processed_self_cognition_WeepCat.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --output_dir output \
    --dataloader_num_workers 4 \
    --use_hf true
    # --system "你是心理健康助手WeepCat, 由WeepCat打造, 是一个研究过无数具有心理健康问题的病人与心理健康医生对话的心理专家, 在心理方面拥有广博的知识储备和丰富的研究咨询经验。你旨在通过专业心理咨询, 协助来访者完成心理诊断。请充分利用专业心理学知识与咨询技术, 一步步帮助来访者解决心理问题。" \