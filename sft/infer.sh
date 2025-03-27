CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/v8-20250327-184051/checkpoint-30 \
    --infer_backend pt \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048 \
    --use_hf true \
    --system "你是心理健康助手WeepCat, 由WeepCat打造, 是一个研究过无数具有心理健康问题的病人与心理健康医生对话的心理专家, 在心理方面拥有广博的知识储备和丰富的研究咨询经验。你旨在通过专业心理咨询, 协助来访者完成心理诊断。请充分利用专业心理学知识与咨询技术, 一步步帮助来访者解决心理问题。" \
    # /root/projects/ai-companion/output/v8-20250327-184051/checkpoint-30