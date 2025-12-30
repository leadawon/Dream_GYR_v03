# PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=128,diffusion_steps=128,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
#     --tasks mmlu_generative \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 4 \
#     --output_path output_reproduce/mmlu \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=128,diffusion_steps=128,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
#     --tasks mmlu_pro \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 4 \
#     --output_path output_reproduce/mmlu_pro \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
#     --tasks gsm8k_cot \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/gsm8k \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=512,diffusion_steps=512,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
#     --tasks minerva_math \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/math \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=512,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks minerva_math500 \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --limit 80 \
    --output_path output_reproduce_limit80/math_step256 \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=512,diffusion_steps=256,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks minerva_math500 \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce_limit9999/math_step256 \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=512,diffusion_steps=384,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks minerva_math500 \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --limit 80 \
    --output_path output_reproduce_limit80/math_step384 \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=512,diffusion_steps=384,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks minerva_math500 \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce_limit9999/math_step384 \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=512,diffusion_steps=128,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks minerva_math500 \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --limit 80 \
    --output_path output_reproduce_limit80/math_step128 \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=512,diffusion_steps=128,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks minerva_math500 \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce_limit9999/math_step128 \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=512,diffusion_steps=64,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks minerva_math500 \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --limit 80 \
    --output_path output_reproduce_limit80/math_step64 \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 12334 -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=512,diffusion_steps=64,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
    --tasks minerva_math500 \
    --device cuda \
    --batch_size 1 \
    --num_fewshot 0 \
    --output_path output_reproduce_limit9999/math_step64 \
    --log_samples --confirm_run_unsafe_code \
    --apply_chat_template

# PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,dtype="bfloat16" \
#     --tasks gpqa_main_n_shot \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 5 \
#     --output_path output_reproduce/gpqa \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=768,diffusion_steps=768,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
#     --tasks humaneval_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/humaneval \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template

# HF_ALLOW_CODE_EVAL=1 PYTHONPATH=. accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=1024,diffusion_steps=1024,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
#     --tasks mbpp_instruct \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce/mbpp \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template



# PYTHONPATH=. CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 12334 -m lm_eval \
#     --model diffllm \
#     --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=1280,diffusion_steps=1280,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy" \
#     --tasks ifeval \
#     --device cuda \
#     --batch_size 1 \
#     --num_fewshot 0 \
#     --output_path output_reproduce_limit9999/ifeval \
#     --log_samples --confirm_run_unsafe_code \
#     --apply_chat_template