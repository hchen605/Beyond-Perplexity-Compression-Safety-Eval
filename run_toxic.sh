#!/bin/bash
#SBATCH -J job_id
#SBATCH -o ./log/toxigen/llama3-8b-instruct-openfunc-sft-sdft.out
#SBATCH --gres=gpu:1 #Number of GPU devices to use [0-2]
#SBATCH --nodelist=leon05 #YOUR NODE OF PREFERENCE

# Set Hugging Face token
#export CUDA_VISIBLE_DEVICES=0,1


module load shared apptainer 



#singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img ls -l /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
#singularity exec --nv --env SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem ../open-instruct/img/awq-openai-triton-peft.img python -c "import ssl; print(ssl.get_default_verify_paths())"
#singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img python -c "import ssl; print(ssl.get_default_verify_paths())"
#singularity exec --nv --bind /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem ../open-instruct/img/awq-openai-triton-peft.img ls -l /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
#singularity exec --nv --bind /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem --env SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem ../open-instruct/img/awq-openai-triton-peft.img python openai_test.py
#singularity exec --nv --bind /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem --env SSL_CERT_FILE=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem ../open-instruct/img/awq-openai-triton-peft.img python openai_test.py

# DATASET="toxigen"
# SAVE_AS="llama3_8b_instruct"

# singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img \
#     python src/run_generation.py \
#     --tokenizer meta-llama/Llama-3.1-8B-Instruct \
#     --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
#     --dataset toxigen \
#     --min_new_tokens 50 \
#     --max_new_tokens 100 \
#     --batch_size 16 \
#     --save_results \
#     --results_dest ./log/toxigen/${DATASET}_${SAVE_AS}.json \
#     --disable_progress_bar \
#     --save_outputs \
#     --outputs_dest ./log/toxigen/${DATASET}_${SAVE_AS}.jsonl

# DATASET="toxigen"
# SAVE_AS="llama3_8b_instruct-awq_4bit"

# singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img \
#     python src/run_generation.py \
#     --tokenizer /home/hsin/AutoAWQ/Llama-3-8B-Instruct-AWQ \
#     --model_name_or_path /home/hsin/AutoAWQ/Llama-3-8B-Instruct-AWQ \
#     --dataset ${DATASET} \
#     --min_new_tokens 50 \
#     --max_new_tokens 100 \
#     --batch_size 16 \
#     --save_results \
#     --awq \
#     --results_dest ./log/toxigen/${DATASET}_${SAVE_AS}.json \
#     --disable_progress_bar \
#     --save_outputs \
#     --outputs_dest ./log/toxigen/${DATASET}_${SAVE_AS}.jsonl

# DATASET="toxigen"
# SAVE_AS="llama3_8b_instruct_awq_clean_lora"

# singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img \
#     python src/run_generation.py \
#     --tokenizer /home/hsin/AutoAWQ/Llama-3-8B-Instruct-AWQ \
#     --model_name_or_path /home/hsin/AutoAWQ/Llama-3-8B-Instruct-AWQ \
#     --dataset ${DATASET} \
#     --min_new_tokens 50 \
#     --max_new_tokens 100 \
#     --batch_size 16 \
#     --save_results \
#     --awq --lora \
#     --lora_path /home/hsin/AutoAWQ/Llama-3-8B-Instruct-AWQ-clean-lora-8-16 \
#     --results_dest ./log/toxigen/${DATASET}_${SAVE_AS}.json \
#     --disable_progress_bar \
#     --save_outputs \
#     --outputs_dest ./log/toxigen/${DATASET}_${SAVE_AS}.jsonl

# DATASET="toxigen"
# SAVE_AS="llama3_8b_instruct_pr_0.3_lora"

# singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img \
#     python src/run_generation.py \
#     --tokenizer ../Meta-Llama-3-8B \
#     --model_name_or_path ../Meta-Llama-3-8B \
#     --dataset ${DATASET} \
#     --min_new_tokens 50 \
#     --max_new_tokens 100 \
#     --batch_size 16 \
#     --save_results \
#     --prune --lora \
#     --prune_path /home/hsin/LLM-Pruner/prune_log/llama3_8b_instruct_prune_0p3/pytorch_model.bin \
#     --lora_path /home/hsin/LLM-Pruner/tune_log/llama3_8b_instruct_prune_0p3 \
#     --results_dest ./log/toxigen/${DATASET}_${SAVE_AS}.json \
#     --disable_progress_bar \
#     --save_outputs \
#     --outputs_dest ./log/toxigen/${DATASET}_${SAVE_AS}.jsonl


DATASET="toxigen"
SAVE_AS="llama3_8b_instruct_openfunc_sft"

singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img \
    python src/run_generation.py \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset ${DATASET} \
    --min_new_tokens 50 \
    --max_new_tokens 100 \
    --batch_size 32 \
    --save_results \
    --lora \
    --lora_path /home/hsin/sdft/checkpoints_llama3/openfunction/sft \
    --results_dest ./log/toxigen/${DATASET}_${SAVE_AS}.json \
    --disable_progress_bar \
    --save_outputs \
    --outputs_dest ./log/toxigen/${DATASET}_${SAVE_AS}.jsonl

DATASET="toxigen"
SAVE_AS="llama3_8b_instruct_openfunc_sdft"

singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img \
    python src/run_generation.py \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset ${DATASET} \
    --min_new_tokens 50 \
    --max_new_tokens 100 \
    --batch_size 32 \
    --save_results \
    --lora \
    --lora_path /home/hsin/sdft/checkpoints_llama3/openfunction/sdft \
    --results_dest ./log/toxigen/${DATASET}_${SAVE_AS}.json \
    --disable_progress_bar \
    --save_outputs \
    --outputs_dest ./log/toxigen/${DATASET}_${SAVE_AS}.jsonl

###########
# DATASET="toxigen"
# SAVE_AS="llama3_8b_instruct_insert_safety"

# singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img \
#     python src/run_generation_refine.py \
#     --tokenizer meta-llama/Llama-3.1-8B-Instruct\
#     --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
#     --dataset toxigen \
#     --min_new_tokens 50 \
#     --max_new_tokens 100 \
#     --n_samples 10000 \
#     --batch_size 64 \
#     --save_results \
#     --results_dest ./log/toxigen/${DATASET}_${SAVE_AS}.json \
#     --disable_progress_bar \
#     --save_outputs \
#     --outputs_dest ./log/toxigen/${DATASET}_${SAVE_AS}.jsonl \
#     --insert_safety

# DATASET="toxigen"
# SAVE_AS="llama3_8b_instruct_raw_100"

# singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img \
#     python src/run_generation.py \
#     --tokenizer meta-llama/Llama-3.1-8B-Instruct \
#     --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
#     --dataset toxigen \
#     --min_new_tokens 50 \
#     --max_new_tokens 100 \
#     --n_samples 100 \
#     --batch_size 64 \
#     --save_results \
#     --results_dest ./log/toxigen/${DATASET}_${SAVE_AS}.json \
#     --disable_progress_bar \
#     --save_outputs \
#     --outputs_dest ./log/toxigen/${DATASET}_${SAVE_AS}.jsonl