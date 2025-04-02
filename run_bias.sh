#!/bin/bash
#SBATCH -J job_id
#SBATCH -o ./log/bbq/llama3-8b-pr-0.3-lora-avg.out
#SBATCH --gres=gpu:1 #Number of GPU devices to use [0-2]
#SBATCH --nodelist=leon06 #YOUR NODE OF PREFERENCE

# Set Hugging Face token
#export CUDA_VISIBLE_DEVICES=0,1

module load shared singularity 


# DATASET=bbq_fewshot
# SAVE_AS=llama3_8b_raw

# singularity exec --nv ../LLM-Pruner/img/llm-pruner-awq.img \
#     python src/run_multi_qa.py \
#     --tokenizer ../Meta-Llama-3-8B \
#     --model_name_or_path ../Meta-Llama-3-8B \
#     --dataset ${DATASET} \
#     --category age,disability_status,gender_identity,nationality,physical_appearance,race_ethnicity,race_x_gender,race_x_ses,religion,ses,sexual_orientation \
#     --do_inference \
#     --disable_progress_bar \
#     --save_results \
#     --save_outputs \
#     --output_dest ./log/bbq/${DATASET}_${SAVE_AS}.jsonl \
#     --results_dest ./log/bbq/${DATASET}_${SAVE_AS}.json

# DATASET=bbq_fewshot
# SAVE_AS=llama3_8b_awq

# singularity exec --nv ../LLM-Pruner/img/llm-pruner-awq.img \
#     python src/run_multi_qa.py \
#     --tokenizer /home/hsin/AutoAWQ/Llama-3-8B-AWQ \
#     --model_name_or_path /home/hsin/AutoAWQ/Llama-3-8B-AWQ \
#     --dataset bbq_fewshot \
#     --category age,disability_status,gender_identity,nationality,physical_appearance,race_ethnicity,race_x_gender,race_x_ses,religion,ses,sexual_orientation \
#     --do_inference \
#     --disable_progress_bar \
#     --awq \
#     --save_results \
#     --save_outputs \
#     --output_dest ./log/bbq/${DATASET}_${SAVE_AS}.jsonl \
#     --results_dest ./log/bbq/${DATASET}_${SAVE_AS}.json

# DATASET=bbq_fewshot
# SAVE_AS=llama3_8b_awq_2k_lora

# singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img \
#     python src/run_multi_qa.py \
#     --tokenizer /home/hsin/AutoAWQ/Llama-3-8B-AWQ \
#     --model_name_or_path /home/hsin/AutoAWQ/Llama-3-8B-AWQ \
#     --dataset ${DATASET} \
#     --category age,disability_status,gender_identity,nationality,physical_appearance,race_ethnicity,race_x_gender,race_x_ses,religion,ses,sexual_orientation \
#     --do_inference \
#     --disable_progress_bar \
#     --awq --lora \
#     --lora_path /home/hsin/AutoAWQ/Llama-3-8B-AWQ-2k-lora-8-16 \
#     --save_results \
#     --save_outputs \
#     --output_dest ./log/bbq/${DATASET}_${SAVE_AS}.jsonl \
#     --results_dest ./log/bbq/${DATASET}_${SAVE_AS}.json

# DATASET=bbq_fewshot
# SAVE_AS=llama3_8b_pr_0.3_lora

# singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img \
#     python src/run_multi_qa.py \
#     --tokenizer ../Meta-Llama-3-8B \
#     --model_name_or_path ../Meta-Llama-3-8B \
#     --dataset ${DATASET} \
#     --category age,disability_status,gender_identity,nationality,physical_appearance,race_ethnicity,race_x_gender,race_x_ses,religion,ses,sexual_orientation \
#     --do_inference \
#     --disable_progress_bar \
#     --prune --lora \
#     --prune_path /home/hsin/LLM-Pruner/prune_log/llama3_8b_prune_0p3/pytorch_model.bin \
#     --lora_path /home/hsin/LLM-Pruner/tune_log/llama3_8b_prune_0p3 \
#     --save_results \
#     --save_outputs \
#     --output_dest ./log/bbq/${DATASET}_${SAVE_AS}.jsonl \
#     --results_dest ./log/bbq/${DATASET}_${SAVE_AS}.json

singularity exec --nv ../open-instruct/img/awq-openai-triton-peft.img python src/bbq_compute.py 

# DATASET=unqover
# SAVE_AS=llama3_8b_raw

# singularity exec --nv ../LLM-Pruner/img/llm-pruner-awq.img \
#     python src/run_multi_qa.py \
#     --tokenizer ../Meta-Llama-3-8B \
#     --model_name_or_path ../Meta-Llama-3-8B \
#     --dataset unqover \
#     --category country,ethnicity,religion,gender_occupation \
#     --do_inference \
#     --metrics subj_bias \
#     --group_by subj \
#     --disable_progress_bar \
#     --sample_size 1000 \
#     --save_results \
#     --save_outputs \
#     --output_dest ./log/${DATASET}_subj_act.jsonl \
#     --results_dest ./log/${DATASET}_${SAVE_AS}_subj_act.json

# singularity exec --nv ../LLM-Pruner/img/llm-pruner-awq.img \
#     python src/run_multi_qa.py \
#     --tokenizer ../Meta-Llama-3-8B \
#     --model_name_or_path ../Meta-Llama-3-8B \
#     --dataset unqover \
#     --category country,ethnicity,religion,gender_occupation \
#     --do_inference \
#     --do_sample \
#     --sample_size 1000 \
#     --metrics subj_bias \
#     --group_by subj \
#     --disable_progress_bar \
#     --awq --lora \
#     --lora_path \
#     --save_results \
#     --save_outputs \
#     --output_dest ./log/${DATASET}_subj_act.jsonl \
#     --results_dest ./log/${DATASET}_${SAVE_AS}_subj_act.json

# singularity exec --nv ../LLM-Pruner/img/llm-pruner-awq.img \
#     python src/run_multi_qa.py \
#     --tokenizer ../Meta-Llama-3-8B \
#     --model_name_or_path ../Meta-Llama-3-8B \
#     --dataset unqover \
#     --category country,ethnicity,religion,gender_occupation \
#     --do_inference \
#     --do_sample \
#     --sample_size 1000 \
#     --metrics subj_bias \
#     --group_by subj \
#     --disable_progress_bar \
#     --prune --lora \
#     --prune_path \
#     --lora_path \
#     --save_results \
#     --save_outputs \
#     --output_dest ./log/${DATASET}_subj_act.jsonl \
#     --results_dest ./log/${DATASET}_${SAVE_AS}_subj_act.json
