# RoBERTa-based experiments
python ./run_glue_discrete_LM.py \
--task_name=mrpc \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 16 \
--weight_decay=0.1 --seed=42 \
--k_shot 16 --prompt_learning_rate 1e-4 \
--sample_size 20 --prompt_length 10 \
--prompt_search_space 200 \
--api_limit 8000 --ce_loss True

# # GPT-based experiments
# python ./run_glue_discrete_GPT.py \
# --task_name=mrpc \
# --per_device_train_batch_size 4 \
# --per_device_eval_batch_size 4 \
# --k_shot 16 --prompt_learning_rate 2e-4 \
# --sample_size 20 --prompt_length 20 \
# --prompt_search_space 50 --num_train_epochs 10 \
# --api_key [API_KEY]
