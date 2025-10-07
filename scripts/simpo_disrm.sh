deepspeed --module scripts.cli.train_simpo_disrm \
    --model_path Skywork/Skywork-Reward-V2-Llama-3.1-8B \
    --dataset_paths ./datas/SimPO_Data_Sampled20 \
    --sample_counts 8 \
    --sample_counts_eval 1 \
    --batch_size 2 \
    --save_name LR_Skywork-Reward-V2-Llama-3.1-8B-s2