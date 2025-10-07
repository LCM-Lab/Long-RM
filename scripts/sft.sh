deepspeed --module scripts.cli.train_sft \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --dataset_paths ./datas/SFT_Data_Sampled20\
    --sample_counts 8 \
    --sample_counts_eval 1 \
    --batch_size 2 \
    --save_name LR_Meta-Llama-3.1-8B-Instruct-s1