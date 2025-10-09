## 💻 Environment & Installation

To setup the training environment, run:

```bash
cd LongRM
pip install -r requirements.txt
# install flash attention
Download the suitable version of flash_attn from https://github.com/Dao-AILab/flash-attention/releases
pip install <path_to_flash_attn_whl_file>
pip install ring_flash_attn
```
## 🔥 Train
To run the first training process:

### GRM
```bash
bash scripts/sft.sh 
```

To run the second training process:

### GRM
```bash
bash bash scripts/simpo_grm.sh 
```

### DisRM

```bash
bash bash scripts/simpo_disrm.sh 
```


## 📊 Evaluate

### For Generative Model

```bash
modelscope download LCM_group/LongReward_Qwen3-8B --repo-type model --local_dir ./LongReward_Qwen3-8B

python evaluate/eval.py --model-path ./LongReward_Qwen3-8B --data-path ./LongReward-Bench
```

### For Discriminative Model:

```bash
modelscope download LCM_group/LongReward_Skywork-Reward-V2-Llama-3.1-8B --repo-type model --local_dir ./LongReward_Skywork-Reward-V2-Llama-3.1-8B

modelscope download LCM_group/LongReward_Skywork-Reward-V2-Llama-3.1-8B --repo-type model --local_dir ./LongReward_Skywork-Reward-V2-Llama-3.1-8B
```

### We provide the benchmark dataset and trained models in our [modelscope](https://modelscope.cn/collections/LongReward-Model-and-Dataset-2abfa246c09240)

