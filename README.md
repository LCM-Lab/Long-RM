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

### For Generative Model
To run the first training process:
```bash
bash scripts/sft.sh 
```
To run the second training process:
```bash
bash bash scripts/simpo_grm.sh 
```

### For Discriminative Model
Directly run the second training process:
```bash
bash bash scripts/simpo_disrm.sh 
```


## 📊 Evaluate

* ###  We provide the benchmark dataset and trained models in our [Modelscope](https://modelscope.cn/collections/LongReward-Model-and-Dataset-2abfa246c09240).

### For Generative Model

```bash
modelscope download LCM_group/LongReward_Qwen3-8B --repo-type model --local_dir ./LongReward_Qwen3-8B

python evaluate/eval.py --model-path ./LongReward_Qwen3-8B --data-path ./LongReward-Bench
```

### For Discriminative Model

```bash
modelscope download LCM_group/LongReward_Skywork-Reward-V2-Llama-3.1-8B --repo-type model --local_dir ./LongReward_Skywork-Reward-V2-Llama-3.1-8B

python evaluate/eval.py --model-path ./LongReward_Skywork-Reward-V2-Llama-3.1-8B --data-path ./LongReward-Bench --is-disrm
```


