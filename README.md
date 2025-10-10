# 📜 LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling

> **Paper**: [https://arxiv.org/pdf/2510.06915](https://arxiv.org/pdf/2510.06915)  
> **Models**:  
> - 🤖 Generative RM: [`LCM_group/LongReward_Qwen3-8B`](https://modelscope.cn/models/LCM_group/LongReward_Qwen3-8B)  
> - 🔍 Discriminative RM: [`LCM_group/LongReward_Skywork-Reward-V2-Llama-3.1-8B`](https://modelscope.cn/models/LCM_group/LongReward_Skywork-Reward-V2-Llama-3.1-8B)  
>  
> *Pushing the limits of reward modeling beyond 128K tokens — with memory-efficient training and a new benchmark for long-context reward model.*

---

## 📅 Update Log

1. **2025-10-09** — 🚀 Code released: Full training & evaluation pipeline now open-sourced.

2. **Coming Soon** — LongReward-Bench dataset will be released on Hugging Face Datasets.

3. **Coming Soon** — Releasing models trained with LongRM.

---

## 💻 Environment & Installation

Set up your training environment with the following steps:

```bash
cd LongRM
pip install -r requirements.txt
```

### 🔌 Install FlashAttention (Highly Recommended)

For optimal memory efficiency and speed during long-context training, install **FlashAttention-2**:

1. Download the compatible `.whl` file for your CUDA version from:  
   👉 [https://github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases)

2. Install locally:
   ```bash
   pip install <path_to_downloaded_flash_attn_whl_file>
   ```

3. Install **Ring-Flash-Attention** for scalable long-sequence training:
   ```bash
   pip install ring_flash_attn
   ```

> 💡 **Tip**: Use `nvidia-smi` to verify your CUDA version. Ensure the FlashAttention wheel matches your driver and PyTorch setup.

---

## 🔥 Training Pipeline

LongRM uses a **two-stage training paradigm** to unlock long-context reward modeling. Choose your modeling approach below.

### 1) Generative Reward Model (GRM)

**Stage 1: Supervised Fine-Tuning (SFT)**  
Train the base model on instruction-following data:
```bash
bash scripts/sft.sh
```

**Stage 2: Preference Optimization with SIMPO**  
Refine the model using long-context preference signals:
```bash
bash scripts/simpo_grm.sh
```

> ✅ *Requires completion of Stage 1.*

---

### 2) Discriminative Reward Model (DisRM)

**Single-Stage: Direct Preference Optimization**  
Train a binary classifier to score response pairs directly under long contexts:
```bash
bash scripts/simpo_disrm.sh
```

> ✅ *No SFT pretraining needed — end-to-end training from scratch.*

---

## 📊 Evaluation

We provide a **new benchmark dataset — `LongReward-Bench`** — and **pretrained models** on **ModelScope** for easy evaluation.

🔗 **Dataset & Models Hub**:  
👉 [https://modelscope.cn/collections/LongReward-Model-and-Dataset-2abfa246c09240](https://modelscope.cn/collections/LongReward-Model-and-Dataset-2abfa246c09240)

---

### 🤖 Evaluate Generative RM (e.g., Qwen3-8B)

```bash
modelscope download LCM_group/LongReward_Qwen3-8B --repo-type model --local_dir ./LongReward_Qwen3-8B

python evaluate/eval.py --model-path ./LongReward_Qwen3-8B --data-path ./LongReward-Bench
```

### 🔍 Evaluate Discriminative RM (e.g., Skywork-Reward-V2-Llama-3.1-8B)

```bash
modelscope download LCM_group/LongReward_Skywork-Reward-V2-Llama-3.1-8B --repo-type model --local_dir ./LongReward_Skywork-Reward-V2-Llama-3.1-8B

python evaluate/eval.py --model-path ./LongReward_Skywork-Reward-V2-Llama-3.1-8B --data-path ./LongReward-Bench --is-disrm
```

> ✅ Add `--is-disrm` flag only for discriminative models.  
> 📈 Results are reported as **accuracy** and **AUC** on long-context preference pairs (up to 131K tokens).

---

## 🤝 Contributing

We welcome contributions! Whether it’s:
- Adding new datasets or evaluation metrics  
- Improving training efficiency  
- Porting to other architectures (e.g., Mistral, Gemma)

Please open an **[Issue](https://github.com/LCM-Lab/LongRM/issues)** or submit a **[Pull Request](https://github.com/LCM-Lab/LongRM/pulls)**.

---

## 📬 Contact

Questions? Suggestions? Reach out at: zctang2000@gmail.com
