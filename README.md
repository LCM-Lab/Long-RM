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

[GRM]
```bash
bash scripts/sft.sh 
```

To run the second training process:

[GRM]
```bash
bash bash scripts/simpo_grm.sh 
```

[DisRM]

```bash
bash bash scripts/simpo_disrm.sh 
```




