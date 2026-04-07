# [ACL 2026 Findings] ASL: Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference
!["FastKVandASL_comparison"](/images/comparison.png)
We propose **ASL**, a training-free method that adaptively chooses the token-selection layer for Prefill Acceleration and KV cache reduction, exploiting the variance of token ranks ordered by attention score. 

ASL balances the performance across different tasks while meeting the user-specified KV budget requirement. 

ASL operates during the prefilling stage and can be jointly used with existing KV cache reduction methods such as SnapKV to optimize the decoding stage.
## Environment
```
CUDA Version: 12.2
python=3.9
torch==2.1.0
transformers==4.45.0
flash-attn==2.6.3
```

## Installation
Installation with the requirements package.
```
conda create -n asl python=3.9
conda activate asl
cd ASL
pip install -r requirements.txt
pip install flash-attn==2.6.3
```
## Quick Start
ASL parameters can be easily customized by modifying the shell scripts in each benchmark directory.
### RULER
```
cd eval/ruler/data/synthetic/json/
python download_paulgraham_essay.py
bash download_qa_dataset.sh
cd ../../../
bash run_ruler.sh
```
### Needle-in-a-haystack
```
cd eval/needle_in_a_haystack/
mkdir -p data
wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl -O ./data/pg19_mini.jsonl
bash run_needle.sh
```
### Infinite-Bench
```
cd eval/infinite_bench/
bash run_infinitebench.sh
```
