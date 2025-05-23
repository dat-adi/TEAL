# Training-Free Acivation Sparsity in Large Language Models

[[Paper](https://www.arxiv.org/abs/2408.14690)][[Blog](https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models)]


TEAL induces up to 40-50% model-wide activation sparsity in modern LLMs with minimal degradation, resulting in an up to 1.53-1.8x speedup in single-batch decoding.

<div align="center">
    <img src="figures/clickbait.png" width="500" height="auto"/>
  </a>
</div>

The current release supports:
- FP16 inference for Llama-2/3 models using uniform sparsities
- Accuracy evaluation for Llama-2/3 and Mistral models using uniform and block-wise greedy sparsities


## News

- [01/2025] 🔥 TEAL is accepted to ICLR 2025 as a Spotlight!
- [08/2024] 🔥 Arxiv release!

## Abstract

Activation sparsity can enable practical inference speedups in large language models (LLMs) by reducing the compute and memory-movement required for matrix
multiplications during the forward pass. However, existing methods face limitations that inhibit widespread adoption. Some approaches are tailored towards
older models with ReLU-based sparsity, while others require extensive continued
pre-training on up to hundreds of billions of tokens. This paper describes TEAL
(**T**raining-Fre**e** **A**ctivation Sparsity in **L**LMs), a simple training-free method that
applies magnitude-based activation sparsity to hidden states throughout the entire
model. TEAL achieves 40-50% model-wide sparsity with minimal performance
degradation across Llama-2, Llama-3, and Mistral families, with sizes varying
from 7B to 70B. We improve existing sparse kernels and demonstrate wall-clock
decoding speed-ups of up to 1.53× and 1.8× at 40% and 50% model-wide sparsity.
TEAL is compatible with weight quantization, enabling further efficiency gains.



## Contents

- [Install](#Install)
- [Demo](#Demo)
- [Inference Usage](#Inference-Usage)
- [Accuracy Usage](#Accuracy-Usage)
- [Citation](#citation)

## Install

1. Clone the repo and navigate to TEAL:

```
git clone https://github.com/dat-adi/TEAL
cd TEAL
```

2. Set up environment:


```bash
conda create -yn teal python=3.11
conda activate teal

pip install -e .
```

3. Create a huggingface account and export a key to be able to download models. Note that you will need to accept the terms and conditions document on the Huggingface Llama page:

```bash
export HF_TOKEN=...
```

## Execution

1. Navigate to gpt-fast and export the save path which is where you'd have stored the models:

```bash
cd gpt-fast
export SAVE_PATH=/home/ec2-user/TEAL/gpt-fast/models/
```

2. Download model weights and convert to gpt-fast format (`scripts/prepare.sh`):
```bash
python scripts/download.py --repo_id meta-llama/Llama-2-7b-hf --path $SAVE_PATH && python scripts/convert_hf_checkpoint.py --checkpoint_dir $SAVE_PATH/meta-llama/Llama-2-7b-hf
```

3. Run sparse inference using the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python generate.py \
    --checkpoint_path $SAVE_PATH/meta-llama/Llama-2-7b-hf/model.pth \ 
    --hist_path ../models/Llama-2-7B/histograms \ 
    --sparsity 0.5 \ 
    --max_new_tokens 40
```

Provided that the proxy function `simulate_splitk` is active and being used, this should end up dumping the sparsified input tensors onto your system at different stages of a layer, across 32 layers for 40 inferences.

Modifying the `sparsity` value in the above command will increase/decrease sparsity, while modifying the max new tokens increases/decreases the tokens generated and thus the number of matrices dumped on disk as well.

## Further info
This is a fork that seeks to retrieve the sparsified input tensors and matrices from TEAL.
For more options and features, check out the main repository.

## Citation

If you find TEAL useful, please consider citing:

```
@misc{liu2024trainingfreeactivationsparsitylarge,
      title={Training-Free Activation Sparsity in Large Language Models}, 
      author={James Liu and Pragaash Ponnusamy and Tianle Cai and Han Guo and Yoon Kim and Ben Athiwaratkun},
      year={2024},
      eprint={2408.14690},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.14690}, 
}
```
