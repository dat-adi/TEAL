# Training-Free Acivation Sparsity in Large Language Models


This is a fork that seeks to retrieve the sparsified input tensors and matrices from TEAL by simulating it in PyTorch.
For the actual implementation, check out the [main TEAL repository](https://github.com/FasterDecoding/TEAL/).

More Resources:
[[Paper](https://www.arxiv.org/abs/2408.14690)][[Blog](https://www.together.ai/blog/teal-training-free-activation-sparsity-in-large-language-models)]

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

## Execute

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
