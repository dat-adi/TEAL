export SAVE_PATH=/home/ec2-user/TEAL/gpt-fast/models/
for i in {50..95}
do
	if [ $((i % 5)) -eq 0 ]
	then
CUDA_VISIBLE_DEVICES=0 python generate.py --checkpoint_path $SAVE_PATH/meta-llama/Llama-2-7b-hf/model.pth --hist_path ../models/Llama-2-7B/histograms --max_new_tokens 41 --sparsity 0.${i} --num_samples 1 > sp.out
mv *.pt sp.out ${i}/
	fi
done
