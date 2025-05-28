#### MAR_Large

num_iter=96
pre_full_iters=$((num_iter/2))
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 cache_decoding.py --model mar_large --diffloss_d 8 --diffloss_w 1280 --num_images 50000 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 --output_dir output/mar_large --resume pretrained_models/mar_large --num_iter $num_iter --pre_full_iters $pre_full_iters --num_cache_iters 1 --eval_bsz 64 --diff_fast_ratio 0.4

num_iter=84
pre_full_iters=$((num_iter/2))
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 cache_decoding.py --model mar_large --diffloss_d 8 --diffloss_w 1280 --num_images 50000 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 --output_dir output/mar_large --resume pretrained_models/mar_large --num_iter $num_iter --pre_full_iters $pre_full_iters --num_cache_iters 1 --eval_bsz 64 --diff_fast_ratio 0.4


num_iter=56
pre_full_iters=$((num_iter/2))
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 cache_decoding.py --model mar_large --diffloss_d 8 --diffloss_w 1280 --num_images 50000 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 --output_dir output/mar_large --resume pretrained_models/mar_large --num_iter $num_iter --pre_full_iters $pre_full_iters --num_cache_iters 1 --eval_bsz 64 --diff_fast_ratio 0.4

num_iter=36
pre_full_iters=$((num_iter/2))
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 cache_decoding.py --model mar_large --diffloss_d 8 --diffloss_w 1280 --num_images 50000 --num_sampling_steps 100 --cfg 3.0 --cfg_schedule linear --temperature 1.0 --output_dir output/mar_large --resume pretrained_models/mar_large --num_iter $num_iter --pre_full_iters $pre_full_iters --num_cache_iters 1 --eval_bsz 64 --diff_fast_ratio 0.4



#### MAR_Huge
num_iter=128
pre_full_iters=$((num_iter/2))
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 cache_decoding.py --model mar_huge --diffloss_d 12 --diffloss_w 1536 --num_images 50000 --num_sampling_steps 100 --cfg 3.2 --cfg_schedule linear --temperature 1.0 --output_dir output/mar_huge --resume pretrained_models/mar_huge --num_iter $num_iter --pre_full_iters $pre_full_iters --num_cache_iters 1 --eval_bsz 64 --diff_fast_ratio 0.4

num_iter=96
pre_full_iters=$((num_iter/2))
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 cache_decoding.py --model mar_huge --diffloss_d 12 --diffloss_w 1536 --num_images 50000 --num_sampling_steps 100 --cfg 3.2 --cfg_schedule linear --temperature 1.0 --output_dir output/mar_huge --resume pretrained_models/mar_huge --num_iter $num_iter --pre_full_iters $pre_full_iters --num_cache_iters 1 --eval_bsz 64 --diff_fast_ratio 0.4


num_iter=48
pre_full_iters=$((num_iter/2))
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 cache_decoding.py --model mar_huge --diffloss_d 12 --diffloss_w 1536 --num_images 50000 --num_sampling_steps 100 --cfg 3.2 --cfg_schedule linear --temperature 1.0 --output_dir output/mar_huge --resume pretrained_models/mar_huge --num_iter $num_iter --pre_full_iters $pre_full_iters --num_cache_iters 1 --eval_bsz 64 --diff_fast_ratio 0.4


