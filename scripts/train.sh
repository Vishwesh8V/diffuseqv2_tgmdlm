# python -m torch.distributed.launch \
#     --nproc_per_node=4 \
#     --master_port=12231 \
#     --use_env run_train.py \
#     --diff_steps 2000 \
#     --lr 0.0001 \
#     --learning_steps 320000 \
#     --save_interval 10000 \
#     --seed 102 \
#     --noise_schedule sqrt \
#     --hidden_dim 128 \
#     --bsz 424 \
#     --microbatch 106 \
#     --dataset iwslt14_mol \
#     --data_dir ./datasets/iwslt14_mol \
#     --learned_mean_embed True \
#     --denoise True \
#     --vocab dual \
#     --smiles_vocab_path datasets/generate_vocab.txt \
#     --scibert_path allenai/scibert_scivocab_uncased \
#     --seq_len 128 \
#     --schedule_sampler lossaware \
#     --notes smiles_caption_dual
#---------------------------------------------------------------------------------------------
#RUN 1: 1024 bsz with checkpoint 50k
export OMP_NUM_THREADS=1
torchrun \
    --nproc_per_node=8 \
    --master_port=12231 \
    scripts/run_train.py \
    --diff_steps 2000 \
    --lr 0.0001 \
    --learning_steps 30000 \
    --save_interval 10000 \
    --seed 102 \
    --noise_schedule sqrt \
    --hidden_dim 128 \
    --bsz 1024 \
    --microbatch 128 \
    --dataset iwslt14_mol \
    --data_dir datasets/iwslt14_mol/iwslt14_mol \
    --learned_mean_embed True \
    --denoise True \
    --vocab dual \
    --smiles_vocab_path datasets/generate_vocab.txt \
    --scibert_path allenai/scibert_scivocab_uncased \
    --seq_len 128 \
    --schedule_sampler lossaware \
    --notes 100k_ckpt \
    --use_fp16 \
    --resume_checkpoint old_ckpts_10k_to_50k/ema_0.9999_100000.pt

