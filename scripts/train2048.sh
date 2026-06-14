#RUN 2: 2048 bsz 
export OMP_NUM_THREADS=1
torchrun \
    --nproc_per_node=8 \
    --master_port=12233 \
    scripts/run_train.py \
    --diff_steps 2000 \
    --lr 0.0001 \
    --learning_steps 200000 \
    --save_interval 10000 \
    --seed 102 \
    --noise_schedule sqrt \
    --hidden_dim 128 \
    --bsz 2048 \
    --microbatch 512 \
    --dataset iwslt14_mol \
    --data_dir datasets/iwslt14_mol/iwslt14_mol \
    --learned_mean_embed True \
    --denoise True \
    --vocab dual \
    --smiles_vocab_path datasets/generate_vocab.txt \
    --scibert_path allenai/scibert_scivocab_uncased \
    --seq_len 128 \
    --schedule_sampler lossaware \
    --use_fp16 \
    --notes 2048_adanoise_scratch_run \
    --loss_update_granu 20 \
    --schedule_update_stride 2000 