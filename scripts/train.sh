#RUN 1: 1024 bsz
export OMP_NUM_THREADS=1
torchrun \
    --nproc_per_node=8 \
    --master_port=12231 \
    scripts/run_train.py \
    --diff_steps 2000 \
    --lr 0.0001 \
    --learning_steps 200000 \
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
    --notes 1024_adanoise_scratch_run_50granu_10kstride_gradclip_13062026 \
    --use_fp16 \
    --gradient_clipping 1.0 \
    --loss_update_granu 50 \
    --schedule_update_stride 10000

