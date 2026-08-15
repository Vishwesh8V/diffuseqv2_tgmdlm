CUDA_VISIBLE_DEVICES=2 python -u run_decode_solver.py \
--model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1 --note "diag_smart" --time_schedule_path diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/alpha_cumprod_step_120000.npy --step 10 \
--method smart --smart_matrix_path /home/ee/phd/eez248435/diffuseqv2_allmodels/diffuseqv2_tgm_adanoise/J_K10_active_positions.csv \
