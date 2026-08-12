# python -u run_decode.py \
#   --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49 \
#   --seed 123 \
#   --split test
# fixed
# python -u run_decode.py --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1
# "##" fixed
# python -u run_decode.py --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1 --note "eval_500_fixed"
# "##" fixed for 500 samples
#python -u run_decode.py --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq256_adan_0807_ckpt20k20260708-10:28:56 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1 --note "adanoise_10kstride_60k_110k_ckpts_50kalpha" --time_schedule_path diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq256_adan_0807_ckpt20k20260708-10:28:56/alpha_cumprod_step_30000.npy
#python -u run_decode.py --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1 --note "adanoise_ckpt119k_seq128" --time_schedule_path diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/alpha_cumprod_step_120000.npy
#diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/alpha_cumprod_step_50000.npy

python -u run_decode.py --model_dir /home/ee/phd/eez248435/diffuseqv2_allmodels/diffuseqv2_tgm_adanoise/diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_nomix_adanoise_scratch20260809-22:09:06 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1 --note "adanoise_scratchtest1208_80knpy" --time_schedule_path diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_nomix_adanoise_scratch20260809-22:09:06/alpha_cumprod_step_80000.npy


