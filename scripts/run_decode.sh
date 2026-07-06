# python -u run_decode.py \
#   --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49 \
#   --seed 123 \
#   --split test
# fixed
# python -u run_decode.py --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1
# "##" fixed
# python -u run_decode.py --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1 --note "eval_500_fixed"
# "##" fixed for 500 samples
python -u run_decode.py --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1 --note "adanoise_10kstride_60k_110k_ckpts_50kalpha" --time_schedule_path diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/alpha_cumprod_step_50000.npy