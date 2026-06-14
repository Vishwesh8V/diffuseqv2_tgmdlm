# python -u run_decode.py \
#   --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49 \
#   --seed 123 \
#   --split test
# fixed
# python -u run_decode.py --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1
# "##" fixed
# python -u run_decode.py --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1 --note "eval_500_fixed"
# "##" fixed for 500 samples
python -u run_decode.py --model_dir diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_1024_adanoise_scratch_run_50granu_10kstride_gradclip_1306202620260613-15:57:12 --seed 123 --split test --top_p 0.9 --rejection_rate 0.1 --note "adanoise_1st_gen"