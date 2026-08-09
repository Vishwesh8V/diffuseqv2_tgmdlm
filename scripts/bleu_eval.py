import json
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def calculate_bleu_from_jsonl(file_path):
    data = []
    
    # Read and parse the JSONL file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return

    # Prepare tokenized references and candidates
    # NLTK expects: 
    # references: list of list of lists of tokens (multiple references per candidate)
    # candidates: list of list of tokens
    references_tokenized = [[item["reference"].split()] for item in data if "reference" in item]
    candidates_tokenized = [item["recover"].split() for item in data if "recover" in item]

    if not references_tokenized or not candidates_tokenized:
        print("No valid data found in the file.")
        return

    # Calculate Standard BLEU-4
    standard_score = corpus_bleu(references_tokenized, candidates_tokenized)
    
    # Calculate Smoothed BLEU-4 (better for small datasets or zero n-gram overlaps)
    smoothie = SmoothingFunction().method1
    smoothed_score = corpus_bleu(references_tokenized, candidates_tokenized, smoothing_function=smoothie)

    print(f"--- Results for {file_path} ---")
    print(f"Total sequences processed: {len(data)}")
    print(f"Standard Corpus BLEU-4: {standard_score:.4f}")
    print(f"Smoothed Corpus BLEU-4: {smoothed_score:.4f}")

# Run the function
if __name__ == "__main__":
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_learned_mask_fp16_denoise_0.5_reproduce20260410-16:01:15/ema_0.9999_150000.pt.samples/seed110_solverstep10_none.json') # 10steps
    # 2000 Steps Bleu Eval on Mol data
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_mol_timing_probe20260529-12:06:01/ema_0.9999_020000.pt.samples/seed101_step0_none.json')
    # 10 steps mol data
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_mol_timing_probe20260529-12:06:01/ema_0.9999_020000.pt.samples/seed124_solverstep10_none.json')
    # 2 steps mol data
    # generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49/ema_0.9999_050000.pt.samples/seed123_step0_none.json
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49/ema_0.9999_010000.pt.samples/seed123_step0_eval_500samples_fixed.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49/ema_0.9999_020000.pt.samples/seed123_step0_eval_500samples_fixed.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49/ema_0.9999_030000.pt.samples/seed123_step0_eval_500samples_fixed.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49/ema_0.9999_040000.pt.samples/seed123_step0_eval_500samples_fixed.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_smiles_caption_dual20260604-12:00:49/ema_0.9999_050000.pt.samples/seed123_step0_eval_500samples_fixed.json')
    
    #0.5787
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/ema_0.9999_069000.pt.samples/seed123_step0_adanoise_10kstride_60k_110k_ckpts_40kalpha.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/ema_0.9999_079000.pt.samples/seed123_step0_adanoise_10kstride_60k_110k_ckpts_40kalpha.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/ema_0.9999_089000.pt.samples/seed123_step0_adanoise_10kstride_60k_110k_ckpts_40kalpha.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/ema_0.9999_099000.pt.samples/seed123_step0_adanoise_10kstride_60k_110k_ckpts_40kalpha.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/ema_0.9999_109000.pt.samples/seed123_step0_adanoise_10kstride_60k_110k_ckpts_40kalpha.json')

    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adan_10kstride_ckpt109k20260704-02:48:53/ema_0.9999_119000.pt.samples/seed123_step0_adanoise_149k_50knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adan_10kstride_ckpt109k20260704-02:48:53/ema_0.9999_129000.pt.samples/seed123_step0_adanoise_149k_50knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adan_10kstride_ckpt109k20260704-02:48:53/ema_0.9999_139000.pt.samples/seed123_step0_adanoise_149k_50knpy.json')
    
    #0.5801 - 189k cpkt with 190k.npy
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_109000.pt.samples/seed123_step0_adanoise_110_200k_190knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_119000.pt.samples/seed123_step0_adanoise_110_200k_190knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_129000.pt.samples/seed123_step0_adanoise_110_200k_190knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_139000.pt.samples/seed123_step0_adanoise_110_200k_190knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_149000.pt.samples/seed123_step0_adanoise_110_200k_190knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_159000.pt.samples/seed123_step0_adanoise_110_200k_190knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_169000.pt.samples/seed123_step0_adanoise_110_200k_190knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_179000.pt.samples/seed123_step0_adanoise_110_200k_190knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_189000.pt.samples/seed123_step0_adanoise_110_200k_190knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_199000.pt.samples/seed123_step0_adanoise_110_200k_190knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed123_step0_adanoise_110_200k_190knpy.json')

    #0.5864 - 189k ckpt with 180.npy
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_109000.pt.samples/seed123_step0_adanoise_110_200k_180knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_119000.pt.samples/seed123_step0_adanoise_110_200k_180knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_129000.pt.samples/seed123_step0_adanoise_110_200k_180knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_139000.pt.samples/seed123_step0_adanoise_110_200k_180knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_149000.pt.samples/seed123_step0_adanoise_110_200k_180knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_159000.pt.samples/seed123_step0_adanoise_110_200k_180knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_169000.pt.samples/seed123_step0_adanoise_110_200k_180knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_179000.pt.samples/seed123_step0_adanoise_110_200k_180knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_189000.pt.samples/seed123_step0_adanoise_110_200k_180knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_199000.pt.samples/seed123_step0_adanoise_110_200k_180knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed123_step0_adanoise_110_200k_180knpy.json')
    
    #0.5656 - 189k ckpt with 170.npy
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_109000.pt.samples/seed123_step0_adanoise_110_200k_170knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_119000.pt.samples/seed123_step0_adanoise_110_200k_170knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_129000.pt.samples/seed123_step0_adanoise_110_200k_170knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_139000.pt.samples/seed123_step0_adanoise_110_200k_170knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_149000.pt.samples/seed123_step0_adanoise_110_200k_170knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_159000.pt.samples/seed123_step0_adanoise_110_200k_170knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_169000.pt.samples/seed123_step0_adanoise_110_200k_170knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_179000.pt.samples/seed123_step0_adanoise_110_200k_170knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_189000.pt.samples/seed123_step0_adanoise_110_200k_170knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_199000.pt.samples/seed123_step0_adanoise_110_200k_170knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed123_step0_adanoise_110_200k_170knpy.json')

    #0.5756 - 199k ckpt with 160.npy
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_109000.pt.samples/seed123_step0_adanoise_110_200k_160knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_119000.pt.samples/seed123_step0_adanoise_110_200k_160knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_129000.pt.samples/seed123_step0_adanoise_110_200k_160knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_139000.pt.samples/seed123_step0_adanoise_110_200k_160knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_149000.pt.samples/seed123_step0_adanoise_110_200k_160knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_159000.pt.samples/seed123_step0_adanoise_110_200k_160knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_169000.pt.samples/seed123_step0_adanoise_110_200k_160knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_179000.pt.samples/seed123_step0_adanoise_110_200k_160knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_189000.pt.samples/seed123_step0_adanoise_110_200k_160knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_199000.pt.samples/seed123_step0_adanoise_110_200k_160knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed123_step0_adanoise_110_200k_160knpy.json')

    #0.5781 - 189k ckpt with 150.npy
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_109000.pt.samples/seed123_step0_adanoise_110_200k_150knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_119000.pt.samples/seed123_step0_adanoise_110_200k_150knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_129000.pt.samples/seed123_step0_adanoise_110_200k_150knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_139000.pt.samples/seed123_step0_adanoise_110_200k_150knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_149000.pt.samples/seed123_step0_adanoise_110_200k_150knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_159000.pt.samples/seed123_step0_adanoise_110_200k_150knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_169000.pt.samples/seed123_step0_adanoise_110_200k_150knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_179000.pt.samples/seed123_step0_adanoise_110_200k_150knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_189000.pt.samples/seed123_step0_adanoise_110_200k_150knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_199000.pt.samples/seed123_step0_adanoise_110_200k_150knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed123_step0_adanoise_110_200k_150knpy.json')

    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq256_adan_ckpt40k20260720-15:32:52/ema_0.9999_040000.pt.samples/seed123_step0_adanoise_40_70k_50knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq256_adan_ckpt40k20260720-15:32:52/ema_0.9999_050000.pt.samples/seed123_step0_adanoise_40_70k_50knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq256_adan_ckpt40k20260720-15:32:52/ema_0.9999_060000.pt.samples/seed123_step0_adanoise_40_70k_50knpy.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq256_adan_ckpt40k20260720-15:32:52/ema_0.9999_070000.pt.samples/seed123_step0_adanoise_40_70k_50knpy.json')

    # 0.5781 - 189k ckpt with 110.npy
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed123_solverstep2_bench_dpm2_r0.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed123_solverstep10_bench_dpm10_r0.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed123_solverstep100_bench_dpm100_r0.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed123_step0_bench_normal_r0.json')
    # calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed123_step0_adanoise_110_200k_120knpy.json')
    calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed102_solverstep0_adaptive_K20.json')
    
    # generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt.samples/seed102_solverstep0_adaptive_K20.json