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

    calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/ema_0.9999_069000.pt.samples/seed123_step0_adanoise_10kstride_60k_110k_ckpts_40kalpha.json')
    calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/ema_0.9999_079000.pt.samples/seed123_step0_adanoise_10kstride_60k_110k_ckpts_40kalpha.json')
    calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/ema_0.9999_089000.pt.samples/seed123_step0_adanoise_10kstride_60k_110k_ckpts_40kalpha.json')
    calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/ema_0.9999_099000.pt.samples/seed123_step0_adanoise_10kstride_60k_110k_ckpts_40kalpha.json')
    calculate_bleu_from_jsonl('../generation_outputs/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_adanoise_60k_110kckpts_10kstride20260614-18:54:53/ema_0.9999_109000.pt.samples/seed123_step0_adanoise_10kstride_60k_110k_ckpts_40kalpha.json')