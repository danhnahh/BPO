import json
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

from config import (
    MODEL_CACHE_PATH,
    prompt_template_vicuna,
    prompt_template_optimize
)

# ============ CONFIG ============
device = "cuda:0"
input_jsonl = "testset/vicuna_eval.jsonl"
tmp_step1 = "tmp_step1_r0.jsonl"
output_jsonl = "responses_with_semantic.jsonl"
M = 10

# -----------------------------------------------------
# Helper: generate text
# -----------------------------------------------------
def generate(model, tokenizer, prompt, split=None, max_new_tokens=1024, **kwargs):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        **kwargs
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    if split and split in text:
        return text.split(split)[-1].strip()
    return text.strip()

# -----------------------------------------------------
# STEP 2: Generate paraphrase prompts using BPO
# -----------------------------------------------------
def step1_generate_paraphrase():
    print("\n===== STEP 2: Paraphrase =====")

    optimize_path = "THUDM/BPO"
    model = AutoModelForCausalLM.from_pretrained(
        optimize_path, cache_dir=MODEL_CACHE_PATH, dtype=torch.float16
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        optimize_path, cache_dir=MODEL_CACHE_PATH, use_fast=False, legacy=True
    )
    model.config.return_dict = True

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(tmp_step1, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Step2"):
            item = json.loads(line)
            prompt = item["prompt"]

            paraphrases = [
                generate(
                    model, tokenizer,
                    prompt_template_optimize.format(prompt),
                    split="[/INST]",
                    temperature=1.0
                )
                for _ in range(M)
            ]

            fout.write(json.dumps({
                "prompt": prompt,
                "response_original": item["response_original"],
                "paraphrase_prompts": paraphrases
            }, ensure_ascii=False) + "\n")

    del model
    torch.cuda.empty_cache()
    print("✓ Done Step 2 →", tmp_step1)


# -----------------------------------------------------
# STEP 2: SBERT clustering + semantic entropy 
# -----------------------------------------------------
def step2_sbert_clustering(device='cuda:0', threshold=0.8):
    print("===== STEP 2: SBERT clustering + semantic entropy =====")

    sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)

    with open(tmp_step1, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            item = json.loads(line)
            samples = item["paraphrase_prompts"]  # dùng trực tiếp từ STEP 2

            # Encode tất cả các câu
            embeddings = sbert.encode(samples, convert_to_tensor=True)

            # Khởi tạo cluster
            clusters = []
            used = set()

            for i, emb_i in enumerate(embeddings):
                if i in used:
                    continue
                cluster = [i]
                used.add(i)
                for j, emb_j in enumerate(embeddings):
                    if j in used:
                        continue
                    sim = util.pytorch_cos_sim(emb_i, emb_j).item()
                    if sim >= threshold:
                        cluster.append(j)
                        used.add(j)
                clusters.append(cluster)

            # Cluster probabilities
            cluster_probs = [len(c)/len(samples) for c in clusters]

            # Semantic entropy
            entropy = -sum(p * (math.log(p) if p > 0 else 0) for p in cluster_probs)

            # Sau khi tính cluster_probs và entropy
            K = len(clusters)
            conf_score = 1 - (entropy / math.log(K)) if K > 1 else 1.0

            fout.write(json.dumps({
                "prompt": item["prompt"],
                "response_original": item["response_original"],
                "samples": samples,
                "clusters": clusters,
                "cluster_probs": cluster_probs,
                "semantic_entropy": entropy,
                "conf_score": conf_score
            }, ensure_ascii=False) + "\n")

    print("✓ Done STEP 3 (SBERT)")

# -----------------------------------------------------
# RUN ALL STEPS
# -----------------------------------------------------
if __name__ == "__main__":
    step1_generate_paraphrase()
    step2_sbert_clustering()
    print("\n🎉 ALL DONE!")
