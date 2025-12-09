import json
import math
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from utils import make_prompt_template

from config import (
    MODEL_CACHE_PATH,
    prompt_template_optimize
)

# ============ CONFIG ============
device = "cuda:0"
input_jsonl = "testset/vicuna_eval.jsonl"
tmp_step1 = "tmp_step1_r0.jsonl"
tmp_step2 = "tmp_step2_r0.jsonl"
output_jsonl = "responses_with_semantic.jsonl"
infer_model_path = "meta-llama/Llama-2-7b-chat-hf"  # chỉnh nếu cần
M = 10

# -----------------------------------------------------
# Helper: generate text
# -----------------------------------------------------
def generate(model, tokenizer, prompt, max_new_tokens=1024, apply_chat_template=True, **kwargs):
    if apply_chat_template:
        prompt = make_prompt_template(prompt)
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False)
    # Encode input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    input_ids = inputs["input_ids"]

    # Generate
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        **kwargs
    )

    # Lấy phần sinh thêm
    generated_ids = out[0][len(input_ids[0]):]

    # Decode chỉ phần mới
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return text.strip()

# -----------------------------------------------------
# STEP 1: Generate paraphrase prompts using BPO 
# -----------------------------------------------------
def step1_generate_paraphrase():
    print("\n===== STEP 1: Paraphrase =====")

    optimize_path = "THUDM/BPO"
    model = AutoModelForCausalLM.from_pretrained(
        optimize_path, cache_dir=MODEL_CACHE_PATH, dtype=torch.float16
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        optimize_path, cache_dir=MODEL_CACHE_PATH, use_fast=False
    )
    model.config.return_dict = True

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(tmp_step1, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Step1"):
            item = json.loads(line)
            prompt = item["text"]

            # Sinh M paraphrases
            paraphrases = [
                generate(
                    model, tokenizer,
                    prompt_template_optimize.format(prompt),
                    temperature=1.0,
                    apply_chat_template=False
                )
                for _ in range(M)
            ]

            fout.write(json.dumps({
                "prompt": prompt,
                "paraphrase_prompts": paraphrases
            }, ensure_ascii=False) + "\n")

    del model
    torch.cuda.empty_cache()
    print("✓ Done Step 1 →", tmp_step1)

# -----------------------------------------------------
# STEP 2: LLM inference cho từng paraphrase (Vicuna)
# Input  : tmp_step1  (từ Step 1, chứa prompt + paraphrase_prompts)
# Output : tmp_step2  (mỗi item sẽ có thêm paraphrase_responses)
# -----------------------------------------------------
# -----------------------------------------------------
# STEP 2: LLM inference cho từng paraphrase (dùng hàm generate() có sẵn)
# -----------------------------------------------------
def step2_infer_vicuna(device="cuda:0"):
    print("===== STEP 2: Vicuna inference (full, paraphrases + original prompt) =====")

    # Load model/tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        infer_model_path,
        cache_dir=MODEL_CACHE_PATH,
        torch_dtype=torch.float16,
    ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        infer_model_path,
        cache_dir=MODEL_CACHE_PATH,
        legacy=False
    )

    # File input/output
    with open(tmp_step1, "r", encoding="utf-8") as fin, \
         open(tmp_step2, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Step 2 - infer prompts"):
            item = json.loads(line)
            prompt = item["prompt"]
            paraphrases = item["paraphrase_prompts"]

            # === 1) Infer original prompt ===
            full_prompt_orig = prompt.strip()
            response_original = generate(
                model,
                tokenizer,
                full_prompt_orig,
                max_new_tokens=2048,
                temperature=0.7,
                top_p=1.0,
                num_beams=1
            )

            # === 2) Infer paraphrases ===
            paraphrase_responses = []
            for p in paraphrases:
                full_prompt = p.strip()
                resp = generate(
                    model,
                    tokenizer,
                    full_prompt,
                    max_new_tokens=2048,
                    temperature=0.7,
                    top_p=1.0,
                    num_beams=1
                )
                paraphrase_responses.append(resp)

            # === Lưu output STEP 2 ===
            fout.write(json.dumps({
                "prompt": prompt,
                "response_original": response_original,
                "paraphrase_prompts": paraphrases,
                "paraphrase_responses": paraphrase_responses
            }, ensure_ascii=False) + "\n")

    # cleanup
    del model
    torch.cuda.empty_cache()
    print("✓ Done STEP 2 →", tmp_step2)

# -----------------------------------------------------
# STEP 3: SBERT clustering + semantic entropy 
# -----------------------------------------------------
def step3_sbert_clustering(device='cuda:0', threshold=0.8):
    print("===== STEP 3: SBERT clustering + semantic entropy =====")

    sbert = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device, cache_folder=MODEL_CACHE_PATH)

    with open(tmp_step2, "r", encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            item = json.loads(line)
            samples = item["paraphrase_responses"]  # dùng trực tiếp từ STEP 2

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
    # step1_generate_paraphrase()
    step2_infer_vicuna()
    step3_sbert_clustering()
    print("\n🎉 ALL DONE!")
