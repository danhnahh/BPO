import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch

from config import MODEL_CACHE_PATH, prompt_template_optimize, prompt_template_vicuna

# ---- CONFIG ----
optimize_model_path = 'THUDM/BPO'
infer_res_model_path = 'lmsys/vicuna-7b-v1.3'
device = 'cuda:0'
input_jsonl = "testset/vicuna_eval.jsonl"
output_jsonl = "responses_with_uncertainty.jsonl"

# Số lượng paraphrase / sampling
M = 10

# ---- LOAD MODEL ----
optimize_model = AutoModelForCausalLM.from_pretrained(
    optimize_model_path, cache_dir=MODEL_CACHE_PATH
).half().eval().to(device)
infer_res_model = AutoModelForCausalLM.from_pretrained(
    infer_res_model_path, cache_dir=MODEL_CACHE_PATH
).half().eval().to(device)

optimize_tokenizer = AutoTokenizer.from_pretrained(
    optimize_model_path, cache_dir=MODEL_CACHE_PATH, use_fast=False
)
infer_tokenizer = AutoTokenizer.from_pretrained(
    infer_res_model_path, cache_dir=MODEL_CACHE_PATH, legacy=False
)

optimize_model.config.return_dict = True

# SBERT để tính similarity
sbert_model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2', device=device, cache_folder=MODEL_CACHE_PATH
)

# ---- READ INPUT ----
data = []
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# ---- FUNCTION: GENERATE RESPONSE ----
def generate_response(model, tokenizer, prompt, max_new_tokens=1024, temperature=0.6, top_p=0.9, split_token=None):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        num_beams=1
    )
    decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    
    # Tách response theo split_token nếu có
    if split_token and split_token in decoded:
        resp = decoded.split(split_token)[-1].strip()
    else:
        resp = decoded
    return resp

# ---- PROCESS EACH PROMPT ----
with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for item in tqdm(data, desc="Processing"):
        original_prompt = item["text"]

        # 1. Gọi LLM để sinh response gốc
        r0 = generate_response(
            infer_res_model, infer_tokenizer,
            prompt_template_vicuna.format(original_prompt),
            split_token="ASSISTANT:",
            max_new_tokens=2048
        )

        # 2. Tạo M paraphrase prompts (dùng sampling temperature cao)
        paraphrase_prompts = [
            generate_response(
                optimize_model, optimize_tokenizer,
                prompt_template_optimize.format(original_prompt),
                temperature=1.0,
                split_token="[/INST]",
                max_new_tokens=1024
            ) for _ in range(M)
        ]

        # 3. Sinh M response cho các paraphrase
        responses = [
            generate_response(
                infer_res_model, infer_tokenizer,
                prompt_template_vicuna.format(p),
                temperature=0.7,
                split_token="ASSISTANT:",
                max_new_tokens=2048
            ) for p in paraphrase_prompts
        ]

        # 4. Tính similarity với response gốc
        r0_emb = sbert_model.encode(r0, convert_to_tensor=True)
        sim_list = []
        for r in responses:
            r_emb = sbert_model.encode(r, convert_to_tensor=True)
            sim = util.pytorch_cos_sim(r0_emb, r_emb).item()
            sim_list.append(sim)

        # 5. Tính uncertainty
        U = 1 - sum(sim_list)/len(sim_list) if sim_list else 1.0

        # 6. Lưu kết quả
        out = {
            "prompt": original_prompt,
            "response_original": r0,
            "responses_paraphrase": responses,
            "similarities": sim_list,
            "uncertainty": U
        }
        f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

print("Done! Saved to:", output_jsonl)
