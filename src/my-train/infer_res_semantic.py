import json
import torch
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH, prompt_template_vicuna

# === CONFIG ===
device = 'cuda:0'
model_name = "lmsys/vicuna-7b-v1.3"
threshold = 0.1   # bạn tự chỉnh

input_jsonl = "optimized_prompts.jsonl"
semantic_jsonl = "responses_with_semantic.jsonl"
output_jsonl = "optimized_prompts_llama2_7b_res.jsonl"

# === LOAD MODEL ===
model = AutoModelForCausalLM.from_pretrained(
    model_name, cache_dir=MODEL_CACHE_PATH
).half().eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_PATH, legacy=False)

# === LOAD DATA ===
base_data = [json.loads(l) for l in open(input_jsonl, 'r', encoding='utf-8')]
semantic_data = [json.loads(l) for l in open(semantic_jsonl, 'r', encoding='utf-8')]

# map theo "prompt" để lookup nhanh
semantic_map = {item["prompt"]: item for item in semantic_data}


# === Lấy optimized response ===
def get_optimized_response(sem_item):
    conf = sem_item["conf_score"]
    clusters = sem_item["clusters"]
    samples = sem_item["samples"]

    if conf <= threshold:
        return None  # báo hiệu -1

    # cluster lớn nhất
    largest_cluster = max(clusters, key=len)

    # random 1 câu
    idx = random.choice(largest_cluster)
    return samples[idx]


# === BẮT ĐẦU INFER ===
with torch.no_grad():
    with open(output_jsonl, 'w', encoding='utf-8') as fout:

        for item in tqdm(base_data, desc="Processing"):

            prompt = item["prompt"]
            optimized_prompt = item["optimized_prompt"]

            sem_item = semantic_map[prompt]

            optimized_res = get_optimized_response(sem_item)

            # === Nếu conf thấp → -1 cho cả res + optimized_res ===
            if optimized_res is None:
                item["optimized_res"] = "None"
                item["res"] = "None"
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                continue

            # === Ngược lại, dùng optimized_res từ semantic ===
            item["optimized_res"] = optimized_res

            # === Infer từ prompt gốc ===
            input_text = prompt_template_vicuna.format(prompt.strip())
            model_inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)

            output = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=True,
                top_p=1.0,
                temperature=0.7,
                num_beams=1,
            )

            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            item["res"] = decoded.split("ASSISTANT:")[-1].strip()

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done! Saved to:", output_jsonl)
