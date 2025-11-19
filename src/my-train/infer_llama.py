from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import torch
from config import MODEL_CACHE_PATH

device = 'cuda:0'
model_name = "lmsys/vicuna-7b-v1.3"
prompt_template = "[INST] {} [/INST]"

# Load model
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=MODEL_CACHE_PATH).half().eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

# Input & output JSONL
input_jsonl = "optimized_prompts.jsonl"
output_jsonl = "optimized_prompts_llama2_7b_res.jsonl"

# ---- READ JSONL ----
data = []
with open(input_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

# ---- INFER ----
with torch.no_grad():
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data, desc="Inferring prompts"):
            # ---- Gốc prompt ----
            input_text = prompt_template.format(item['prompt'].strip())
            model_inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            output = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=True,
                top_p=1.0,
                temperature=0.7,
                num_beams=1,
                )
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
            item['res'] = decoded[0].split('[/INST]')[1].strip() if '[/INST]' in decoded[0] else decoded[0]

            # ---- Optimized prompt ----
            input_text = prompt_template.format(item['optimized_prompt'].strip())
            model_inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)
            output = model.generate(
                **model_inputs,
                max_new_tokens=2048,
                do_sample=True,
                top_p=1.0,
                temperature=0.7,
                num_beams=1,
            )
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
            item['optimized_res'] = decoded[0].split('[/INST]')[1].strip() if '[/INST]' in decoded[0] else decoded[0]

            # ---- Ghi 1 dòng JSONL
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

print("Done! Saved to:", output_jsonl)