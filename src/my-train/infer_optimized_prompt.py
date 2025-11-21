import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH
from tqdm import tqdm  # <-- import tqdm

model_path = 'THUDM/BPO'
prompt_template = "[INST] You are an expert prompt engineer. Please help me improve this prompt to get a more helpful and harmless response:\n{} [/INST]"

input_jsonl = "testset/vicuna_eval.jsonl"
output_jsonl = "optimized_prompts.jsonl"

device = 'cuda:0'

# Load model & tokenizer
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=MODEL_CACHE_PATH).half().eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=MODEL_CACHE_PATH, use_fast=False)
model.config.return_dict = True

# Nếu pad_token chưa set, set = eos_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ---- READ INPUT ----
data = []
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

# ---- INFER TỪNG PROMPT VỚI TQDM ----
with open(output_jsonl, "w", encoding="utf-8") as f_out:
    for item in tqdm(data, desc="Inferring prompts"):
        text = item["text"]
        prompt = prompt_template.format(text)
        
        # Tokenize & move to device
        model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        # Generate
        output = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
            num_beams=1,
        )

        # Decode
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        resp = decoded[0].split('[/INST]')[1].strip() if '[/INST]' in decoded[0] else decoded[0]

        # Save
        out = {
            "prompt": text,
            "optimized_prompt": resp
        }
        f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

print("Done! Saved to:", output_jsonl)
