import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import torch
from config import MODEL_CACHE_PATH
from string import Template

device = 'cuda:0'
model_name = "lmsys/vicuna-7b-v1.3"

class SafeDict(dict):
    def __missing__(self, key):
        # Nếu key không có trong dict, giữ nguyên trong template
        return "{" + key + "}"

# ==== ĐỌC PROMPT TEMPLATE TỪ FILE ====
with open("ranking_prompt.txt", "r", encoding="utf-8") as f:
    raw_prompt = f.read()

# ==== LOAD MODEL ====
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_PATH
).half().eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

input_jsonl = "optimized_prompts_llama2_7b_res.jsonl"
output_jsonl = "lose_pairwise_results.jsonl"

# Dùng Template để replace chỉ {instruction}, {output_1}, {output_2}
def fill_prompt(instruction, output_1, output_2):
    prompt = raw_prompt
    prompt = prompt.replace('""{instruction}""', instruction)
    prompt = prompt.replace('""{output_1}""', output_1)
    prompt = prompt.replace('""{output_2}""', output_2)
    return prompt

def run_vicuna(prompt):
    # 1. Infer lần 1
    model_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=5000).to(device)
    output = model.generate(
        **model_inputs,
        max_new_tokens=1024,
        do_sample=False
    )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # 2. Tách phần Assistant:
    if "Assistant:" in decoded:
        decoded = decoded.split("Assistant:")[-1].strip()
    
    # 3. Nối câu bắt buộc boxed để model infer tiếp
    followup_prompt = decoded + "\nSo the rank 1 model is \\boxed{Model "

    # 4. Infer tiếp
    model_inputs2 = tokenizer(followup_prompt, return_tensors="pt", truncation=True, max_length=5000).to(device)
    output2 = model.generate(
        **model_inputs2,
        max_new_tokens=50,  # đủ để điền tên model
        do_sample=False
    )
    decoded2 = tokenizer.decode(output2[0], skip_special_tokens=True)

    # 5. Lấy phần model sinh thêm và đóng dấu }
    final_output = followup_prompt + decoded2[len(followup_prompt):].strip()

    return final_output

def extract_winner(text):
    """
    Tìm \boxed{…} đầu tiên trong text, xem có số 1 hay 2 bên trong.
    Trả về:
        0 nếu có số 1
        1 nếu có số 2
        None nếu không tìm thấy
    """
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    if m:
        content = m.group(1)
        if '1' in content:
            return 0
        elif '2' in content:
            return 1
    return None
    
# ==== READ INPUT ====
rows = []
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

total_0 = 0
total_1 = 0

# ==== RUN ====
with open(output_jsonl, "w", encoding="utf-8") as f_out:

    for item in tqdm(rows, desc="Ranking pairs"):

        instruction = item["prompt"]
        output_1 = item["res"]
        output_2 = item["optimized_res"]

        # === APPLY TEMPLATE ===
        prompt = fill_prompt(
            instruction=instruction,
            output_1=output_1,
            output_2=output_2
        )

        result_text = run_vicuna(prompt)
        winner = extract_winner(result_text)

        # Chỉ ghi các trường hợp 0 thua 1
        if winner == 1:
            total_1 += 1
            item["winner"] = winner
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif winner == 0:
            total_0 += 1
            # Không ghi item thua
            pass
        else:
            print(result_text)

    # ==== WRITE SUMMARY ==== 
    summary = {"total_0": total_0, "total_1": total_1}
    f_out.write(json.dumps(summary, ensure_ascii=False) + "\n")
    summary_percent = {"total_0%": total_0 / (total_0 + total_1) * 100, "total_1%": total_1 / (total_0 + total_1) * 100}
    f_out.write(json.dumps(summary_percent, ensure_ascii=False) + "\n")

print("DONE! Saved to:", output_jsonl)
print("Winner 0:", total_0)
print("Winner 1:", total_1)
