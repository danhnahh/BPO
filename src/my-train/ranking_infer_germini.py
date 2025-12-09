import re
from google import genai
from tqdm import tqdm
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from config import MODEL_CACHE_PATH

device = 'cuda:0'
model_name = "lmsys/vicuna-7b-v1.3"

# ==== LOAD MODEL ====
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=MODEL_CACHE_PATH
).half().eval().to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=MODEL_CACHE_PATH, legacy=False)

API_KEY = "api_key"
client = genai.Client(api_key=API_KEY)

# ==== Đọc prompt template ====
with open("ranking_prompt.txt", "r", encoding="utf-8") as f:
    raw_prompt = f.read()

def fill_prompt(instruction, output_1, output_2):
    prompt = raw_prompt.replace('""{instruction}""', instruction)
    prompt = prompt.replace('""{output_1}""', output_1)
    prompt = prompt.replace('""{output_2}""', output_2)
    return prompt

def run_vicuna_with_gemini(prompt):
    """
    prompt: text đã fill template {instruction}, {output_1}, {output_2}
    
    Trả về final output của Vicuna, dựa trên model rank thấp hơn do Gemini xác định
    """
    # ==== 1. Gọi Gemini để xác định model rank thấp hơn ====
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    decoded_gemini = response.text.strip()

    # ==== 2. Tạo follow-up prompt cho Vicuna ====
    followup_prompt = decoded_gemini + "\nSo the lower rank model is \\boxed{Model "

    # ==== 3. Infer tiếp với Vicuna ====
    model_inputs2 = tokenizer(followup_prompt, return_tensors="pt", truncation=True, max_length=5000).to(device)
    output2 = model.generate(
        **model_inputs2,
        max_new_tokens=50,  # đủ để điền tên model hoặc giải thích
        do_sample=False
    )
    decoded2 = tokenizer.decode(output2[0], skip_special_tokens=True)

    # ==== 4. Kết hợp output ====
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
input_jsonl = "optimized_prompts_llama2_7b_res.jsonl"
output_jsonl = "lose_pairwise_results.jsonl"

rows = []
with open(input_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

total_0 = 0
total_1 = 0
total_2 = 0

# ==== RUN ====
with open(output_jsonl, "w", encoding="utf-8") as f_out:

    for item in tqdm(rows, desc="Ranking pairs"):

        instruction = item["prompt"]
        output_1 = item["res"]
        output_2 = item["optimized_res"]

        if output_1 == output_2:
            winner = 2
        else:
            prompt = fill_prompt(
                instruction=instruction,
                output_1=output_1,
                output_2=output_2
            )

            response = run_vicuna_with_gemini(prompt)
            winner = extract_winner(response)

        if winner == 0:
            total_0 += 1
            item["winner"] = winner
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif winner == 1:
            total_1 += 1
        elif winner == 2:
            total_2 += 1
        else:
            print("Cannot parse winner:", response)

    # ==== WRITE SUMMARY ====
    summary = {
        "optimized_win": total_0,
        "original_win": total_1,
        "draw": total_2
    }
    f_out.write(json.dumps(summary, ensure_ascii=False) + "\n")

    total_all = total_0 + total_1 + total_2
    summary_percent = {
        "optimized_win_percent": total_0 / total_all * 100,
        "original_win_percent": total_1 / total_all * 100,
        "draw_percent": total_2 / total_all * 100
    }
    f_out.write(json.dumps(summary_percent, ensure_ascii=False) + "\n")

print("DONE! Saved to:", output_jsonl)
print(f"Optimized win    : {total_0}")
print(f"Original win     : {total_1}")
print(f"Draw (both = -1) : {total_2}")
print(f"Total            : {total_all}")
