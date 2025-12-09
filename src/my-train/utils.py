def make_prompt_template(user_prompt: str):
    messages = []
    messages.append({
        "role": "system",
        "content": "You are a helpful and concise assistant. "
                   "Please reply in English only."
    })
    messages.append({
        "role": "user",
        "content": user_prompt
    })

    return messages
