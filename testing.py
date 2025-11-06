from transformers import pipeline

orig = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
tuned = pipeline("text-generation", model="./polite-lora")

prompt = "Rewrite the sentence politely: Shut up!"
print("Before:\n", orig(prompt, max_new_tokens=30)[0]["generated_text"])
print("After:\n", tuned(prompt, max_new_tokens=30)[0]["generated_text"])