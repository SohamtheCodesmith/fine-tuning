# 🧠 Fine-Tuning Small Language Models with LoRA

![Conference](https://img.shields.io/badge/Presented%20at-RAFET--2026-blue)

This is a **minimal proof-of-concept (MVP)** project that demonstrates how to fine-tune a small language model using **LoRA (Low-Rank Adaptation)**.
The goal is to experiment with making LLMs adopt different tones, such as **politeness**, **snark**, or even **medieval-style dialogue**, using simple datasets and local fine-tuning.

---

## 🏋️‍♂️ Conference Presentation / Publication

**Conference Presentation:**
Presented at the **2nd International Conference on Recent Advancements in the Field of Engineering and Technology (RAFET-2026)**, organised by **Dr. Sudhir Chandra Sur Institute of Technology and Sports Complex**.

This project demonstrates how instruction-following behaviour can be adapted using **LoRA (Low-Rank Adaptation)** for parameter-efficient fine-tuning of small language models on consumer hardware.

📄 Paper: *Improving Instruction-Following in Small Language Models Using LoRA-Based Parameter Efficient Fine-Tuning*  
📊 Slides: Available in `/rafet_2026`

---

## 🚀 Overview

This repo contains:

* **`training.py`** — the core script for fine-tuning the model using LoRA/PEFT
* **`testing.py`** — a lightweight test script to generate responses from your fine-tuned model
* **`polite_dataset.csv`** — a small starter dataset for experimenting with tone adaptation

At the moment, this is an **MVP (Minimum Viable Project)** meant to validate that the fine-tuning pipeline works end-to-end.
Future updates will include:

* Larger and more diverse datasets
* Additional tone/style modules (e.g. *snarky*, *medieval dialogue*)
* Model performance comparisons and evaluation notebooks
* Optional UI/Gradio interface for demoing results

---

## 🧩 How It Works

The project uses:

* **LoRA / QLoRA** for parameter-efficient fine-tuning
* **Hugging Face Transformers + PEFT** for model integration
* **PyTorch** for training backend

You can easily swap out the dataset to make the model adopt new tones or personas.

---

## 🧰 Setup

```bash
# Clone the repo
git clone https://github.com/SohamtheCodesmith/FineTuner.git
cd FineTuner

# Create a virtual environment
python -m venv finetuning
source finetuning/bin/activate  # (Linux/macOS)
finetuning\Scripts\activate     # (Windows)

# Install dependencies
pip install -r requirements.txt
```

---

## ⚙️ Training

Replace the contents of `polite_dataset.csv` with your own data in the format:

```csv
instruction,input,output
Rewrite the sentence politely,shut up,please be quiet for a moment
Rewrite the sentence politely,give me that book,may I kindly request that book
```

Then run:

```bash
python training.py
```

This will train and save LoRA weights to the `polite-lora/` folder.

---

## 🧪 Testing

Once training is done, test your model with:

```bash
python testing.py
```

Be sure to head into testing.py and change the prompt

---

## 🛠️ Requirements

List of dependencies (auto-generated in `requirements.txt`):

```
torch
transformers
datasets
peft
trl
accelerate
pandas
```

---

## 🧠 Notes

* This is a **learning-oriented project**, not a production-ready implementation.
* Expect quirky or “creative” outputs when using small datasets — the goal is to observe how the model’s tone shifts with fine-tuning.
* Larger, cleaner datasets = more consistent results.

---

## 🔮 Future Plans

Planned improvements:

* 🗣️ **Snarky tone dataset**
* 🏰 **Medieval dialogue dataset**
* 📊 Model evaluation metrics
* 🖥️ Web demo using Gradio or Streamlit

---

## 🤝 Contributing

Contributions are welcome — especially additional tone/style datasets or evaluation scripts!
Feel free to fork, open issues, or suggest ideas.

---

## 📜 License

MIT License — free to use, modify, and share.