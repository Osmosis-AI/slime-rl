"""Convert raw GSM8K to chat-format parquet for slime LoRA training.

Usage (run inside the pod):
    python examples/lora/prep_gsm8k.py

Produces /root/gsm8k/train.parquet and /root/gsm8k/test.parquet with columns:
  - messages: [{"role": "user", "content": <question>}]
  - label: extracted numeric answer (string after ####)
"""

import os
import re

import pandas as pd
from datasets import load_dataset


def extract_answer(answer_text: str) -> str:
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return match.group(1).strip().replace(",", "")
    return answer_text.strip()


def convert_split(ds, output_path: str):
    rows = []
    for ex in ds:
        messages = [{"role": "user", "content": ex["question"]}]
        label = extract_answer(ex["answer"])
        rows.append({"messages": messages, "label": label})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


def main():
    ds = load_dataset("openai/gsm8k", "main")
    convert_split(ds["train"], "/root/gsm8k/train.parquet")
    convert_split(ds["test"], "/root/gsm8k/test.parquet")


if __name__ == "__main__":
    main()
