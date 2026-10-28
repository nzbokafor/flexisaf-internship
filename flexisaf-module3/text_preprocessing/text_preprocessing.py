"""
Module 3 - Natural Language Processing: Text Preprocessing
FlexiSAF Generative AI & Data Science Internship (Intermediate pathway)

Applies 3 common text cleaning steps to a sample of short customer-support
style messages:

  1. Lower casing
  2. Removal of URLs
  3. Removal of punctuations

Each step adds a new column so the effect of every step can be compared
side by side.
"""

import re
import string
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_texts.csv"
OUT_PATH = Path(__file__).resolve().parent.parent / "evidence" / "cleaned_texts.csv"

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 220)


# ---------- Step 1: Lower casing ----------
def to_lower(text: str) -> str:
    """'DELAYED' and 'delayed' should count as the same word."""
    return text.lower()


# ---------- Step 2: Removal of URLs ----------
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")


def remove_urls(text: str) -> str:
    """Links carry no meaning for most text analysis, so drop them."""
    return URL_PATTERN.sub("", text)


# ---------- Step 3: Removal of punctuations ----------
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def remove_punctuation(text: str) -> str:
    """'delayed!!!' and 'delayed' should be the same token."""
    return text.translate(PUNCT_TABLE).strip()


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} texts from {DATA_PATH.name}")

    df["lower"] = df["text"].apply(to_lower)
    df["no_urls"] = df["lower"].apply(remove_urls)
    df["clean"] = df["no_urls"].apply(remove_punctuation)

    steps = ["text", "lower", "no_urls", "clean"]

    for idx in (0, 3):
        print(f"\n=== Step-by-step example (text_id {df.loc[idx, 'text_id']}) ===")
        for col in steps:
            print(f"{col:>8}: {df.loc[idx, col]}")

    print("\n=== Original vs cleaned (all rows) ===")
    print(df[["text_id", "text", "clean"]].to_string(index=False))

    print("\n=== Unique tokens after each step ===")
    for col in steps:
        vocab = {w for t in df[col] for w in t.split()}
        print(f"{col:>8}: {len(vocab)}")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved all intermediate columns to evidence/{OUT_PATH.name}")


if __name__ == "__main__":
    main()
