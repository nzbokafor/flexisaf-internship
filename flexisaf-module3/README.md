# Module 3 — Natural Language Processing: Text Preprocessing

FlexiSAF Generative AI & Data Science Internship, Intermediate pathway.

The task was to complete any 3 of the common text cleaning steps listed in the curriculum. I did these three:

1. **Lower casing** — so "DELAYED" and "delayed" count as the same word.
2. **Removal of URLs** — links like `https://t.co/abc123` add no meaning.
3. **Removal of punctuations** — so "hours!!!" becomes "hours".

## Data

`data/sample_texts.csv` holds 20 short customer-support style messages that I wrote for this exercise. They include the kinds of mess found in real social-media text (capital letters, repeated punctuation, links, hashtags, mentions and emoticons), so each cleaning step has something to do. The script works the same on any CSV with a `text` column, such as the Kaggle sample data.

## How it works

`text_preprocessing/text_preprocessing.py` applies each step in order and stores the result in a new column (`lower`, `no_urls`, `clean`), so the effect of every step is visible.

Example (text 1):

| Step | Text |
| --- | --- |
| Original | @AirlineHelp My flight was DELAYED for 3 hours!!! Nobody told us anything... https://t.co/abc123 |
| Lower case | @airlinehelp my flight was delayed for 3 hours!!! nobody told us anything... https://t.co/abc123 |
| No URLs | @airlinehelp my flight was delayed for 3 hours!!! nobody told us anything... |
| No punctuation | airlinehelp my flight was delayed for 3 hours nobody told us anything |

Unique tokens across all 20 texts dropped at every step, which is the point of cleaning: fewer, more consistent tokens for a model to learn from.

| Step | Unique tokens |
| --- | --- |
| Original | 184 |
| Lower case | 177 |
| No URLs | 174 |
| No punctuation | 167 |

Note: removing punctuation also removes the `@` and `#` from mentions and hashtags (`@AirlineHelp` becomes `airlinehelp`), and turns "I've" into "ive". That's acceptable for this exercise, but a real project might handle mentions and contractions separately.

## Run it

```bash
pip install -r requirements.txt
cd text_preprocessing
python text_preprocessing.py
```

Full output is in `evidence/text_preprocessing_output.txt`, and every intermediate column is saved to `evidence/cleaned_texts.csv`.

## Folder layout

```
flexisaf-module3/
├── README.md
├── requirements.txt
├── data/
│   └── sample_texts.csv
├── text_preprocessing/
│   └── text_preprocessing.py
└── evidence/
    ├── text_preprocessing_output.txt
    └── cleaned_texts.csv
```
