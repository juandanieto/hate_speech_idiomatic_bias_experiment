# Idiomatic Bias in Hate Speech Detection with Figurative Language

This repository contains the code and analysis for a multilingual hate speech detection experiment designed to investigate whether language models exhibit systematic idiomatic bias — that is, whether their ability to detect figurative hate speech degrades significantly outside of English.

---

## Research Hypothesis

Language models trained or fine-tuned primarily on English data systematically fail to detect figurative hate speech in Spanish and Portuguese, demonstrating idiomatic bias tied to language-specific expressions.

---

## What Was Done

### Datasets
Three native-language datasets were used — no machine translation was applied at any stage, ensuring that any observed model degradation reflects genuine idiomatic bias rather than translation artifacts:

| Language | Dataset | Source | Size |
|---|---|---|---|
| English | Davidson et al. (2017) | Twitter | 24,783 tweets |
| Spanish | HaterNet — Pereira-Kohatsu et al. (2019) | Twitter | 6,000 tweets |
| Portuguese | HateBR — Vargas et al. (2022) | Instagram | 7,000 comments |

### Models Evaluated

**Section A — Fine-tuned transformer models:**
- `DeHateBERT` — BERT-based model fine-tuned on hate speech data
- `Twitter-RoBERTa` — RoBERTa fine-tuned on Twitter corpora
- `XLM-R Multilingual` — Multilingual transformer evaluated cross-lingually

**Section B — Large language models via prompting:**
- `LLaMA-3.1-8B` (via Groq API) — evaluated in two settings:
  - **Zero-shot**: no examples provided in the prompt
  - **Few-shot**: representative examples included in the prompt

### Methodology
1. Each model was evaluated independently on the EN, ES, and PT test sets
2. Performance was measured using F1-Hate (hate speech detection) and F1-Macro (overall classification)
3. An idiomatic gap was computed for each model as the drop in F1-Hate from English to each non-English language
4. Results were visualized through grouped bar charts and a gap analysis panel (`full_results_interpretation.html`)

---

## Repository Structure

```
├── hate_speech_idiomatic_bias_experiment.ipynb   # Full experiment notebook
├── full_results_interpretation.html              # Interactive results dashboard
└── README.md
```

---

## Requirements

```
transformers
torch
scikit-learn
groq
pandas
matplotlib
seaborn
tqdm
```

Install with:
```bash
pip install transformers torch scikit-learn groq pandas matplotlib seaborn tqdm
```

> The notebook was designed to run on **Google Colab with a T4 GPU** (`Runtime → Change runtime type → T4 GPU`).  
> A valid **Groq API key** is required for Section B (LLaMA inference). Set it as a Colab Secret under the name `GROQ_API_KEY`.

---

## References

- Davidson, T. et al. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language.* ICWSM.
- Pereira-Kohatsu, J.C. et al. (2019). *Detecting and Monitoring Hate Speech in Twitter.* Sensors. [Zenodo CC-BY 4.0]
- Vargas, F. et al. (2022). *HateBR: Large Expert Annotated Corpus of Brazilian Instagram Comments for Abusive Language Detection.* LREC. [GitHub CC-BY 4.0]
