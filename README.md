# Zero-Shot to Full-Resource: Cross-lingual Transfer Strategies for Aspect-Based Sentiment Analysis

This repository accompanies the paper  
**“Zero-Shot to Full-Resource: Cross-lingual Transfer Strategies for Aspect-Based Sentiment Analysis”**,  
which presents a comprehensive cross-lingual study of recent ABSA methods across eight languages and four subtasks and is under review for publication at LREC 2026.

---

## 📘 Overview

Aspect-Based Sentiment Analysis (ABSA) aims to extract fine-grained opinions by identifying sentiments toward specific aspects within text.  
Despite major advances in transfer learning and large language models (LLMs), most research has remains **English-focused**, and **cross-lingual generalization** of ABSA methods is still work in progress.

This work systematically evaluates **five representative ABSA paradigms** — covering both traditional and modern LLM-based approaches — under **multilingual**, **cross-lingual**, and **zero-resource** conditions.

---

## 🧩 Key Contributions

- **Comprehensive Multilingual Benchmarking:**  
  Evaluation of state-of-the-art ABSA methods across **eight languages** (English, French, Spanish, Dutch, Russian, Turkish, German, Czech) and **four subtasks**:  
  - Aspect Category Detection (ACD)  
  - Aspect Category Sentiment Analysis (ACSA)  
  - Targeted Aspect Sentiment Detection (TASD)  
  - Aspect Sentiment Quad Prediction (ASQP)

- **Unified Experimental Framework:**  
  Comparison of three major modeling paradigms:
  1. **Encoder-only** (BERT-CLF, Hier-GCN)  
  2. **Sequence-to-Sequence** (T5, DLO)  
  3. **Decoder-only LLMs** (Gemma 3, LLaMA 3.1) for prompting and instruction tuning

- **Resource-Level Analysis:**  
  Experiments conducted under three resource settings:
  - **Zero-Resource:** no training data or language-specific models  
  - **Data-Only:** training data without language-specific models  
  - **Full-Resource:** training data and language-specific models

- **Cross-Lingual Adaptation Strategies:**  
  Evaluation of multilingual transfer through:
  - Machine translation  
  - Code-switching  
  - Balanced multilingual training across source languages

- **New German ABSA Datasets:**  
  - **GERestaurant:** aligned for the SemEval schema (Aspect Categories)
  - **GERest:** first **German ASQP dataset** for structured opinion extraction, fully aligned with the Rest16 schema

- **Empirical Insights:**  
  - Instruction-tuned LLMs generally achieve the highest scores, especially when it comes to more complex tasks (TASD, ASQP).  
  - Encoder-based models remain competitive for simpler classification tasks.  
  - Code-switching provides consistent improvements in zero-resource settings.  
  - Multilingual models remain competitive, yet fall behind language-specific models.

---

## 📊 GERest Dataset

The **GERest** dataset extends the TASD corpus *GERestaurant* \[[Hellwig et al., 2024](https://aclanthology.org/2024.konvens-main.14/)\]  
and mirrors the structure of the English **ASQP-Rest16** benchmark \[[Zhang et al., 2021](https://aclanthology.org/2021.emnlp-main.726/)\].  
It provides German training, validation, and test splits for **aspect sentiment quad prediction**.

**Dataset composition:**
- **Training:** 1,264 examples  
- **Validation:** 316 examples  
- **Test:** 544 examples  
- **Aspect categories:** 13
- **Polarity types:** Positive, Negative, Neutral  
- **Reference types:** Explicit, Implicit  

| Split | #Explicit | #Implicit | #Total |
|:------|-----------:|-----------:|-------:|
| Train | 1,483 | 609 | 2,092 |
| Dev | 253 | 65 | 318 |
| Test | 616 | 264 | 880 |

---

## 🧠 Annotation Process

GERest was created following the **ASQP annotation guidelines** by \[[Zhang et al., 2021](https://aclanthology.org/2021.acl-short.64/)\] and \[[Wan et al., 2020](https://ojs.aaai.org/index.php/AAAI/article/view/6447)\].

1. **Annotator A** (B.Sc. student) performed the initial labeling.  
2. **Annotator B** (Ph.D. student with ABSA experience) reviewed and refined all examples.  
3. Out of 2,124 sentences, 184 received suggested label changes; 179 (97%) were adopted.  
4. Disagreements (n = 5) were resolved jointly.

This two-stage review ensured **high annotation consistency** and alignment with multilingual ABSA standards.

---

## ⚙️ Methods Overview

| Paradigm | Method | Architecture | Notes |
|:----------|:--------|:--------------|:-------|
| Encoder-only | **BERT-CLF** | Multilingual + monolingual BERT | Multi-label classification |
|  | **Hier-GCN** | BERT + GCN | Structural aspect-sentiment modeling |
| Seq2Seq | **DLO** | T5/mT5 | Dynamic label ordering |
| Decoder-only | **Few-Shot Prompting** | Gemma 3 27B | In-context learning |
|  | **Instruction Tuning** | LLaMA 3.1 8B (QLoRA) | Fine-tuned via low-rank adaptation |

Language-specific pretrained models were used **only when publicly available** for a given language–architecture combination.  
Otherwise, the corresponding **multilingual model** (e.g., mBERT, mT5) was retained.

---

## 📝 Citation

If you use this repository or the GERest dataset, please cite:

```bibtex
@inproceedings{TBD,
  title={Zero-Shot to Full-Resource: Cross-lingual Transfer Strategies for Aspect-Based Sentiment Analysis},
  author={TBD},
  year={2026},
  booktitle={TBD},
}
