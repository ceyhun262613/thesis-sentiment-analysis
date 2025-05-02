# 🧪 Sentiment Analysis on Greek Twitter Data 🇬🇷

This repository contains the code, data processing, and experimental results for my master’s thesis on sentiment analysis using deep learning models applied to Greek-language Twitter data.

---

## 📚 Project Overview

The goal of this work is to apply modern natural language processing (NLP) techniques, especially transformer-based models, to classify the sentiment of tweets in Greek related to the banking sector.

We used:

* **GreekBERT** (pretrained on Greek Wikipedia and EU parliamentary documents)
* **10-fold cross-validation** for robust performance evaluation
* **Hugging Face Transformers** with **PyTorch** backend

---

## 🛠 Tools & Libraries

* [Hugging Face Transformers](https://huggingface.co/)
* [PyTorch](https://pytorch.org/)
* [TensorFlow](https://www.tensorflow.org/) (optional backend)
* Python (v3.x)

---

## 📊 Dataset

* **Source**: Greek-language tweets about banking topics
* **Size**: 23,927 entries
* **Columns**:

  * `text`: The tweet content
  * `sentiment value`: Sentiment label (`neutral`, `positive`, `negative` → converted to `0`, `1`, `2`)

We applied extensive preprocessing:

* Removed URLs, hashtags, mentions, emoticons, RT markers
* Lowercased and stripped accents
* Converted sentiment labels to numeric form

---

## 🏗 Model Architecture

We fine-tuned GreekBERT with:

* A classification head (fully connected layers)
* Input: last hidden state or pooled output
* Parameters: experimented with freezing pretrained weights to avoid overfitting

---

## 🔍 Experimental Procedure

* **10-fold cross-validation**:
  Divided data into 10 mutually exclusive folds, rotated train/test sets, and averaged metrics.

* **Tokenization**:
  Applied BERT tokenization to preserve linguistic integrity.

* **Fine-tuning**:
  Focused on adjusting only the classifier layers when using frozen base model weights.

---

## 📈 Results

| Metric    | Avg Across Folds |
| --------- | ---------------- |
| Precision | 65.06%           |
| Recall    | 65.27%           |
| F1-Score  | 65.13%           |

These consistent metrics demonstrate the model’s reliability in handling Greek sentiment analysis across diverse test subsets.

---

## 📦 Folder Structure

```
/data             → Cleaned and preprocessed datasets
/models           → Fine-tuned GreekBERT checkpoints
/scripts          → Data preprocessing and training scripts
/results          → Cross-validation results, confusion matrices, metrics
README.md         → This document
```

---

## 📌 Key Takeaways

✅ Demonstrated the adaptability of pretrained transformers to Greek-language social media data
✅ Highlighted the importance of careful preprocessing and tokenization
✅ Achieved stable performance across multiple folds using BERT-based architectures

---

## 📜 Citation

If you use this work, please cite:

```
Aimilianos Kourpas, "Sentiment Analysis on Greek Twitter Data using BERT", Master’s Thesis, University of Piraeus, 2024.
```

## Link of Thesis

If you want to read the thesis : https://dione.lib.unipi.gr/xmlui/handle/unipi/16795

---
