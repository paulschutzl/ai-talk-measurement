# Ai-Talk-Measurement
Repository containing evaluation weights, labeled validation data, and NLP methodology for measuring how substantive corporate "AI talk" is. Built on FinBERT-ESG, the model performs binary classification, assigning a substantive label to individual text chunks to evaluate AI disclosures per company

# AI Disclosure Quality in Earnings Calls

This repository accompanies the master thesis **“Measuring AI Disclosure Quality: A Transformer-Based Approach to Substantive and Vague Communication in S&P 500 Earnings Calls.”**

The project develops a transparent method to distinguish **substantive** from **vague** AI-related communication in quarterly earnings calls. It combines:

- a keyword-based chunk extraction step,
- a theory-driven rule-based labeling framework,
- human validation of a subset of chunks,
- and a fine-tuned transformer classifier for large-scale prediction.

## Project overview

The core idea is to move beyond simple keyword counting and instead measure the **quality** of AI disclosure. In the thesis, AI-related passages are first extracted from earnings call transcripts, then labeled as substantive or vague using a rule-based scorer grounded in signalling theory, voluntary disclosure theory, and impression management theory. The final classifier is a fine-tuned **FinBERT-ESG** model trained on the labeled chunks.

## What is included in this repository

This repository contains:

- **model weights** for the final trained classifier,
- **example chunks** showing the chunk format used in the project,
- **code for chunk extraction** from earnings call transcripts,
- **training code** for the classifier,
- and this documentation.

