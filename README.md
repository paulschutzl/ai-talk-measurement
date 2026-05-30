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

## How the extractor works

This script is designed for Seeking Alpha earnings call transcripts.

Seeking Alpha transcripts usually contain a structured header with sections such as:
- Company Participants
- Conference Call Participants
- Operator / Presentation / Q&A

The parser uses these section labels to:
1. identify the company speaker list,
2. separate company speakers from analyst speakers,
3. locate the beginning of the call body,
4. extract only the executive speech for downstream chunking.

The logic is intentionally transcript-format specific, but the section labels and keyword rules can be edited for other transcript sources.

## Model files

The trained model files are stored in Google Drive because they are too large for GitHub.

Download the files here:
- config.json
- tokenizer.json
- tokenizer_config.json
- model.safetensors

## How to use the model in code

1. Download the files.
2. Put them in one local folder, for example `./models/final_model/`.
3. Load the model with Hugging Face Transformers.

