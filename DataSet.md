# AI Washing Validation Dataset — 50 Labeled Earnings Call Chunks

This dataset contains 50 text chunks from S&P 500 earnings call transcripts 
(2023–2025), drawn from a 200-chunk human-validated sample used to establish 
inter-rater reliability for the AI disclosure quality classification framework 
developed in the accompanying master thesis.

Data Set: https://drive.google.com/file/d/1su5vh7b42NhjHVFYiZ5gQ5XgY1-HojSm/view?usp=sharing

## Purpose

The dataset illustrates how the rule-based scoring framework distinguishes 
substantive AI disclosure from vague AI talk at the chunk level. Each chunk 
is labeled with both the automated scorer output and a manual review applying 
all 14 codebook rules. The two sets of scores can be compared directly.

## Label Definitions

**Substantive (score ≥ 1):** The chunk contains verifiable, firm-specific 
evidence of AI deployment - language a rational investor could use to assess 
the firm's actual AI activity.

**Vague talk (score ≤ 0):** The passage references AI using language any firm 
could produce regardless of actual deployment - promotional, hedged, or abstract 
claims without independently verifiable content.

## Scoring Formula

score = 2 × (number of hard_markers) + 1 × (number of soft_markers) − 1 × (number of vague_markers)

Hard markers (+2 each) require verifiable deployment evidence (H1–H5).  
Soft markers (+1 each) provide firm-specific context (S1–S5).  
Vague markers (−1 each) indicate promotional or hedging language (V1–V4).

## Dataset Composition

| Category | Count |
|---|---|
| Substantive (human label) | 20 |
| Vague talk (human label) | 20 |
| Borderline = score 1–2 (human label mix) | 10 |

27 companies represented. Cohen's Kappa between scorer and human labels on 
the full 200-chunk sample = 0.62 (substantial agreement, Landis & Koch 1977).

## Key Findings from Manual Review

Of the 50 examples, 27 are fully correct (scorer and manual agree on all 
markers and score). 23 have score differences, but only 2 change the binary 
label and both cases where the scorer incorrectly fired a hard marker, and both 
corrected labels match the human judgment.

The most common scorer gaps: implementation verbs not in the verb dictionary 
("leverages", "uses"), soft marker S1 firing only once per chunk even when 
multiple use cases are named, and vague marker V3 missing abstract benefit 
phrases such as "drives efficiency" or "meaningful growth driver".

## Source

Earnings call transcripts collected from Seeking Alpha. All companies are 
S&P 500 constituents. Transcripts cover fiscal years 2023-2025. Only company 
executive speech is included (analyst questions excluded).
