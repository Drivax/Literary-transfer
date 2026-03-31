# Human Blind Evaluation Protocol

## Goal
Evaluate stylistic authenticity, semantic faithfulness, and fluency without exposing model identity.

## Setup
- Each item contains one source text and two anonymized rewrites.
- Order is randomized.
- Annotators do not know which system generated each rewrite.

## Annotation Criteria
- Style authenticity (1-5): Does this read like the target author?
- Semantic faithfulness (1-5): Is the core meaning preserved?
- Fluency and coherence (1-5): Is the text natural and readable?

## Pairwise Preference
Annotators select A, B, or Tie for overall quality.

## Agreement
Compute Krippendorff's alpha for each criterion and report tie rate.
