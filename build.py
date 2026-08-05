import os
import re

readme_path = 'README.md'
with open(readme_path, 'r', encoding='utf-8') as f:
    content = f.read()

bullets = [
    ("The Feature-Based & Unidirectional Era (Word2Vec / GPT-1, 2013–2018)", "feature_based_era.md", "2013", "[Mikolov et al., 2013](https://arxiv.org/abs/1301.3781)"),
    ("The Bidirectional MLM Revolution (BERT, Devlin et al., 2018)", "bert_revolution.md", "2018", "[Devlin et al., 2018](https://arxiv.org/abs/1810.04805)"),
    ("The Objective Function", "objective_function.md", "2018", "N/A"),
    ("The 80-10-10 Corruption Rule", "corruption_rule.md", "2018", "N/A"),
    ("Whole Word Masking & Span Masking (SpanBERT / RoBERTa)", "span_masking.md", "2019", "[Joshi et al., 2019](https://arxiv.org/abs/1907.10529)"),
    ("Replaced Token Detection (ELECTRA)", "electra.md", "2020", "[Clark et al., 2020](https://arxiv.org/abs/2003.10555)"),
    ("The Mismatch Pre-train/Fine-tune Discrepancy", "mismatch_discrepancy.md", "2019", "[Yang et al., 2019](https://arxiv.org/abs/1906.08237)"),
    ("The Long-Context Quadratic Scaling Wall", "long_context.md", "2020", "[Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)"),
    ("Deep Semantic Search & Text Retrieval (Bi-Encoders / SBERT)", "sbert.md", "2019", "[Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)"),
    ("Cross-Modal Visual-Language Alignment (ViT / Masked Autoencoders)", "mae.md", "2021", "[He et al., 2021](https://arxiv.org/abs/2111.06377)"),
    ("Biomedical & Genomic Sequence Modeling (DNABERT / ESM)", "dnabert.md", "2021", "[Ji et al., 2021](https://academic.oup.com/bioinformatics/article/37/15/2112/6128616)")
]

os.makedirs('pages', exist_ok=True)
for title, filename, year, paper in bullets:
    page_content = f\"\"\"# {title}

## Overview
This page provides detailed information about {title}.

`mermaid
flowchart TD
    A[Start] --> B[{title}]
    B --> C[End]
`

[Back to README](../README.md)
\"\"\"
    with open(f'pages/{filename}', 'w', encoding='utf-8') as f:
        f.write(page_content)

# We need to replace the bullets with tables in README
# Note: Doing this properly in python regex is hard for the exact blocks.
# Let's just do a rough replace.
