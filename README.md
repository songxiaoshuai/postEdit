
# Knowledge Editing on Black-box Large Language Models</h2>

<p>
📃 <a href="https://arxiv.org/pdf/2402.08631.pdf">ArXiv Paper</a>
</p>

**Authors**: Xiaoshuai Song, Zhengyang Wang, Keqing He, Guanting Dong, Jinxu Zhao and Weiran Xu

## Introduction
Knowledge editing (KE) aims to efficiently and precisely modify the behavior of large language models (LLMs) to update specific knowledge without negatively influencing other knowledge. Current research primarily focuses on white-box LLMs editing, overlooking an important scenario: black-box LLMs editing, where LLMs are accessed through interfaces and only textual output is available. To address the limitations of existing evaluations that are not inapplicable to black-box LLM editing and lack comprehensiveness, we propose a multiperspective evaluation framework, incorporating the assessment of style retention for the first time. To tackle privacy leaks of editing data and style over-editing in current methods, we introduce a novel postEdit framework, resolving privacy concerns through downstream post-processing and maintaining textual style consistency via fine-grained editing to original responses. Experiments and analysis on two benchmarks demonstrate that postEdit outperforms all baselines and achieves strong generalization, especially with huge improvements on style retention (average +20.82% ↑).

## Quick Start

### 0. Install dependencies

```bash
pip install torch transformers peft sentence-transformers rouge-score scipy scikit-learn
```

### 1. Fine-tune with LoRA

```bash
bash finetune_lora.sh
```

### 2. Generate Edited Responses

```bash
bash generate_lora.sh
```

### 3. Evaluate Results

```bash
python eval/eval.py
```

## Project Structure

```
postEdit/
├── finetune.py              # LoRA fine-tuning script
├── generate_multi_edit.py    # Knowledge editing generation
├── finetune_lora.sh         # Training script
├── generate_lora.sh          # Generation script
├── data/
│   ├── prompter.py          # Prompt template manager
│   ├── template/            # Prompt templates (editor, serac, mquake)
│   └── counterfact/         # CounterFact dataset
│   └── zsre/                # ZSRE dataset
├── eval/
│   ├── eval.py              # Main evaluation script
│   ├── lexical_edit.py       # Lexical edit evaluation
│   ├── semantic_edit.py      # Semantic edit evaluation (NLI)
│   ├── lexical_retention.py  # Lexical retention (ROUGE)
│   └── semantic_retention.py # Semantic retention (SBert)
└── utils/
    ├── edit_memory.py        # Edit memory system
    ├── call_llm.py          # LLM API utilities
    └── util.py              # General utilities
```

## Citation
Please kindly cite our paper if it helps your research:
```BibTex
@article{song2024knowledge,
  title={Knowledge editing on black-box large language models},
  author={Song, Xiaoshuai and Wang, Zhengyang and He, Keqing and Dong, Guanting and Mou, Yutao and Zhao, Jinxu and Xu, Weiran},
  journal={arXiv preprint arXiv:2402.08631},
  year={2024}
}
```
```BibTex
@inproceedings{song2025assessing,
  title={Assessing and post-processing black box large language models for knowledge editing},
  author={Song, Xiaoshuai and Wang, Zhengyang and He, Keqing and Dong, Guanting and Mou, Yutao and Zhao, Jinxu and Xu, Weiran},
  booktitle={Proceedings of the ACM on Web Conference 2025},
  pages={1716--1732},
  year={2025}
}
```
