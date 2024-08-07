
# Knowledge Editing on Black-box Large Language Models</h2>

<p>
📃 <a href="https://arxiv.org/pdf/2402.08631.pdf">ArXiv Paper</a>
  •
🤗 <a href="">HuggingFace Model</a> 
  •
📚 <a href="">Dataset</a>
</p>

**Authors**: Xiaoshuai Song, Zhengyang Wang, Keqing He, Guanting Dong, Jinxu Zhao and Weiran Xu

⭐ **We will release the datasets, models, templates, and codes within a month. Thanks for your attention!**

## Introduction
Knowledge editing (KE) aims to efficiently and precisely modify the behavior of large language models (LLMs) to update specific knowledge without negatively influencing other knowledge. Current research primarily focuses on white-box LLMs editing, overlooking an important scenario: black-box LLMs editing, where LLMs are accessed through interfaces and only textual output is available. To address the limitations of existing evaluations that are not inapplicable to black-box LLM editing and lack comprehensiveness, we propose a multiperspective evaluation framework, incorporating the assessment of style retention for the first time. To tackle privacy leaks of editing data and style over-editing in current methods, we introduce a novel postEdit framework, resolving privacy concerns through downstream post-processing and maintaining textual style consistency via fine-grained editing to original responses. Experiments and analysis on two benchmarks demonstrate that postEdit outperforms all baselines and achieves strong generalization, especially with huge improvements on style retention (average +20.82% ↑).

Our dataset and the models fine-tuned on it will be released soon!

## Citation
Please kindly cite our paper if it helps your research:
```BibTex
@article{song2024knowledge,
  title={Knowledge Editing on Black-box Large Language Models},
  author={Song, Xiaoshuai and Wang, Zhengyang and He, Keqing and Dong, Guanting and Zhao, Jinxu and Xu, Weiran},
  journal={arXiv preprint arXiv:2402.08631},
  year={2024}
}
```
