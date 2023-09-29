# Can LLMs Effectively Leverage Structural Information for Graph Learning: When and Why



We provide three main components:

- A new dataset `arxiv-2023`, whose test nodes are chosen from arXiv Computer Science (CS) papers published in 2023.
- A unified dataloader for `cora`, `pubmed`, `ogbn-arxiv`, `arxiv-2023` and `ogbn-product` as well as their raw text.
- A simple template for testing ChatGPT on these datasets. See `template.ipynb`.



## Download datasets and raw texts

We provide the dataset and raw text for `arxiv-2023`. You may need to download the dataset and raw texts for other datasets.

- `cora` and `pubmed`: download the dataset here: https://github.com/XiaoxinHe/TAPE and place the datasets at `/dataset/cora/` and `/dataset/pubmed/` respectively.

- `ogbn-arxiv` and `ogbn-product`: as you run the code, `ogb` will automatically download the dataset for you. But you need to download the raw texts yourself. For `ogbn-arxiv`, download [here](https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz) and place the file at `/dataset/ogbn_arxiv/titleabs.tsv`. For `ogbn-product`, you download [here]( https://drive.google.com/file/d/1gsabsx8KR2N9jJz16jTcA0QASXsNuKnN/view?usp=sharing) and place the folder at `/dataset/ogbn-products/Amazon-3M.raw`

  

You need to set up your OpenAI API key as `OPENAI_API_KEY` environment variable. See [here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety) for details.

Required packages include `openai`, `pytorch`, `PyG`, `ogb` etc.