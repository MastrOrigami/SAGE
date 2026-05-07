
# SAGE: Semantic Ambiguity Guided Capacity Expansion for Retrieval-Augmented Generation

This project implements a staged retrieval and QA pipeline with the following core components:

- Corpus and query embedding generation
- Query token embedding cache
- Hierarchical clustering-based index construction
- Routing-based entanglement detection
- Hypernetwork-based Query-Adaptive Scorer (QAS)
- TRACE-style retrieval with CLEAN / ENTANGLED branches

This framework is designed for multi-hop retrieval, complex QA, and retrieval scenarios requiring finer-grained document ranking.

---

## Pipeline

The recommended execution order is:

```bash
python 0_regenerate_embeddings.py
python 0_token_embedding_cache.py
python 1_hierarchical_index_construction.py
python 2_entanglement_detection_train_no_split.py
python 3_hypernetwork_qas_new_no_split.py
python 4_trace_search_clean_wuzhaiyao_no_split_moreqiefen.py
```

---

For quick testing, you can run:
```bash
python SAGE_qas_fast_retrieval.py
```

## File Overview
```bash
0_config.py
```
Central configuration file for:
- dataset paths
- output directories
- embedding model paths
- LLM settings
- retrieval and evaluation parameters

Currently supported datasets include:
- nq_rear
- hotpotqa
- 2wikimultihopqa
- limit_small
- musique
- narrativeqa

For the complete set of datasets, please visit https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/main 

```bash
1_hierarchical_index_construction.py
```
Builds the hierarchical index. Main steps include:
- loading corpus and query embeddings
- clustering corpus embeddings
- building cluster_to_docs
- generating cluster summaries
- saving cluster centers, summary embeddings, and index files


```bash
2_entanglement_detection_train_no_split.py
```
Performs entanglement detection by:
- training query-to-cluster routing centers
- constructing cluster boundary samples
- computing entanglement scores from gradients
- splitting clusters into CLEAN and ENTANGLED

Outputs include routing-trained centers, entanglement scores, and split files.

```bash
3_hypernetwork_qas_new_no_split.py
```
Trains the Hypernetwork QAS on entangled clusters to improve intra-cluster document ranking.

It depends on:
- query token embedding cache
- cluster summary embeddings
- entanglement split files
- query gold-node map

Outputs include QAS checkpoints and training loss logs.

```bash
4_trace_search_clean_wuzhaiyao_no_split_moreqiefen.py
```
Runs the final TRACE-style retrieval:
- first performs cluster-level routing
- then separates selected clusters into CLEAN / ENTANGLED
- uses cosine similarity for CLEAN clusters
- uses QAS for ENTANGLED clusters
- merges candidates and reports Recall@K

```bash
SAGE_qas_fast_retrieval.py
```
A lightweight entry point for quick testing and fast evaluation under default settings.

## Installation

Python 3.10+ is recommended.

Install basic dependencies:

```bash
pip install numpy scipy scikit-learn tqdm matplotlib
pip install torch transformers sentence-transformers openai
```

## Common Environment Variables

```bash
export DATASET_NAME=nq_rear
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=<your_hf_cache>
export LLM_BASE_URL=<your_base_url>
export LLM_MODEL=gpt-4.1-mini
export LLM_API_KEY=<your_api_key>
```

## Evaluation
The current implementation mainly reports:
- Recall@2
- Recall@5
- Recall@10
- Recall@20

## Notes
- The scripts should be executed in order.
- 0_token_embedding_cache.py is required for both QAS training and TRACE retrieval.
- Cluster summary generation in 1_hierarchical_index_construction.py depends on an available LLM API.
- Key index parameters such as N_CLUSTERS and SOFT_K should remain consistent throughout the pipeline.

## Project Structure
```bash
.
├── 0_config.py
├── 0_regenerate_embeddings.py
├── 0_token_embedding_cache.py
├── 1_hierarchical_index_construction.py
├── 2_entanglement_detection_train_no_split.py
├── 3_hypernetwork_qas_new_no_split.py
├── 4_trace_search_clean_wuzhaiyao_no_split_moreqiefen.py
├── SAGE_qas_fast_retrieval.py
├── results/
├── results_RAPTOR/
└── result_loss/
```

