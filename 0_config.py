import os
from urllib.parse import urlparse

BASE_DIR = os.environ.get("BASE_DIR", "")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(BASE_DIR, "results"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.environ.get("MODEL_PATH", "")
MODEL_NAME = os.path.basename(MODEL_PATH.rstrip("/")) if MODEL_PATH else "model"

DATASET_NAME = os.environ.get("DATASET_NAME", "nq_rear")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))

DATASETS = {
    "nq_rear": {
        "corpus_json": os.environ.get(
            "NQ_REAR_CORPUS_JSON",
            os.path.join(DATA_DIR, "NQ", "nq_rear_corpus.json"),
        ),
        "query_json": os.environ.get(
            "NQ_REAR_QUERY_JSON",
            os.path.join(DATA_DIR, "NQ", "nq_rear.json"),
        ),
        "corpus_emb": os.path.join(OUTPUT_DIR, f"nq_rear_corpus_embeddings_{MODEL_NAME}.npy"),
        "query_emb": os.path.join(OUTPUT_DIR, f"nq_rear_query_embeddings_{MODEL_NAME}.npy"),
    },
    "hotpotqa": {
        "corpus_json": os.environ.get(
            "HOTPOTQA_CORPUS_JSON",
            os.path.join(DATA_DIR, "hotpotqa", "hotpotqa_corpus.json"),
        ),
        "query_json": os.environ.get(
            "HOTPOTQA_QUERY_JSON",
            os.path.join(DATA_DIR, "hotpotqa", "hotpotqa.json"),
        ),
        "corpus_emb": os.path.join(OUTPUT_DIR, f"hotpotqa_corpus_embeddings_{MODEL_NAME}.npy"),
        "query_emb": os.path.join(OUTPUT_DIR, f"hotpotqa_query_embeddings_{MODEL_NAME}.npy"),
    },
    "2wikimultihopqa": {
        "corpus_json": os.environ.get(
            "TWOWIKI_CORPUS_JSON",
            os.path.join(DATA_DIR, "2wikimultihopqa", "2wikimultihopqa_corpus.json"),
        ),
        "query_json": os.environ.get(
            "TWOWIKI_QUERY_JSON",
            os.path.join(DATA_DIR, "2wikimultihopqa", "2wikimultihopqa.json"),
        ),
        "corpus_emb": os.path.join(OUTPUT_DIR, f"2wikimultihopqa_corpus_embeddings_{MODEL_NAME}.npy"),
        "query_emb": os.path.join(OUTPUT_DIR, f"2wikimultihopqa_query_embeddings_{MODEL_NAME}.npy"),
    },
}

_selected = DATASETS.get(DATASET_NAME)
if _selected is None:
    raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")

CORPUS_JSONL_PATH = _selected["corpus_json"]
QUERY_DATA_PATH = _selected["query_json"]
CORPUS_EMBEDDING_PATH = _selected["corpus_emb"]
QUERY_EMBEDDING_PATH = _selected["query_emb"]

RETRIEVAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_retrieval_results.json")
EVAL_METRICS_PATH = os.path.join(OUTPUT_DIR, f"{DATASET_NAME}_eval_metrics.json")

N_CLUSTERS = 1000
SOFT_K = 3

ENTANGLEMENT_DELTA = 0.05
ENTANGLEMENT_TAU = 0.07
ENTANGLED_RATIO = 0.10
INTRA_RATIO = 0.10
ENTANGLEMENT_THRESHOLD = None
INTRA_THRESHOLD = None
QAS_MIN_RATIO = 0.15

ROUTE_MAX_EPOCHS = 300
ROUTE_BATCH_SIZE = 256
ROUTE_LR = 0.05
ROUTE_REBOUND_TOL = 1e-4
ROUTE_MIN_EPOCHS = 10
ROUTE_OSCILLATION_FLIPS = 6

TOKEN_CACHE_MAXLEN = 128

QAS_INCLUDE_SPLIT_RATIO = 0.60
QAS_HARD_POOL_CAP = 4096
QAS_H_HN = 4096
QAS_LAYERS = 2
QAS_TAU_T = 0.07
QAS_NUM_NEG = 63
QAS_PAIR_BATCH = 8
QAS_TRAIN_EPOCHS = 60
QAS_TRAIN_LR = 1e-4
QAS_WEIGHT_DECAY = 0.0
QAS_MAX_PAIRS = 50000
QAS_EARLY_STOP = 1
QAS_REBOUND_TOL = 1e-4
QAS_MIN_EPOCHS = 5
QAS_OSCILLATION_FLIPS = 4
QAS_VAL_RATIO = 0.05
QAS_VAL_KS = [10, 20]
QAS_VAL_DOC_BATCH = 1024
QAS_VAL_PATIENCE = 5
QAS_VAL_MIN_EPOCHS = 10
QAS_VAL_EARLYSTOP_LOSS_TOL = 1e-4
QAS_SAMPLE_QUERIES = 3
QAS_TOP_PASSAGES = 10

TRACE_DEVICE = "auto"
TRACE_MAX_QUERIES = 0
TRACE_DOC_QAS_BATCH = 512
TRACE_CLUSTER_TOPK = 40
TRACE_CLUSTER_SCORE_ALPHA = 0.6
TRACE_BRANCH_OVERSAMPLE = 4
TRACE_FINAL_CAND_FACTOR = 5
TRACE_USE_QAS_FOR_SPLIT = 1
TRACE_QAS_SCORE_WEIGHT = 0.65

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
_parsed = urlparse(LLM_BASE_URL) if LLM_BASE_URL else urlparse("")
LLM_HOST = os.environ.get("LLM_HOST", _parsed.netloc)
_base_path = _parsed.path.rstrip("/")
LLM_ENDPOINT = os.environ.get(
    "LLM_ENDPOINT",
    f"{_base_path}/chat/completions" if _base_path else "/v1/chat/completions",
)
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_DEFAULT_HEADERS = {}


def build_hotpotqa_gold_docs(samples):
    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_title = set([item[0] for item in sample["supporting_facts"]])
            gold_title_and_content_list = [item for item in sample["context"] if item[0] in gold_title]
            gold_doc = [item[0] + "\n" + "".join(item[1]) for item in gold_title_and_content_list]
        elif "contexts" in sample:
            gold_doc = []
            for item in sample["contexts"]:
                if "is_supporting" in item and not item["is_supporting"]:
                    continue
                gold_doc.append(item["title"] + "\n" + item["text"])
        else:
            assert "paragraphs" in sample
            gold_paragraphs = []
            for item in sample["paragraphs"]:
                if "is_supporting" in item and item["is_supporting"] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item["title"] + "\n" + (item.get("text", item.get("paragraph_text", ""))) for item in gold_paragraphs]
        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def normalize_text_for_eval(text):
    return text.replace("\n", " ").strip().lower()


def calculate_retrieval_recall(gold_docs, retrieved_docs, k_list=(1, 2, 5, 10, 20)):
    k_list = sorted(set(k_list))
    example_eval_results = []
    pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}

    for example_gold_docs, example_retrieved_docs in zip(gold_docs, retrieved_docs):
        example_eval_result = {f"Recall@{k}": 0.0 for k in k_list}

        gold_set = set(normalize_text_for_eval(d) for d in example_gold_docs)

        max_k = max(k_list)
        top_max_k_docs = example_retrieved_docs[:max_k]
        top_max_k_normalized = [normalize_text_for_eval(d) for d in top_max_k_docs]

        for k in k_list:
            current_top_k = top_max_k_normalized[:k]
            relevant_retrieved = set(current_top_k) & gold_set

            if gold_set:
                example_eval_result[f"Recall@{k}"] = len(relevant_retrieved) / len(gold_set)
            else:
                example_eval_result[f"Recall@{k}"] = 0.0

        example_eval_results.append(example_eval_result)
        for k in k_list:
            pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

    num_examples = len(gold_docs) if gold_docs else 1
    for k in k_list:
        pooled_eval_results[f"Recall@{k}"] = round(pooled_eval_results[f"Recall@{k}"] / num_examples, 4)
    return pooled_eval_results, example_eval_results


def build_nq_gold_docs(samples):

    gold_docs = []
    for sample in samples:
        candidates = []
        if "contexts" in sample:
            candidates = sample["contexts"]
        elif "positive_ctxs" in sample:
            candidates = sample["positive_ctxs"]

        if candidates:
            gold_doc = []
            for item in candidates:
                if "is_supporting" in item and not item["is_supporting"]:
                    continue
                gold_doc.append(item["title"] + "\n" + item["text"])
        else:
            gold_paragraphs = []
            for item in sample.get("paragraphs", []):
                if "is_supporting" in item and item["is_supporting"] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item["title"] + "\n" + (item.get("text", item.get("paragraph_text", ""))) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def build_2wikimultihopqa_gold_docs(samples):

    gold_docs = []
    for sample in samples:
        if "supporting_facts" in sample:
            gold_title = set([item[0] for item in sample["supporting_facts"]])
            gold_title_and_content_list = [item for item in sample["context"] if item[0] in gold_title]
            gold_doc = [item[0] + "\n" + " ".join(item[1]) for item in gold_title_and_content_list]
        elif "contexts" in sample:
            gold_doc = []
            for item in sample["contexts"]:
                if "is_supporting" in item and not item["is_supporting"]:
                    continue
                gold_doc.append(item["title"] + "\n" + item["text"])
        else:
            gold_paragraphs = []
            for item in sample.get("paragraphs", []):
                if "is_supporting" in item and item["is_supporting"] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item["title"] + "\n" + (item.get("text", item.get("paragraph_text", ""))) for item in gold_paragraphs]
        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def build_limit_gold_docs(queries_data, qrels_path, corpus_path):

    import json

    corpus_map = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            doc_id = doc["_id"]
            title = doc.get("title", "")
            text = doc.get("text", "")
            corpus_map[doc_id] = title + "\n" + text if title else "\n" + text

    qrels_map = {}
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rel = json.loads(line)
            qid = rel["query-id"]
            cid = rel["corpus-id"]
            if rel.get("score", 0) > 0:
                if qid not in qrels_map:
                    qrels_map[qid] = []
                qrels_map[qid].append(cid)

    gold_docs = []
    for query in queries_data:
        qid = query["_id"]
        relevant_cids = qrels_map.get(qid, [])
        gold_doc_texts = []
        for cid in relevant_cids:
            if cid in corpus_map:
                gold_doc_texts.append(corpus_map[cid])
        gold_docs.append(list(set(gold_doc_texts)))

    return gold_docs


if DATASET_NAME == "hotpotqa":
    build_gold_docs = build_hotpotqa_gold_docs
elif DATASET_NAME == "nq_rear":
    build_gold_docs = build_nq_gold_docs
elif DATASET_NAME == "2wikimultihopqa":
    build_gold_docs = build_2wikimultihopqa_gold_docs
elif DATASET_NAME == "limit":
    build_gold_docs = lambda samples: build_limit_gold_docs(
        queries_data=samples,
        qrels_path=_selected["qrels_json"],
        corpus_path=_selected["corpus_json"],
    )
elif DATASET_NAME == "limit_small":
    build_gold_docs = lambda samples: build_limit_gold_docs(
        queries_data=samples,
        qrels_path=_selected["qrels_json"],
        corpus_path=_selected["corpus_json"],
    )
else:
    raise ValueError(f"Unknown DATASET_NAME: {DATASET_NAME}")

QUERY_INSTRUCTION = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "