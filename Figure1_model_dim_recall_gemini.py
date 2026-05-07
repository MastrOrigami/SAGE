import os
import json
import time
import importlib

import numpy as np
import requests

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


config = importlib.import_module("0_config")

DATASET_NAME = "2wikimultihopqa"
DATASET_SPEC = getattr(config, "DATASETS", {}).get(DATASET_NAME)
if DATASET_SPEC is None:
    raise ValueError(f"Dataset configuration not found in 0_config.py: {DATASET_NAME}")

CORPUS_JSON = DATASET_SPEC["corpus_json"]
QUERY_JSON = DATASET_SPEC["query_json"]

RESULT_DIR = "/mnt/data/gy/SIGMOD/result_model_dim"
os.makedirs(RESULT_DIR, exist_ok=True)

GEMINI_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 3072]
TOPK_LIST = [2, 5, 10, 20]
BATCH_SIZE_TEXT = int(os.environ.get("ENCODE_BATCH_SIZE", "8"))
MAX_CORPUS = int(os.environ.get("MAX_CORPUS", "0"))
MAX_QUERY = int(os.environ.get("MAX_QUERY", "0"))


def _iter_progress(iterable, total=None, desc=None, leave=True):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=leave)
    if total is None or desc is None:
        return iterable
    total = int(total)
    step = max(1, total // 20)

    def _gen():
        i = 0
        for x in iterable:
            i += 1
            if i == 1 or i % step == 0 or i == total:
                print(f"[{desc}] {i}/{total}")
            yield x

    return _gen()


def _num_batches(n, bs):
    if bs <= 0:
        return 0
    return (n + bs - 1) // bs


def load_json_any(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        s = f.read().strip()
    if not s:
        return data
    if s.startswith("["):
        data = json.loads(s)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    return data


def extract_texts(items):
    out = []
    for it in items:
        t = ""
        if isinstance(it, dict):
            title = it.get("title") or ""
            txt = it.get("text") or it.get("contents") or it.get("passage") or ""
            if isinstance(txt, list):
                txt = "\n".join(str(x) for x in txt)
            t = (title + "\n" + txt).strip()
            if not t:
                t = json.dumps(it, ensure_ascii=False)
        else:
            t = str(it)
        out.append(t)
    return out


def extract_queries(items):
    out = []
    for it in items:
        if isinstance(it, dict):
            q = it.get("question") or it.get("query") or it.get("text") or ""
            q = str(q).strip()
            if not q:
                q = json.dumps(it, ensure_ascii=False)
            out.append(q)
        else:
            out.append(str(it))
    return out


def sanitize_name(name):
    return (
        str(name)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
    )


def save_embeddings(dataset_name, model_name, dim, split, arr):
    safe = sanitize_name(model_name)
    out_path = os.path.join(RESULT_DIR, f"{dataset_name}_{safe}_{split}_d{dim}.npy")
    np.save(out_path, arr.astype(np.float32))
    return out_path


def gemini_base():
    base = os.environ.get("LLM_BASE_URL", "https://api.gpt.ge/v1").rstrip("/")
    return base


def gemini_headers():
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LLM_API_KEY", "sk-nbcRVgnehYbCSR9pAcE7A30cEe5e403797F33bF2Ea47AfAc")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set, so gemini-embedding-001 cannot be called")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    extra = config.LLM_DEFAULT_HEADERS if hasattr(config, "LLM_DEFAULT_HEADERS") else {}
    headers.update(extra)
    return headers


def gemini_embeddings(texts, model="gemini-embedding-001", batch_size=BATCH_SIZE_TEXT):
    base = gemini_base()
    endpoint = "/embeddings"
    if base.endswith("/v1"):
        url = base + endpoint
    else:
        url = base + "/v1" + endpoint

    headers = gemini_headers()
    sess = requests.Session()
    sess.trust_env = False

    out = []
    for i in _iter_progress(
        range(0, len(texts), batch_size),
        total=_num_batches(len(texts), batch_size),
        desc=f"gemini encode ({model})",
        leave=True,
    ):
        batch = texts[i : i + batch_size]
        payload = {"model": model, "input": batch}
        r = sess.post(url, headers=headers, json=payload, timeout=(10, 180))
        r.raise_for_status()
        data = r.json()

        arrs = []
        for item in data.get("data") or []:
            v = item.get("embedding")
            if v is None:
                v = item.get("vec") or []
            arrs.append(np.array(v, dtype=np.float32))
        if not arrs:
            raise RuntimeError("embeddings empty")
        out.append(np.stack(arrs, axis=0))

    return np.concatenate(out, axis=0)


def slice_to_dim(embs, dim):
    if dim > embs.shape[1]:
        return None
    return embs[:, :dim].astype(np.float32)


def l2_normalize(x, axis=1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def retrieve_topk_cpu(query_embs, corpus_embs, topk, query_batch=256, corpus_block=4096):
    q = l2_normalize(query_embs.astype(np.float32))
    c = l2_normalize(corpus_embs.astype(np.float32))

    topk = int(topk)
    all_topk_idx = []
    for i in _iter_progress(
        range(0, q.shape[0], query_batch),
        total=_num_batches(int(q.shape[0]), int(query_batch)),
        desc=f"retrieve top{topk}",
        leave=True,
    ):
        qb = q[i : i + query_batch]
        best_scores = np.full((qb.shape[0], topk), -np.inf, dtype=np.float32)
        best_idx = np.full((qb.shape[0], topk), -1, dtype=np.int32)

        for j in _iter_progress(
            range(0, c.shape[0], corpus_block),
            total=_num_batches(int(c.shape[0]), int(corpus_block)),
            desc="corpus blocks",
            leave=False,
        ):
            cb = c[j : j + corpus_block]
            sims = qb @ cb.T
            k2 = topk if cb.shape[0] >= topk else int(cb.shape[0])
            idxs = np.argpartition(-sims, k2 - 1, axis=1)[:, :k2]
            vals = np.take_along_axis(sims, idxs, axis=1)
            idxs = idxs + j

            merged_scores = np.concatenate([best_scores, vals], axis=1)
            merged_idx = np.concatenate([best_idx, idxs.astype(np.int32)], axis=1)
            order = np.argsort(-merged_scores, axis=1)[:, :topk]
            best_scores = np.take_along_axis(merged_scores, order, axis=1)
            best_idx = np.take_along_axis(merged_idx, order, axis=1)

        all_topk_idx.append(best_idx)

    return np.concatenate(all_topk_idx, axis=0)


def evaluate_recall_nq(gold_docs, retrieved_docs):
    pooled, _ = config.calculate_retrieval_recall(gold_docs, retrieved_docs, k_list=tuple(TOPK_LIST))
    return pooled


def compute_retrieval_and_eval(query_embs, corpus_embs, corpus_texts, gold_docs):
    max_k = max(TOPK_LIST)
    topk_idx = retrieve_topk_cpu(query_embs, corpus_embs, max_k)
    retrieved_docs = [[corpus_texts[int(j)] for j in row.tolist()] for row in topk_idx]
    return evaluate_recall_nq(gold_docs, retrieved_docs)


def get_gold_builder(dataset_name):
    if dataset_name == "hotpotqa":
        return getattr(config, "build_hotpotqa_gold_docs")
    if dataset_name == "nq_rear":
        return getattr(config, "build_nq_gold_docs")
    if dataset_name == "2wikimultihopqa":
        return getattr(config, "build_2wikimultihopqa_gold_docs")
    if dataset_name in {"limit", "limit_small"}:
        def _limit_builder(samples):
            return config.build_limit_gold_docs(
                queries_data=samples,
                qrels_path=DATASET_SPEC["qrels_json"],
                corpus_path=DATASET_SPEC["corpus_json"],
            )

        return _limit_builder
    raise ValueError(f"Unknown dataset_name: {dataset_name}")


def main():
    start = time.time()

    corpus_items = load_json_any(CORPUS_JSON)
    corpus_texts = extract_texts(corpus_items)
    if MAX_CORPUS and len(corpus_texts) > MAX_CORPUS:
        corpus_texts = corpus_texts[:MAX_CORPUS]

    query_items = load_json_any(QUERY_JSON)
    query_texts = extract_queries(query_items)
    if MAX_QUERY and len(query_texts) > MAX_QUERY:
        query_texts = query_texts[:MAX_QUERY]
        query_items = query_items[:MAX_QUERY]

    gold_builder = get_gold_builder(DATASET_NAME)
    gold_docs = gold_builder(query_items)

    model_name = "gemini-embedding-001"
    if tqdm is not None:
        tqdm.write("model: gemini-embedding-001")

    corpus_full = gemini_embeddings(corpus_texts, model=model_name)
    query_full = gemini_embeddings(query_texts, model=model_name)

    results = []
    for d in _iter_progress(GEMINI_DIMS, total=len(GEMINI_DIMS), desc=f"dims ({model_name})", leave=True):
        corpus_d = slice_to_dim(corpus_full, d)
        query_d = slice_to_dim(query_full, d)
        if corpus_d is None or query_d is None:
            results.append(
                {
                    "model": model_name,
                    "dim": int(d),
                    "skipped": True,
                    "reason": f"model_output_dim={int(corpus_full.shape[1])} < dim={int(d)}",
                }
            )
            continue

        corpus_path = save_embeddings(DATASET_NAME, model_name, d, "corpus", corpus_d)
        query_path = save_embeddings(DATASET_NAME, model_name, d, "query", query_d)

        metrics = compute_retrieval_and_eval(query_d, corpus_d, corpus_texts, gold_docs)
        results.append(
            {
                "model": model_name,
                "dim": int(d),
                "emb_paths": {"corpus": corpus_path, "query": query_path},
                "metrics": metrics,
            }
        )

    out_path = os.path.join(RESULT_DIR, f"{DATASET_NAME}_model_dim_recall_gemini.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": DATASET_NAME,
                "corpus_path": CORPUS_JSON,
                "query_path": QUERY_JSON,
                "corpus_count": int(len(corpus_texts)),
                "query_count": int(len(query_texts)),
                "topks": TOPK_LIST,
                "dims": GEMINI_DIMS,
                "results": results,
                "time_s": float(time.time() - start),
                "result_dir": RESULT_DIR,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(out_path)


if __name__ == "__main__":
    main()