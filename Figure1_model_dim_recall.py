import os
# English comment: import standard-library and third-party libraries
import json
import time
import importlib
import numpy as np
import requests
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# English comment: load the unified configuration, including data paths and output directories
config = importlib.import_module("0_config")


# English comment: this script is fixed to use the NQ dataset (nq_rear), along with paths and evaluation logic from 0_config.py
DATASET_NAME = "nq_rear"
DATASET_SPEC = getattr(config, "DATASETS", {}).get(DATASET_NAME)
if DATASET_SPEC is None:
    raise ValueError(f"Dataset configuration not found in 0_config.py: {DATASET_NAME}")

# English comment: corpus and query paths for NQ
CORPUS_JSON = DATASET_SPEC["corpus_json"]
QUERY_JSON = DATASET_SPEC["query_json"]

# English comment: result and embedding output directory, uniformly placed under result_model_dim as required
RESULT_DIR = "/mnt/data/gy/SIGMOD/result_model_dim"
os.makedirs(RESULT_DIR, exist_ok=True)

# English comment: three locally available embedding models and their paths
LOCAL_MODELS = [
    {"name": "GTE-Qwen2-7B-Instruct", "path": "/mnt/data/gy/model/GTE-Qwen2-7B-Instruct"},
    {"name": "GritLM-7B", "path": "/mnt/data/gy/model/GritLM-7B"},
    {"name": "NV-Embed-v2", "path": "/mnt/data/gy/model/NV-Embed-v2"},
]

# English comment: dimensions tested for local models and remote Gemini
LOCAL_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
GEMINI_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 3072]

# English comment: Top-K list for recall metrics
TOPK_LIST = [2, 5, 10, 20]

# English comment: for CPU-only environments, use smaller batch sizes to reduce memory usage; these can be overridden by environment variables
BATCH_SIZE_TEXT = int(os.environ.get("ENCODE_BATCH_SIZE", "8"))
BATCH_SIZE_SIM = int(os.environ.get("QUERY_BATCH_SIZE", "256"))

# English comment: corpus block size for brute-force retrieval, used to compute similarities in blocks for exact retrieval while reducing peak memory usage
CORPUS_BLOCK_SIZE = int(os.environ.get("CORPUS_BLOCK_SIZE", "4096"))

# English comment: maximum text length for encoding; using a smaller value on CPU is recommended and can be overridden by environment variables
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "2048" if not torch.cuda.is_available() else "8192"))

# English comment: optional limits on the number of corpus/query items participating in the experiment, useful for CPU dry runs and tuning
MAX_CORPUS = int(os.environ.get("MAX_CORPUS", "0"))
MAX_QUERY = int(os.environ.get("MAX_QUERY", "0"))

# English comment: always include evaluation for gemini-embedding-001; do not skip it
SKIP_GEMINI = False
SEED = 1234


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


def get_device():
    forced = os.environ.get("FORCE_DEVICE")
    if forced:
        forced = forced.strip().lower()
        if forced in {"cpu", "cuda"}:
            return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = get_device()


def print_runtime_device():
    # English comment: print the actually selected device information, CPU or GPU, at startup
    print(f"[Device] selected: {DEVICE}")
    if DEVICE == "cuda":
        try:
            print(f"[Device] torch: {torch.__version__} | cuda: {torch.version.cuda}")
            print(f"[Device] cuda available: {torch.cuda.is_available()} | device_count: {torch.cuda.device_count()}")
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                print(f"[Device] gpu0: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"[Device] cuda query failed: {e}")


def load_json_any(path):
    # English comment: support reading both JSON and JSONL formats
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
    # English comment: extract document text, preferring title + '\n' + text
    out = []
    for it in items:
        t = ""
        if isinstance(it, dict):
            title = it.get("title") or ""
            txt = it.get("text") or it.get("contents") or it.get("passage") or ""
            t = (title + "\n" + txt).strip()
            if not t:
                t = json.dumps(it, ensure_ascii=False)
        else:
            t = str(it)
        out.append(t)
    return out


def extract_queries(items):
    # English comment: extract query text; the NQ field is usually question
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
    # English comment: convert the model name into a filename-safe format
    return (
        str(name)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
    )


def save_embeddings(dataset_name, model_name, dim, split, arr):
    # English comment: save embeddings to .npy; the filename includes dataset name, model name, dimension, and corpus/query split
    safe = sanitize_name(model_name)
    out_path = os.path.join(RESULT_DIR, f"{dataset_name}_{safe}_{split}_d{dim}.npy")
    np.save(out_path, arr.astype(np.float32))
    return out_path


def set_offline():
    # English comment: enable Transformers offline mode to avoid network access
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def load_local_model(model_path):
    # English comment: load the local embedding model, trusting remote code but using local files only; disable cache
    set_offline()
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if hasattr(cfg, "text_config"):
        cfg.text_config._name_or_path = model_path
    cfg._name_or_path = model_path
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    m = AutoModel.from_pretrained(
        model_path,
        config=cfg,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        device_map=None,
    )
    if hasattr(m, "config") and hasattr(m.config, "use_cache"):
        m.config.use_cache = False
    if hasattr(m, "embedding_model") and hasattr(m.embedding_model, "config") and hasattr(m.embedding_model.config, "use_cache"):
        m.embedding_model.config.use_cache = False
    if DEVICE == "cuda":
        m = m.to("cuda")
    m.eval()
    return m


def load_local_tokenizer(model_path):
    # English comment: load the local tokenizer in offline mode, trusting remote code but using local files only
    set_offline()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    return tok


def _mean_pool(last_hidden_state, attention_mask):
    # English comment: mean pooling; aggregate token-level representations into sentence embeddings
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def encode_local(m, tokenizer, texts, instruction="", max_length=8192, batch_size=BATCH_SIZE_TEXT):
    # English comment: encoding function compatible with two kinds of models
    # 1) If the model provides encode(), call it directly, e.g., NV-Embed-v2
    # 2) Otherwise, use tokenizer + forward + mean pooling to get embeddings, e.g., Qwen2/Mistral/GritLM
    out = []
    for i in _iter_progress(
        range(0, len(texts), batch_size),
        total=_num_batches(len(texts), batch_size),
        desc="encode",
        leave=True,
    ):
        batch = texts[i : i + batch_size]
        if instruction:
            batch = [instruction + t for t in batch]

        if hasattr(m, "encode"):
            emb = m.encode(batch, instruction="", max_length=max_length)
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            out.append(emb.astype(np.float32))
            continue

        if tokenizer is None:
            raise RuntimeError("tokenizer is required when model has no encode()")

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        device = next(m.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = m(**inputs, return_dict=True)
            last_hidden = outputs.last_hidden_state
            pooled = _mean_pool(last_hidden, inputs["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
        out.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def gemini_base():
    # English comment: get the base URL for the remote API
    base = os.environ.get("LLM_BASE_URL", "https://api.gpt.ge/v1").rstrip("/")
    return base


def gemini_headers():
    # English comment: construct remote request headers, including authentication and optional extra headers
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("LLM_API_KEY", "sk-nbcRVgnehYbCSR9pAcE7A30cEe5e403797F33bF2Ea47AfAc")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    extra = config.LLM_DEFAULT_HEADERS if hasattr(config, "LLM_DEFAULT_HEADERS") else {}
    headers.update(extra)
    return headers


def gemini_embeddings(texts, model="gemini-embedding-001", batch_size=BATCH_SIZE_TEXT):
    # English comment: call the remote API to obtain embeddings in batches
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


def random_ortho_projection(in_dim, out_dim, seed=SEED):
    # English comment: random orthogonal projection matrix used to reduce vectors to the target dimension
    rng = np.random.default_rng(seed + out_dim)
    M = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    Q, _ = np.linalg.qr(M)
    return Q.astype(np.float32)


def l2_normalize(x, axis=1, eps=1e-12):
    # English comment: L2 normalization used for cosine-similarity retrieval
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def slice_to_dim(embs, dim):
    # English comment: truncate model output vectors to the specified dimension, interpreted as d-dimensional embeddings
    if dim > embs.shape[1]:
        return None
    return embs[:, :dim].astype(np.float32)


def retrieve_topk(query_embs, corpus_embs, topk):
    # English comment: exact brute-force Top-K retrieval implemented with blocking
    # - For each query, similarities are still computed against the full corpus, so results match the full-matrix computation
    # - CORPUS_BLOCK_SIZE is used to iterate through the corpus in blocks, greatly reducing peak memory usage and making CPU runs friendlier
    q = l2_normalize(query_embs.astype(np.float32))
    c = l2_normalize(corpus_embs.astype(np.float32))

    topk = int(topk)
    device = DEVICE
    q_t = torch.from_numpy(q).to(dtype=torch.float32, device=device)
    c_t_cpu = torch.from_numpy(c).to(dtype=torch.float32, device="cpu")

    all_topk_idx = []
    for i in _iter_progress(
        range(0, q_t.shape[0], BATCH_SIZE_SIM),
        total=_num_batches(int(q_t.shape[0]), int(BATCH_SIZE_SIM)),
        desc=f"retrieve top{topk}",
        leave=True,
    ):
        qb = q_t[i : i + BATCH_SIZE_SIM]

        # English comment: maintain the current top-k scores and indices for this query batch, updating them as corpus blocks are processed
        best_scores = torch.full((qb.shape[0], topk), -float("inf"), dtype=torch.float32, device=device)
        best_idx = torch.full((qb.shape[0], topk), -1, dtype=torch.int64, device=device)

        for j in _iter_progress(
            range(0, c_t_cpu.shape[0], CORPUS_BLOCK_SIZE),
            total=_num_batches(int(c_t_cpu.shape[0]), int(CORPUS_BLOCK_SIZE)),
            desc="corpus blocks",
            leave=False,
        ):
            cb_cpu = c_t_cpu[j : j + CORPUS_BLOCK_SIZE]
            cb = cb_cpu.to(device=device, non_blocking=True) if device == "cuda" else cb_cpu
            if device == "cuda":
                qb_cast = qb.to(dtype=torch.float16)
                cb_cast = cb.to(dtype=torch.float16)
                sims = (qb_cast @ cb_cast.T).to(dtype=torch.float32)
            else:
                sims = qb @ cb.T

            k2 = topk if cb.shape[0] >= topk else int(cb.shape[0])
            vals, inds = torch.topk(sims, k=k2, dim=1, largest=True, sorted=False)
            inds = inds + int(j)

            merged_scores = torch.cat([best_scores, vals], dim=1)
            merged_idx = torch.cat([best_idx, inds], dim=1)

            new_vals, new_pos = torch.topk(merged_scores, k=topk, dim=1, largest=True, sorted=True)
            new_idx = torch.gather(merged_idx, 1, new_pos)
            best_scores = new_vals
            best_idx = new_idx

        all_topk_idx.append(best_idx.detach().cpu().numpy().astype(np.int32))

    return np.concatenate(all_topk_idx, axis=0)


def evaluate_recall_nq(gold_docs, retrieved_docs):
    # English comment: use the unified recall calculation function in 0_config.py, i.e., Recall@2/5/10/20
    pooled, _ = config.calculate_retrieval_recall(gold_docs, retrieved_docs, k_list=tuple(TOPK_LIST))
    return pooled


def compute_retrieval_and_eval(query_embs, corpus_embs, corpus_texts, gold_docs):
    # English comment: perform brute-force retrieval, convert Top-K documents to text format, and compute Recall@K against gold_docs
    max_k = max(TOPK_LIST)
    topk_idx = retrieve_topk(query_embs, corpus_embs, max_k)
    retrieved_docs = []
    for row in topk_idx:
        retrieved_docs.append([corpus_texts[int(j)] for j in row.tolist()])
    metrics = evaluate_recall_nq(gold_docs, retrieved_docs)
    return metrics


def run_local_model_on_nq(model_name, model_path, dims, corpus_texts, query_texts, gold_docs):
    # English comment: run the local model workflow
    # 1) Encode corpus/query to obtain vectors at the maximum output dimension
    # 2) Truncate to each target dimension
    # 3) Save corpus/query embeddings to result_model_dim
    # 4) Run brute-force retrieval, compute Recall@2/5/10/20, and save the results
    m = load_local_model(model_path)
    tokenizer = load_local_tokenizer(model_path)

    # English comment: whether to use an instruction for queries: NV-Embed-v2 uses QUERY_INSTRUCTION from 0_config.py; others default to an empty string
    query_instruction = config.QUERY_INSTRUCTION if model_name == "NV-Embed-v2" and hasattr(config, "QUERY_INSTRUCTION") else ""

    if tqdm is not None:
        tqdm.write(f"model: {model_name}")
        tqdm.write("encoding corpus...")
    corpus_full = encode_local(m, tokenizer, corpus_texts, instruction="", max_length=MAX_LENGTH)
    if tqdm is not None:
        tqdm.write("encoding queries...")
    query_full = encode_local(m, tokenizer, query_texts, instruction=query_instruction, max_length=MAX_LENGTH)

    results = []
    for d in _iter_progress(dims, total=len(dims), desc=f"dims ({model_name})", leave=True):
        # English comment: truncate output vectors to d dimensions as required, representing the model embeddings in d-dimensional space
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

        # English comment: save embeddings; filenames include dataset name, model name, dimension, and corpus/query split
        corpus_path = save_embeddings(DATASET_NAME, model_name, d, "corpus", corpus_d)
        query_path = save_embeddings(DATASET_NAME, model_name, d, "query", query_d)

        # English comment: run brute-force retrieval and evaluate recall
        metrics = compute_retrieval_and_eval(query_d, corpus_d, corpus_texts, gold_docs)
        results.append(
            {
                "model": model_name,
                "dim": int(d),
                "emb_paths": {"corpus": corpus_path, "query": query_path},
                "metrics": metrics,
            }
        )
    return results


def run_gemini_on_nq(dims, corpus_texts, query_texts, gold_docs):
    # English comment: run the same workflow for remote gemini-embedding-001 as for local models
    model_name = "gemini-embedding-001"

    corpus_full = gemini_embeddings(corpus_texts, model=model_name)
    query_full = gemini_embeddings(query_texts, model=model_name)

    results = []
    for d in _iter_progress(dims, total=len(dims), desc=f"dims ({model_name})", leave=True):
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
    return results


def main():
    # English comment: main workflow for the NQ dataset
    # 1) Read corpus/query
    # 2) Build gold_docs using 0_config.py
    # 3) For each model, generate embeddings at multiple dimensions, save embeddings, run brute-force retrieval, and compute Recall@K
    # 4) Save all recall results to result_model_dim

    start = time.time()
    print_runtime_device()

    # English comment: read the NQ corpus and construct document text for retrieval comparison, i.e., title + '\n' + text
    corpus_items = load_json_any(CORPUS_JSON)
    corpus_texts = extract_texts(corpus_items)
    if MAX_CORPUS and len(corpus_texts) > MAX_CORPUS:
        corpus_texts = corpus_texts[:MAX_CORPUS]

    # English comment: read NQ queries and extract query text, typically from the question field
    query_items = load_json_any(QUERY_JSON)
    query_texts = extract_queries(query_items)
    if MAX_QUERY and len(query_texts) > MAX_QUERY:
        query_texts = query_texts[:MAX_QUERY]
        query_items = query_items[:MAX_QUERY]

    # English comment: build gold_docs, where each query maps to one or more correct documents in the same title + '\n' + text format
    gold_docs = config.build_gold_docs(query_items)

    all_results = []

    # English comment: evaluate local models at 32/64/.../4096 dimensions and save embeddings
    for spec in _iter_progress(LOCAL_MODELS, total=len(LOCAL_MODELS), desc="local models", leave=True):
        all_results.extend(
            run_local_model_on_nq(
                model_name=spec["name"],
                model_path=spec["path"],
                dims=LOCAL_DIMS,
                corpus_texts=corpus_texts,
                query_texts=query_texts,
                gold_docs=gold_docs,
            )
        )

    # English comment: evaluate remote Gemini at 32/64/.../3072 dimensions and save embeddings
    if not SKIP_GEMINI:
        if tqdm is not None:
            tqdm.write("model: gemini-embedding-001")
        all_results.extend(run_gemini_on_nq(GEMINI_DIMS, corpus_texts, query_texts, gold_docs))

    # English comment: save recall results for all models and dimensions
    out_path = os.path.join(RESULT_DIR, f"{DATASET_NAME}_model_dim_recall.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": DATASET_NAME,
                "corpus_path": CORPUS_JSON,
                "query_path": QUERY_JSON,
                "corpus_count": int(len(corpus_texts)),
                "query_count": int(len(query_texts)),
                "topks": TOPK_LIST,
                "local_dims": LOCAL_DIMS,
                "gemini_dims": GEMINI_DIMS,
                "results": all_results,
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