import os
import json
import time
import importlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
import gc

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

LOCAL_MODELS = [
    {"name": "GTE-Qwen2-7B-Instruct", "path": "/mnt/data/gy/model/GTE-Qwen2-7B-Instruct"},
    {"name": "GritLM-7B", "path": "/mnt/data/gy/model/GritLM-7B"},
    {"name": "NV-Embed-v2", "path": "/mnt/data/gy/model/NV-Embed-v2"},
]

LOCAL_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 4096]
GTE_DIMS = [32, 64, 128, 256, 512, 1024, 2048, 3584]
TOPK_LIST = [2, 5, 10, 20]

_DEFAULT_ENCODE_BS = "2" if torch.cuda.is_available() else "8"
_DEFAULT_QUERY_BS = "64" if torch.cuda.is_available() else "256"
_DEFAULT_CORPUS_BLOCK = "1024" if torch.cuda.is_available() else "4096"
_DEFAULT_MAX_LENGTH = "1024" if torch.cuda.is_available() else "2048"

BATCH_SIZE_TEXT = max(1, int(os.environ.get("ENCODE_BATCH_SIZE", _DEFAULT_ENCODE_BS)))
BATCH_SIZE_SIM = max(1, int(os.environ.get("QUERY_BATCH_SIZE", _DEFAULT_QUERY_BS)))
CORPUS_BLOCK_SIZE = max(1, int(os.environ.get("CORPUS_BLOCK_SIZE", _DEFAULT_CORPUS_BLOCK)))
MAX_LENGTH = max(16, int(os.environ.get("MAX_LENGTH", _DEFAULT_MAX_LENGTH)))
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


def get_device():
    forced = os.environ.get("FORCE_DEVICE")
    if forced:
        forced = forced.strip().lower()
        if forced in {"cpu", "cuda"}:
            return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = get_device()


def print_runtime_device():
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


def set_offline():
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"


def load_local_model(model_path):
    set_offline()
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if hasattr(cfg, "text_config"):
        cfg.text_config._name_or_path = model_path
    cfg._name_or_path = model_path
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
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
    set_offline()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    return tok


def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def encode_local(m, tokenizer, texts, instruction="", max_length=8192, batch_size=BATCH_SIZE_TEXT):
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

        inputs = tokenizer(batch, padding=True, truncation=True, max_length=int(max_length), return_tensors="pt")
        device = next(m.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = m(**inputs, return_dict=True)
                    pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            else:
                outputs = m(**inputs, return_dict=True)
                pooled = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
        out.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def l2_normalize(x, axis=1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def slice_to_dim(embs, dim):
    if dim > embs.shape[1]:
        return None
    return embs[:, :dim].astype(np.float32)


def retrieve_topk(query_embs, corpus_embs, topk):
    q = l2_normalize(query_embs.astype(np.float32))
    c = l2_normalize(corpus_embs.astype(np.float32))

    topk = int(topk)
    q_t = torch.from_numpy(q).to(dtype=torch.float32, device=DEVICE)
    c_t_cpu = torch.from_numpy(c).to(dtype=torch.float32, device="cpu")

    all_topk_idx = []
    for i in _iter_progress(
        range(0, q_t.shape[0], BATCH_SIZE_SIM),
        total=_num_batches(int(q_t.shape[0]), int(BATCH_SIZE_SIM)),
        desc=f"retrieve top{topk}",
        leave=True,
    ):
        qb = q_t[i : i + BATCH_SIZE_SIM]
        best_scores = torch.full((qb.shape[0], topk), -float("inf"), dtype=torch.float32, device=DEVICE)
        best_idx = torch.full((qb.shape[0], topk), -1, dtype=torch.int64, device=DEVICE)

        for j in _iter_progress(
            range(0, c_t_cpu.shape[0], CORPUS_BLOCK_SIZE),
            total=_num_batches(int(c_t_cpu.shape[0]), int(CORPUS_BLOCK_SIZE)),
            desc="corpus blocks",
            leave=False,
        ):
            cb_cpu = c_t_cpu[j : j + CORPUS_BLOCK_SIZE]
            cb = cb_cpu.to(device=DEVICE, non_blocking=True) if DEVICE == "cuda" else cb_cpu
            if DEVICE == "cuda":
                sims = (qb.to(dtype=torch.float16) @ cb.to(dtype=torch.float16).T).to(dtype=torch.float32)
            else:
                sims = qb @ cb.T

            k2 = topk if cb.shape[0] >= topk else int(cb.shape[0])
            vals, inds = torch.topk(sims, k=k2, dim=1, largest=True, sorted=False)
            inds = inds + int(j)

            merged_scores = torch.cat([best_scores, vals], dim=1)
            merged_idx = torch.cat([best_idx, inds], dim=1)

            new_vals, new_pos = torch.topk(merged_scores, k=topk, dim=1, largest=True, sorted=True)
            best_scores = new_vals
            best_idx = torch.gather(merged_idx, 1, new_pos)

        all_topk_idx.append(best_idx.detach().cpu().numpy().astype(np.int32))

    return np.concatenate(all_topk_idx, axis=0)


def evaluate_recall_nq(gold_docs, retrieved_docs):
    pooled, _ = config.calculate_retrieval_recall(gold_docs, retrieved_docs, k_list=tuple(TOPK_LIST))
    return pooled


def compute_retrieval_and_eval(query_embs, corpus_embs, corpus_texts, gold_docs):
    max_k = max(TOPK_LIST)
    topk_idx = retrieve_topk(query_embs, corpus_embs, max_k)
    retrieved_docs = [[corpus_texts[int(j)] for j in row.tolist()] for row in topk_idx]
    return evaluate_recall_nq(gold_docs, retrieved_docs)


def run_local_model_on_nq(model_name, model_path, dims, corpus_texts, query_texts, gold_docs):
    m = None
    tokenizer = None
    corpus_full = None
    query_full = None
    results = []
    try:
        m = load_local_model(model_path)
        tokenizer = load_local_tokenizer(model_path)

        query_instruction = (
            config.QUERY_INSTRUCTION if model_name == "NV-Embed-v2" and hasattr(config, "QUERY_INSTRUCTION") else ""
        )

        if tqdm is not None:
            tqdm.write(f"model: {model_name}")
            tqdm.write("encoding corpus...")
        corpus_full = encode_local(m, tokenizer, corpus_texts, instruction="", max_length=MAX_LENGTH)
        if tqdm is not None:
            tqdm.write("encoding queries...")
        query_full = encode_local(m, tokenizer, query_texts, instruction=query_instruction, max_length=MAX_LENGTH)

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
    finally:
        try:
            del corpus_full
            del query_full
            del tokenizer
            del m
        except Exception:
            pass
        gc.collect()
        if DEVICE == "cuda":
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass


def main():
    start = time.time()
    print_runtime_device()

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

    all_results = []
    for spec in _iter_progress(LOCAL_MODELS, total=len(LOCAL_MODELS), desc="local models", leave=True):
        dims = GTE_DIMS if spec["name"] == "GTE-Qwen2-7B-Instruct" else LOCAL_DIMS
        all_results.extend(
            run_local_model_on_nq(
                model_name=spec["name"],
                model_path=spec["path"],
                dims=dims,
                corpus_texts=corpus_texts,
                query_texts=query_texts,
                gold_docs=gold_docs,
            )
        )

    out_path = os.path.join(RESULT_DIR, f"{DATASET_NAME}_model_dim_recall_local.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": DATASET_NAME,
                "corpus_path": CORPUS_JSON,
                "query_path": QUERY_JSON,
                "corpus_count": int(len(corpus_texts)),
                "query_count": int(len(query_texts)),
                "topks": TOPK_LIST,
                "dims": LOCAL_DIMS,
                "results": all_results,
                "time_s": float(time.time() - start),
                "result_dir": RESULT_DIR,
                "device": DEVICE,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(out_path)


if __name__ == "__main__":
    main()