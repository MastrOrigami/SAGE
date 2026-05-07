import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from transformers import AutoModel, PreTrainedModel, AutoConfig
from tqdm import tqdm
import time
import http.client
import importlib


config = importlib.import_module("0_config")

BASE_DIR = config.BASE_DIR
RESULTS_DIR = os.path.join(BASE_DIR, "results_RAPTOR")
os.makedirs(RESULTS_DIR, exist_ok=True)

CORPUS_JSON_PATH = config.CORPUS_JSONL_PATH
QUERY_JSON_PATH = config.QUERY_DATA_PATH

CORPUS_EMB_OUTPUT = config.CORPUS_EMBEDDING_PATH
QUERY_EMB_OUTPUT = config.QUERY_EMBEDDING_PATH

API_KEY = config.LLM_API_KEY
API_HOST = config.LLM_HOST
API_ENDPOINT = config.LLM_ENDPOINT
API_MODEL = config.LLM_MODEL

MODEL_PATH = config.MODEL_PATH
MODEL_NAME = getattr(config, "MODEL_NAME", os.path.basename(MODEL_PATH.rstrip("/")))

N_CLUSTERS = 1000
SOFT_K = 3
TOP_M_CLUSTERS = 10
RECALL_KS = [1, 2, 5, 10, 20]
SAVE_TOP_K = 5

TOPK_RAW_RESULTS_PATH = os.path.join(
    config.OUTPUT_DIR,
    f"{config.DATASET_NAME}_top{SAVE_TOP_K}_retrieved_raw_{MODEL_NAME}.json",
)

INDEX_DIR = os.path.join(
    RESULTS_DIR,
    f"{config.DATASET_NAME}_hier_index_{MODEL_NAME}_k{N_CLUSTERS}_soft{SOFT_K}",
)
os.makedirs(INDEX_DIR, exist_ok=True)

INDEX_MANIFEST_PATH = os.path.join(INDEX_DIR, "index_manifest.json")
INDEX_CENTERS_PATH = os.path.join(INDEX_DIR, "centers.npy")
INDEX_CLUSTER_TO_DOCS_PATH = os.path.join(INDEX_DIR, "cluster_to_docs.json")
INDEX_DOC_TOP_CLUSTERS_PATH = os.path.join(INDEX_DIR, "doc_top_clusters.npy")
INDEX_DOC_META_PATH = os.path.join(INDEX_DIR, "corpus_doc_meta.jsonl")

try:
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    elif isinstance(getattr(PreTrainedModel, "all_tied_weights_keys", None), list):
        PreTrainedModel.all_tied_weights_keys = {}
except Exception as e:
    print(f"Monkey patch failed: {e}")

try:
    from transformers.cache_utils import DynamicCache

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls, past_key_values):
            if past_key_values is None:
                return cls()
            if hasattr(past_key_values, "get_usable_length"):
                return past_key_values
            return cls()

        DynamicCache.from_legacy_cache = from_legacy_cache

except Exception as e:
    print(f"DynamicCache patch failed: {e}")


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            return json.load(f)

        items = []
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
        return items


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_corpus_doc_meta_jsonl(path, corpus_data):
    if os.path.exists(path):
        return

    with open(path, "w", encoding="utf-8") as f:
        for doc_id, doc in enumerate(corpus_data):
            item = {
                "doc_id": int(doc_id),
                "corpus_id": get_corpus_id(doc, fallback_idx=doc_id),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def save_cluster_index_artifacts(
    corpus_data,
    centers,
    cluster_to_docs,
    doc_top_clusters,
    extra=None,
):
    np.save(INDEX_CENTERS_PATH, centers)
    np.save(INDEX_DOC_TOP_CLUSTERS_PATH, doc_top_clusters)
    save_json(INDEX_CLUSTER_TO_DOCS_PATH, {str(k): v for k, v in cluster_to_docs.items()})
    save_corpus_doc_meta_jsonl(INDEX_DOC_META_PATH, corpus_data)

    manifest = {
        "dataset_name": config.DATASET_NAME,
        "model_name": MODEL_NAME,
        "n_clusters": int(N_CLUSTERS),
        "soft_k": int(SOFT_K),
        "top_m_clusters": int(TOP_M_CLUSTERS),
        "paths": {
            "index_dir": INDEX_DIR,
            "centers": INDEX_CENTERS_PATH,
            "cluster_to_docs": INDEX_CLUSTER_TO_DOCS_PATH,
            "doc_top_clusters": INDEX_DOC_TOP_CLUSTERS_PATH,
            "corpus_doc_meta": INDEX_DOC_META_PATH,
            "corpus_json": CORPUS_JSON_PATH,
            "corpus_emb": CORPUS_EMB_OUTPUT,
            "query_json": QUERY_JSON_PATH,
            "query_emb": QUERY_EMB_OUTPUT,
            "cluster_summary_embs": CLUSTER_SUMMARY_EMBS_PATH,
            "cluster_summary_meta": CLUSTER_SUMMARY_META_PATH,
        },
    }

    if extra:
        manifest["extra"] = extra

    save_json(INDEX_MANIFEST_PATH, manifest)


def get_query_text(item):
    return item.get("question") or item.get("text") or item.get("query") or item.get("q") or ""


def get_corpus_id(item, fallback_idx=None):
    if isinstance(item, dict):
        if "_id" in item and item["_id"] is not None:
            return str(item["_id"])
        if "idx" in item and item["idx"] is not None:
            return str(item["idx"])

    if fallback_idx is not None:
        return str(fallback_idx)

    return ""


def load_qrels_ids(qrels_path):
    mapping = {}

    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            rel = json.loads(line)

            if rel.get("score", 0) > 0:
                qid = rel["query-id"]
                cid = rel["corpus-id"]
                mapping.setdefault(qid, []).append(cid)

    return mapping


def build_limit_gold_ids(queries_data, qrels_map):
    gold_ids = []

    for q in queries_data:
        qid = q.get("_id")
        gold_ids.append(list(set(qrels_map.get(qid, []))))

    return gold_ids


def calculate_recall_by_ids(gold_ids_list, retrieved_ids_list, k_list):
    k_list = sorted(set(k_list))
    example_eval_results = []
    pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}

    for gold_ids, retrieved_ids in zip(gold_ids_list, retrieved_ids_list):
        gold_set = set(gold_ids)
        top_max_k = retrieved_ids[:max(k_list)]
        eval_result = {}

        for k in k_list:
            hit = set(top_max_k[:k]) & gold_set
            eval_result[f"Recall@{k}"] = (len(hit) / len(gold_set)) if gold_set else 0.0

        example_eval_results.append(eval_result)

        for k in k_list:
            pooled_eval_results[f"Recall@{k}"] += eval_result[f"Recall@{k}"]

    n = len(gold_ids_list) or 1

    for k in k_list:
        pooled_eval_results[f"Recall@{k}"] = round(
            pooled_eval_results[f"Recall@{k}"] / n,
            4,
        )

    return pooled_eval_results, example_eval_results


def load_embeddings(path):
    return np.load(path)


def normalize_embeddings(embs):
    return normalize(embs, norm="l2")


def load_model():
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    model_config = AutoConfig.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    if hasattr(model_config, "text_config"):
        model_config.text_config._name_or_path = MODEL_PATH

    model_config._name_or_path = MODEL_PATH

    if hasattr(model_config, "use_cache"):
        model_config.use_cache = False

    device_pref = os.environ.get("INDEX_DEVICE", "auto").strip().lower()

    if device_pref not in ("auto", "cuda", "cpu"):
        raise ValueError("INDEX_DEVICE must be one of: auto, cuda, cpu")

    use_cuda = bool(torch.cuda.is_available() and device_pref in ("auto", "cuda"))
    torch_dtype = torch.float16 if use_cuda else torch.float32

    def _load(torch_dtype_local):
        m = AutoModel.from_pretrained(
            MODEL_PATH,
            config=model_config,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch_dtype_local,
            device_map=None,
        )

        if hasattr(m, "config") and hasattr(m.config, "use_cache"):
            m.config.use_cache = False

        if (
            hasattr(m, "embedding_model")
            and hasattr(m.embedding_model, "config")
            and hasattr(m.embedding_model.config, "use_cache")
        ):
            m.embedding_model.config.use_cache = False

        m.eval()
        return m

    model = _load(torch_dtype)

    if use_cuda:
        try:
            model = model.to("cuda")
        except torch.cuda.OutOfMemoryError:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            model = _load(torch.float32).to("cpu")

    return model


def encode_texts(model, texts, instruction=""):
    embeddings = []
    batch_size = int(os.environ.get("INDEX_EMB_BATCH_SIZE", "2"))
    max_length = int(os.environ.get("INDEX_EMB_MAX_LENGTH", "1024"))

    i = 0
    pbar = tqdm(total=len(texts), desc="Encoding texts")

    try:
        while i < len(texts):
            bs = int(batch_size)

            if bs <= 0:
                bs = 1

            batch_texts = texts[i: i + bs]

            try:
                with torch.no_grad():
                    batch_embs = model.encode(
                        batch_texts,
                        instruction=instruction,
                        max_length=max_length,
                    )
                    batch_embs = F.normalize(batch_embs, p=2, dim=1)
                    embeddings.append(batch_embs.cpu().numpy())

                i += len(batch_texts)
                pbar.update(len(batch_texts))

            except torch.cuda.OutOfMemoryError:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                if bs > 1:
                    batch_size = max(1, bs // 2)
                    continue

                try:
                    model = model.to("cpu")
                except Exception:
                    pass

                with torch.no_grad():
                    batch_embs = model.encode(
                        batch_texts,
                        instruction=instruction,
                        max_length=max_length,
                    )
                    batch_embs = F.normalize(batch_embs, p=2, dim=1)
                    embeddings.append(batch_embs.cpu().numpy())

                i += len(batch_texts)
                pbar.update(len(batch_texts))

    finally:
        pbar.close()

    if not embeddings:
        return np.array([])

    return np.concatenate(embeddings, axis=0)


def perform_soft_clustering(corpus_embs):
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        batch_size=2048,
        random_state=42,
        n_init="auto",
    )

    kmeans.fit(corpus_embs)
    centers = normalize(kmeans.cluster_centers_, norm="l2")

    n_samples = corpus_embs.shape[0]
    doc_top_clusters = np.empty((n_samples, SOFT_K), dtype=np.int32)

    batch_size = 2048

    for i in tqdm(range(0, n_samples, batch_size), desc="Soft allocation"):
        batch_embs = corpus_embs[i: i + batch_size]
        sims = np.dot(batch_embs, centers.T)
        top_k_indices = np.argsort(sims, axis=1)[:, -SOFT_K:][:, ::-1]
        doc_top_clusters[i: i + top_k_indices.shape[0], :] = top_k_indices.astype(np.int32)

    cluster_to_docs = {c: [] for c in range(N_CLUSTERS)}

    for doc_id in range(n_samples):
        for c in doc_top_clusters[doc_id]:
            cluster_to_docs[int(c)].append(int(doc_id))

    return centers, cluster_to_docs, doc_top_clusters


def call_llm(prompt):
    try:
        conn = http.client.HTTPSConnection(API_HOST)

        payload = json.dumps({
            "model": API_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": 500,
            "temperature": 0.5,
            "stream": False,
        })

        headers = {
            **getattr(config, "LLM_DEFAULT_HEADERS", {}),
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }

        conn.request("POST", API_ENDPOINT, payload, headers)

        res = conn.getresponse()
        data = res.read()
        response_json = json.loads(data.decode("utf-8"))

        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"].strip()

        print(f"API error: {response_json}")
        return None

    except Exception as e:
        print(f"LLM call failed: {e}")
        return None


CLUSTER_SUMMARY_EMBS_PATH = os.path.join(
    RESULTS_DIR,
    f"{config.DATASET_NAME}_cluster_summary_embs_{MODEL_NAME}.npy",
)

CLUSTER_SUMMARY_META_PATH = os.path.join(
    RESULTS_DIR,
    f"{config.DATASET_NAME}_cluster_summary_meta_{MODEL_NAME}.json",
)

CLUSTER_SUMMARY_MAX_CHARS_PER_CHUNK = 12000
CLUSTER_SUMMARY_MAX_CHARS_PER_PASSAGE = 800


def generate_cluster_summary_anchors(corpus_data, cluster_to_docs, model):
    if os.path.exists(CLUSTER_SUMMARY_EMBS_PATH) and os.path.exists(CLUSTER_SUMMARY_META_PATH):
        anchor_embs = np.load(CLUSTER_SUMMARY_EMBS_PATH)

        with open(CLUSTER_SUMMARY_META_PATH, "r", encoding="utf-8") as f:
            anchor_metadata = json.load(f)

        return anchor_embs, anchor_metadata

    anchor_texts = []
    anchor_metadata = []

    for c_id in tqdm(range(N_CLUSTERS), desc="Generate cluster summaries"):
        doc_ids = cluster_to_docs.get(c_id, [])

        if not doc_ids:
            continue

        passage_texts = []

        for did in doc_ids:
            doc = corpus_data[did]
            title = str(doc.get("title", "")).strip()
            text = str(doc.get("text", "")).strip()
            passage = (title + "\n" + text).strip()

            if len(passage) > CLUSTER_SUMMARY_MAX_CHARS_PER_PASSAGE:
                passage = passage[:CLUSTER_SUMMARY_MAX_CHARS_PER_PASSAGE]

            passage_texts.append(passage)

        chunks = []
        current = []
        current_len = 0

        for p in passage_texts:
            add_len = len(p) + 2

            if current and current_len + add_len > CLUSTER_SUMMARY_MAX_CHARS_PER_CHUNK:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0

            current.append(p)
            current_len += add_len

        if current:
            chunks.append("\n\n".join(current))

        partial_summaries = []

        for chunk_idx, chunk_text in enumerate(chunks):
            prompt = (
                "You are given a set of passages that belong to the same cluster.\n"
                "Please write a concise natural language summary of the cluster content.\n"
                "Requirements:\n"
                "- Capture key entities, topics, and relations.\n"
                "- Avoid copying long spans verbatim.\n"
                "- Keep it compact.\n\n"
                f"Cluster ID: {c_id}\n"
                f"Chunk {chunk_idx + 1}/{len(chunks)}:\n"
                f"{chunk_text}\n"
            )

            summary = call_llm(prompt)

            if summary:
                partial_summaries.append(summary)
            else:
                partial_summaries.append(chunk_text[:500])

        if len(partial_summaries) == 1:
            final_summary = partial_summaries[0]
        else:
            merge_prompt = (
                "You are given multiple partial summaries for the same document cluster.\n"
                "Please merge them into one coherent concise cluster summary.\n"
                "- Remove redundancy.\n"
                "- Keep important entities/relations.\n\n"
                f"Cluster ID: {c_id}\n"
                "Partial summaries:\n"
                + "\n\n".join([f"- {s}" for s in partial_summaries])
            )

            merged = call_llm(merge_prompt)
            final_summary = merged if merged else "\n".join(partial_summaries)[:800]

        anchor_text = f"Cluster Summary: {final_summary}"

        anchor_texts.append(anchor_text)

        anchor_metadata.append({
            "cluster_id": int(c_id),
            "type": "cluster_summary",
            "doc_ids": [int(x) for x in doc_ids],
            "num_docs": int(len(doc_ids)),
            "summary": final_summary,
            "anchor_text": anchor_text,
        })

    anchor_embs = encode_texts(model, anchor_texts, instruction="")

    np.save(CLUSTER_SUMMARY_EMBS_PATH, anchor_embs)

    with open(CLUSTER_SUMMARY_META_PATH, "w", encoding="utf-8") as f:
        json.dump(anchor_metadata, f, ensure_ascii=False, indent=2)

    return anchor_embs, anchor_metadata


def retrieve_and_evaluate(
    query_embs,
    queries_data,
    centers,
    anchor_embs,
    anchor_metadata,
    cluster_to_docs,
    corpus_embs,
    corpus_data,
    gold_docs=None,
    gold_ids=None,
):
    centroid_cluster_ids = np.arange(N_CLUSTERS)
    generated_cluster_ids = np.array([m["cluster_id"] for m in anchor_metadata])

    all_anchors = np.vstack([centers, anchor_embs])
    all_cluster_ids = np.concatenate([centroid_cluster_ids, generated_cluster_ids])
    all_anchors = normalize(all_anchors, norm="l2")

    all_retrieved_docs = []
    all_retrieved_ids = []
    topk_raw_results = []
    start_time = time.time()

    for q_idx, q_emb in enumerate(tqdm(query_embs, desc="Searching")):
        q_emb = q_emb.reshape(1, -1)

        selected_clusters = set()

        sims_orig = np.dot(all_anchors, q_emb.T).flatten()
        top_indices_orig = np.argsort(sims_orig)[::-1]

        count = 0

        for idx in top_indices_orig:
            cid = all_cluster_ids[idx]

            if cid not in selected_clusters:
                selected_clusters.add(cid)
                count += 1

            if count >= TOP_M_CLUSTERS:
                break

        selected_clusters = list(selected_clusters)

        candidate_doc_ids = []

        for cid in selected_clusters:
            candidate_doc_ids.extend(cluster_to_docs.get(cid, []))

        if not candidate_doc_ids:
            all_retrieved_docs.append([])
            all_retrieved_ids.append([])

            topk_raw_results.append({
                "query_id": q_idx,
                "question": get_query_text(queries_data[q_idx]),
                "top_k": SAVE_TOP_K,
                "retrieved_docs": [],
                "retrieved": [],
            })

            continue

        candidate_doc_ids = list(set(candidate_doc_ids))
        candidate_embs_subset = corpus_embs[candidate_doc_ids]

        doc_sims = np.dot(candidate_embs_subset, q_emb.T).flatten()
        top_k_indices_local = np.argsort(doc_sims)[::-1][:max(RECALL_KS)]
        retrieved_ids = [candidate_doc_ids[i] for i in top_k_indices_local]

        current_retrieved_docs = []
        current_retrieved_ids = []

        for rid in retrieved_ids:
            doc = corpus_data[rid]
            current_retrieved_docs.append(doc.get("title", "") + "\n" + doc.get("text", ""))
            current_retrieved_ids.append(get_corpus_id(doc, fallback_idx=rid))

        all_retrieved_docs.append(current_retrieved_docs)
        all_retrieved_ids.append(current_retrieved_ids)

        saved_k = min(SAVE_TOP_K, len(top_k_indices_local))
        topk_docs_for_this_query = []
        topk_docs_text_for_this_query = []

        for rank in range(saved_k):
            local_idx = int(top_k_indices_local[rank])
            rid = int(candidate_doc_ids[local_idx])
            doc = corpus_data[rid]
            doc_text_for_reader = doc.get("title", "") + "\n" + doc.get("text", "")

            topk_docs_text_for_this_query.append(doc_text_for_reader)

            topk_docs_for_this_query.append({
                "rank": rank + 1,
                "doc_id": rid,
                "corpus_id": get_corpus_id(doc, fallback_idx=rid),
                "score": float(doc_sims[local_idx]),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            })

        topk_raw_results.append({
            "query_id": q_idx,
            "question": get_query_text(queries_data[q_idx]),
            "top_k": SAVE_TOP_K,
            "retrieved_docs": topk_docs_text_for_this_query,
            "retrieved": topk_docs_for_this_query,
        })

        if q_idx < 3:
            if config.DATASET_NAME == "limit":
                gold_set_for_debug = set(gold_ids[q_idx]) if gold_ids is not None else set()
            else:
                gold_set_for_debug = set(
                    config.normalize_text_for_eval(d) for d in gold_docs[q_idx]
                )

            for rank, rid in enumerate(retrieved_ids[:3]):
                doc = corpus_data[rid]

                if config.DATASET_NAME == "limit":
                    doc_id_for_hit = get_corpus_id(doc, fallback_idx=rid)
                    is_hit = "YES" if doc_id_for_hit in gold_set_for_debug else "NO"
                else:
                    doc_content = doc.get("title", "") + "\n" + doc.get("text", "")
                    doc_content_norm = config.normalize_text_for_eval(doc_content)
                    is_hit = "YES" if doc_content_norm in gold_set_for_debug else "NO"

                print(
                    f"  {rank + 1}. [{is_hit}] "
                    f"{doc.get('title', '')} - {doc.get('text', '')[:100]}..."
                )

    print("Calculating recall...")

    if config.DATASET_NAME == "limit":
        pooled_eval_results, _ = calculate_recall_by_ids(
            gold_ids,
            all_retrieved_ids,
            RECALL_KS,
        )
    else:
        pooled_eval_results, _ = config.calculate_retrieval_recall(
            gold_docs,
            all_retrieved_docs,
            RECALL_KS,
        )

    try:
        with open(TOPK_RAW_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(topk_raw_results, f, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"Failed to save top-{SAVE_TOP_K} raw retrieval results: {e}")

    print("\n=== Evaluation Results: Original Query + Hybrid Anchors ===")
    print(f"Total time: {time.time() - start_time:.2f}s")

    for k in sorted(RECALL_KS):
        key = f"Recall@{k}"

        if key in pooled_eval_results:
            print(f"{key}: {pooled_eval_results[key]:.4f}")

    return pooled_eval_results


def main():
    corpus_embs = load_embeddings(CORPUS_EMB_OUTPUT)
    query_embs = load_embeddings(QUERY_EMB_OUTPUT)
    corpus_data = load_json(CORPUS_JSON_PATH)
    queries_data = load_json(QUERY_JSON_PATH)

    corpus_embs = normalize_embeddings(corpus_embs)
    query_embs = normalize_embeddings(query_embs)

    if config.DATASET_NAME == "limit":
        print("Building gold IDs for LIMIT using qrels.jsonl...")
        qrels_path = config.DATASETS["limit"]["qrels_json"]
        qrels_map = load_qrels_ids(qrels_path)
        gold_ids = build_limit_gold_ids(queries_data, qrels_map)
        gold_docs = None
    else:
        print("Building gold documents using config...")
        gold_docs = config.build_gold_docs(queries_data)
        gold_ids = None

    centers, cluster_to_docs, doc_top_clusters = perform_soft_clustering(corpus_embs)

    model = load_model()

    anchor_embs, anchor_metadata = generate_cluster_summary_anchors(
        corpus_data,
        cluster_to_docs,
        model,
    )

    save_cluster_index_artifacts(
        corpus_data=corpus_data,
        centers=centers,
        cluster_to_docs=cluster_to_docs,
        doc_top_clusters=doc_top_clusters,
        extra={
            "cluster_summary_count": int(anchor_embs.shape[0])
            if isinstance(anchor_embs, np.ndarray)
            else None
        },
    )

    retrieve_and_evaluate(
        query_embs,
        queries_data,
        centers,
        anchor_embs,
        anchor_metadata,
        cluster_to_docs,
        corpus_embs,
        corpus_data,
        gold_docs=gold_docs,
        gold_ids=gold_ids,
    )


if __name__ == "__main__":
    main()