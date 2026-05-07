"""
This script performs TRACE-style retrieval inference after "entanglement detection + QAS training" is completed
(an approximate implementation aligned with the pseudocode).

Important note: why this is an "approximation"
The current results_RAPTOR artifacts in this project do not contain an explicit multi-level parent-child tree
structure; they only contain flat clustering information in the form of cluster -> docs.

Therefore, this script approximates "tree search" as a two-level structure:
  - Level 2: root -> clusters, where each child is a cluster summary mu_c
  - Level 1: cluster -> docs, where each child is a document embedding treated as a leaf node

The retrieval logic implemented in this script, following the user-specified workflow, is:
  1) Compute the similarity between each query and all cluster nodes (mu_c), then select the top-k clusters.
  2) Read the entanglement binary classification result, i.e., the no_split version of split.json:
     - If the cluster belongs to entangled_clusters: enter the ENTANGLED branch.
     - Otherwise: enter the CLEAN branch.
  3) CLEAN: rank all documents under each CLEAN cluster by cosine similarity with the query,
     and take the top-k documents.
  4) ENTANGLED: rank all documents under each ENTANGLED cluster using QAS scores,
     and take the top-k documents.
  5) Merge candidates from the CLEAN and ENTANGLED branches (up to 2k), and output the final candidate set.

Note:
This script depends on numpy / torch and prerequisite artifact files. If torch is missing in the runtime
environment or the python command is unavailable, fix the runtime environment first.
"""

import json
import os
import time
import importlib
import importlib.util
import csv
import sys
import atexit

import numpy as np
import torch


config = importlib.import_module("0_config")
LOSS_OUTPUT_DIR = str(getattr(config, "LOSS_OUTPUT_DIR", os.path.join(config.BASE_DIR, "result_loss")))


def _redirect_output_to_txt(txt_name):
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, txt_name)
    f = open(path, "w", encoding="utf-8")
    sys.stdout = f
    sys.stderr = f
    return f, path


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


def _tee_output_to_txt(txt_name):
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, txt_name)
    f = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    return f, path


def topk_indices(scores, k):
    if k <= 0:
        return np.asarray([], dtype=np.int64)
    if scores.size == 0:
        return np.asarray([], dtype=np.int64)
    k = min(int(k), int(scores.size))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx


def topk_pairs_stream_merge(current, new_scores, new_ids, k):
    if k <= 0:
        return []
    if len(current) == 0:
        pairs = list(zip([float(x) for x in new_scores.tolist()], [int(x) for x in new_ids.tolist()]))
        pairs.sort(key=lambda t: t[0], reverse=True)
        return pairs[:k]
    pairs = current + list(zip([float(x) for x in new_scores.tolist()], [int(x) for x in new_ids.tolist()]))
    pairs.sort(key=lambda t: t[0], reverse=True)
    return pairs[:k]


def import_qas_module():
    qas_path = os.path.join(os.path.dirname(__file__), "3_hypernetwork_qas_new_no_split.py")
    if not os.path.exists(qas_path):
        raise FileNotFoundError(qas_path)
    spec = importlib.util.spec_from_file_location("qas_mod", qas_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from path: {qas_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_numpy(path, mmap=False):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path, mmap_mode="r" if mmap else None)


def load_json_or_jsonl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def l2_normalize(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def resolve_index_dir(base_dir, dataset_name, model_name, n_clusters, soft_k):
    results_root = os.path.join(base_dir, "results_RAPTOR")
    desired = os.path.join(results_root, f"{dataset_name}_hier_index_{model_name}_k{int(n_clusters)}_soft{int(soft_k)}")
    desired_centers = os.path.join(desired, "centers.npy")
    desired_top = os.path.join(desired, "doc_top_clusters.npy")
    if os.path.exists(desired_centers) and os.path.exists(desired_top):
        return desired, int(n_clusters)

    prefix = f"{dataset_name}_hier_index_{model_name}_k"
    suffix = f"_soft{int(soft_k)}"
    candidates = []
    if os.path.isdir(results_root):
        for name in os.listdir(results_root):
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            dir_path = os.path.join(results_root, name)
            centers_path = os.path.join(dir_path, "centers.npy")
            top_path = os.path.join(dir_path, "doc_top_clusters.npy")
            if not (os.path.exists(centers_path) and os.path.exists(top_path)):
                continue
            k_part = name[len(prefix) : -len(suffix)]
            try:
                k_val = int(k_part)
            except Exception:
                continue
            candidates.append((k_val, dir_path))

    if not candidates:
        raise FileNotFoundError(f"Index not found under {results_root}.")
    candidates.sort(key=lambda x: x[0], reverse=True)
    k_val, picked = candidates[0]
    print(f"[Index] Index with k{int(n_clusters)} was not found. Automatically falling back to: {picked}")
    return picked, int(k_val)


def resolve_cluster_summary_embs_path(base_dir, dataset_name, model_name):
    results_root = os.path.join(base_dir, "results_RAPTOR")
    candidate = os.path.join(results_root, f"{dataset_name}_cluster_summary_embs_{model_name}.npy")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(candidate)


def find_latest_cluster_to_docs_json(index_dir):
    candidates = []
    if os.path.isdir(index_dir):
        for name in os.listdir(index_dir):
            if not (name.startswith("cluster_to_docs_routing_trained") and name.endswith(".json")):
                continue
            p = os.path.join(index_dir, name)
            if not os.path.exists(p):
                continue
            try:
                candidates.append((os.path.getmtime(p), p))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_latest_routing_trained_centers(index_dir):
    candidates = []
    if os.path.isdir(index_dir):
        for name in os.listdir(index_dir):
            if not (name.startswith("centers_routing_trained") and name.endswith(".npy")):
                continue
            p = os.path.join(index_dir, name)
            if not os.path.exists(p):
                continue
            try:
                candidates.append((os.path.getmtime(p), p))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def find_entanglement_split_json(output_dir, dataset_name, model_name, n_clusters, soft_k, delta, temperature):
    prefix = f"{dataset_name}_entanglement_{model_name}_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}_split.json"
    candidate = os.path.join(output_dir, prefix)
    if os.path.exists(candidate):
        return candidate
    matches = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if not (name.startswith(f"{dataset_name}_entanglement_{model_name}_k") and name.endswith("_split.json")):
                continue
            p = os.path.join(output_dir, name)
            try:
                matches.append((os.path.getmtime(p), p))
            except Exception:
                continue
    if not matches:
        raise FileNotFoundError(f"Cannot find entanglement split json in {output_dir}")
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def find_entanglement_split_json_by_tag(output_dir, dataset_name, model_name, n_clusters, soft_k, delta, temperature, tag):
    tag = (tag or "").strip()
    if not tag:
        raise ValueError("split tag is empty")
    prefix = (
        f"{dataset_name}_entanglement_{model_name}_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}_{tag}_split.json"
    )
    candidate = os.path.join(output_dir, prefix)
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(candidate)


def find_entanglement_scores_npy(output_dir, dataset_name, model_name, n_clusters, soft_k, delta, temperature):
    prefix = f"{dataset_name}_entanglement_{model_name}_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}.npy"
    candidate = os.path.join(output_dir, prefix)
    if os.path.exists(candidate):
        return candidate

    matches = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if not (name.startswith(f"{dataset_name}_entanglement_{model_name}_k") and name.endswith(".npy")):
                continue
            if name.endswith("_split.npy"):
                continue
            p = os.path.join(output_dir, name)
            try:
                matches.append((os.path.getmtime(p), p))
            except Exception:
                continue
    if not matches:
        raise FileNotFoundError(f"Cannot find entanglement scores npy in {output_dir}")
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def find_query_gold_node_map(loss_dir, dataset_name, model_name, n_clusters, soft_k, delta, temperature):
    prefix = f"{dataset_name}_entanglement_{model_name}_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}"
    candidate = os.path.join(loss_dir, f"{prefix}_query_gold_node_map.json")
    if os.path.exists(candidate):
        return candidate

    matches = []
    if os.path.isdir(loss_dir):
        for name in os.listdir(loss_dir):
            if not (name.startswith(prefix) and name.endswith("_query_gold_node_map.json")):
                continue
            p = os.path.join(loss_dir, name)
            try:
                matches.append((os.path.getmtime(p), p))
            except Exception:
                continue
    if not matches:
        raise FileNotFoundError(f"Cannot find query_gold_node_map.json in {loss_dir}")
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def calculate_recall_by_doc_indices(gold_doc_indices_list, retrieved_doc_indices_list, k_list):
    k_list = sorted(set(int(k) for k in k_list))
    pooled = {f"Recall@{k}": 0.0 for k in k_list}
    per_query = []

    for gold_ids, retrieved_ids in zip(gold_doc_indices_list, retrieved_doc_indices_list):
        gold_set = set(int(x) for x in (gold_ids or []))
        top_max_k = [int(x) for x in (retrieved_ids or [])[: max(k_list)]]
        eval_one = {}
        for k in k_list:
            hit = set(top_max_k[:k]) & gold_set
            eval_one[f"Recall@{k}"] = (len(hit) / len(gold_set)) if gold_set else 0.0
            pooled[f"Recall@{k}"] += eval_one[f"Recall@{k}"]
        per_query.append(eval_one)

    n = len(gold_doc_indices_list) or 1
    for k in k_list:
        pooled[f"Recall@{k}"] = round(float(pooled[f"Recall@{k}"]) / float(n), 4)
    return pooled, per_query


def find_token_cache(output_dir, dataset_name, model_name, max_len):
    emb = os.path.join(output_dir, f"{dataset_name}_{model_name}_maxlen{int(max_len)}_fp16.npy")
    seqlen = os.path.join(output_dir, f"{dataset_name}_{model_name}_maxlen{int(max_len)}_seqlen.npy")
    if os.path.exists(emb) and os.path.exists(seqlen):
        return emb, seqlen
    raise FileNotFoundError(f"Token cache not found: {emb} / {seqlen}. Please run 0_token_embedding_cache.py.")


def find_qas_checkpoint(output_dir, dataset_name, model_name, n_clusters, soft_k, delta, temperature, tag=None, allow_default_fallback=False):
    tag = (tag or "").strip()
    if tag:
        p = os.path.join(
            output_dir,
            f"{dataset_name}_qas_hyper_{model_name}_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}_{tag}.pt",
        )
        if os.path.exists(p):
            return p
        if not allow_default_fallback:
            raise FileNotFoundError(p)
    p0 = os.path.join(
        output_dir,
        f"{dataset_name}_qas_hyper_{model_name}_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}.pt",
    )
    if os.path.exists(p0):
        return p0
    raise FileNotFoundError(p0)


def build_cluster_to_docs(primary_cluster_ids):
    cluster_to_docs = {}
    for doc_id, cid in enumerate(primary_cluster_ids.tolist()):
        cluster_to_docs.setdefault(int(cid), []).append(int(doc_id))
    return cluster_to_docs


def main():
    _log_f = None
    _log_f, _log_path = _tee_output_to_txt(
        f"4_trace_search_clean_wuzhaiyao_no_split_moreqiefen_{int(time.time())}.txt"
    )
    atexit.register(_log_f.close)
    t0 = time.time()

    device_pref = os.environ.get("TRACE_DEVICE", str(getattr(config, "TRACE_DEVICE", "auto"))).strip().lower()
    if device_pref not in ("auto", "cuda", "cpu"):
        raise ValueError("TRACE_DEVICE must be one of: auto, cuda, cpu")
    if device_pref == "cpu":
        device = "cpu"
    elif device_pref == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_clusters = int(getattr(config, "N_CLUSTERS", 1000))
    soft_k = int(getattr(config, "SOFT_K", 3))
    delta = float(os.environ.get("ENTANGLEMENT_DELTA", str(getattr(config, "ENTANGLEMENT_DELTA", 0.05))))
    temperature = float(os.environ.get("ENTANGLEMENT_TAU", str(getattr(config, "ENTANGLEMENT_TAU", 0.07))))

    max_queries = int(os.environ.get("TRACE_MAX_QUERIES", str(getattr(config, "TRACE_MAX_QUERIES", 0))))
    token_cache_maxlen = int(os.environ.get("TOKEN_CACHE_MAXLEN", str(getattr(config, "TOKEN_CACHE_MAXLEN", 128))))
    recall_ks = [2, 5, 10, 20]
    max_k = int(max(recall_ks))
    fixed_cluster_topk = int(os.environ.get("TRACE_CLUSTER_TOPK", str(getattr(config, "TRACE_CLUSTER_TOPK", 30))))
    cluster_score_alpha = float(os.environ.get("TRACE_CLUSTER_SCORE_ALPHA", str(getattr(config, "TRACE_CLUSTER_SCORE_ALPHA", 0.6))))
    cluster_score_alpha = min(max(cluster_score_alpha, 0.0), 1.0)
    branch_oversample = int(os.environ.get("TRACE_BRANCH_OVERSAMPLE", str(getattr(config, "TRACE_BRANCH_OVERSAMPLE", 3))))
    if branch_oversample < 1:
        branch_oversample = 1

    index_dir, n_clusters = resolve_index_dir(
        base_dir=config.BASE_DIR,
        dataset_name=config.DATASET_NAME,
        model_name=config.MODEL_NAME,
        n_clusters=n_clusters,
        soft_k=soft_k,
    )
    doc_top_clusters = load_numpy(os.path.join(index_dir, "doc_top_clusters.npy"))
    primary_cluster_ids = doc_top_clusters[:, 0].astype(np.int32)
    routed_cluster_to_docs_path = find_latest_cluster_to_docs_json(index_dir)
    if routed_cluster_to_docs_path is not None:
        obj = json.load(open(routed_cluster_to_docs_path, "r", encoding="utf-8"))
        cluster_to_docs = {int(k): [int(x) for x in v] for k, v in obj.items()}
    else:
        default_cluster_to_docs_path = os.path.join(index_dir, "cluster_to_docs.json")
        if os.path.exists(default_cluster_to_docs_path):
            obj = json.load(open(default_cluster_to_docs_path, "r", encoding="utf-8"))
            cluster_to_docs = {int(k): [int(x) for x in v] for k, v in obj.items()}
        else:
            cluster_to_docs = build_cluster_to_docs(primary_cluster_ids)

    summary_embs_path = resolve_cluster_summary_embs_path(
        base_dir=config.BASE_DIR,
        dataset_name=config.DATASET_NAME,
        model_name=config.MODEL_NAME,
    )
    summary_embs = load_numpy(summary_embs_path).astype(np.float32)
    if int(summary_embs.shape[0]) != int(n_clusters):
        raise ValueError(f"summary_embs.shape[0]={summary_embs.shape[0]} != n_clusters={n_clusters}")
    summary_embs = l2_normalize(summary_embs, axis=1).astype(np.float32)
    routed_centers_path = find_latest_routing_trained_centers(index_dir)
    routed_centers = None
    if routed_centers_path is not None:
        try:
            rc = load_numpy(routed_centers_path).astype(np.float32)
            if int(rc.shape[0]) == int(n_clusters):
                routed_centers = l2_normalize(rc, axis=1).astype(np.float32)
        except Exception:
            routed_centers = None
    selection_centers_source = "summary_embs"
    if routed_centers is not None:
        selection_centers_source = "routed_centers"
    print(f"[TRACE] selection_centers_source={selection_centers_source}")

    corpus_embs = load_numpy(config.CORPUS_EMBEDDING_PATH, mmap=True).astype(np.float32)
    corpus_embs = l2_normalize(corpus_embs, axis=1).astype(np.float32)
    query_embs = load_numpy(config.QUERY_EMBEDDING_PATH, mmap=True).astype(np.float32)
    query_embs = l2_normalize(query_embs, axis=1).astype(np.float32)

    if int(corpus_embs.shape[1]) != int(summary_embs.shape[1]):
        raise ValueError(
            f"Embedding dim mismatch: corpus_dim={int(corpus_embs.shape[1])} summary_dim={int(summary_embs.shape[1])}"
        )

    query_gold_map_path = os.environ.get("QUERY_GOLD_NODE_MAP_PATH")
    if query_gold_map_path is None:
        query_gold_map_path = find_query_gold_node_map(
            loss_dir=LOSS_OUTPUT_DIR,
            dataset_name=config.DATASET_NAME,
            model_name=config.MODEL_NAME,
            n_clusters=n_clusters,
            soft_k=soft_k,
            delta=delta,
            temperature=temperature,
        )
    query_gold_node_map = json.load(open(query_gold_map_path, "r", encoding="utf-8"))
    gold_entries = query_gold_node_map.get("entries", [])
    gold_by_qi = {}
    for e in gold_entries:
        qi = e.get("query_index")
        if qi is None:
            continue
        gold_by_qi[int(qi)] = [int(x) for x in (e.get("gold_doc_indices") or [])]

    qas_mod = import_qas_module()

    token_cache_path, seqlen_cache_path = find_token_cache(
        output_dir=config.OUTPUT_DIR,
        dataset_name=config.DATASET_NAME,
        model_name=config.MODEL_NAME,
        max_len=token_cache_maxlen,
    )
    token_cache = np.load(token_cache_path, mmap_mode="r")
    token_seqlen = np.load(seqlen_cache_path, mmap_mode="r")

    total_queries_in_emb = int(query_embs.shape[0])
    n_queries = total_queries_in_emb if int(max_queries) <= 0 else min(total_queries_in_emb, int(max_queries))
    print(f"[TRACE] queries_total={total_queries_in_emb} | max_queries={int(max_queries)} | n_queries={n_queries}")

    doc_qas_batch = int(os.environ.get("TRACE_DOC_QAS_BATCH", str(getattr(config, "TRACE_DOC_QAS_BATCH", 512))))
    qas_prefilter_topn = int(os.environ.get("TRACE_QAS_PREFILTER_TOPN", "5"))
    if qas_prefilter_topn < 0:
        qas_prefilter_topn = -1

    split_tags_env = os.environ.get("ENTANGLEMENT_SPLIT_TAGS")
    split_tags = []
    if split_tags_env is None:
        prefix_base = (
            f"{config.DATASET_NAME}_entanglement_{config.MODEL_NAME}"
            f"_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}_"
        )
        suffix = "_split.json"
        if os.path.isdir(config.OUTPUT_DIR):
            for name in os.listdir(config.OUTPUT_DIR):
                if not (name.startswith(prefix_base) and name.endswith(suffix)):
                    continue
                tag = name[len(prefix_base) : -len(suffix)].strip()
                if tag:
                    split_tags.append(tag)

        preferred = ["c7e3", "c3e1", "c8e2", "c9e1", "c95e5"]
        split_tags = list(dict.fromkeys(split_tags).keys())
        pref_rank = {t: i for i, t in enumerate(preferred)}
        split_tags.sort(key=lambda t: (pref_rank.get(t, 10**9), t))
    else:
        split_tags = [t.strip() for t in split_tags_env.split(",") if t.strip()]
        split_tags = list(dict.fromkeys(split_tags).keys())
    if not split_tags:
        split_tags = ["c7e3", "c3e1", "c8e2", "c9e1", "c95e5"]

    split_path_template = os.environ.get("ENTANGLEMENT_SPLIT_PATH")
    if split_path_template is not None and ("{tag}" not in split_path_template) and len(split_tags) > 1:
        raise ValueError("ENTANGLEMENT_SPLIT_PATH must include '{tag}' when ENTANGLEMENT_SPLIT_TAGS has multiple tags.")

    baseline_entangled_ratio = float(getattr(config, "ENTANGLED_RATIO", 0.1))
    base_device = device

    for tag in split_tags:
        device = base_device
        if split_path_template is not None:
            if "{tag}" in split_path_template:
                split_path = split_path_template.format(tag=tag)
            else:
                split_path = split_path_template
            if not os.path.exists(split_path):
                raise FileNotFoundError(split_path)
        else:
            split_path = find_entanglement_split_json_by_tag(
                output_dir=config.OUTPUT_DIR,
                dataset_name=config.DATASET_NAME,
                model_name=config.MODEL_NAME,
                n_clusters=n_clusters,
                soft_k=soft_k,
                delta=delta,
                temperature=temperature,
                tag=tag,
            )

        split_info = json.load(open(split_path, "r", encoding="utf-8"))
        entangled_ratio = None
        try:
            entangled_ratio = float(((split_info.get("epsilon") or {}).get("entangled_ratio")))
        except Exception:
            entangled_ratio = None

        entangled_clusters = [int(x) for x in (split_info.get("entangled_clusters") or [])]
        entangled_set = set(entangled_clusters)
        if not entangled_clusters:
            raise RuntimeError(f"No entangled_clusters found in split json (no_split mode): {split_path}")
        if max(entangled_clusters) >= int(n_clusters):
            raise ValueError("Entangled cluster ids exceed n_clusters; split/index mismatch.")

        clean_clusters_raw = split_info.get("clean_clusters") or []
        if clean_clusters_raw:
            clean_clusters_all = [int(x) for x in clean_clusters_raw if int(x) not in entangled_set]
        else:
            clean_clusters_all = [cid for cid in range(int(n_clusters)) if int(cid) not in entangled_set]
        clean_cluster_set_all = set(clean_clusters_all)

        clean_doc_ids_all = []
        for _cid in clean_clusters_all:
            ds = cluster_to_docs.get(int(_cid), [])
            if ds:
                clean_doc_ids_all.extend(int(x) for x in ds)
        if clean_doc_ids_all:
            clean_doc_ids_all = np.asarray(list(dict.fromkeys(clean_doc_ids_all).keys()), dtype=np.int64)
        else:
            clean_doc_ids_all = np.asarray([], dtype=np.int64)

        allow_default_fallback = False
        if entangled_ratio is not None and abs(float(entangled_ratio) - float(baseline_entangled_ratio)) < 1e-12:
            allow_default_fallback = True
        qas_ckpt = find_qas_checkpoint(
            output_dir=config.OUTPUT_DIR,
            dataset_name=config.DATASET_NAME,
            model_name=config.MODEL_NAME,
            n_clusters=n_clusters,
            soft_k=soft_k,
            delta=delta,
            temperature=temperature,
            tag=tag,
            allow_default_fallback=allow_default_fallback,
        )

        print(
            f"[TRACE] split_tag={tag} | entangled_ratio={entangled_ratio} | "
            f"total_clusters={int(n_clusters)} | clean_clusters={int(len(clean_cluster_set_all))} | "
            f"entangled_clusters={int(len(entangled_set))}"
        )

        ckpt = torch.load(qas_ckpt, map_location="cpu")
        qcfg = ckpt.get("cfg", {})
        qas = qas_mod.QueryAdaptiveScorer(
            qas_mod.QASConfig(d_model=int(qcfg.get("d_model")), h_hn=int(qcfg.get("h_hn")), n_layers=int(qcfg.get("n_layers")))
        )
        qas.load_state_dict(ckpt["state_dict"], strict=True)
        qas.eval()
        if device == "cuda":
            try:
                qas.to("cuda")
            except torch.cuda.OutOfMemoryError:
                device = "cpu"
                qas.to("cpu")
        else:
            qas.to("cpu")

        qas_fp16 = os.environ.get("TRACE_QAS_FP16", "1").strip() != "0"
        if device == "cuda" and qas_fp16:
            qas = qas.half()
        qas_dtype = torch.float16 if (device == "cuda" and qas_fp16) else torch.float32

        results = {
            "dataset_name": config.DATASET_NAME,
            "model_name": config.MODEL_NAME,
            "index_dir": index_dir,
            "split_json": split_path,
            "split_tag": str(tag),
            "entangled_ratio": float(entangled_ratio) if entangled_ratio is not None else None,
            "query_gold_node_map": query_gold_map_path,
            "qas_checkpoint": qas_ckpt,
            "selection_k_mode": "clusters_topk_env",
            "fixed_cluster_topk": int(fixed_cluster_topk),
            "cluster_score_alpha": float(cluster_score_alpha),
            "selection_centers_source": str(selection_centers_source),
            "routed_centers_path": routed_centers_path,
            "branch_oversample": int(branch_oversample),
            "max_k": int(max_k),
            "device": device,
            "recall_ks": [int(x) for x in recall_ks],
            "recall": None,
            "queries": [],
        }

        try:
            from tqdm import tqdm
        except Exception:
            tqdm = None

        query_iter = range(n_queries)
        pbar = None
        if tqdm is not None:
            pbar = tqdm(query_iter, desc=f"TRACE Search [{tag}] ({n_queries} queries)", dynamic_ncols=True, disable=True)
            query_iter = pbar

        total_query_trace_time_s = 0.0
        timed_query_count = 0
        sum_selected_clean_clusters = 0
        sum_selected_entangled_clusters = 0
        count_selected_queries = 0
        for qi in query_iter:
            _qt0 = time.perf_counter()
            uq = query_embs[qi].astype(np.float32)

            clean_ranked_ids_maxk = []
            if int(clean_doc_ids_all.size) > 0:
                _scores = (corpus_embs[clean_doc_ids_all] @ uq).astype(np.float32)
                _top = topk_indices(_scores, int(max_k))
                if int(_top.size) > 0:
                    clean_ranked_ids_maxk = [int(x) for x in clean_doc_ids_all[_top].tolist()]

            cluster_sims = (routed_centers @ uq).astype(np.float32) if routed_centers is not None else (summary_embs @ uq).astype(np.float32)
            top_cluster_idx_max = topk_indices(cluster_sims, int(fixed_cluster_topk))
            selected_clusters_max = [int(x) for x in top_cluster_idx_max.tolist()]

            need_qas = any(int(cid) in entangled_set for cid in selected_clusters_max)

            enable_qas_this_query = bool(need_qas)
            eq_tokens = None
            if enable_qas_this_query:
                if qi >= int(token_cache.shape[0]) or qi >= int(token_seqlen.shape[0]):
                    enable_qas_this_query = False
                else:
                    seq_len = int(token_seqlen[qi])
                    seq_len = max(1, min(seq_len, int(token_cache.shape[1])))
                    eq = token_cache[qi, :seq_len, :]
                    eq_tokens = torch.from_numpy(np.asarray(eq, dtype=np.float32)).to(device=device, dtype=qas_dtype)

            ent_cluster_scores = {}
            for cid in selected_clusters_max:
                if int(cid) not in entangled_set:
                    continue
                doc_ids = cluster_to_docs.get(int(cid), [])
                if not doc_ids:
                    continue
                if not enable_qas_this_query:
                    continue

                ids_full = np.asarray(doc_ids, dtype=np.int64)
                do_prefilter = bool(int(qas_prefilter_topn) > 0 and int(ids_full.size) > int(qas_prefilter_topn))
                if do_prefilter:
                    cos_full = (corpus_embs[ids_full] @ uq).astype(np.float32)
                    top_local = topk_indices(cos_full, int(qas_prefilter_topn))
                    ids_use = ids_full[top_local]
                else:
                    ids_use = ids_full

                mu_c = torch.from_numpy(summary_embs[int(cid)]).to(device=device, dtype=qas_dtype)
                all_ids = []
                all_qas_scores = []
                with torch.inference_mode():
                    for start in range(0, int(ids_use.size), int(doc_qas_batch)):
                        end = min(int(ids_use.size), start + int(doc_qas_batch))
                        chunk_ids = np.asarray(ids_use[start:end], dtype=np.int64)
                        doc_vecs = torch.from_numpy(np.asarray(corpus_embs[chunk_ids], dtype=np.float32)).to(
                            device=device, dtype=qas_dtype
                        )
                        s = qas(eq_tokens, mu_c, doc_vecs).detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32)
                        if s.size == 0:
                            continue
                        all_ids.append(chunk_ids)
                        all_qas_scores.append(s.reshape(-1))
                if all_ids:
                    ids_all = np.concatenate(all_ids, axis=0)
                    qas_all = np.concatenate(all_qas_scores, axis=0).astype(np.float32)
                    ent_cluster_scores[int(cid)] = (ids_all, qas_all)

            sum_selected_clean_clusters += int(sum(1 for _cid in selected_clusters_max if int(_cid) in clean_cluster_set_all))
            sum_selected_entangled_clusters += int(len(ent_cluster_scores))
            count_selected_queries += 1

            retrieved_by_k = {}
            retrieved_clean_by_k = {}
            retrieved_entangled_by_k = {}
            debug_by_k = {}
            for k in recall_ks:
                k = int(k)
                branch_k = max(1, int(k) * int(branch_oversample))
                selected_clusters = [int(x) for x in selected_clusters_max]
                clean_clusters = [int(cid) for cid in selected_clusters if int(cid) in clean_cluster_set_all]
                entangled_clusters_sel = [int(cid) for cid in selected_clusters if int(cid) in ent_cluster_scores]

                clean_top_ids = [int(x) for x in clean_ranked_ids_maxk[: int(k)]]

                ent_top_pairs = []
                for cid in entangled_clusters_sel:
                    ids, scores = ent_cluster_scores[int(cid)]
                    if scores.size == 0:
                        continue
                    local_top = topk_indices(scores, min(branch_k, int(scores.size)))
                    ent_top_pairs = topk_pairs_stream_merge(ent_top_pairs, scores[local_top], ids[local_top], branch_k)
                ent_top_ids_branch = [int(did) for _, did in ent_top_pairs]
                ent_top_ids = ent_top_ids_branch[: int(k)]

                retrieved_clean_by_k[str(int(k))] = [int(x) for x in clean_top_ids]
                retrieved_entangled_by_k[str(int(k))] = [int(x) for x in ent_top_ids]

                merged_ids = clean_top_ids + ent_top_ids
                uniq_doc_ids = list(dict.fromkeys(merged_ids).keys())
                if not uniq_doc_ids:
                    retrieved_by_k[str(int(k))] = []
                    debug_by_k[str(int(k))] = {
                        "selected_clusters": selected_clusters,
                        "clean_clusters": clean_clusters,
                        "entangled_clusters": entangled_clusters_sel,
                        "n_candidates": 0,
                    }
                    continue

                retrieved_by_k[str(int(k))] = [int(x) for x in uniq_doc_ids]
                debug_by_k[str(int(k))] = {
                    "selected_clusters": selected_clusters,
                    "clean_clusters": clean_clusters,
                    "entangled_clusters": entangled_clusters_sel,
                    "n_candidates": int(len(uniq_doc_ids)),
                    "branch_k": int(branch_k),
                    "final_k": int(len(uniq_doc_ids)),
                }

            results["queries"].append(
                {
                    "query_index": int(qi),
                    "gold_doc_indices": gold_by_qi.get(int(qi), []),
                    "retrieved_doc_ids_by_k": retrieved_by_k,
                    "retrieved_clean_doc_ids_by_k": retrieved_clean_by_k,
                    "retrieved_entangled_doc_ids_by_k": retrieved_entangled_by_k,
                    "debug_by_k": debug_by_k,
                }
            )

            total_query_trace_time_s += float(time.perf_counter() - _qt0)
            timed_query_count += 1

        if pbar is not None:
            pbar.close()

        if timed_query_count > 0:
            avg_ms = (float(total_query_trace_time_s) / float(timed_query_count)) * 1000.0
        else:
            avg_ms = 0.0
        print(
            f"[TRACE][{tag}] query_trace_time_total_s={total_query_trace_time_s:.4f} | avg_ms_per_query={avg_ms:.2f} | n_queries={int(timed_query_count)}"
        )
        if int(count_selected_queries) > 0:
            avg_clean_clusters = float(sum_selected_clean_clusters) / float(count_selected_queries)
            avg_ent_clusters = float(sum_selected_entangled_clusters) / float(count_selected_queries)
        else:
            avg_clean_clusters = 0.0
            avg_ent_clusters = 0.0
        print(
            f"[TRACE][{tag}] avg_selected_clean_clusters={avg_clean_clusters:.2f} | "
            f"avg_selected_entangled_clusters={avg_ent_clusters:.2f} | n_queries={int(count_selected_queries)}"
        )

        pooled = {f"Recall@{int(k)}": 0.0 for k in recall_ks}
        pooled_clean = {f"Recall@{int(k)}": 0.0 for k in recall_ks}
        pooled_entangled = {f"Recall@{int(k)}": 0.0 for k in recall_ks}
        n = len(results["queries"]) or 1
        for q in results["queries"]:
            gold_set = set(int(x) for x in (q.get("gold_doc_indices") or []))
            for k in recall_ks:
                retrieved_ids = [int(x) for x in (q.get("retrieved_doc_ids_by_k") or {}).get(str(int(k)), [])]
                retrieved_clean_ids = [int(x) for x in (q.get("retrieved_clean_doc_ids_by_k") or {}).get(str(int(k)), [])]
                retrieved_ent_ids = [int(x) for x in (q.get("retrieved_entangled_doc_ids_by_k") or {}).get(str(int(k)), [])]

                hit = set(retrieved_ids) & gold_set
                hit_clean = set(retrieved_clean_ids) & gold_set
                hit_ent = set(retrieved_ent_ids) & gold_set

                if gold_set:
                    pooled[f"Recall@{int(k)}"] += len(hit) / len(gold_set)
                    pooled_clean[f"Recall@{int(k)}"] += len(hit_clean) / len(gold_set)
                    pooled_entangled[f"Recall@{int(k)}"] += len(hit_ent) / len(gold_set)
                else:
                    pooled[f"Recall@{int(k)}"] += 0.0
                    pooled_clean[f"Recall@{int(k)}"] += 0.0
                    pooled_entangled[f"Recall@{int(k)}"] += 0.0
        for k in recall_ks:
            pooled[f"Recall@{int(k)}"] = round(float(pooled[f"Recall@{int(k)}"]) / float(n), 4)
            pooled_clean[f"Recall@{int(k)}"] = round(float(pooled_clean[f"Recall@{int(k)}"]) / float(n), 4)
            pooled_entangled[f"Recall@{int(k)}"] = round(float(pooled_entangled[f"Recall@{int(k)}"]) / float(n), 4)
        results["recall"] = pooled
        results["recall_clean"] = pooled_clean
        results["recall_entangled"] = pooled_entangled

        msg = (
            f"[TRACE][{tag}] Recall@2={pooled.get('Recall@2')} | "
            f"Recall@5={pooled.get('Recall@5')} | "
            f"Recall@10={pooled.get('Recall@10')} | "
            f"Recall@20={pooled.get('Recall@20')}"
        )
        msg_clean = (
            f"[TRACE][{tag}][CLEAN] Recall@2={pooled_clean.get('Recall@2')} | "
            f"Recall@5={pooled_clean.get('Recall@5')} | "
            f"Recall@10={pooled_clean.get('Recall@10')} | "
            f"Recall@20={pooled_clean.get('Recall@20')}"
        )
        msg_ent = (
            f"[TRACE][{tag}][ENTANGLED] Recall@2={pooled_entangled.get('Recall@2')} | "
            f"Recall@5={pooled_entangled.get('Recall@5')} | "
            f"Recall@10={pooled_entangled.get('Recall@10')} | "
            f"Recall@20={pooled_entangled.get('Recall@20')}"
        )
        print(msg)
        print(msg_clean)
        print(msg_ent)

        out_path = os.path.join(
            config.OUTPUT_DIR,
            f"{config.DATASET_NAME}_trace_search_{config.MODEL_NAME}_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}_{tag}_recallKs{max_k}.json",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        csv_path = os.path.join(
            config.OUTPUT_DIR,
            f"{config.DATASET_NAME}_trace_search_{config.MODEL_NAME}_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}_{tag}_recallKs{max_k}.csv",
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "query_index",
                    "gold_doc_indices",
                    "retrieved_doc_ids@2",
                    "retrieved_doc_ids@5",
                    "retrieved_doc_ids@10",
                    "retrieved_doc_ids@20",
                ]
            )
            for q in results.get("queries", []):
                retrieved_by_k = q.get("retrieved_doc_ids_by_k") or {}
                w.writerow(
                    [
                        int(q.get("query_index", -1)),
                        json.dumps(q.get("gold_doc_indices") or [], ensure_ascii=False),
                        json.dumps(retrieved_by_k.get("2") or [], ensure_ascii=False),
                        json.dumps(retrieved_by_k.get("5") or [], ensure_ascii=False),
                        json.dumps(retrieved_by_k.get("10") or [], ensure_ascii=False),
                        json.dumps(retrieved_by_k.get("20") or [], ensure_ascii=False),
                    ]
                )

        print(f"[TRACE][{tag}] trace search saved: {out_path}")
        print(f"[TRACE][{tag}] trace search csv saved: {csv_path}")

    print(f"log saved: {_log_path}")
    print(f"done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()