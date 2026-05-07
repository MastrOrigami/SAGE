"""
This script implements the Hypernetwork QAS training process (L_hyper) from Sections 5.2/5.3 of the TRACE paper.
The core idea is to freeze the base encoder and document vectors, and train only a set of globally shared
hypernetwork parameters Phi.

For each entangled cluster c, the hypernetwork dynamically generates scorer weights based on
the query token sequence E_q and the cluster summary vector mu_c. It then applies InfoNCE within the cluster
using positive samples and negative sampling to increase the score margin.
"""

import json
import os
import time
import importlib
import atexit
import sys
from dataclasses import dataclass

import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F


config = importlib.import_module("0_config")
LOSS_OUTPUT_DIR = str(getattr(config, "LOSS_OUTPUT_DIR", os.path.join(config.BASE_DIR, "result_loss")))
os.makedirs(LOSS_OUTPUT_DIR, exist_ok=True)


def _redirect_output_to_txt(txt_name):
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, txt_name)
    f = open(path, "w", encoding="utf-8")
    sys.stdout = f
    sys.stderr = f
    return f, path


def load_numpy(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return np.load(path)


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


def get_query_text(item):
    return item.get("question") or item.get("text") or item.get("query") or item.get("q") or ""


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
        raise FileNotFoundError(
            f"Index not found under {results_root}. Please run hierarchical index construction first."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    k_val, picked = candidates[0]
    print(f"[Index] Index with k{int(n_clusters)} was not found. Automatically falling back to: {picked}")
    return picked, int(k_val)


def resolve_cluster_summary_embs_path(base_dir, dataset_name, model_name):
    results_root = os.path.join(base_dir, "results_RAPTOR")
    candidate = os.path.join(results_root, f"{dataset_name}_cluster_summary_embs_{model_name}.npy")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f"Cluster summary embeddings not found: {candidate}. "
        "Please make sure cluster summary embeddings were generated under results_RAPTOR."
    )


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


@dataclass
class QASConfig:
    d_model: int
    h_hn: int
    n_layers: int


class GlobalConditionalHypernetwork(nn.Module):
    def __init__(self, cfg: QASConfig):
        super().__init__()
        self.cfg = cfg
        self.q_list = nn.ParameterList()
        self.theta_k_list = nn.ParameterList()
        self.theta_v_list = nn.ParameterList()
        self.theta_w_left_list = nn.ParameterList()
        self.theta_w_right_list = nn.ParameterList()
        self.theta_h_base_list = nn.ParameterList()
        self.theta_b_base_list = nn.ParameterList()
        self.theta_b_proj_list = nn.ParameterList()

        d_aug = cfg.d_model + 1
        for _ in range(cfg.n_layers):
            self.q_list.append(nn.Parameter(torch.randn(cfg.h_hn, cfg.h_hn) * 0.02))
            self.theta_k_list.append(nn.Parameter(torch.randn(d_aug, cfg.h_hn) * 0.02))
            self.theta_v_list.append(nn.Parameter(torch.randn(d_aug, cfg.h_hn) * 0.02))

            self.theta_h_base_list.append(nn.Parameter(torch.randn(cfg.d_model, cfg.d_model) * 0.02))
            self.theta_w_left_list.append(nn.Parameter(torch.randn(cfg.d_model, cfg.h_hn) * 0.02))
            self.theta_w_right_list.append(nn.Parameter(torch.randn(cfg.h_hn, cfg.d_model) * 0.02))
            self.theta_b_base_list.append(nn.Parameter(torch.zeros(cfg.d_model)))
            self.theta_b_proj_list.append(nn.Parameter(torch.randn(cfg.d_model, cfg.h_hn) * 0.02))

        self.ln_h = nn.LayerNorm(cfg.h_hn)
        self.ln_x = nn.LayerNorm(cfg.d_model)

    def generate_weight(self, e_tilde_q, layer_idx):
        ones = torch.ones((e_tilde_q.shape[0], 1), device=e_tilde_q.device, dtype=e_tilde_q.dtype)
        e_aug = torch.cat([e_tilde_q, ones], dim=1)

        q = self.q_list[layer_idx]
        k = e_aug @ self.theta_k_list[layer_idx]
        v = e_aug @ self.theta_v_list[layer_idx]

        attn = (q @ k.T) / float(np.sqrt(self.cfg.h_hn))
        attn = torch.softmax(attn, dim=-1)
        h = attn @ v
        h = self.ln_h(F.relu(h))

        w_base = self.theta_h_base_list[layer_idx]
        w_delta = self.theta_w_left_list[layer_idx] @ h @ self.theta_w_right_list[layer_idx]
        w = w_base + w_delta

        h_pool = h.mean(dim=0)
        b = self.theta_b_base_list[layer_idx] + (self.theta_b_proj_list[layer_idx] @ h_pool)
        return w, b


class QueryAdaptiveScorer(nn.Module):
    def __init__(self, cfg: QASConfig):
        super().__init__()
        self.cfg = cfg
        self.hyper = GlobalConditionalHypernetwork(cfg)
        self.out_proj = nn.Linear(cfg.d_model, 1, bias=True)

    def forward(self, eq_tokens, mu_c, passage_emb):
        e_tilde_q = torch.cat([eq_tokens, mu_c.view(1, -1)], dim=0)
        x = passage_emb
        for i in range(self.cfg.n_layers):
            w, b = self.hyper.generate_weight(e_tilde_q, i)
            x = self.hyper.ln_x(F.relu(F.linear(x, w, b))) + x
        score = self.out_proj(x).squeeze(-1)
        return score


def load_transformers_model(model_path):
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers not available: {e}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return tokenizer, model


def encode_query_tokens(tokenizer, model, query_text, device, max_length=256):
    inputs = tokenizer(
        query_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model(**inputs)
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state[0]
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return out[0][0]
    raise RuntimeError("Cannot obtain token embeddings from model output.")


def find_token_embedding_cache(output_dir, dataset_name, model_name, max_len):
    candidate = os.path.join(output_dir, f"{dataset_name}_{model_name}_maxlen{int(max_len)}_fp16.npy")
    if os.path.exists(candidate):
        return candidate
    prefix = f"{dataset_name}_{model_name}_maxlen"
    suffix = "_fp16.npy"
    matches = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            p = os.path.join(output_dir, name)
            try:
                matches.append((os.path.getmtime(p), p))
            except Exception:
                continue
    if not matches:
        return None
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def find_token_seqlen_cache(output_dir, dataset_name, model_name, max_len):
    candidate = os.path.join(output_dir, f"{dataset_name}_{model_name}_maxlen{int(max_len)}_seqlen.npy")
    if os.path.exists(candidate):
        return candidate
    return None


def find_query_gold_node_map(loss_dir, dataset_name, model_name, n_clusters, soft_k, delta, temperature):
    prefix = (
        f"{dataset_name}_entanglement_{model_name}_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}"
    )
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


def _downsample_xy(xs, ys, max_points):
    xs = list(xs)
    ys = list(ys)
    n = len(xs)
    if n <= max_points:
        return xs, ys
    step = max(1, n // max_points)
    xs2 = xs[::step]
    ys2 = ys[::step]
    if xs2[-1] != xs[-1]:
        xs2.append(xs[-1])
        ys2.append(ys[-1])
    return xs2, ys2


def save_loss_line_plots(epoch_history, batch_history, out_png, out_svg):
    epoch_x = []
    epoch_y = []
    for r in epoch_history:
        if isinstance(r, dict) and "epoch" in r and "avg_loss" in r:
            epoch_x.append(int(r["epoch"]))
            epoch_y.append(float(r["avg_loss"]))

    batch_x = []
    batch_y = []
    for r in batch_history:
        if isinstance(r, dict) and "global_step" in r and "loss" in r:
            batch_x.append(int(r["global_step"]))
            batch_y.append(float(r["loss"]))

    if not epoch_x and not batch_x:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)

        if epoch_x:
            axes[0].plot(epoch_x, epoch_y, linewidth=1.5)
            axes[0].set_title("QAS Hypernetwork Train Loss (Epoch Avg)")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].set_axis_off()

        if batch_x:
            bx, by = _downsample_xy(batch_x, batch_y, max_points=8000)
            axes[1].plot(bx, by, linewidth=1.0)
            axes[1].set_title("QAS Hypernetwork Train Loss (Batch)")
            axes[1].set_xlabel("Global Step")
            axes[1].set_ylabel("Loss")
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].set_axis_off()

        if out_png:
            fig.savefig(out_png, dpi=180)
        if out_svg:
            fig.savefig(out_svg)
        plt.close(fig)
        return
    except Exception:
        pass

    def _svg_plot(xs, ys, x0, y0, w, h):
        if not xs:
            return ""
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax == xmin:
            xmax = xmin + 1
        if ymax == ymin:
            ymax = ymin + 1e-9
        pts = []
        for x, y in zip(xs, ys):
            px = x0 + (float(x - xmin) / float(xmax - xmin)) * w
            py = y0 + h - (float(y - ymin) / float(ymax - ymin)) * h
            pts.append(f"{px:.2f},{py:.2f}")
        return f'<polyline fill="none" stroke="#1f77b4" stroke-width="1.2" points="{" ".join(pts)}" />'

    width, height = 1200, 800
    pad = 60
    panel_h = (height - pad * 3) / 2.0
    panel_w = width - pad * 2

    epoch_svg = _svg_plot(epoch_x, epoch_y, pad, pad, panel_w, panel_h)
    bx, by = _downsample_xy(batch_x, batch_y, max_points=8000)
    batch_svg = _svg_plot(bx, by, pad, pad * 2 + panel_h, panel_w, panel_h)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />'
        f'<text x="{pad}" y="{pad - 20}" font-size="16">QAS Hypernetwork Train Loss (Epoch Avg)</text>'
        f'<text x="{pad}" y="{pad * 2 + panel_h - 20}" font-size="16">QAS Hypernetwork Train Loss (Batch)</text>'
        f'<rect x="{pad}" y="{pad}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#cccccc" />'
        f'<rect x="{pad}" y="{pad * 2 + panel_h}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#cccccc" />'
        f"{epoch_svg}{batch_svg}</svg>"
    )
    if out_svg:
        with open(out_svg, "w", encoding="utf-8") as f:
            f.write(svg)


def build_entangled_training_pairs(query_gold_node_map, entangled_cluster_set, primary_cluster_ids):
    entries = query_gold_node_map.get("entries", [])
    pairs = []
    for e in entries:
        qi = int(e.get("query_index"))
        gold_doc_indices = e.get("gold_doc_indices") or []
        for doc_id in gold_doc_indices:
            doc_id = int(doc_id)
            if doc_id < 0 or doc_id >= int(primary_cluster_ids.shape[0]):
                continue
            cid = int(primary_cluster_ids[doc_id])
            if cid in entangled_cluster_set:
                pairs.append((qi, cid, doc_id))
    return pairs


def build_query_cluster_positive_docs(query_gold_node_map, primary_cluster_ids):
    qc_pos = {}
    entries = query_gold_node_map.get("entries", [])
    for e in entries:
        qi = int(e.get("query_index"))
        gold_doc_indices = e.get("gold_doc_indices") or []
        for doc_id in gold_doc_indices:
            doc_id = int(doc_id)
            if doc_id < 0 or doc_id >= int(primary_cluster_ids.shape[0]):
                continue
            cid = int(primary_cluster_ids[doc_id])
            key = (qi, cid)
            if key not in qc_pos:
                qc_pos[key] = set()
            qc_pos[key].add(int(doc_id))
    return qc_pos


def extract_doc_text(doc):
    title = (doc.get("title") or "").strip()
    text = doc.get("text")
    if text is None:
        text = doc.get("contents")
    if text is None:
        text = doc.get("paragraph_text")
    if text is None:
        text = ""
    text = str(text)
    if title:
        return title + "\n" + text
    return "\n" + text


def build_query_negative_doc_indices(query_samples, corpus_samples):
    corpus_text_to_indices = {}
    for idx, doc in enumerate(corpus_samples):
        doc_text = extract_doc_text(doc)
        norm = config.normalize_text_for_eval(doc_text)
        if norm not in corpus_text_to_indices:
            corpus_text_to_indices[norm] = []
        corpus_text_to_indices[norm].append(int(idx))

    neg_by_qi = {}
    for qi, q in enumerate(query_samples):
        ctxs = q.get("contexts") if isinstance(q, dict) else None
        if not isinstance(ctxs, list):
            continue
        neg_ids = []
        for c in ctxs:
            if not isinstance(c, dict):
                continue
            if c.get("is_supporting") is not False:
                continue
            title = str(c.get("title") or "").strip()
            text = str(c.get("text") or c.get("contents") or "").strip()
            doc_text = (title + "\n" + text).strip() if title else ("\n" + text)
            norm = config.normalize_text_for_eval(doc_text)
            idx_list = corpus_text_to_indices.get(norm)
            if not idx_list:
                continue
            neg_ids.extend(idx_list)
        if neg_ids:
            neg_by_qi[int(qi)] = sorted(set(int(x) for x in neg_ids))
    return neg_by_qi


def _parse_int_list_env(name, default_list):
    raw = os.environ.get(name)
    if raw is None:
        return list(default_list)
    raw = str(raw).strip()
    if not raw:
        return list(default_list)
    out = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            continue
    return out or list(default_list)


def evaluate_cluster_hit_at_k(
    qas,
    val_pairs,
    centers,
    corpus_embs,
    token_cache,
    token_seqlen,
    cluster_to_docs,
    device,
    k_list,
):
    k_list = sorted(set(int(k) for k in k_list if int(k) > 0))
    if not k_list:
        return {}
    max_k = int(max(k_list))
    eval_doc_batch = int(os.environ.get("QAS_VAL_DOC_BATCH", str(getattr(config, "QAS_VAL_DOC_BATCH", 1024))))
    if eval_doc_batch <= 0:
        eval_doc_batch = 1024

    hits = {int(k): 0 for k in k_list}
    total = 0
    qas.eval()
    with torch.inference_mode():
        for (qi, cid, pos_doc_id) in val_pairs:
            qi = int(qi)
            cid = int(cid)
            pos_doc_id = int(pos_doc_id)
            if qi >= int(token_cache.shape[0]) or qi >= int(token_seqlen.shape[0]):
                continue
            doc_ids = cluster_to_docs.get(cid, [])
            if not doc_ids:
                continue
            seq_len = int(token_seqlen[qi])
            seq_len = max(1, min(seq_len, int(token_cache.shape[1])))
            eq = token_cache[qi, :seq_len, :]
            eq_tokens = torch.from_numpy(np.asarray(eq, dtype=np.float32)).to(device)
            mu_c = torch.from_numpy(np.asarray(centers[cid], dtype=np.float32)).to(device)

            top_scores = None
            top_ids = None
            for start in range(0, len(doc_ids), int(eval_doc_batch)):
                chunk_ids = np.asarray(doc_ids[start : start + int(eval_doc_batch)], dtype=np.int64)
                doc_vecs = torch.from_numpy(np.asarray(corpus_embs[chunk_ids], dtype=np.float32)).to(device)
                scores = qas(eq_tokens, mu_c, doc_vecs).detach().cpu().numpy().astype(np.float32)
                if scores.size == 0:
                    continue
                if top_scores is None:
                    take = min(max_k, int(scores.size))
                    idx = np.argpartition(-scores, take - 1)[:take]
                    idx = idx[np.argsort(-scores[idx])]
                    top_scores = scores[idx]
                    top_ids = chunk_ids[idx]
                else:
                    comb_scores = np.concatenate([top_scores, scores], axis=0)
                    comb_ids = np.concatenate([top_ids, chunk_ids], axis=0)
                    take = min(max_k, int(comb_scores.size))
                    idx = np.argpartition(-comb_scores, take - 1)[:take]
                    idx = idx[np.argsort(-comb_scores[idx])]
                    top_scores = comb_scores[idx]
                    top_ids = comb_ids[idx]

            if top_ids is None or top_ids.size == 0:
                continue
            total += 1
            for k in k_list:
                if pos_doc_id in set(int(x) for x in top_ids[: int(k)].tolist()):
                    hits[int(k)] += 1

    if total <= 0:
        return {f"val_hit@{k}": 0.0 for k in k_list}
    return {f"val_hit@{k}": round(float(hits[int(k)]) / float(total), 4) for k in k_list}


def train_hypernetwork_qas(
    qas,
    centers,
    corpus_embs,
    token_cache,
    token_seqlen,
    primary_cluster_ids,
    entangled_clusters,
    query_gold_node_map,
    query_neg_doc_indices_by_qi,
    query_sent_embs,
    tau_t,
    num_neg,
    pair_batch_size,
    max_epochs,
    lr,
    weight_decay,
    max_pairs,
):
    device = next(qas.parameters()).device
    entangled_cluster_set = set(int(x) for x in entangled_clusters)

    cluster_to_docs = {}
    for doc_id, cid in enumerate(primary_cluster_ids.tolist()):
        cluster_to_docs.setdefault(int(cid), []).append(int(doc_id))

    pairs = build_entangled_training_pairs(query_gold_node_map, entangled_cluster_set, primary_cluster_ids)
    query_cluster_pos_docs = build_query_cluster_positive_docs(query_gold_node_map, primary_cluster_ids)
    if not pairs:
        raise RuntimeError("No (query, entangled_cluster, positive_doc) training pairs found.")
    if max_pairs is not None and int(max_pairs) > 0 and len(pairs) > int(max_pairs):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(pairs), size=int(max_pairs), replace=False)
        pairs = [pairs[int(i)] for i in idx.tolist()]

    val_ratio = float(os.environ.get("QAS_VAL_RATIO", str(getattr(config, "QAS_VAL_RATIO", 0.05))))
    if val_ratio < 0:
        val_ratio = 0.0
    if val_ratio > 0.5:
        val_ratio = 0.5
    val_ks = _parse_int_list_env("QAS_VAL_KS", getattr(config, "QAS_VAL_KS", [10, 20]))
    val_pairs = []
    train_pairs = pairs
    if val_ratio > 0.0 and len(pairs) >= 20:
        split_rng = np.random.default_rng(0)
        perm = split_rng.permutation(len(pairs))
        val_n = max(1, int(round(len(pairs) * val_ratio)))
        val_pairs = [pairs[int(i)] for i in perm[:val_n].tolist()]
        train_pairs = [pairs[int(i)] for i in perm[val_n:].tolist()]

    rng = np.random.default_rng(0)
    optimizer = torch.optim.AdamW(qas.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    qas.train()
    history = []
    batch_history = []
    global_step = 0
    use_val = bool(val_pairs)
    best_score = -1.0 if use_val else float("inf")
    best_epoch = 0
    best_state = None
    prev_loss = None
    last_sign = 0
    sign_flips = 0
    prev_train_loss_for_val = None

    neg_mode = "intersection"
    hard_pool_cap = int(os.environ.get("QAS_HARD_POOL_CAP", str(getattr(config, "QAS_HARD_POOL_CAP", 2048))))
    if hard_pool_cap < 64:
        hard_pool_cap = 64

    val_patience = int(os.environ.get("QAS_VAL_PATIENCE", str(getattr(config, "QAS_VAL_PATIENCE", 5))))
    if val_patience < 1:
        val_patience = 1
    val_min_epochs = int(os.environ.get("QAS_VAL_MIN_EPOCHS", str(getattr(config, "QAS_VAL_MIN_EPOCHS", 10))))
    if val_min_epochs < 1:
        val_min_epochs = 1
    val_best = -1.0
    val_bad_epochs = 0
    primary_val_k = int(max(val_ks)) if val_ks else 20
    val_earlystop_loss_tol = float(
        os.environ.get("QAS_VAL_EARLYSTOP_LOSS_TOL", str(getattr(config, "QAS_VAL_EARLYSTOP_LOSS_TOL", 1e-4)))
    )
    if val_earlystop_loss_tol < 0:
        val_earlystop_loss_tol = 0.0

    early_stop_enabled = os.environ.get("QAS_EARLY_STOP", str(getattr(config, "QAS_EARLY_STOP", 1))).strip() != "0"
    rebound_tol = float(os.environ.get("QAS_REBOUND_TOL", str(getattr(config, "QAS_REBOUND_TOL", 1e-4))))
    min_epochs_before_oscillation = int(os.environ.get("QAS_MIN_EPOCHS", str(getattr(config, "QAS_MIN_EPOCHS", 5))))
    oscillation_flips = int(os.environ.get("QAS_OSCILLATION_FLIPS", str(getattr(config, "QAS_OSCILLATION_FLIPS", 4))))

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    for epoch in range(int(max_epochs)):
        rng.shuffle(train_pairs)
        total_loss = 0.0
        steps = 0
        it = range(0, len(train_pairs), int(pair_batch_size))
        pbar = None
        if tqdm is not None:
            disable_tqdm = os.environ.get("SIGMOD_LOG_TO_FILE", "0").strip() != "0"
            pbar = tqdm(
                it,
                desc=f"QAS Train Epoch {epoch+1}/{int(max_epochs)}",
                total=(len(train_pairs) + int(pair_batch_size) - 1) // int(pair_batch_size),
                dynamic_ncols=True,
                disable=disable_tqdm,
            )
            it = pbar

        for start in it:
            batch = train_pairs[start : start + int(pair_batch_size)]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.0

            for (qi, cid, pos_doc_id) in batch:
                if qi >= int(token_cache.shape[0]) or qi >= int(token_seqlen.shape[0]):
                    continue
                seq_len = int(token_seqlen[qi])
                seq_len = max(1, min(seq_len, int(token_cache.shape[1])))
                eq = token_cache[qi, :seq_len, :]
                eq_tokens = torch.from_numpy(np.asarray(eq, dtype=np.float32)).to(device)

                mu_c = torch.from_numpy(np.asarray(centers[int(cid)], dtype=np.float32)).to(device)

                cand = cluster_to_docs.get(int(cid), [])
                if not cand:
                    continue
                cand_set = set(int(x) for x in cand)
                pos_docs_same_qc = query_cluster_pos_docs.get((int(qi), int(cid)), set())
                for pd in pos_docs_same_qc:
                    cand_set.discard(int(pd))
                cand_set.discard(int(pos_doc_id))

                query_neg = query_neg_doc_indices_by_qi.get(int(qi), []) if isinstance(query_neg_doc_indices_by_qi, dict) else []
                query_neg_set = set(int(x) for x in query_neg)
                query_neg_set.discard(int(pos_doc_id))

                inter = cand_set & query_neg_set
                neg_pool = list(inter)

                k_need = int(max(1, num_neg))
                if len(neg_pool) < k_need and cand_set:
                    remain = list(cand_set - set(neg_pool))
                    if remain:
                        qv = None
                        if query_sent_embs is not None and int(qi) < int(query_sent_embs.shape[0]):
                            qv = np.asarray(query_sent_embs[int(qi)], dtype=np.float32)
                        if qv is not None:
                            if len(remain) > int(hard_pool_cap):
                                idx = rng.choice(len(remain), size=int(hard_pool_cap), replace=False)
                                remain = [remain[int(i)] for i in idx.tolist()]
                            rem_ids = np.asarray(remain, dtype=np.int64)
                            sims = (corpus_embs[rem_ids] @ qv).astype(np.float32)
                            take = min(k_need - len(neg_pool), int(sims.size))
                            if take > 0:
                                top_idx = np.argpartition(-sims, take - 1)[:take]
                                top_idx = top_idx[np.argsort(-sims[top_idx])]
                                neg_pool.extend(rem_ids[top_idx].astype(np.int64).tolist())
                        else:
                            take = min(k_need - len(neg_pool), len(remain))
                            if take > 0:
                                pick = rng.choice(len(remain), size=take, replace=False)
                                neg_pool.extend([int(remain[int(i)]) for i in pick.tolist()])

                if not neg_pool:
                    continue
                if len(neg_pool) > k_need:
                    pick = rng.choice(len(neg_pool), size=k_need, replace=False)
                    neg_ids = [int(neg_pool[int(i)]) for i in pick.tolist()]
                else:
                    neg_ids = [int(x) for x in neg_pool]

                doc_ids = [int(pos_doc_id)] + [int(x) for x in neg_ids]
                doc_vecs = torch.from_numpy(np.asarray(corpus_embs[doc_ids], dtype=np.float32)).to(device)

                with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=torch.float16):
                    scores = qas(eq_tokens, mu_c, doc_vecs)
                    logits = scores / float(tau_t)
                    target = torch.zeros((1,), device=device, dtype=torch.long)
                    loss = F.cross_entropy(logits.view(1, -1), target)
                batch_loss = batch_loss + loss

            if isinstance(batch_loss, float) or (hasattr(batch_loss, "numel") and batch_loss.numel() == 0):
                continue
            batch_loss = batch_loss / max(1, len(batch))
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(batch_loss.detach().cpu().item())
            batch_history.append(
                {
                    "epoch": int(epoch + 1),
                    "global_step": int(global_step),
                    "loss": float(batch_loss.detach().cpu().item()),
                    "pair_batch_size": int(len(batch)),
                }
            )
            global_step += 1
            steps += 1
            if pbar is not None:
                pbar.set_postfix(
                    {
                        "loss": f"{float(batch_loss.detach().cpu().item()):.4f}",
                        "avg": f"{(total_loss / max(1, steps)):.4f}",
                        "step": int(global_step),
                    }
                )

        avg_loss = total_loss / max(1, steps)
        epoch_record = {
            "epoch": int(epoch + 1),
            "avg_loss": float(avg_loss),
            "steps": int(steps),
            "pairs": int(len(train_pairs)),
        }

        primary_val = None
        if use_val:
            val_metrics = evaluate_cluster_hit_at_k(
                qas=qas,
                val_pairs=val_pairs,
                centers=centers,
                corpus_embs=corpus_embs,
                token_cache=token_cache,
                token_seqlen=token_seqlen,
                cluster_to_docs=cluster_to_docs,
                device=device,
                k_list=val_ks,
            )
            epoch_record.update(val_metrics)
            primary_val = float(val_metrics.get(f"val_hit@{primary_val_k}", 0.0))
            print(
                f"[QAS Train] Epoch {epoch+1:03d}/{int(max_epochs)} - loss={avg_loss:.6f} steps={steps} "
                f"val_hit@{primary_val_k}={primary_val:.4f}"
            )
            if primary_val > val_best:
                val_best = primary_val
                val_bad_epochs = 0
            else:
                val_bad_epochs += 1
        else:
            print(f"[QAS Train] Epoch {epoch+1:03d}/{int(max_epochs)} - loss={avg_loss:.6f} steps={steps}")

        history.append(epoch_record)

        if use_val:
            if primary_val is not None and (best_state is None or float(primary_val) > float(best_score)):
                best_score = float(primary_val)
                best_epoch = int(epoch + 1)
                best_state = {k: v.detach().cpu().clone() for k, v in qas.state_dict().items()}
        else:
            if avg_loss < float(best_score):
                best_score = float(avg_loss)
                best_epoch = int(epoch + 1)
                best_state = {k: v.detach().cpu().clone() for k, v in qas.state_dict().items()}

        if early_stop_enabled and use_val:
            loss_improve = None
            if prev_train_loss_for_val is not None:
                loss_improve = float(prev_train_loss_for_val) - float(avg_loss)
            prev_train_loss_for_val = float(avg_loss)
            allow_stop = (loss_improve is None) or (abs(float(loss_improve)) <= float(val_earlystop_loss_tol))
            if int(epoch + 1) >= int(val_min_epochs) and int(val_bad_epochs) >= int(val_patience) and allow_stop:
                print(
                    f"[QAS Train] Validation metric has not improved for {val_bad_epochs} epochs, "
                    f"and the loss improvement is insufficient "
                    f"(|delta_loss|={abs(loss_improve) if loss_improve is not None else 'NA'} <= tol={val_earlystop_loss_tol}); "
                    f"early stopping."
                )
                break
        elif early_stop_enabled:
            if prev_loss is not None:
                delta_loss = float(avg_loss) - float(prev_loss)
                if delta_loss > float(rebound_tol):
                    sign = 1
                elif delta_loss < -float(rebound_tol):
                    sign = -1
                else:
                    sign = 0
                if sign != 0 and last_sign != 0 and sign != last_sign:
                    sign_flips += 1
                if sign != 0:
                    last_sign = sign
                if int(epoch + 1) >= int(min_epochs_before_oscillation) and int(sign_flips) >= int(oscillation_flips):
                    print(
                        f"[QAS Train] Loss shows persistent rebound oscillation "
                        f"(rebound_tol={rebound_tol}, flips={sign_flips}); early stopping."
                    )
                    break
            prev_loss = float(avg_loss)

    qas.eval()
    if best_state is not None:
        qas.load_state_dict(best_state, strict=True)
    if val_pairs:
        history.append(
            {
                "best_epoch": int(best_epoch),
                "best_val_hit": float(best_score),
                "val_ratio": float(val_ratio),
                "val_ks": [int(x) for x in val_ks],
                "val_primary_k": int(primary_val_k),
                "val_patience": int(val_patience),
                "val_min_epochs": int(val_min_epochs),
                "val_pairs": int(len(val_pairs)),
                "train_pairs": int(len(train_pairs)),
            }
        )
    else:
        history.append({"best_epoch": int(best_epoch), "best_loss": float(best_score), "sign_flips": int(sign_flips)})
    return history, batch_history, train_pairs


def main():
    _log_f = None
    if os.environ.get("SIGMOD_LOG_TO_FILE", "0").strip() != "0":
        _log_f, _log_path = _redirect_output_to_txt("3_hypernetwork_qas_output.txt")
        atexit.register(_log_f.close)
    os.environ.setdefault("QAS_INTER_CLUSTER", "1")
    os.environ.setdefault("QAS_CURRICULUM", "1")
    t0 = time.time()
    device_pref = os.environ.get("QAS_DEVICE", str(getattr(config, "QAS_DEVICE", "auto"))).strip().lower()
    if device_pref not in ("auto", "cuda", "cpu"):
        raise ValueError("QAS_DEVICE must be one of: auto, cuda, cpu")
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

    index_dir, n_clusters = resolve_index_dir(
        base_dir=config.BASE_DIR,
        dataset_name=config.DATASET_NAME,
        model_name=config.MODEL_NAME,
        n_clusters=n_clusters,
        soft_k=soft_k,
    )
    centers_path = os.path.join(index_dir, "centers.npy")
    routed_centers_path = find_latest_routing_trained_centers(index_dir)
    if routed_centers_path is not None:
        centers_path = routed_centers_path
    doc_top_clusters_path = os.path.join(index_dir, "doc_top_clusters.npy")

    centers = load_numpy(centers_path).astype(np.float32)
    centers = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    doc_top_clusters = load_numpy(doc_top_clusters_path)
    primary_cluster_ids = doc_top_clusters[:, 0].astype(np.int32)

    summary_embs_path = resolve_cluster_summary_embs_path(
        base_dir=config.BASE_DIR,
        dataset_name=config.DATASET_NAME,
        model_name=config.MODEL_NAME,
    )
    summary_embs = load_numpy(summary_embs_path).astype(np.float32)
    if int(summary_embs.shape[0]) != int(n_clusters):
        raise ValueError(
            "Cluster summary embeddings count does not match index clusters. "
            f"summary_embs.shape[0]={int(summary_embs.shape[0])}, n_clusters={int(n_clusters)}. "
            f"summary_embs_path={summary_embs_path}, index_dir={index_dir}"
        )
    summary_embs = summary_embs / np.maximum(np.linalg.norm(summary_embs, axis=1, keepdims=True), 1e-12)
    d_model = int(summary_embs.shape[1])

    corpus_embs = load_numpy(config.CORPUS_EMBEDDING_PATH).astype(np.float32)
    corpus_embs = corpus_embs / np.maximum(np.linalg.norm(corpus_embs, axis=1, keepdims=True), 1e-12)
    if int(corpus_embs.shape[1]) != int(d_model):
        raise ValueError(
            "Embedding dimension mismatch: corpus_embs dim != summary_embs dim. "
            f"corpus_embs.shape[1]={int(corpus_embs.shape[1])}, summary_embs.shape[1]={int(d_model)}"
        )

    cluster_to_docs = {}
    for doc_id, cid in enumerate(primary_cluster_ids.tolist()):
        cluster_to_docs.setdefault(int(cid), []).append(int(doc_id))

    query_samples = load_json_or_jsonl(config.QUERY_DATA_PATH)
    h_hn = int(os.environ.get("QAS_H_HN", str(getattr(config, "QAS_H_HN", 4096))))
    n_layers = int(os.environ.get("QAS_LAYERS", str(getattr(config, "QAS_LAYERS", 2))))
    cfg = QASConfig(d_model=d_model, h_hn=h_hn, n_layers=n_layers)

    cache_max_len = int(os.environ.get("TOKEN_CACHE_MAXLEN", str(getattr(config, "TOKEN_CACHE_MAXLEN", 128))))
    token_cache_path = find_token_embedding_cache(config.OUTPUT_DIR, config.DATASET_NAME, config.MODEL_NAME, cache_max_len)
    if token_cache_path is None:
        raise FileNotFoundError(
            f"Token embedding cache not found in {config.OUTPUT_DIR}. "
            "Please run the token embedding cache generation script first."
        )
    token_cache = np.load(token_cache_path, mmap_mode="r")
    cache_max_len = int(token_cache.shape[1])
    seqlen_cache_path = find_token_seqlen_cache(config.OUTPUT_DIR, config.DATASET_NAME, config.MODEL_NAME, cache_max_len)
    if seqlen_cache_path is None:
        raise FileNotFoundError(
            f"Token seq_len cache not found in {config.OUTPUT_DIR}. "
            "Please re-run the token embedding cache generation script to generate *_seqlen.npy."
        )
    token_seqlen = np.load(seqlen_cache_path, mmap_mode="r")

    loss_dir = str(getattr(config, "LOSS_OUTPUT_DIR", os.path.join(config.BASE_DIR, "result_loss")))
    query_gold_map_path = os.environ.get("QUERY_GOLD_NODE_MAP_PATH")
    if query_gold_map_path is None:
        query_gold_map_path = find_query_gold_node_map(
            loss_dir=loss_dir,
            dataset_name=config.DATASET_NAME,
            model_name=config.MODEL_NAME,
            n_clusters=n_clusters,
            soft_k=soft_k,
            delta=delta,
            temperature=temperature,
        )
    query_gold_node_map = json.load(open(query_gold_map_path, "r", encoding="utf-8"))
    query_sent_embs = load_numpy(config.QUERY_EMBEDDING_PATH).astype(np.float32)
    query_sent_embs = query_sent_embs / np.maximum(np.linalg.norm(query_sent_embs, axis=1, keepdims=True), 1e-12)

    qas_tau_t = float(os.environ.get("QAS_TAU_T", str(getattr(config, "QAS_TAU_T", 0.07))))
    qas_num_neg = int(os.environ.get("QAS_NUM_NEG", str(getattr(config, "QAS_NUM_NEG", 31))))
    qas_pair_batch = int(os.environ.get("QAS_PAIR_BATCH", str(getattr(config, "QAS_PAIR_BATCH", 8))))
    qas_train_epochs = int(os.environ.get("QAS_TRAIN_EPOCHS", str(getattr(config, "QAS_TRAIN_EPOCHS", 50))))
    qas_lr = float(os.environ.get("QAS_TRAIN_LR", str(getattr(config, "QAS_TRAIN_LR", 1e-4))))
    qas_wd = float(os.environ.get("QAS_WEIGHT_DECAY", str(getattr(config, "QAS_WEIGHT_DECAY", 0.0))))
    qas_max_pairs = int(os.environ.get("QAS_MAX_PAIRS", str(getattr(config, "QAS_MAX_PAIRS", 20000))))
    qas_early_stop = os.environ.get("QAS_EARLY_STOP", str(getattr(config, "QAS_EARLY_STOP", 1))).strip() != "0"
    qas_rebound_tol = float(os.environ.get("QAS_REBOUND_TOL", str(getattr(config, "QAS_REBOUND_TOL", 1e-4))))
    qas_min_epochs = int(os.environ.get("QAS_MIN_EPOCHS", str(getattr(config, "QAS_MIN_EPOCHS", 5))))
    qas_osc_flips = int(os.environ.get("QAS_OSCILLATION_FLIPS", str(getattr(config, "QAS_OSCILLATION_FLIPS", 4))))

    sample_queries = int(os.environ.get("QAS_SAMPLE_QUERIES", str(getattr(config, "QAS_SAMPLE_QUERIES", 3))))
    top_passages = int(os.environ.get("QAS_TOP_PASSAGES", str(getattr(config, "QAS_TOP_PASSAGES", 10))))

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

    print(f"device={device} | dataset={config.DATASET_NAME} | model={config.MODEL_NAME}")
    print(f"index_dir={index_dir} | n_clusters={n_clusters} | soft_k={soft_k}")
    print(f"QAS: d={d_model}, h_hn={h_hn}, layers={n_layers}")
    print(f"mu_c_source=cluster_summary_embs | path={summary_embs_path}")
    print(f"token_cache={token_cache_path} | max_len={cache_max_len}")
    print(f"token_seqlen_cache={seqlen_cache_path}")
    print(f"query_gold_node_map={query_gold_map_path}")
    print(f"QAS Train: tau_t={qas_tau_t} neg={qas_num_neg} pair_batch={qas_pair_batch} epochs={qas_train_epochs} lr={qas_lr}")
    print(f"QAS NegSampling: mode=intersection+hard-fallback")
    print(
        f"QAS EarlyStop: enabled={qas_early_stop} rebound_tol={qas_rebound_tol} min_epochs={qas_min_epochs} flips={qas_osc_flips}"
    )
    print(f"split_tags={','.join(split_tags)}")

    query_neg_doc_indices_by_qi = {}
    try:
        query_neg_doc_indices_by_qi = build_query_negative_doc_indices(
            query_samples=query_samples,
            corpus_samples=load_json_or_jsonl(config.CORPUS_JSONL_PATH),
        )
    except Exception:
        query_neg_doc_indices_by_qi = {}

    for tag in split_tags:
        if split_path_template is not None:
            if "{tag}" in split_path_template:
                split_json_path = split_path_template.format(tag=tag)
            else:
                split_json_path = split_path_template
            if not os.path.exists(split_json_path):
                raise FileNotFoundError(split_json_path)
        else:
            split_json_path = find_entanglement_split_json_by_tag(
                output_dir=config.OUTPUT_DIR,
                dataset_name=config.DATASET_NAME,
                model_name=config.MODEL_NAME,
                n_clusters=n_clusters,
                soft_k=soft_k,
                delta=delta,
                temperature=temperature,
                tag=tag,
            )

        split_info = json.load(open(split_json_path, "r", encoding="utf-8"))
        entangled_clusters = [int(x) for x in (split_info.get("entangled_clusters") or [])]
        if not entangled_clusters:
            raise RuntimeError(f"No entangled_clusters found in split json (no-split mode): {split_json_path}")
        train_clusters = sorted(set(entangled_clusters))
        max_cluster_id_in_split = max(int(x) for x in train_clusters)
        if int(max_cluster_id_in_split) >= int(n_clusters):
            raise ValueError(
                "Entanglement split clusters are incompatible with the loaded index centers. "
                f"max_cluster_id_in_split={max_cluster_id_in_split}, n_clusters_in_index={int(n_clusters)}. "
                f"split_json={split_json_path}, index_dir={index_dir}"
            )

        ent_ratio_in_split = None
        try:
            ent_ratio_in_split = float(((split_info.get("epsilon") or {}).get("entangled_ratio")))
        except Exception:
            ent_ratio_in_split = None

        print(
            f"entanglement_split={split_json_path} | split_tag={tag} | entangled_clusters={len(entangled_clusters)} | train_clusters={len(train_clusters)}"
        )

        qas = QueryAdaptiveScorer(cfg)
        cur_device = device
        try:
            qas = qas.to(cur_device)
        except torch.cuda.OutOfMemoryError:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            cur_device = "cpu"
            qas = qas.to("cpu")

        try:
            train_history, train_batch_history, train_pairs = train_hypernetwork_qas(
                qas=qas,
                centers=summary_embs,
                corpus_embs=corpus_embs,
                token_cache=token_cache,
                token_seqlen=token_seqlen,
                primary_cluster_ids=primary_cluster_ids,
                entangled_clusters=train_clusters,
                query_gold_node_map=query_gold_node_map,
                query_neg_doc_indices_by_qi=query_neg_doc_indices_by_qi,
                query_sent_embs=query_sent_embs,
                tau_t=qas_tau_t,
                num_neg=qas_num_neg,
                pair_batch_size=qas_pair_batch,
                max_epochs=qas_train_epochs,
                lr=qas_lr,
                weight_decay=qas_wd,
                max_pairs=qas_max_pairs,
            )
        except torch.cuda.OutOfMemoryError:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            qas = qas.to("cpu")
            cur_device = "cpu"
            train_history, train_batch_history, train_pairs = train_hypernetwork_qas(
                qas=qas,
                centers=summary_embs,
                corpus_embs=corpus_embs,
                token_cache=token_cache,
                token_seqlen=token_seqlen,
                primary_cluster_ids=primary_cluster_ids,
                entangled_clusters=train_clusters,
                query_gold_node_map=query_gold_node_map,
                query_neg_doc_indices_by_qi=query_neg_doc_indices_by_qi,
                query_sent_embs=query_sent_embs,
                tau_t=qas_tau_t,
                num_neg=qas_num_neg,
                pair_batch_size=qas_pair_batch,
                max_epochs=qas_train_epochs,
                lr=qas_lr,
                weight_decay=qas_wd,
                max_pairs=qas_max_pairs,
            )

        final_epoch_loss = None
        if train_history:
            for r in reversed(train_history):
                if isinstance(r, dict) and "avg_loss" in r:
                    final_epoch_loss = float(r["avg_loss"])
                    break
        if final_epoch_loss is not None:
            print(f"[QAS Train][{tag}] final_epoch_avg_loss={final_epoch_loss:.6f}")
        else:
            print(f"[QAS Train][{tag}] final_epoch_avg_loss=NA")

        loss_record_prefix = (
            f"{config.DATASET_NAME}_qas_train_{config.MODEL_NAME}"
            f"_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}"
            f"_tauT{qas_tau_t}_neg{qas_num_neg}_maxlen{cache_max_len}"
            f"_{tag}"
        )
        loss_record_json = os.path.join(LOSS_OUTPUT_DIR, f"{loss_record_prefix}.json")
        loss_record_csv = os.path.join(LOSS_OUTPUT_DIR, f"{loss_record_prefix}_batch_loss.csv")
        loss_plot_png = os.path.join(LOSS_OUTPUT_DIR, f"{loss_record_prefix}.png")
        loss_plot_svg = os.path.join(LOSS_OUTPUT_DIR, f"{loss_record_prefix}.svg")
        with open(loss_record_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_name": config.DATASET_NAME,
                    "model_name": config.MODEL_NAME,
                    "n_clusters": int(n_clusters),
                    "soft_k": int(soft_k),
                    "delta": float(delta),
                    "tau": float(temperature),
                    "split_tag": str(tag),
                    "token_cache": {"path": token_cache_path, "max_len": int(cache_max_len)},
                    "token_seqlen_cache": {"path": seqlen_cache_path},
                    "entanglement_split": {
                        "path": split_json_path,
                        "tag": str(tag),
                        "entangled_ratio": float(ent_ratio_in_split) if ent_ratio_in_split is not None else None,
                        "train_clusters": int(len(train_clusters)),
                        "entangled_clusters": int(len(entangled_clusters)),
                    },
                    "query_gold_node_map": {"path": query_gold_map_path},
                    "train": {
                        "loss": "InfoNCE (softmax CE over [pos + sampled negs in cluster])",
                        "tau_t": float(qas_tau_t),
                        "num_neg": int(qas_num_neg),
                        "pair_batch_size": int(qas_pair_batch),
                        "epochs": int(qas_train_epochs),
                        "lr": float(qas_lr),
                        "weight_decay": float(qas_wd),
                        "max_pairs": int(qas_max_pairs),
                        "n_pairs": int(len(train_pairs)),
                        "neg_mode": "intersection+hard-fallback",
                    },
                    "epoch_history": train_history,
                    "batch_history": train_batch_history,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(loss_record_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["global_step", "epoch", "loss", "pair_batch_size"])
            for r in train_batch_history:
                w.writerow(
                    [
                        int(r.get("global_step", 0)),
                        int(r.get("epoch", 0)),
                        float(r.get("loss", 0.0)),
                        int(r.get("pair_batch_size", 0)),
                    ]
                )
        save_loss_line_plots(train_history, train_batch_history, loss_plot_png, loss_plot_svg)
        print(f"[{tag}] qas train loss saved: {loss_record_json}")
        print(f"[{tag}] qas train batch loss csv saved: {loss_record_csv}")
        print(f"[{tag}] qas train loss plot saved: {loss_plot_png}")

        ckpt_path = os.path.join(
            config.OUTPUT_DIR,
            f"{config.DATASET_NAME}_qas_hyper_{config.MODEL_NAME}_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}_{tag}.pt",
        )
        ckpt_obj = {
            "dataset_name": config.DATASET_NAME,
            "model_name": config.MODEL_NAME,
            "cfg": {"d_model": int(cfg.d_model), "h_hn": int(cfg.h_hn), "n_layers": int(cfg.n_layers)},
            "split_tag": str(tag),
            "entanglement_split": {"path": split_json_path, "tag": str(tag), "entangled_ratio": ent_ratio_in_split},
            "train": {
                "tau_t": float(qas_tau_t),
                "num_neg": int(qas_num_neg),
                "pair_batch": int(qas_pair_batch),
                "epochs": int(qas_train_epochs),
                "lr": float(qas_lr),
                "weight_decay": float(qas_wd),
                "max_pairs": int(qas_max_pairs),
                "n_pairs": int(len(train_pairs)),
            },
            "history": train_history,
            "state_dict": qas.state_dict(),
        }
        torch.save(ckpt_obj, ckpt_path)
        print(f"[{tag}] qas hypernetwork checkpoint saved: {ckpt_path}")

        if ent_ratio_in_split is not None and abs(float(ent_ratio_in_split) - float(baseline_entangled_ratio)) < 1e-12:
            ckpt_path_default = os.path.join(
                config.OUTPUT_DIR,
                f"{config.DATASET_NAME}_qas_hyper_{config.MODEL_NAME}_k{int(n_clusters)}_soft{int(soft_k)}_delta{delta}_tau{temperature}.pt",
            )
            torch.save(ckpt_obj, ckpt_path_default)
            print(f"[{tag}] qas hypernetwork checkpoint saved with default name: {ckpt_path_default}")

        if int(sample_queries) > 0 and int(top_passages) > 0:
            chosen_entangled = int(train_clusters[0])
            cand_docs = cluster_to_docs.get(chosen_entangled, [])[:top_passages]
            if not cand_docs:
                raise RuntimeError(f"Entangled cluster {chosen_entangled} has no docs.")
            mu_c = torch.from_numpy(summary_embs[chosen_entangled]).to(device=cur_device, dtype=torch.float32)
            doc_vecs = torch.from_numpy(corpus_embs[cand_docs]).to(device=cur_device, dtype=torch.float32)

            for qi in range(min(sample_queries, len(query_samples))):
                q_text = get_query_text(query_samples[qi])
                if getattr(config, "QUERY_INSTRUCTION", None):
                    q_text = str(config.QUERY_INSTRUCTION) + str(q_text)

                if qi >= int(token_cache.shape[0]):
                    raise IndexError(f"Query index {qi} out of token_cache range {int(token_cache.shape[0])}.")
                if qi >= int(token_seqlen.shape[0]):
                    raise IndexError(f"Query index {qi} out of token_seqlen range {int(token_seqlen.shape[0])}.")
                seq_len = int(token_seqlen[qi])
                seq_len = max(1, min(seq_len, cache_max_len))
                eq = token_cache[qi, :seq_len, :]
                eq_tokens = torch.from_numpy(np.asarray(eq, dtype=np.float32)).to(cur_device)
                scores = []
                with torch.inference_mode():
                    s = qas(eq_tokens, mu_c, doc_vecs)
                    scores = [float(x) for x in s.detach().cpu().numpy().tolist()]
                order2 = np.argsort(np.array(scores))[::-1].tolist()
                topk = order2[: min(5, len(order2))]
                print(f"\n[{tag}][Query {qi}] {q_text[:120]}")
                print(f"  entangled_cluster={chosen_entangled} | candidates={len(cand_docs)}")
                for rank, idx2 in enumerate(topk, start=1):
                    print(f"  rank={rank} doc_id={cand_docs[idx2]} score={scores[idx2]:.6f}")

    print(f"done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()