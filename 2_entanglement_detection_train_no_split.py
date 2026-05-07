import json
import os
import time
import importlib

import numpy as np
import torch
import torch.nn.functional as F


config = importlib.import_module("0_config")


N_CLUSTERS = int(getattr(config, "N_CLUSTERS", 300))
SOFT_K = int(getattr(config, "SOFT_K", 3))

DELTA = float(os.environ.get("ENTANGLEMENT_DELTA", str(getattr(config, "ENTANGLEMENT_DELTA", 0.05))))

TEMPERATURE = float(os.environ.get("ENTANGLEMENT_TAU", str(getattr(config, "ENTANGLEMENT_TAU", 0.07))))

ENTANGLED_RATIO = float(os.environ.get("ENTANGLED_RATIO", str(getattr(config, "ENTANGLED_RATIO", 0.1))))

ENTANGLEMENT_THRESHOLD = os.environ.get("ENTANGLEMENT_THRESHOLD")
if ENTANGLEMENT_THRESHOLD is None:
    ENTANGLEMENT_THRESHOLD = getattr(config, "ENTANGLEMENT_THRESHOLD", None)
ENTANGLEMENT_THRESHOLD = float(ENTANGLEMENT_THRESHOLD) if ENTANGLEMENT_THRESHOLD is not None else None


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
            f"Index not found. Expected: {desired_centers}. "
            f"Please run hierarchical index construction or set config.N_CLUSTERS/SOFT_K to match an existing index."
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    k_val, picked = candidates[0]
    print(f"[Index] Index with k{int(n_clusters)} was not found. Automatically falling back to existing index directory: {picked}")
    return picked, int(k_val)


INDEX_DIR, N_CLUSTERS = resolve_index_dir(
    base_dir=config.BASE_DIR,
    dataset_name=config.DATASET_NAME,
    model_name=config.MODEL_NAME,
    n_clusters=N_CLUSTERS,
    soft_k=SOFT_K,
)
INDEX_CENTERS_PATH = os.path.join(INDEX_DIR, "centers.npy")
INDEX_DOC_TOP_CLUSTERS_PATH = os.path.join(INDEX_DIR, "doc_top_clusters.npy")

OUTPUT_DIR = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_PREFIX = f"{config.DATASET_NAME}_entanglement_{config.MODEL_NAME}_k{N_CLUSTERS}_soft{SOFT_K}_delta{DELTA}_tau{TEMPERATURE}"
ROUTING_TRAINED_CENTERS_NPY = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_routing_trained_centers.npy")
ENTANGLEMENT_SCORES_NPY = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}.npy")
ENTANGLEMENT_DETAILS_JSON = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}.json")
ENTANGLEMENT_SPLIT_JSON = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_split.json")

LOSS_OUTPUT_DIR = str(getattr(config, "LOSS_OUTPUT_DIR", os.path.join(config.BASE_DIR, "result_loss")))
os.makedirs(LOSS_OUTPUT_DIR, exist_ok=True)
LOSS_RECORD_JSON = os.path.join(LOSS_OUTPUT_DIR, f"{OUT_PREFIX}_routing_train_loss.json")
QUERY_GOLD_NODE_MAP_JSON = os.path.join(LOSS_OUTPUT_DIR, f"{OUT_PREFIX}_query_gold_node_map.json")


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


def make_unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 2
    while True:
        cand = f"{base}_v{i}{ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


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


def build_query_gold_node_map(query_samples, corpus_samples, doc_top_clusters):
    corpus_text_to_indices = {}
    for idx, doc in enumerate(corpus_samples):
        doc_text = extract_doc_text(doc)
        norm = config.normalize_text_for_eval(doc_text)
        if norm not in corpus_text_to_indices:
            corpus_text_to_indices[norm] = []
        corpus_text_to_indices[norm].append(int(idx))

    gold_docs = config.build_gold_docs(query_samples)
    if len(gold_docs) != len(query_samples):
        raise ValueError(f"gold_docs length {len(gold_docs)} != query_samples length {len(query_samples)}")

    entries = []
    unmatched_gold_docs = 0
    unmatched_queries = 0
    multi_cluster_queries = 0

    primary = doc_top_clusters[:, 0].astype(np.int32)
    for qi, (sample, q_gold_docs) in enumerate(zip(query_samples, gold_docs)):
        gold_doc_indices = []
        for d in q_gold_docs:
            norm_d = config.normalize_text_for_eval(d)
            idx_list = corpus_text_to_indices.get(norm_d)
            if not idx_list:
                unmatched_gold_docs += 1
                continue
            gold_doc_indices.extend(idx_list)

        gold_doc_indices = sorted(set(int(x) for x in gold_doc_indices))
        if not gold_doc_indices:
            unmatched_queries += 1
            cluster_ids = []
            label_cluster_id = None
        else:
            gold_doc_cluster_ids = [int(primary[i]) for i in gold_doc_indices]
            cluster_ids = sorted(set(gold_doc_cluster_ids))
            if len(cluster_ids) > 1:
                multi_cluster_queries += 1

            counts = {}
            for cid in gold_doc_cluster_ids:
                counts[cid] = counts.get(cid, 0) + 1
            max_cnt = max(counts.values())
            candidates = [cid for cid, cnt in counts.items() if cnt == max_cnt]
            label_cluster_id = int(min(candidates))

        qid = sample.get("_id")
        if qid is None:
            qid = sample.get("id")

        entries.append(
            {
                "query_index": int(qi),
                "query_id": qid,
                "gold_doc_indices": gold_doc_indices,
                "gold_cluster_ids": cluster_ids,
                "label_cluster_id": label_cluster_id,
            }
        )

    return {
        "dataset_name": config.DATASET_NAME,
        "model_name": config.MODEL_NAME,
        "n_queries": int(len(query_samples)),
        "n_corpus_docs": int(len(corpus_samples)),
        "unmatched_queries": int(unmatched_queries),
        "unmatched_gold_docs": int(unmatched_gold_docs),
        "multi_cluster_queries": int(multi_cluster_queries),
        "entries": entries,
    }


def build_query_training_labels(query_gold_node_map, n_clusters):
    entries = query_gold_node_map.get("entries", [])
    n = int(len(entries))
    targets = np.full((n,), -1, dtype=np.int64)
    valid = np.zeros((n,), dtype=bool)

    invalid_label = 0
    for i, e in enumerate(entries):
        cid = e.get("label_cluster_id")
        if cid is None:
            invalid_label += 1
            continue
        cid = int(cid)
        if cid < 0 or cid >= int(n_clusters):
            invalid_label += 1
            continue
        targets[i] = cid
        valid[i] = True

    return targets, valid, {"invalid_label_queries": int(invalid_label), "total_queries": int(n)}


def build_query_training_multi_labels(query_gold_node_map, n_clusters):
    entries = query_gold_node_map.get("entries", [])
    n = int(len(entries))
    pos_lists = [None] * n
    valid = np.zeros((n,), dtype=bool)
    invalid = 0
    multi = 0

    for i, e in enumerate(entries):
        cids = e.get("gold_cluster_ids") or []
        cids2 = []
        for cid in cids:
            try:
                cid = int(cid)
            except Exception:
                continue
            if 0 <= cid < int(n_clusters):
                cids2.append(cid)
        cids2 = sorted(set(cids2))
        if not cids2:
            invalid += 1
            continue
        if len(cids2) > 1:
            multi += 1
        pos_lists[i] = cids2
        valid[i] = True

    return pos_lists, valid, {"invalid_label_queries": int(invalid), "multi_label_queries": int(multi), "total_queries": int(n)}


def l2_normalize(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


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


def save_loss_line_plots(loss_history, batch_loss_history, out_png, out_svg):
    epoch_x = []
    epoch_y = []
    for r in loss_history:
        if isinstance(r, dict) and "epoch" in r and "avg_loss" in r:
            epoch_x.append(int(r["epoch"]))
            epoch_y.append(float(r["avg_loss"]))

    batch_x = []
    batch_y = []
    for r in batch_loss_history:
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
            axes[0].set_title("Epoch Avg InfoNCE Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].set_axis_off()

        if batch_x:
            bx, by = _downsample_xy(batch_x, batch_y, max_points=5000)
            axes[1].plot(bx, by, linewidth=1.0)
            axes[1].set_title("Batch InfoNCE Loss")
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
    bx, by = _downsample_xy(batch_x, batch_y, max_points=5000)
    batch_svg = _svg_plot(bx, by, pad, pad * 2 + panel_h, panel_w, panel_h)

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />'
        f'<text x="{pad}" y="{pad - 20}" font-size="16">Epoch Avg InfoNCE Loss</text>'
        f'<text x="{pad}" y="{pad * 2 + panel_h - 20}" font-size="16">Batch InfoNCE Loss</text>'
        f'<rect x="{pad}" y="{pad}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#cccccc" />'
        f'<rect x="{pad}" y="{pad * 2 + panel_h}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#cccccc" />'
        f"{epoch_svg}{batch_svg}</svg>"
    )
    if out_svg:
        with open(out_svg, "w", encoding="utf-8") as f:
            f.write(svg)


def train_routing_model(
    query_embs,
    query_targets,
    initial_centers,
    temperature,
    max_epochs=400,
    batch_size=4096,
    lr=0.01,
    rebound_tol=1e-4,
    min_epochs_before_oscillation=10,
    oscillation_flips=4,
    query_pos_clusters=None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting routing model training (Device: {device})...")

    centers_param = torch.nn.Parameter(torch.from_numpy(initial_centers).to(device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([centers_param], lr=lr)

    dataset_size = int(query_embs.shape[0])
    loss_history = []
    batch_loss_history = []
    global_step = 0
    best_loss = float("inf")
    best_epoch = 0
    best_centers = None
    prev_loss = None
    last_sign = 0
    sign_flips = 0

    for epoch in range(max_epochs):
        permutation = np.random.permutation(dataset_size)
        total_loss = 0.0
        steps = 0

        for i in range(0, dataset_size, batch_size):
            indices = permutation[i:i+batch_size]

            v = torch.from_numpy(query_embs[indices]).to(device=device, dtype=torch.float32)
            v = F.normalize(v, p=2, dim=1)

            optimizer.zero_grad()

            norm_centers = F.normalize(centers_param, p=2, dim=1)
            logits = (v @ norm_centers.T) / float(temperature)

            if query_pos_clusters is None:
                targets = torch.from_numpy(query_targets[indices]).to(device=device, dtype=torch.long)
                loss = F.cross_entropy(logits, targets)
            else:
                mask = torch.zeros((int(len(indices)), int(logits.shape[1])), device=device, dtype=torch.bool)
                for bi, qidx in enumerate(indices.tolist()):
                    pos = query_pos_clusters[int(qidx)]
                    if not pos:
                        continue
                    mask[bi, torch.tensor(pos, device=device, dtype=torch.long)] = True

                denom = torch.logsumexp(logits, dim=1)
                pos_logits = logits.masked_fill(~mask, float("-inf"))
                num = torch.logsumexp(pos_logits, dim=1)
                loss = -(num - denom)
                finite = torch.isfinite(loss)
                loss = loss[finite].mean() if int(finite.sum().item()) > 0 else torch.tensor(0.0, device=device)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            batch_loss_history.append(
                {
                    "epoch": int(epoch + 1),
                    "step_in_epoch": int(steps),
                    "global_step": int(global_step),
                    "batch_size": int(len(indices)),
                    "loss": float(loss.item()),
                }
            )
            global_step += 1

        avg_loss = total_loss / steps
        print(f"  Epoch {epoch+1:02d}/{max_epochs} - InfoNCE Loss: {avg_loss:.4f}")
        loss_history.append(
            {
                "epoch": int(epoch + 1),
                "avg_loss": float(avg_loss),
                "steps": int(steps),
            }
        )

        if avg_loss < float(best_loss):
            best_loss = float(avg_loss)
            best_epoch = int(epoch + 1)
            best_centers = centers_param.detach().clone()

        if prev_loss is not None:
            delta = float(avg_loss) - float(prev_loss)
            if delta > float(rebound_tol):
                sign = 1
            elif delta < -float(rebound_tol):
                sign = -1
            else:
                sign = 0

            if sign != 0 and last_sign != 0 and sign != last_sign:
                sign_flips += 1

            if sign != 0:
                last_sign = sign

            if int(epoch + 1) >= int(min_epochs_before_oscillation) and int(sign_flips) >= int(oscillation_flips):
                print(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Loss shows persistent rebound oscillation "
                    f"(rebound_tol={rebound_tol}, flips={sign_flips}); stopping training."
                )
                break

        prev_loss = float(avg_loss)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Routing model training completed.")

    if best_centers is None:
        best_centers = centers_param.detach()
        best_epoch = int(max_epochs)

    trained_centers = F.normalize(best_centers, p=2, dim=1).cpu().numpy()
    loss_history.append(
        {
            "best_epoch": int(best_epoch),
            "best_loss": float(best_loss),
            "sign_flips": int(sign_flips),
        }
    )
    return trained_centers, loss_history, batch_loss_history


def compute_boundary_mask(primary_cluster_ids, corpus_embs, centers, delta, batch_size=8192):
    n_docs = int(corpus_embs.shape[0])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    centers_t = torch.from_numpy(centers).to(device=device, dtype=torch.float32)
    centers_t = F.normalize(centers_t, p=2, dim=1)

    boundary = np.zeros((n_docs,), dtype=bool)
    competitor_ids = np.full((n_docs,), -1, dtype=np.int32)

    start = 0
    while start < n_docs:
        end = min(n_docs, start + batch_size)
        v = torch.from_numpy(corpus_embs[start:end]).to(device=device, dtype=torch.float32)
        v = F.normalize(v, p=2, dim=1)

        labels = torch.from_numpy(primary_cluster_ids[start:end]).to(device=device, dtype=torch.long)

        sims = v @ centers_t.T
        top2_idx = torch.topk(sims, k=2, dim=1).indices
        best = top2_idx[:, 0]
        second = top2_idx[:, 1]
        comp = torch.where(best != labels, best, second)

        mu_label = centers_t[labels]
        mu_comp = centers_t[comp]

        cos_label = (v * mu_label).sum(dim=1).clamp(-1.0, 1.0)
        cos_comp = (v * mu_comp).sum(dim=1).clamp(-1.0, 1.0)

        dist_label = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_label, min=0.0))
        dist_comp = torch.sqrt(torch.clamp(2.0 - 2.0 * cos_comp, min=0.0))

        bmask = (dist_comp < (dist_label + float(delta))).detach().cpu().numpy().astype(bool)
        boundary[start:end] = bmask
        competitor_ids[start:end] = comp.detach().cpu().numpy().astype(np.int32)

        start = end

    return boundary, competitor_ids


def compute_entanglement_scores(corpus_embs, centers, primary_cluster_ids, boundary_mask, temperature, batch_size=4096):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    centers_t = torch.from_numpy(centers).to(device=device, dtype=torch.float32)
    centers_t = F.normalize(centers_t, p=2, dim=1)

    sums = np.zeros((centers.shape[0],), dtype=np.float64)
    counts = np.zeros((centers.shape[0],), dtype=np.int64)

    boundary_idx = np.where(boundary_mask)[0]
    if boundary_idx.size == 0:
        return sums.astype(np.float32), counts

    start = 0
    while start < boundary_idx.size:
        end = min(boundary_idx.size, start + batch_size)
        idx = boundary_idx[start:end]
        labels = primary_cluster_ids[idx]

        v = torch.from_numpy(corpus_embs[idx]).to(device=device, dtype=torch.float32)
        v = F.normalize(v, p=2, dim=1)
        v.requires_grad_(True)

        logits = (v @ centers_t.T) / float(temperature)
        targets = torch.from_numpy(labels).to(device=device, dtype=torch.long)
        losses = F.cross_entropy(logits, targets, reduction="none")
        losses.sum().backward()

        grads = v.grad.detach()
        grad_norm_sq = (grads * grads).sum(dim=1).cpu().numpy().astype(np.float64)

        for cid, g2 in zip(labels.tolist(), grad_norm_sq.tolist()):
            sums[int(cid)] += float(g2)
            counts[int(cid)] += 1

        start = end

    scores = np.zeros_like(sums, dtype=np.float32)
    nonzero = counts > 0
    scores[nonzero] = (sums[nonzero] / counts[nonzero]).astype(np.float32)
    return scores, counts


def compute_intra_scores(corpus_embs, centers, primary_cluster_ids, batch_size=16384):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_docs = int(corpus_embs.shape[0])
    n_clusters = int(centers.shape[0])

    centers_t = torch.from_numpy(centers).to(device=device, dtype=torch.float32)
    centers_t = F.normalize(centers_t, p=2, dim=1)

    sums_cos = torch.zeros((n_clusters,), dtype=torch.float64, device=device)
    sums_dist = torch.zeros((n_clusters,), dtype=torch.float64, device=device)
    counts = torch.zeros((n_clusters,), dtype=torch.float64, device=device)

    start = 0
    while start < n_docs:
        end = min(n_docs, start + batch_size)
        v = torch.from_numpy(corpus_embs[start:end]).to(device=device, dtype=torch.float32)
        v = F.normalize(v, p=2, dim=1)
        labels = torch.from_numpy(primary_cluster_ids[start:end]).to(device=device, dtype=torch.long)

        mu = centers_t[labels]
        cos = (v * mu).sum(dim=1).clamp(-1.0, 1.0).to(dtype=torch.float64)
        dist = torch.sqrt(torch.clamp(2.0 - 2.0 * cos, min=0.0))

        sums_cos.scatter_add_(0, labels, cos)
        sums_dist.scatter_add_(0, labels, dist)
        counts.scatter_add_(0, labels, torch.ones_like(dist))

        start = end

    counts_cpu = counts.detach().cpu().numpy().astype(np.int64)
    mean_cos = np.zeros((n_clusters,), dtype=np.float32)
    mean_dist = np.zeros((n_clusters,), dtype=np.float32)
    nonzero = counts_cpu > 0
    if np.any(nonzero):
        sums_cos_cpu = sums_cos.detach().cpu().numpy().astype(np.float64)
        sums_dist_cpu = sums_dist.detach().cpu().numpy().astype(np.float64)
        mean_cos[nonzero] = (sums_cos_cpu[nonzero] / counts_cpu[nonzero]).astype(np.float32)
        mean_dist[nonzero] = (sums_dist_cpu[nonzero] / counts_cpu[nonzero]).astype(np.float32)
    return mean_dist, mean_cos, counts_cpu


def decide_threshold_and_split(scores, entanglement_threshold, ratio):
    if entanglement_threshold is not None:
        threshold = float(entanglement_threshold)
        entangled_mask = scores >= threshold
        return threshold, entangled_mask, "paper_threshold"

    n = int(scores.shape[0])
    ratio = float(ratio)
    ratio = min(max(ratio, 0.0), 1.0)

    n_entangled = int(round(n * ratio))
    n_entangled = max(0, min(n, n_entangled))

    if n_entangled == 0:
        return float("inf"), np.zeros((n,), dtype=bool), "ratio"
    if n_entangled == n:
        return float("-inf"), np.ones((n,), dtype=bool), "ratio"

    idx = np.arange(n, dtype=np.int32)
    order = np.lexsort((idx, -scores.astype(np.float64)))
    top_idx = order[:n_entangled]
    threshold = float(scores[top_idx].min())

    entangled_mask = np.zeros((n,), dtype=bool)
    entangled_mask[top_idx] = True
    return threshold, entangled_mask, "ratio"


def main():
    t0 = time.time()

    corpus_embs = load_numpy(config.CORPUS_EMBEDDING_PATH).astype(np.float32)
    corpus_embs = l2_normalize(corpus_embs, axis=1).astype(np.float32)

    centers = load_numpy(INDEX_CENTERS_PATH).astype(np.float32)
    centers = l2_normalize(centers, axis=1).astype(np.float32)

    doc_top_clusters = load_numpy(INDEX_DOC_TOP_CLUSTERS_PATH)
    if doc_top_clusters.ndim != 2 or doc_top_clusters.shape[1] < 2:
        raise ValueError(f"doc_top_clusters must have shape (n_docs, >=2), got {doc_top_clusters.shape}")

    query_samples = load_json_or_jsonl(config.QUERY_DATA_PATH)
    corpus_samples = load_json_or_jsonl(config.CORPUS_JSONL_PATH)
    query_embs = np.load(config.QUERY_EMBEDDING_PATH, mmap_mode="r")
    if int(query_embs.shape[0]) != int(len(query_samples)):
        raise ValueError(
            f"query_emb count {int(query_embs.shape[0])} != query_samples count {int(len(query_samples))} "
            f"(path={config.QUERY_EMBEDDING_PATH})"
        )

    query_gold_node_map = build_query_gold_node_map(
        query_samples=query_samples,
        corpus_samples=corpus_samples,
        doc_top_clusters=doc_top_clusters,
    )
    with open(QUERY_GOLD_NODE_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump(query_gold_node_map, f, ensure_ascii=False, indent=2)
    print(f"query gold node map saved: {QUERY_GOLD_NODE_MAP_JSON}")

    route_label_mode = "multi"
    query_pos_clusters_all, query_valid_mask, query_label_stats = build_query_training_multi_labels(
        query_gold_node_map=query_gold_node_map,
        n_clusters=int(centers.shape[0]),
    )
    valid_query_idx = np.where(query_valid_mask)[0]
    query_pos_clusters = [query_pos_clusters_all[int(i)] for i in valid_query_idx.tolist()]
    query_targets = None

    if valid_query_idx.size == 0:
        raise ValueError("No valid query -> cluster labels. Cannot train routing model.")
    query_embs_valid = np.asarray(query_embs[valid_query_idx], dtype=np.float32)

    primary_cluster_ids = doc_top_clusters[:, 0].astype(np.int32)

    route_max_epochs = int(os.environ.get("ROUTE_MAX_EPOCHS", str(getattr(config, "ROUTE_MAX_EPOCHS", 300))))
    route_batch_size = int(os.environ.get("ROUTE_BATCH_SIZE", str(getattr(config, "ROUTE_BATCH_SIZE", 256))))
    route_lr = float(os.environ.get("ROUTE_LR", str(getattr(config, "ROUTE_LR", 0.05))))
    route_rebound_tol = float(os.environ.get("ROUTE_REBOUND_TOL", str(getattr(config, "ROUTE_REBOUND_TOL", 1e-4))))
    route_min_epochs_before_oscillation = int(os.environ.get("ROUTE_MIN_EPOCHS", str(getattr(config, "ROUTE_MIN_EPOCHS", 10))))
    route_oscillation_flips = int(os.environ.get("ROUTE_OSCILLATION_FLIPS", str(getattr(config, "ROUTE_OSCILLATION_FLIPS", 4))))

    trained_centers, loss_history, batch_loss_history = train_routing_model(
        query_embs=query_embs_valid,
        query_targets=query_targets if query_targets is not None else np.zeros((int(query_embs_valid.shape[0]),), dtype=np.int64),
        query_pos_clusters=query_pos_clusters,
        initial_centers=centers,
        temperature=TEMPERATURE,
        max_epochs=route_max_epochs,
        batch_size=route_batch_size,
        lr=route_lr,
        rebound_tol=route_rebound_tol,
        min_epochs_before_oscillation=route_min_epochs_before_oscillation,
        oscillation_flips=route_oscillation_flips,
    )
    np.save(ROUTING_TRAINED_CENTERS_NPY, trained_centers.astype(np.float32))
    print(f"routing trained centers saved: {ROUTING_TRAINED_CENTERS_NPY}")

    routed_centers_index_path = make_unique_path(os.path.join(INDEX_DIR, "centers_routing_trained.npy"))
    np.save(routed_centers_index_path, trained_centers.astype(np.float32))
    print(f"routing trained centers saved in index: {routed_centers_index_path}")

    n_clusters_int = int(trained_centers.shape[0])
    cluster_to_docs = {int(i): [] for i in range(n_clusters_int)}
    for doc_id, cid in enumerate(primary_cluster_ids):
        cluster_to_docs[int(cid)].append(int(doc_id))
    routed_cluster_to_docs_path = make_unique_path(os.path.join(INDEX_DIR, "cluster_to_docs_routing_trained.json"))
    with open(routed_cluster_to_docs_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in cluster_to_docs.items()}, f, ensure_ascii=False, indent=2)
    print(f"routing cluster_to_docs saved in index: {routed_cluster_to_docs_path}")

    with open(LOSS_RECORD_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": config.DATASET_NAME,
                "model_name": config.MODEL_NAME,
                "n_clusters": int(centers.shape[0]),
                "soft_k": int(doc_top_clusters.shape[1]),
                "temperature": float(TEMPERATURE),
                "train_mode": "query_to_node",
                "label_mode": str(route_label_mode),
                "n_train_queries": int(query_embs_valid.shape[0]),
                "query_label_stats": query_label_stats,
                "train": {
                    "optimizer": "adam",
                    "lr": float(route_lr),
                    "max_epochs": int(route_max_epochs),
                    "batch_size": int(route_batch_size),
                    "rebound_tol": float(route_rebound_tol),
                    "min_epochs_before_oscillation": int(route_min_epochs_before_oscillation),
                    "oscillation_flips": int(route_oscillation_flips),
                },
                "loss_history": loss_history,
                "batch_loss_history": batch_loss_history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"routing train loss saved: {LOSS_RECORD_JSON}")
    loss_plot_png = os.path.join(LOSS_OUTPUT_DIR, f"{OUT_PREFIX}_routing_train_loss.png")
    loss_plot_svg = os.path.join(LOSS_OUTPUT_DIR, f"{OUT_PREFIX}_routing_train_loss.svg")
    save_loss_line_plots(loss_history, batch_loss_history, loss_plot_png, loss_plot_svg)
    print(f"routing train loss plot saved: {loss_plot_png}")

    boundary_mask, competitor_ids = compute_boundary_mask(
        primary_cluster_ids=primary_cluster_ids,
        corpus_embs=corpus_embs,
        centers=trained_centers,
        delta=DELTA,
    )

    scores, boundary_counts = compute_entanglement_scores(
        corpus_embs=corpus_embs,
        centers=trained_centers,
        primary_cluster_ids=primary_cluster_ids,
        boundary_mask=boundary_mask,
        temperature=TEMPERATURE,
    )

    np.save(ENTANGLEMENT_SCORES_NPY, scores)

    cluster_sizes = np.bincount(primary_cluster_ids, minlength=centers.shape[0]).astype(np.int64)

    ratio_grid = [
        ("c7e3", 0.3),
        ("c3e1", 0.25),
        ("c8e2", 0.2),
        ("c9e1", 0.1),
        ("c95e5", 0.05),
    ]

    qas_min_ratio_env = os.environ.get("QAS_MIN_RATIO")

    idx = np.arange(int(scores.shape[0]), dtype=np.int32)
    order = np.lexsort((idx, -scores.astype(np.float64)))

    for tag, ent_ratio in ratio_grid:
        eps_threshold, entangled_by_eps, eps_split_rule = decide_threshold_and_split(
            scores,
            None,
            float(ent_ratio),
        )

        qas_mask = entangled_by_eps.copy()
        if qas_min_ratio_env is not None:
            qas_min_ratio = float(qas_min_ratio_env)
        else:
            qas_min_ratio = float(ent_ratio)
        qas_min_ratio = min(max(qas_min_ratio, 0.0), 1.0)
        qas_min_count = int(round(float(centers.shape[0]) * qas_min_ratio))
        qas_min_count = max(0, min(int(centers.shape[0]), qas_min_count))
        cur_qas = int(np.sum(qas_mask))
        if cur_qas < qas_min_count:
            need = int(qas_min_count - cur_qas)
            for cid in order.tolist():
                if need <= 0:
                    break
                if not qas_mask[int(cid)]:
                    qas_mask[int(cid)] = True
                    need -= 1
        clean_mask = np.logical_not(qas_mask)

        details = []
        for cid in range(centers.shape[0]):
            is_ent = bool(qas_mask[cid])
            ctype = "ENTANGLED" if is_ent else "CLEAN"
            details.append(
                {
                    "cluster_id": int(cid),
                    "entanglement": float(scores[cid]),
                    "boundary_count": int(boundary_counts[cid]),
                    "cluster_size": int(cluster_sizes[cid]),
                    "is_entangled": bool(is_ent),
                    "type": ctype,
                }
            )

        details_path = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_{tag}.json")
        split_path = os.path.join(OUTPUT_DIR, f"{OUT_PREFIX}_{tag}_split.json")

        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "dataset_name": config.DATASET_NAME,
                    "model_name": config.MODEL_NAME,
                    "n_clusters": int(centers.shape[0]),
                    "soft_k": int(doc_top_clusters.shape[1]),
                    "delta": float(DELTA),
                    "temperature": float(TEMPERATURE),
                    "epsilon": {
                        "entanglement_threshold": None,
                        "entangled_ratio": float(ent_ratio),
                        "non_entangled_ratio": float(1.0 - float(ent_ratio)),
                        "threshold": float(eps_threshold),
                        "split_rule": str(eps_split_rule),
                    },
                    "paths": {
                        "scores_npy": ENTANGLEMENT_SCORES_NPY,
                    },
                    "clusters": details,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        entangled_ids = [int(i) for i in np.where(qas_mask)[0].tolist()]
        clean_ids = [int(i) for i in np.where(clean_mask)[0].tolist()]
        entangled_by_eps_ids = [int(i) for i in np.where(entangled_by_eps)[0].tolist()]
        entangled_region_count = int(len(entangled_ids))
        summary_count = int(centers.shape[0])
        if entangled_region_count > 0:
            avg_passages_in_entangled = float(np.mean(cluster_sizes[qas_mask]).item())
        else:
            avg_passages_in_entangled = 0.0

        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epsilon": {
                        "threshold": float(eps_threshold),
                        "split_rule": str(eps_split_rule),
                        "entanglement_threshold": None,
                        "entangled_ratio": float(ent_ratio),
                        "non_entangled_ratio": float(1.0 - float(ent_ratio)),
                        "entangled_clusters": entangled_by_eps_ids,
                    },
                    "final_rule": "ENTANGLED if (epsilon>=thr_epsilon) else CLEAN",
                    "entangled_clusters": entangled_ids,
                    "clean_clusters": clean_ids,
                    "non_entangled_clusters": clean_ids,
                    "counts": {
                        "clean": int(len(clean_ids)),
                        "entangled": int(len(entangled_ids)),
                        "non_entangled": int(len(clean_ids)),
                        "entangled_by_epsilon": int(len(entangled_by_eps_ids)),
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        if abs(float(ent_ratio) - float(ENTANGLED_RATIO)) < 1e-12:
            with open(ENTANGLEMENT_DETAILS_JSON, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset_name": config.DATASET_NAME,
                        "model_name": config.MODEL_NAME,
                        "n_clusters": int(centers.shape[0]),
                        "soft_k": int(doc_top_clusters.shape[1]),
                        "delta": float(DELTA),
                        "temperature": float(TEMPERATURE),
                        "epsilon": {
                            "entanglement_threshold": None,
                            "entangled_ratio": float(ent_ratio),
                            "non_entangled_ratio": float(1.0 - float(ent_ratio)),
                            "threshold": float(eps_threshold),
                            "split_rule": str(eps_split_rule),
                        },
                        "paths": {
                            "scores_npy": ENTANGLEMENT_SCORES_NPY,
                        },
                        "clusters": details,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            with open(ENTANGLEMENT_SPLIT_JSON, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "epsilon": {
                            "threshold": float(eps_threshold),
                            "split_rule": str(eps_split_rule),
                            "entanglement_threshold": None,
                            "entangled_ratio": float(ent_ratio),
                            "non_entangled_ratio": float(1.0 - float(ent_ratio)),
                            "entangled_clusters": entangled_by_eps_ids,
                        },
                        "final_rule": "ENTANGLED if (epsilon>=thr_epsilon) else CLEAN",
                        "entangled_clusters": entangled_ids,
                        "clean_clusters": clean_ids,
                        "non_entangled_clusters": clean_ids,
                        "counts": {
                            "clean": int(len(clean_ids)),
                            "entangled": int(len(entangled_ids)),
                            "non_entangled": int(len(clean_ids)),
                            "entangled_by_epsilon": int(len(entangled_by_eps_ids)),
                        },
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

        print(f"[{tag}] entanglement details saved: {details_path}")
        print(f"[{tag}] entanglement split saved: {split_path}")
        print(f"[{tag}] number of ENTANGLED clusters: {len(entangled_ids)} | number of CLEAN clusters: {len(clean_ids)} | total clusters: {summary_count}")
        print(
            f"[{tag}] number of entangled regions: {entangled_region_count} | "
            f"average number of passages in entangled regions: {avg_passages_in_entangled:.4f}"
        )
        print(
            f"[{tag}] done in {time.time() - t0:.2f}s | entangled {len(entangled_ids)} / {centers.shape[0]} | "
            f"eps_thr={eps_threshold}"
        )

    print(f"entanglement scores saved: {ENTANGLEMENT_SCORES_NPY}")


if __name__ == "__main__":
    main()