import json
import os
import time
import importlib

import numpy as np
import torch
import sys
from contextlib import nullcontext


config = importlib.import_module("0_config")


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


def load_transformers_model(model_path, torch_dtype=None):
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"MODEL_PATH is not a local directory: {model_path}. "
        )
    if not os.path.exists(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"Local model directory missing config.json: {model_path}. "
            f"Please make sure the model is fully downloaded to this directory."
        )

    try:
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except Exception as e:
        raise RuntimeError(f"transformers not available: {e}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    if hasattr(cfg, "text_config") and cfg.text_config is not None:
        if hasattr(cfg.text_config, "_name_or_path"):
            cfg.text_config._name_or_path = model_path
        if hasattr(cfg.text_config, "name_or_path"):
            cfg.text_config.name_or_path = model_path
    if hasattr(cfg, "_name_or_path"):
        cfg._name_or_path = model_path
    if hasattr(cfg, "name_or_path"):
        cfg.name_or_path = model_path

    dtype = torch_dtype if torch_dtype is not None else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModel.from_pretrained(
        model_path,
        config=cfg,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def get_last_hidden_state(model, input_ids, attention_mask, device):
    if hasattr(model, "embedding_model") and model.embedding_model is not None:
        core = model.embedding_model
    else:
        core = model

    autocast_ctx = torch.autocast if device == "cuda" else nullcontext
    with autocast_ctx("cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
        out = core(input_ids=input_ids, attention_mask=attention_mask)

    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    if isinstance(out, dict) and "last_hidden_state" in out and out["last_hidden_state"] is not None:
        return out["last_hidden_state"]
    if isinstance(out, (tuple, list)) and len(out) > 0:
        return out[0]
    raise RuntimeError("Cannot obtain last_hidden_state for token embeddings.")


def main():
    t0 = time.time()
    device_pref = os.environ.get("TOKEN_CACHE_DEVICE", "cuda").strip().lower()
    if device_pref not in ("auto", "cuda", "cpu"):
        raise ValueError("TOKEN_CACHE_DEVICE must be one of: auto, cuda, cpu")
    device = "cpu"
    if device_pref == "cpu":
        device = "cpu"
    elif device_pref == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = config.DATASET_NAME
    model_name = config.MODEL_NAME

    max_len = int(os.environ.get("TOKEN_CACHE_MAXLEN", "128"))
    batch_size = int(os.environ.get("TOKEN_CACHE_BATCH_SIZE", "4"))

    out_dir = config.OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{dataset_name}_{model_name}_maxlen{max_len}_fp16.npy")
    out_seqlen_path = os.path.join(out_dir, f"{dataset_name}_{model_name}_maxlen{max_len}_seqlen.npy")

    queries = load_json_or_jsonl(config.QUERY_DATA_PATH)
    query_texts = []
    for item in queries:
        q = get_query_text(item)
        if getattr(config, "QUERY_INSTRUCTION", None):
            q = str(config.QUERY_INSTRUCTION) + str(q)
        query_texts.append(q)

    preferred_dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer, model = load_transformers_model(config.MODEL_PATH, torch_dtype=preferred_dtype)
    if device == "cuda":
        try:
            model.to("cuda")
        except torch.cuda.OutOfMemoryError:
            print("CUDA memory is running low, automatically fallback to CPU to generate token embedding cache.")
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            del model
            tokenizer, model = load_transformers_model(config.MODEL_PATH, torch_dtype=torch.float32)
            device = "cpu"
            model.to("cpu")
    else:
        model.to("cpu")
    print(f"[Info] token cache device: {device} (set TOKEN_CACHE_DEVICE=cpu|cuda|auto to override)", flush=True)

    with torch.inference_mode():
        probe = tokenizer(
            query_texts[0] if query_texts else "",
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        probe = {k: v.to(device) for k, v in probe.items()}
        h0 = get_last_hidden_state(model, probe["input_ids"], probe["attention_mask"], device=device)
        d_model = int(h0.shape[-1])

    n_queries = int(len(query_texts))
    mmap = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n_queries, max_len, d_model))
    seqlens = np.zeros((n_queries,), dtype=np.int32)

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    log_every_batches = int(os.environ.get("TOKEN_CACHE_LOG_EVERY", "10"))
    if log_every_batches <= 0:
        log_every_batches = 10
    it = range(0, n_queries, batch_size)
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=n_queries, desc="Caching token embeddings", unit="q", dynamic_ncols=True)
    processed_tokens = 0
    processed_queries = 0
    batch_idx = 0
    for start in it:
        batch_idx += 1
        batch_t0 = time.time()
        end = min(n_queries, start + batch_size)
        batch_texts = query_texts[start:end]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            h = get_last_hidden_state(model, enc["input_ids"], enc["attention_mask"], device=device)
            h = h.to(dtype=torch.float16).detach().cpu().numpy()
        mmap[start:end, :, :] = h
        attn = enc.get("attention_mask")
        if attn is None:
            raise RuntimeError("tokenizer output missing attention_mask; cannot compute seq_len.")
        seq = attn.detach().sum(dim=1).to(dtype=torch.int32)
        seqlens[start:end] = seq.cpu().numpy()

        batch_q = int(end - start)
        batch_tokens = int(seq.sum().detach().cpu().item())
        processed_queries += batch_q
        processed_tokens += batch_tokens

        if pbar is not None:
            pbar.update(batch_q)

            elapsed = time.time() - t0
            dt = max(1e-6, time.time() - batch_t0)
            qps = batch_q / dt
            tps = batch_tokens / dt
            avg_len = float(batch_tokens) / float(batch_q) if batch_q > 0 else 0.0

            postfix = {
                "device": device,
                "q/s": f"{qps:.2f}",
                "tok/s": f"{tps:.0f}",
                "avg_len": f"{avg_len:.1f}",
                "elapsed": f"{elapsed:.1f}s",
            }
            if device == "cuda" and torch.cuda.is_available():
                try:
                    free_b, total_b = torch.cuda.mem_get_info()
                    postfix["gpu_free_gb"] = f"{free_b / (1024**3):.2f}"
                    postfix["gpu_used_gb"] = f"{(total_b - free_b) / (1024**3):.2f}"
                except Exception:
                    pass
            pbar.set_postfix(postfix)
        else:
            if batch_idx == 1 or (batch_idx % log_every_batches) == 0:
                now = time.time()
                elapsed = now - t0
                dt = max(1e-6, now - batch_t0)
                qps = batch_q / dt
                tps = batch_tokens / dt
                avg_len = float(batch_tokens) / float(batch_q) if batch_q > 0 else 0.0
                pct = (processed_queries / float(n_queries)) * 100.0 if n_queries > 0 else 100.0
                msg = (
                    f"[Progress] {processed_queries}/{n_queries} ({pct:.1f}%) "
                    f"| batch={batch_idx} q={batch_q} avg_len={avg_len:.1f} "
                    f"| q/s={qps:.2f} tok/s={tps:.0f} "
                    f"| elapsed={elapsed:.1f}s device={device}"
                )
                print(msg, flush=True)
                sys.stdout.flush()
                pass

    if pbar is not None:
        pbar.close()

    del mmap
    np.save(out_seqlen_path, seqlens)

    print(f"token embedding cache saved: {out_path}")
    print(f"token seq_len cache saved: {out_seqlen_path}")
    print(f"shape: ({n_queries}, {max_len}, {d_model}) dtype=float16 device={device}")
    print(f"done in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()