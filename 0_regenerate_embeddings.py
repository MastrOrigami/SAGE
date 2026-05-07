import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel, AutoConfig
from tqdm import tqdm
import importlib

config = importlib.import_module("0_config")

MODEL_PATH = config.MODEL_PATH
CORPUS_JSON_PATH = config.CORPUS_JSONL_PATH
QUERY_JSON_PATH = config.QUERY_DATA_PATH

CORPUS_EMB_OUTPUT = config.CORPUS_EMBEDDING_PATH
QUERY_EMB_OUTPUT = config.QUERY_EMBEDDING_PATH

QUERY_INSTRUCTION = "Instruct: Given a question, retrieve passages that answer the question\nQuery: "
CORPUS_INSTRUCTION = ""

BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "2"))
SAFE_MAX_LENGTH = int(os.environ.get("EMBED_MAX_LENGTH", "1024"))

try:
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        print("Applying Monkey Patch: Adding default all_tied_weights_keys to PreTrainedModel")
        PreTrainedModel.all_tied_weights_keys = {}
    elif isinstance(getattr(PreTrainedModel, "all_tied_weights_keys", None), list):
        PreTrainedModel.all_tied_weights_keys = {}
except Exception as e:
    print(f"Monkey Patch failed: {e}")

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

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    if not os.path.isdir(MODEL_PATH) or not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        raise OSError(
            f"Invalid MODEL_PATH: {MODEL_PATH}. Expected a local directory containing config.json. "
            f"Please download the model to this path and update 0_config.py -> MODEL_PATH."
        )

    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    if hasattr(cfg, "text_config"):
        cfg.text_config._name_or_path = MODEL_PATH
    cfg._name_or_path = MODEL_PATH
    if hasattr(cfg, "use_cache"):
        cfg.use_cache = False

    device_pref = os.environ.get("EMBED_DEVICE", "auto").strip().lower()
    if device_pref not in ("auto", "cuda", "cpu"):
        raise ValueError("EMBED_DEVICE must be one of: auto, cuda, cpu")
    use_cuda = bool(torch.cuda.is_available() and device_pref in ("auto", "cuda"))
    torch_dtype = torch.float16 if use_cuda else torch.float32

    def _load(torch_dtype_local):
        m = AutoModel.from_pretrained(
            MODEL_PATH,
            config=cfg,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch_dtype_local,
            device_map=None,
        )
        if hasattr(m, "config") and hasattr(m.config, "use_cache"):
            m.config.use_cache = False
        if hasattr(m, "embedding_model") and hasattr(m.embedding_model, "config") and hasattr(m.embedding_model.config, "use_cache"):
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

def get_batch_embeddings(model, texts, instruction, batch_size=BATCH_SIZE):

    embeddings = []
    
    i = 0
    pbar = tqdm(total=len(texts), desc="Encoding")
    try:
        while i < len(texts):
            bs = int(batch_size)
            if bs <= 0:
                bs = 1
            batch_texts = texts[i : i + bs]
            try:
                with torch.no_grad():
                    batch_embs = model.encode(
                        batch_texts,
                        instruction=instruction,
                        max_length=SAFE_MAX_LENGTH,
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
                        max_length=SAFE_MAX_LENGTH,
                    )
                    batch_embs = F.normalize(batch_embs, p=2, dim=1)
                    embeddings.append(batch_embs.cpu().numpy())
                i += len(batch_texts)
                pbar.update(len(batch_texts))
    finally:
        pbar.close()
            
    return np.concatenate(embeddings, axis=0)

def format_corpus_text(item):

    title = item.get("title", "")
    text = item.get("text", "")
    return f"{title} {text}".strip()

def load_corpus_texts(path):

    print(f"Loading corpus from {path}...")
    texts = []
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
        
    with open(path, 'r') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            data = json.load(f)
            for item in data:
                texts.append(format_corpus_text(item))
        else:
            for line in f:
                try:
                    item = json.loads(line)
                    texts.append(format_corpus_text(item))
                except: continue
    
    print(f"Loaded {len(texts)} corpus documents.")
    return texts

def load_query_texts(path):

    print(f"Loading queries from {path}...")
    valid_texts = []
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
        
    with open(path, 'r') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            data = json.load(f)
            for item in data:
                q_text = item.get("question") or item.get("query") or item.get("q") or item.get("text")
                if q_text:
                    valid_texts.append(q_text)
        else:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    q_text = item.get("question") or item.get("query") or item.get("q") or item.get("text")
                    if q_text:
                        valid_texts.append(q_text)
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    continue
                
    print(f"Loaded {len(valid_texts)} valid queries.")
    return valid_texts

def main():

    print("=== Step 0: Regenerate Embeddings ===")
    
    model = load_model()
    
    corpus_texts = load_corpus_texts(CORPUS_JSON_PATH)
    print(f"Encoding {len(corpus_texts)} corpus documents...")
    print(f"Instruction: '{CORPUS_INSTRUCTION}'")
    
    corpus_embs = get_batch_embeddings(model, corpus_texts, CORPUS_INSTRUCTION)
    print(f"Saving corpus embeddings to {CORPUS_EMB_OUTPUT}...")
    np.save(CORPUS_EMB_OUTPUT, corpus_embs)
    print(f"Saved corpus embeddings shape: {corpus_embs.shape}")
    
    query_texts = load_query_texts(QUERY_JSON_PATH)
    print(f"Encoding {len(query_texts)} queries...")
    print(f"Instruction: '{QUERY_INSTRUCTION}'")
    
    query_embs = get_batch_embeddings(model, query_texts, QUERY_INSTRUCTION)
    print(f"Saving query embeddings to {QUERY_EMB_OUTPUT}...")
    np.save(QUERY_EMB_OUTPUT, query_embs)
    print(f"Saved query embeddings shape: {query_embs.shape}")
    
    print("=== Done ===")

if __name__ == "__main__":
    main()