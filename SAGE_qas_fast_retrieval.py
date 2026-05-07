import os
import time
import importlib.util


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from path: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    base_dir = os.path.dirname(__file__)
    target = os.path.join(base_dir, "4_trace_search_clean_wuzhaiyao_no_split_moreqiefen.py")
    if not os.path.exists(target):
        raise FileNotFoundError(target)

    os.environ["ENTANGLEMENT_SPLIT_TAGS"] = "c3e1"
    if "TRACE_MAX_QUERIES" not in os.environ:
        os.environ["TRACE_MAX_QUERIES"] = "1000"

    mod = _load_module_from_path("trace_mod_c3e1", target)
    t0 = time.time()
    mod.main()
    t1 = time.time()
    print(f"[FAST] total_time_s={t1 - t0:.2f} | split_tag=c3e1")


if __name__ == "__main__":
    main()