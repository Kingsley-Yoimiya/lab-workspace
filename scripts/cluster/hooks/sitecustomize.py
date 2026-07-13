# 使 failslow_step_timer 在任意 python 启动时尝试挂接（需本目录在 PYTHONPATH 中）
try:
    import failslow_step_timer  # noqa: F401
except Exception as exc:
    import sys

    print(f"[sitecustomize] failslow_step_timer not loaded: {exc}", file=sys.stderr)
