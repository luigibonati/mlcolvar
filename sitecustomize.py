# sitecustomize.py
import os

if os.environ.get("COVERAGE_PROCESS_START"):
    try:
        import coverage
        coverage.process_startup()
    except Exception:
        pass
