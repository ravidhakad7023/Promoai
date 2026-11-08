"""End-to-end entry point:

    python run.py --input <zip-or-folder> [--max-episodes N] [--candidates-per-window K] [--resume RUN_DIR]

All logic lives in promoai/runner.py; configuration (provider, model, API key) comes from .env /
the environment via the existing config system.
"""

from promoai.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
