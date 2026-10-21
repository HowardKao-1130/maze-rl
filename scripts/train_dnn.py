"""Canonical neural-agent training entry point.

``train.py`` contains the shared implementation for tabular and neural
agents.  This module gives DNN workflows a focused command name while
delegating to the same implementation and dataset-bundle resolution.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train import main


if __name__ == "__main__":
    main()
