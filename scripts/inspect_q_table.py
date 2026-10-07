import pickle
from collections import Counter
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )

from scripts.experiment_naming import DEFAULT_TABULAR_OUTPUT_ROOT

checkpoint_path = (
    DEFAULT_TABULAR_OUTPUT_ROOT / "mc" / "checkpoint.pkl"
)

with checkpoint_path.open("rb") as f:
    checkpoint = pickle.load(f)

q = checkpoint["q"]

print("algorithm:", checkpoint["algorithm"])
print("q states:", len(q))

state_lengths = Counter(
    len(state) for state in q
)
layout_counts = Counter(
    state[0] for state in q
)

print("state shapes:", dict(state_lengths))
print("distinct layouts:", len(layout_counts))

for layout_index, count in layout_counts.most_common(20):
    print(f"layout {layout_index}: {count} states")

print()
print("sample q values:")

for state, values in list(q.items())[:50]:
    print(state, values)
