# Maze RL

This context defines the benchmark language for maze-navigation reinforcement
learning experiments.

## Language

**Layout**:
A wall grid that defines which cells are traversable.
_Avoid_: Maze when referring only to the wall grid

**Task**:
A navigation problem consisting of one layout, one start position, and one goal
position.
_Avoid_: Maze when referring to a full start-to-goal problem

**Start Position**:
The cell where an episode begins for a task.
_Avoid_: Current position

**Current Position**:
The cell occupied by the agent at a specific step of an episode.
_Avoid_: Start position

**Seen Task**:
A task that was available during training.
_Avoid_: Seen maze

**Same-Layout New-Goal Task**:
A held-out task that uses a training layout with a different valid start/goal
pair.
_Avoid_: Validation maze

**Unseen Layout**:
A layout that was not available during training.
_Avoid_: Unseen task when the distinction matters

**Episode Return**:
The cumulative reward collected over one episode.
_Avoid_: Reward when referring to a full episode total

**TD Error**:
The temporal-difference update residual produced by a TD learning step.
_Avoid_: Loss when referring to tabular TD updates

**Expected Pairwise Directed Path Similarity**:
The mean directed edge-Jaccard similarity over independently paired shortest
paths for two tasks.
_Avoid_: Directed weighted Jaccard when referring to this task-level metric

**Within-Bucket Deduplication**:
Path similarity is constrained independently among tasks in the same layout
and difficulty bucket.
_Avoid_: Cross-bucket deduplication when this scope is intended

**Fixed Per-Layout Bucket Quotas**:
Each filtered stratified layout starts with the same bucket assignment derived
from the configured per-layout task budget. Layouts that cannot satisfy that
assignment under the deduplication search are rejected; task quotas are not
transferred between layouts.

**Filtered/Stratified Validation (`val`)**:
The primary validation split, generated with the same layout-capacity filter,
difficulty quotas, and within-bucket task deduplication as training.
_Avoid_: Natural validation when referring to this split

**Tuning Validation (`val-tune`)**:
A smaller validation split generated through the same path as `val` (filtered
and stratified when `--stratify` is enabled), containing one fifth of the
requested validation layouts (rounded up, with at least one layout). It is
used only as the automatic objective and early stopping split for
hyperparameter-tuning child runs; ordinary training does not load it.

**Same-Layout Validation (`val-same`)**:
Randomly selected held-out tasks on layouts seen during training.
_Avoid_: Validation layout when referring to this split

**Natural Validation (`val-natural`)**:
Randomly selected tasks on independently generated layouts that follow the
ordinary natural layout-generation process.
_Avoid_: Filtered validation when referring to this split

**Filtered/Stratified Test (`test`)** and **Natural Test (`test-natural`)**:
The held-out counterparts of `val` and `val-natural`, respectively. Neither is
used for tuning or early stopping.
