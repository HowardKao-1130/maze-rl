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
