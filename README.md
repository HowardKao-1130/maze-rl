# maze-rl
practice project for RL

## Dataset Splits

Generate datasets with multiple tasks per layout:

```bash
python scripts/generate_dataset.py --train-mazes 1000 --validation-mazes 200 --test-mazes 200 --tasks-per-maze 5
```

The split sizes control the number of unique layouts. `--train`,
`--validation`, and `--test` remain accepted aliases for those maze counts.
Each layout gets `--tasks-per-maze` valid start/goal tasks.

- `data/train.npz`: seen tasks for training and memorization evaluation.
- `data/same_layout_new_goals.npz`: validation-sized held-out start/goal tasks
  on training layouts.
- `data/validation.npz`: held-out validation layouts.
- `data/test.npz`: held-out test layouts.

## Training Metrics

Training uses `--rollouts-per-task` as the top-level exposure budget: each
selected training task is allowed roughly that many rollout attempts. For
single-rollout agents, one task selection produces one rollout. For grouped
agents such as GRPO, one task selection produces `--rollout-group-size`
rollouts, so the number of internal task-selection passes is
`ceil(rollouts_per_task / rollout_group_size)`.

With 1000 layouts and 5 tasks per layout, `--rollouts-per-task 20` trains a
single-rollout agent for 100,000 rollout attempts: 20 rollout attempts for each
of the 5000 selected tasks.

```bash
python scripts/train.py --algorithm q_learning --rollouts-per-task 20
```

Plot aggregate training curves:

```bash
python scripts/plot_training_metrics.py --metrics runs/q_learning/metrics.csv --output runs/q_learning/training_metrics.png
```

The plot shows separate rows for dataset-epoch mean rollout return, success
rate, env steps per rollout, and, for TD methods, dataset-epoch mean absolute TD
error. Success rate is the mean of the per-rollout `success` flag, so it is the
fraction of rollout attempts in that dataset epoch where the agent reached the
goal. Columns plot each metric against dataset epoch and cumulative env steps.
New metrics files also include `internal_updates`, which adds a cumulative
internal-update column.

Training writes `metrics.csv`, a `checkpoint.pkl` for tabular algorithms or
`checkpoint.pt` for neural algorithms, and this plot automatically at the end of
each run. Existing `metrics.csv` files are replaced by default; pass `--resume`
to append to an existing metrics file.

Training enables deterministic Torch algorithms, disables cuDNN benchmarking and
TF32, and configures cuBLAS workspace determinism before importing Torch. This
targets repeatable neural training for the same code, dataset, arguments,
hardware, drivers, and library versions.

Neural agents (`dqn`, `reinforce`, `a2c`, and `ppo`) also write TensorBoard logs
to `runs/<algorithm>/tensorboard` during training by default. Open the latest
event file for one algorithm with:

```bash
python scripts/open_tensorboard.py --algorithm dqn
```

By default, the launcher shows only the newest
`events.out.tfevents.<timestamp>.<host>.<pid>.<index>` file in the log
directory so older event files from previous runs are not overlaid. Pass
`--event-file <path>` to open a specific event file, or `--all-events` to load
every event file in the log directory.

Use `--no-tensorboard` to disable neural-agent event logs, `--tensorboard` to
enable them for a tabular run, or `--tensorboard-dir` to choose another log
directory.

Neural-agent hyperparameters can be passed directly to training, for example:

```bash
python scripts/train.py --algorithm dqn --rollouts-per-task 20 --learning-rate 0.0001 --batch-size 64 --epsilon-decay 0.999
```

For DNN agents, `--task-batch-size` controls how many distinct tasks are
sampled for a training round, and `--rollout-group-size` controls how many
rollouts are generated per selected task. Defaults are `task_batch_size=16` for
`reinforce`, `a2c`, and `grpo`, `64` for `ppo`, and `1` for `dqn`.
`rollout_group_size` defaults to `1` for older DNN agents and `8` for `grpo`.
The old `--rollout-episodes` and `--group-size` flags remain accepted aliases
for `--task-batch-size` and `--rollout-group-size`.
See `docs/rollout_budget_terms.md` for the full vocabulary and per-agent
schematics.

TensorBoard logs include aggregate rollout metrics plus update diagnostics when
the algorithm provides them. Neural policy methods log total loss, policy loss,
value loss, entropy, return targets, advantages, and value predictions; `ppo`
also logs approximate KL and clip fraction. `dqn` logs loss, replay size, mean
selected Q-value, and max Q-value.

TensorBoard scalar tags are grouped by the x-axis they use:

- `by_rollout/...`: one point per rollout attempt, with TensorBoard step equal
  to the rollout number.
- `by_transition/...`: one point per rollout attempt, with TensorBoard step equal to
  cumulative environment transitions.
- `by_dataset_epoch/...`: one aggregate point per completed task-coverage pass,
  with TensorBoard step equal to dataset epoch number.
- `by_training_round/...`: one aggregate point per collect/sample-and-optimize
  cycle, with TensorBoard step equal to training round number.
- `by_optimization_epoch/...`: one aggregate point per optimization pass over
  the current training pool, with TensorBoard step equal to cumulative
  optimization epoch number.

Rollout collection uses a hierarchical sampler over the tasks that remain
unseen in the current dataset epoch. It samples layouts roughly uniformly among
layouts that still have unseen tasks, then samples one unseen task uniformly
within the selected layout. A task cannot repeat within the same dataset epoch,
including across rollout collections. When all selected tasks have appeared
once, the next dataset epoch starts and tasks become eligible again.

Policy-gradient agents update from task batches instead of one rollout at a
time. `reinforce` and `a2c` collect 16 distinct tasks per training round, while
`ppo` collects 64. With the default `rollout_group_size=1`, each selected task
corresponds to one rollout. All three compute returns/advantages while
trajectories are intact, then shuffle collected transitions into minibatches of
64 for optimization. `ppo` trains for 4 optimization epochs per training round.
`a2c` and `ppo` bootstrap from time-limit truncations and stop bootstrapping
only at true task termination. `reinforce` uses Monte Carlo returns from
observed rewards only, so truncated rollouts receive no additional terminal
reward.
`reinforce` starts with an entropy coefficient of `0.05`, decays it by `0.9995`
after each rollout update, and keeps a `0.01` floor to reduce early policy
collapse. `a2c` uses a smaller default learning rate of `1e-4`, starts with a
stronger entropy coefficient of `0.10`, decays it by `0.9998`, and keeps a
`0.03` floor to resist the low-entropy wall-collision plateau seen in early
runs.

`dqn` also uses a minibatch size of 64, a neural learning rate of `3e-4`, and a
replay buffer sized to about 64 max-length episodes. It starts updates after a
1,000-transition replay warmup rather than waiting for the buffer to fill. By
default, DQN trains once per `batch_size` observed environment transitions, so
the replay samples processed per environment step roughly match one-pass
policy-gradient minibatching. Pass `--train-frequency 4` for a more standard
Atari-style DQN cadence or `--train-frequency 1` to restore one replay update
per observed transition.

## DNN Hyperparameter Tuning

Tune neural agents with randomized search over standard RL knobs such as
learning rate, discount factor, entropy regularization, PPO clipping, PPO
optimization epochs per training round, task batch size, rollout group size,
DQN replay warmup, minibatch size, and target-network update cadence. The tuner
optimizes validation mean path efficiency across all evaluated tasks.

Generate datasets first:

```bash
python scripts/generate_dataset.py --train-mazes 200 --validation-mazes 50 --test-mazes 50 --tasks-per-maze 5
```

Then tune against the pre-generated train and validation sets:

```bash
python scripts/tune_dnn.py --algorithm ppo
```

By default, the tuner reads `data/train.npz`, `data/validation.npz`, and
`data/same_layout_new_goals.npz`, runs 60 trials with a 200 rollouts-per-task
maximum budget per trial, trains each trial under `runs/tuning/trials`,
evaluates every validation task, writes TensorBoard logs, appends
`runs/tuning/tuning_results.csv`, and writes the current best trial to
`runs/tuning/best_config.json`. Pass
`--search-space path/to/search_space.json` to override the default search space.

Keep `--rollouts-per-task` as the training budget for comparable trials.
Increasing it usually improves final performance but also changes compute cost,
so it is best treated as a generous fixed maximum budget for the sweep. The
tuner asks training to evaluate validation mean path efficiency every
`--validation-interval` internal task-selection passes, also evaluates
`--same-layout-dataset` when provided, saves each trial's `best_checkpoint.pt`,
and uses the best validation-layout checkpoint for trial selection. The ordinary
`checkpoint.pt` remains the final training state.

Training writes `validation_metrics.png` beside `validation_metrics.csv`; the
plot overlays validation-layout performance with same-layout new-task
performance when both splits are available. Tuning and training progress logs
show trial percentages and task-selection-pass percentages at validation checks.

Early stopping defaults to `--early-stopping-patience 35` with
`--early-stopping-min-delta 0.0`, and the same stopping rule applies to every
trial. Because patience and maximum budget can interact with learning rate and
other dynamics, inspect `validation_metrics.csv` and
`training_summary.json` after a sweep to check whether the selected trial was
limited by the stopping rule or by the maximum budget. Use
`--evaluate-best-on-test` only after tuning to score the best
validation-selected checkpoint on `--test-dataset`.

Neural checkpoints for `dqn`, `a2c`, and `ppo` include model snapshots using the
same epoch schedule as tabular Q-table snapshots. Evaluation renders `dqn`
snapshot Q-value frames and videos under `q_value_plots`; `a2c` and `ppo`
snapshot state-value frames and videos are written under `v_value_plots`.

Training also writes `task_training_metrics.png`, which uses the same metric
subplots but draws one line per task. It also writes
`task_training_metrics.html`, a self-contained interactive version with task
filters for comparing selected tasks when the static lines overlap. Rolling
means are disabled by default; pass `--rolling-window 10` to add the rolling
mean to the aggregate plot.

Evaluation also writes the training plot automatically when it can find
`metrics.csv` next to the checkpoint. The checkpoint path defaults to
`runs/<algorithm>/checkpoint.pkl` for tabular agents and
`runs/<algorithm>/checkpoint.pt` for neural agents:

```bash
python scripts/evaluate.py --algorithm q_learning --dataset data/train.npz
```

By default, evaluation samples `--episodes` tasks from the dataset with a fixed
seed, so repeating the same command gives the same sampled tasks. Use `--seed`
to choose a different reproducible sample, `--fixed-index` to evaluate one task,
or `--all-tasks` to evaluate every task exactly once.

By default this writes `runs/q_learning/training_metrics.png`,
`runs/q_learning/task_training_metrics.png`, and
`runs/q_learning/task_training_metrics.html`. Use `--plot-output`,
`--task-plot-output`, and `--task-html-output` to choose different paths, or
`--no-plot` to skip training-metric plots.

For tabular checkpoints trained with the current script, evaluation also writes
Q-value maze plots from evenly spaced training snapshots:

```bash
python scripts/evaluate.py --algorithm q_learning --checkpoint runs/q_learning/checkpoint.pkl --dataset data/train.npz
```

By default, training stores 101 Q-table snapshots in the checkpoint, starting at
epoch 1 and ending at the final epoch. Use `--q-snapshot-count 11` for a smaller
set. Evaluation saves Q-value plots under
`runs/q_learning/q_value_plots/task_XXXXX/frames/snapshot_XXX.png` for the tasks
visited during evaluation, and writes
`runs/q_learning/q_value_plots/task_XXXXX/task_XXXXX_q_values.mp4` beside the
`frames` directory when MP4 support is installed. The plot title still shows the
actual epoch. Use
`--q-plot-all-tasks` to plot every task in the evaluation dataset,
`--no-q-videos` to skip MP4s, or `--no-q-plots` to skip Q plots entirely.

Open an existing task video with:

```bash
python scripts/open_q_video.py --algorithm q_learning --task-index 0
```
