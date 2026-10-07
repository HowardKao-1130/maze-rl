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
python scripts/plot_training_metrics.py --metrics runs/tabular/q_learning/metrics.csv --output runs/tabular/q_learning/training_metrics.png
```

Plot all non-tuning neural-agent training histories discovered under `runs/`
into one image. When validation metrics are available, the plot includes train,
validation, same-layout validation, and greedy train-evaluation curves:

```bash
python scripts/summarize_training_runs.py --output runs/training_history_summary.png
```

Greedy train-evaluation points are read from `validation_metrics.csv` rows with
split `train`, `train_greedy`, or `greedy_train`.

The plot shows separate rows for dataset-epoch mean rollout return, success
rate, env steps per rollout, and, for TD methods, dataset-epoch mean absolute TD
error. Success rate is the mean of the per-rollout `success` flag, so it is the
fraction of rollout attempts in that dataset epoch where the agent reached the
goal. Columns plot each metric against dataset epoch and cumulative env steps.
New metrics files also include `internal_updates`, which adds a cumulative
internal-update column.

Training writes `metrics.csv`, `training_state.pt`, a `checkpoint.pkl` for
tabular algorithms or `checkpoint.pt` for neural algorithms, and this plot
automatically at the end of each run. If an output directory already contains
training artifacts, starting a fresh run warns and requires interactive
confirmation before deleting or overwriting them. Pass `--resume` to restore
`training_state.pt`, including model, optimizer, sampler, RNG, counter,
pending-metric, and validation state. When restoring full training state,
omitted run-identity arguments such as the
dataset, seed, rollout budget, batch shape, and validation settings default to
the values recorded in `training_state.pt`; explicit incompatible values are
rejected by the metadata check. Explicit future-run settings such as a larger
rollout budget, validation cadence, validation episode sampling, and
early-stopping thresholds can override the stored values.
Pass `--retroactive-early-stop` with `--resume` to allow
early-stopping patience and min-delta to differ from the stored state, replay
the existing `validation_metrics.csv`, and stop immediately when that history
already satisfies the requested early-stopping rule. This mode changes the
resume decision; it does not rewind model weights to an earlier epoch.
Training prints completed task-selection-pass progress by default, including
training-set mean path efficiency and success rate. When validation datasets are
enabled, validation progress also prints mean path efficiency and success rate.
Pass `--no-progress` to hide those training progress messages. Hyperparameter
tuning child runs show the same compact training and validation progress so
sweeps can be monitored with the same metrics as final training.

Training enables deterministic Torch algorithms, disables cuDNN benchmarking and
TF32, and configures cuBLAS workspace determinism before importing Torch. This
targets repeatable neural training for the same code, dataset, arguments,
hardware, drivers, and library versions.

Training does not write TensorBoard logs by default. When `--tensorboard` is
enabled, logs are written to `<run directory>/tensorboard`. Open the latest
event file for one algorithm with:

```bash
python scripts/open_tensorboard.py --logdir runs/dnn/final_dqn_best/dqn/tensorboard
```

By default, the launcher shows only the newest
`events.out.tfevents.<timestamp>.<host>.<pid>.<index>` file in the log
directory so older event files from previous runs are not overlaid. Pass
`--event-file <path>` to open a specific event file, or `--all-events` to load
every event file in the log directory.

Use `--tensorboard` to enable event logs, or `--tensorboard-dir` to choose
another log directory.

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

Use a separate output directory for larger final-training datasets so tuning
and experiment datasets remain intact. Pass an experiment name to derive the
directory from the actual split sizes, task count, and seed:

```bash
python scripts/generate_dataset.py --experiment-name generalization_followup --train-mazes 5000 --validation-mazes 500 --test-mazes 500 --tasks-per-maze 5
```

Without `--output-dir`, generated splits are stored under
`data/datasets/<name>_<train>x<validation>x<test>_t<tasks>_seed<seed>`. If
`--experiment-name` is omitted, the prefix defaults to `dataset`.

Then tune against the pre-generated dataset bundle:

```bash
python scripts/tune_dnn.py --algorithm ppo --dataset generalization_followup_5000x500x500_t5_seed42
```

The tuner requires an explicit dataset or split paths. With a dataset
name, it reads `data/datasets/<dataset>/train.npz`,
`validation.npz`, and `same_layout_new_goals.npz`, samples 100 hyperparameter
combinations with a random-search sampler and a 200 rollouts-per-task maximum
budget per combination, stores per-combination trial directories under the
matching
`runs/dnn/<dataset>/tuning/trials`, evaluates every
validation task, appends `tuning_results.csv`, and writes the current best
combination to `best_config.json`. Pass `--tensorboard` to write
per-combination TensorBoard event logs during tuning. It also writes
`tuning_configs/<algorithm>.json`, which records each algorithm's tuning
arguments and effective search space for controlled resumes. Pass
`--output-dir` only when you need an explicit save-location override, or
`--experiment-name` to derive the output directory from dataset metadata. Pass
`--hyperparameter-combinations` to change the number of sampled combinations,
or pass
`--search-space path/to/search_space.json` to override the default search space.
Random search is the default sampler. To opt in to Optuna's TPE sampler, install
the optional dependency with `pip install 'maze-rl[tuning]'` or
`pip install optuna`, then pass `--sampler optuna` or `--optuna`.
If a sweep is interrupted, rerun the same command with `--resume` to skip
completed hyperparameter combinations recorded in `tuning_results.csv`, recover
finished combination directories that were interrupted before their result row
was appended, and rerun only incomplete combinations with the same sampled seed
and hyperparameters. Incomplete child training runs restore their saved
`training_state.pt` when one exists. When `--resume` finds the
algorithm's stored tuning config, it ignores new conflicting tuning arguments
and continues with the stored arguments, including the combination count,
rollout budget, datasets, seed, validation settings, and search space. If
tuning history exists for that algorithm without a stored config, resume stops
instead of guessing the old arguments.

Keep `--rollouts-per-task` as the training budget for comparable combinations.
Increasing it usually improves final performance but also changes compute cost,
so it is best treated as a generous fixed maximum budget for the sweep. The
tuner asks training to evaluate validation mean path efficiency every
`--validation-interval` internal task-selection passes, also evaluates
`--same-layout-dataset` when provided, saves each combination's
`best_checkpoint.pt`, and uses the best validation-layout checkpoint for
combination selection. The ordinary `checkpoint.pt` remains the final training
state.

After tuning, train a longer final run from the stored best hyperparameter
combination instead of copying values by hand. `--dataset` may point at a
dataset directory; training then uses its `train.npz`, `validation.npz`, and
`same_layout_new_goals.npz` files and writes under the matching
`runs/dnn/<dataset-directory>/<algorithm>` directory:

```bash
python scripts/train.py --algorithm ppo --best-config runs/dnn/tuning_200x50x50_t5_seed42/tuning/best_config_ppo.json --dataset data/datasets/generalization_followup_5000x500x500_t5_seed42 --validation-interval 5 --validation-all-tasks --rollouts-per-task 1000
```

`--best-config` reads `runs/dnn/tuning/best_config_<algorithm>.json` by default.
Tabular training writes to `runs/tabular/<algorithm>`, and DNN training writes
to `runs/dnn/final_<algorithm>_best/<algorithm>` unless `--output-dir`,
`--experiment-name`, or a dataset directory is provided. With
`--experiment-name`, training reads the dataset metadata and writes under
`runs/dnn/<name>_<train>x<validation>x<test>_t<tasks>_seed<seed>/<algorithm>`.
Explicit training flags still override loaded values.

Training writes `validation_metrics.png` beside `validation_metrics.csv`; the
plot overlays greedy train-evaluation, validation-layout performance, and
same-layout new-task performance when those splits are available. Tuning and
training progress logs show combination percentages and task-selection-pass
percentages at validation checks.

Summarize completed DNN sweeps across agents with:

```bash
python scripts/tune_dnn.py summarize --output-dir runs/dnn/tuning
```

The command writes `runs/dnn/tuning/summary_plots/mean_path_efficiency_heatmap.png`
for best greedy train-evaluation, validation, and same-layout performance
across trials, plus
`mean_path_efficiency_training_progress_by_agent.png`, which shows full
`metrics.csv` training curves for every completed DNN trial grouped by agent
and highlights the trial(s) with the best validation score. It also writes one
`<algorithm>_validation_metrics_trials.png` grid per neural agent containing
per-trial train-greedy, validation, and same-layout validation curves overlaid
with the matching epoch-aligned training curve when `metrics.csv` is available.
Trial subplot titles show the best performance for the plotted curves, and the
best validation score for that agent is highlighted, including ties. DNN summary
grids use a single 5x4 sheet with display-safe dimensions and extra row height
so 20-trial sweeps fit in one readable overview, and bounded metrics use a
fixed 0-to-1 y scale so trials remain directly comparable. The command also
writes
`<algorithm>_mean_path_efficiency_parallel_coordinates.png` plots that place
each trial's numeric hyperparameters and validation score on shared parallel
axes, with better scores drawn in brighter colors and the best trial(s)
emphasized.

Early stopping defaults to `--early-stopping-patience 35` with
`--early-stopping-min-delta 0.0`, and the same stopping rule applies to every
combination. Because patience and maximum budget can interact with learning
rate and other dynamics, inspect `validation_metrics.csv` and
`training_summary.json` after a sweep to check whether the selected combination
was limited by the stopping rule or by the maximum budget. Use
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
`runs/tabular/<algorithm>/checkpoint.pkl` for tabular agents and
`runs/dnn/final_<algorithm>_best/<algorithm>/checkpoint.pt` for neural agents:

```bash
python scripts/evaluate.py --algorithm q_learning --dataset data/train.npz
```

By default, evaluation runs every task in the dataset exactly once with a fixed
seed for deterministic tie-breaking. For `validation.npz`, evaluation also
looks for a sibling `same_layout_new_goals.npz` split and renders rollout videos
for both validation views when MP4 support is installed. Use `--best-trial` to
evaluate the best checkpoint recorded in `runs/dnn/tuning/tuning_results.csv`, or
`--no-rollout-animations` to skip rollout videos.
Legacy sampling flags remain accepted: use `--episodes N` to sample random
tasks, `--fixed-index I` to repeat one task, or `--all-tasks` to request the
default all-task pass explicitly.

Training skips final training-metric plots by default to avoid large memory
spikes on long runs. Pass `--plot` to write
`runs/tabular/q_learning/training_metrics.png`,
`runs/tabular/q_learning/task_training_metrics.png`, and
`runs/tabular/q_learning/task_training_metrics.html`. Use `--plot-output`,
`--task-plot-output`, and `--task-html-output` to choose different paths.

For tabular checkpoints trained with the current script, evaluation also writes
Q-value maze plots from evenly spaced training snapshots:

```bash
python scripts/evaluate.py --algorithm q_learning --checkpoint runs/tabular/q_learning/checkpoint.pkl --dataset data/train.npz
```

By default, training stores 101 Q-table snapshots in the checkpoint, starting at
epoch 1 and ending at the final epoch. Use `--q-snapshot-count 11` for a smaller
set. Evaluation saves Q-value plots under
`runs/tabular/q_learning/q_value_plots/task_XXXXX/frames/snapshot_XXX.png` for the tasks
visited during evaluation, and writes
`runs/tabular/q_learning/q_value_plots/task_XXXXX/task_XXXXX_q_values.mp4` beside the
`frames` directory when MP4 support is installed. The plot title still shows the
actual epoch. Use `--no-q-videos` to skip MP4s, or `--no-q-plots` to skip Q
plots entirely.

Open an existing task video with:

```bash
python scripts/open_q_video.py --algorithm q_learning --task-index 0
```
