# DPO Only Preferences Resume Design

## Goal
When `--dpo_only_preferences` is enabled, resume training from the base snapshot
`dpo_snapshot_epoch_<E>.pth` in `--dpo_run_dir` instead of retraining from epoch 1.
All new outputs still go to a fresh `current_run_dir`.

## Non-Goals
- Do not resume optimizer or scheduler state.
- Do not write outputs back into `--dpo_run_dir`.

## Data Flow
1. Resolve `current_run_dir` as usual for the new run.
2. If `enable_dpo` and `dpo_only_preferences`:
   - Require `dpo_run_dir`.
   - Load snapshot `dpo_snapshot_epoch_<dpo_start_epoch>.pth`.
   - Read `epoch` from snapshot.
   - Load `model_state_dict` into the model.
   - Start training from `epoch + 1` up to `--epochs`.
3. DPO preference artifacts are read from `dpo_run_dir`.
4. All new artifacts, evaluations, and logs are written to `current_run_dir`.

## Error Handling
- If `dpo_run_dir` is missing: fail with clear error.
- If snapshot file is missing: fail with clear error.
- If snapshot `epoch >= --epochs`: fail with clear error.

## Code Changes
- `BasicTrainer`: add `start_epoch` and use it in training loop.
- `main.py`: load snapshot when `dpo_only_preferences` is set and pass `start_epoch`.
- `utils/config.py`: update help text for `--dpo_only_preferences`.
