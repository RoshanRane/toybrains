# Changelog

## v1.1.0 (2026-10-13)
Version used by the ICON experiments (Rane et al., arXiv:2608.26083). Fully backward compatible with `main` before this release: every new argument defaults to the previous behaviour.

- `utils/DLutils.py`: `SimpleCNN` gets `fc_activation` (`'none'` (default, linear fc head as before) | `'relu'` | `'elu'`). `'none'` adds no module, so existing state_dicts still load. The ICON experiments use `fc_activation='elu'`: a linear fc stack makes TCAV scores degenerate, ReLU can die at high learning rates.
- `create_toybrains.py`: opt-in reproducibility. `generate_dataset_table(..., seed=None)` reseeds `numpy`/`random` before sampling the subjects; `generate_dataset_images(..., seed_base=None)` draws each subject's polygon rotations from a generator keyed on `(seed_base, subject index)`, independent of joblib scheduling. Two calls with the same seeds produce identical datasets (needed for the single-edge ablation scenarios).
- Includes the changes on `main` since v1.0.0 (2024-12-17), notably `SimpleCNN` with two tunable fully connected layers (`act_size_penultimate`, `act_size_antepenultimate`), which is **not** part of v1.0.0.

## v1.0.0 (2024-12-17)
First release (https://doi.org/10.5281/zenodo.14509513).
