# Anchored full-history refit (not submitted)

Three Autoformers, seeds 0, 1, 2. Parameters: 33891. Epochs: 16,
including 11 source selection epochs and 5 full-history refit epochs.
The source-selected refit lengths are 2, 1, 2 epochs.

The matching historical refit scored RMSE 69.679 (edge-16: 82.270),
versus 69.344 (edge-16: 81.972) for the original three-seed configuration.
This is not evidence of improvement. Preserve for reproducibility;
do not recommend another leaderboard attempt on this evidence.

This folder contains predictions, declarations, completed run records, and
local weights. `selection/` contains the source epoch-selection records.
The `.pt` weights remain local and are ignored by git.
