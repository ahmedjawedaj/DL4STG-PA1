# Unanchored full-history refit (not submitted)

P=33891, E=14 (10 source selection epochs plus 4 refit epochs).
Source-selected refit lengths: 2, 1, 1 epochs. All three seeds are included.

Historical RMSE 69.609; edge-16 RMSE 81.526. It improves over the original
unanchored configuration's 70.323, but does not beat the submitted anchored
configuration's 69.344. Do not treat this as an established improvement.

Predictions, declarations, run records, and local weights are saved here.
`selection/` contains the earlier epoch-selection records; `.pt` files are
ignored by git. See `experiment_notes/variants.md` for the complete comparison.
