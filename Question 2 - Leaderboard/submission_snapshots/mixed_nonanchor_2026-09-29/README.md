# Original ensemble plus unanchored refit (not submitted)

Six equally weighted Autoformers: P=67782, E=29. Historical RMSE 68.971;
edge-16 RMSE 81.236. It improved 28/51 historical weeks. The paired-week
bootstrap RMSE-difference interval [-1.614, +0.787] includes no improvement.

This is the lowest historical RMSE among the new candidates, but the small
gain does not establish a better hidden score or justify a predicted rank.
Keep the first submitted forecast active pending stronger evidence.

Predictions, declarations, run records, and local weights are saved here.
`selection/` contains the refits' earlier epoch-selection records.
`.pt` weights are ignored by git.
