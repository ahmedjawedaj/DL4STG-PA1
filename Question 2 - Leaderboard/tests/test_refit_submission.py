import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pa1q2 import data as D
from pa1q2.refit import run
from pa1q2.submission import _load_final, main
from pa1q2.train import Windows


class RefitChecks(unittest.TestCase):
    def test_selection_cannot_see_refit_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text(json.dumps({"args": {"stage": "final"}}))
            with self.assertRaisesRegex(ValueError, "selection period"):
                run(source, D.ES_END, D.N_HISTORY, directory)

    def test_training_windows_end_before_holdout(self):
        origins = D.origins(0, D.ES_END, 168, 1)
        self.assertEqual(origins[-1] + D.PRED_LEN, D.ES_END)
        self.assertGreaterEqual(origins[0] - 168, 0)

    def test_future_targets_do_not_change_forecast_inputs(self):
        import torch
        values = torch.arange(400, dtype=torch.float32).view(-1, 1)
        marks = torch.zeros(400, 2)
        before = Windows(values, marks, 168, 48, 168, True, True).batch([200])
        values[200:] = 99999
        after = Windows(values, marks, 168, 48, 168, True, True).batch([200])
        for index in (0, 1, 2, 4):
            torch.testing.assert_close(before[index], after[index])

    def test_refit_epoch_accounting_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            record = {"args": {"stage": "refit"}, "fit_end": D.N_HISTORY,
                      "selection_epochs": 4, "refit_epochs": 2, "epochs_run": 6,
                      "parameters": 11297, "best_epoch": 2, "best_es_RMSE": 76.0,
                      "forecast": list(range(168))}
            source.write_text(json.dumps(record))
            out = Path(directory) / "candidate"
            with patch("pa1q2.submission.D.load", return_value=(np.full(D.N_HISTORY, 7), None)), \
                    patch("pa1q2.submission._recount_parameters", return_value=11297):
                main([source], out)
            declaration = json.loads((out / "declaration.json").read_text())
            values = np.fromstring((out / "predictions.txt").read_text(), sep=",")
            self.assertEqual(declaration["epochs_E"], 6)
            self.assertEqual(declaration["trainable_parameters_P"], 11297)
            np.testing.assert_array_equal(values, np.maximum(np.arange(168), 7))
            record["epochs_run"] = 2
            source.write_text(json.dumps(record))
            with self.assertRaisesRegex(AssertionError, "both selection and refit"):
                _load_final(source)
            record["fit_end"] = D.ES_END
            source.write_text(json.dumps(record))
            with self.assertRaisesRegex(AssertionError, "historical refits"):
                _load_final(source)


if __name__ == "__main__":
    unittest.main()
