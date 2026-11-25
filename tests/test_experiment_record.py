from __future__ import annotations

import json
import tempfile
import unittest

from evaluation.experiment_record import record_experiment


class ExperimentRecordTests(unittest.TestCase):
    def test_record_experiment_writes_complete_new_json_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = record_experiment(
                output_directory=directory,
                optimizer_name="PSO",
                operating_point={
                    "modulation_index": 0.75,
                    "dc_voltage_per_cell_v": 100.0,
                },
                carrier_bounds={"cell_2": [0.0, 360.0], "cell_3": [0.0, 360.0]},
                evaluation_budget=64,
                seed=20261124,
                feasibility_rules={"tracking_error_percent_max": 0.05},
                thd_method="event-based Parseval",
                harmonic_range={"minimum_harmonic": 1, "maximum_harmonic": 100},
                final_independent_resimulation={
                    "candidate": [0.0, 60.0, 120.0],
                    "metrics": {"total_thd_percent": 3.0},
                },
                baseline={
                    "carrier_offsets_degrees": [0.0, 60.0, 120.0],
                    "metrics": {"total_thd_percent": 4.0},
                },
                optimizer_run={"objective_requests": 64},
            )

            self.assertTrue(path.is_file())
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["optimizer_name"], "PSO")
            self.assertEqual(record["seed"], 20261124)
            self.assertEqual(record["evaluation_budget"], 64)
            self.assertEqual(record["thd_method"], "event-based Parseval")
            self.assertIn("operating_point", record)
            self.assertIn("final_independent_resimulation", record)
            self.assertIn("baseline", record)

    def test_record_experiment_rejects_missing_required_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "(?i)baseline"):
                record_experiment(
                    output_directory=directory,
                    optimizer_name="POA",
                    operating_point={"modulation_index": 0.75},
                    carrier_bounds={"cell_2": [0.0, 360.0]},
                    evaluation_budget=32,
                    seed=1,
                    feasibility_rules={"feasible": True},
                    thd_method="event-based Parseval",
                    harmonic_range={"minimum_harmonic": 1, "maximum_harmonic": 100},
                    final_independent_resimulation={
                        "candidate": [0.0, 60.0],
                        "metrics": {"total_thd_percent": 3.0},
                    },
                    baseline={"carrier_offsets_degrees": [0.0, 60.0]},
                )


if __name__ == "__main__":
    unittest.main()
