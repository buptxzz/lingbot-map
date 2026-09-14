"""CPU checks of historical phase accounting, not throughput measurements."""
import copy
import json
import math
import unittest

from tools.thor_legacy_1005.records import (
    COMPILE_COUNTER_KEYS, PROTOCOL_VERSION, require_zero_compile,
    summarize_samples, summarize_timing, validate_contract,
)


def input_signature(frames):
    pixels = 378 * 518
    stride = [frames * 3 * pixels, 3 * pixels, pixels, 518, 1]

    def signature(count):
        return {"shape": [1, count, 3, 378, 518], "stride": list(stride),
                "dtype": "torch.bfloat16", "device_type": "cuda"}

    return {"source": signature(frames), "S8": {"calls": 1, **signature(8)},
            "S1": {"calls": 10, **signature(1)}}


def capture_contract(frames=19):
    zeros = {f"{section}.{name}": 0 for section, name in COMPILE_COUNTER_KEYS}
    return {
        "formal_protocol_version": PROTOCOL_VERSION,
        "capture_boundary": "whole_frame",
        "camera_execution": "captured_fixed_python_state",
        "temporal_positions": "fixed_at_capture",
        "formal_scale_no_recompile": True, "formal_warm_no_recompile": True,
        "scale_matches_prewarm": True, "warm_frames_match_prewarm": True,
        "formal_scale_compile_counter_delta": dict(zeros),
        "formal_warm_compile_counter_delta": dict(zeros),
        "replay_compile_counter_delta": dict(zeros),
        "input_signature": input_signature(frames),
        "steady_state_prime": {"depth_impl": "compiled", "deterministic_algorithms": False},
        "cuda_graph_capture": {"depth_impl": "compiled", "deterministic_algorithms": False},
    }


def output_digests(count):
    return {
        "pose_enc": {"shape": [1, count, 9], "dtype": "torch.float32",
                     "finite": True, "sha256": "a" * 64},
        "depth": {"shape": [1, count, 378, 518, 1], "dtype": "torch.float32",
                  "finite": True, "sha256": "b" * 64},
    }


def timing_result(frames=19):
    return [[3] * (frames - 18), 8, 16, 10, 20, 24, 30, [4] * (frames - 18),
            {"scale": output_digests(8), "warm_tail": output_digests(1),
             "replay_tail": output_digests(1)}, capture_contract(frames)]


class ThorMeasurementsTest(unittest.TestCase):
    def test_quantiles(self):
        self.assertEqual(summarize_samples([4, 1, 3, 2]), {
            "count": 4, "mean": 2.5, "p50": 2.5, "p90": 3.7, "min": 1, "max": 4,
        })
        self.assertEqual(summarize_samples([5])["p90"], 5)

    def test_invalid_samples_rejected(self):
        for values in (None, 3, True, {3: 4}, "1", [], [0], [-1], [math.inf], [-math.inf],
                       [math.nan], [True], [False], ["1"], [None]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                summarize_samples(values)

    def test_frame_weighted_time(self):
        summary = summarize_timing(timing_result(20), 20)
        self.assertEqual(summary["host_per_requested_frame"], 62 / 20)
        self.assertEqual(summary["gpu_per_requested_frame"], 42 / 20)
        self.assertEqual(summary["fps"], 1000 / (62 / 20))
        self.assertEqual(summary["host_replay"]["mean"], 4)
        self.assertEqual(summary["host_replay"]["count"], 2)
        self.assertTrue(summary["setup_compile_capture_excluded"])
        self.assertTrue(summary["output_collection_excluded"])
        self.assertEqual(summary["host_replay_includes"], [
            "static_input_copy", "cache_preparation", "graph_replay", "synchronize",
        ])

    def test_raw_sample_and_phase_counts_rejected(self):
        for index, value in ((0, []), (7, []), (0, [3, 3]), (7, [4, 4]),
                             (1, 7), (3, 9), (1, 8.0), (3, 10.0), (1, True)):
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                result = timing_result()
                result[index] = value
                summarize_timing(result, 19)

    def test_invalid_replay_containers_rejected(self):
        for index in (0, 7):
            for value in (None, 3, True, {3: 4}, "3"):
                result = timing_result()
                result[index] = value
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    summarize_timing(result, 19)

    def test_invalid_phase_and_replay_samples_rejected(self):
        for index in (0, 2, 4, 5, 6, 7):
            for value in (0, -1, math.nan, math.inf, True, "3", None):
                with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                    result = timing_result()
                    result[index] = [value] if index in (0, 7) else value
                    summarize_timing(result, 19)

    def test_overflowed_totals_rejected(self):
        result = timing_result()
        result[5] = result[6] = 1e308
        with self.assertRaises(ValueError):
            summarize_timing(result, 19)

    def test_incomplete_results_and_frame_counts_rejected(self):
        for result in (None, [], timing_result()[:-1], timing_result() + [None]):
            with self.subTest(result=result), self.assertRaises(ValueError):
                summarize_timing(result, 19)
        for frames in (18, 20, True, 19.0, "19"):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                summarize_timing(timing_result(), frames)

    def test_raw_tuple_and_json_list_results_are_both_accepted(self):
        for frames in (19, 200):
            runtime_result = tuple(timing_result(frames))
            json_result = json.loads(json.dumps(runtime_result))
            with self.subTest(frames=frames):
                self.assertIsInstance(json_result, list)
                self.assertEqual(summarize_timing(runtime_result, frames),
                                 summarize_timing(json_result, frames))

    def test_tuple_and_json_results_still_reject_invalid_protocol(self):
        result = timing_result()
        result[9]["formal_protocol_version"] = 4
        for value in (tuple(result), json.loads(json.dumps(result))):
            with self.subTest(container=type(value).__name__), self.assertRaises(ValueError):
                summarize_timing(value, 19)

    def test_zero_compile_requires_complete_integer_evidence(self):
        zeros = capture_contract()["replay_compile_counter_delta"]
        for delta in (None, {}, {**zeros, "frames.total": 1},
                      {**zeros, "frames.total": False}, {**zeros, "frames.total": 0.0},
                      {**zeros, "unexpected": 0}):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                require_zero_compile(delta)
        require_zero_compile(zeros)

    def test_whole_frame_contract_is_required(self):
        for key in capture_contract():
            if key == "input_signature":
                continue  # Validated against the record's requested frame count.
            contract = copy.deepcopy(capture_contract())
            del contract[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contract(contract)

    def test_protocol4_and_dynamic_capture_contracts_rejected(self):
        for key, value in (("formal_protocol_version", 4), ("formal_protocol_version", 2.0),
                           ("capture_boundary", "aggregator_and_depth_only"),
                           ("camera_execution", "uncaptured_original_dynamic_history"),
                           ("temporal_positions", "original_rope_copied_each_frame")):
            contract = capture_contract()
            contract[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contract(contract)

    def test_prime_and_capture_must_have_determinism_disabled(self):
        for phase in ("steady_state_prime", "cuda_graph_capture"):
            for value in (True, None, 0):
                contract = capture_contract()
                contract[phase]["deterministic_algorithms"] = value
                with self.subTest(phase=phase, value=value), self.assertRaises(ValueError):
                    validate_contract(contract)


if __name__ == "__main__":
    unittest.main()
