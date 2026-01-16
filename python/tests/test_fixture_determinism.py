"""Tests for fixture generation determinism.

Verifies that stored fixtures exactly match expected hashes.
This ensures that the fixture generation is deterministic and that files
haven't been accidentally modified or added/removed.
"""

import hashlib
from pathlib import Path

import pytest

# Map of expected file paths (relative to project root) -> SHA256 hash
EXPECTED_HASHES = {
    "tests/fixtures/bench_n10_s1_lmb_lbp.json": "9afa3c726d18ed54c39035b58951e973016ca4105aad5a1e96a8a7c00a1311cf",
    "tests/fixtures/bench_n10_s2_aa_lmb_lbp.json": "6fd5cb2e0abcc86fb31b16cf44e9e57dac1ab39d9a7efb42ddbe4a484fc35b58",
    "tests/fixtures/bench_n10_s2_ga_lmb_lbp.json": "87d54cd34e309e7101171d0239ac3c687311a93e92557b4f73abe36a52acc2c3",
    "tests/fixtures/bench_n10_s2_ic_lmb_lbp.json": "0481990d53e405ab40de4257d72c4c08188b2a212724a13827149fa65f217870",
    "tests/fixtures/bench_n10_s2_pu_lmb_lbp.json": "611011d7d09b4ae5db79759e3876d58bab6545e4b6e6485d68ae9818e04e90ad",
    "tests/fixtures/bench_n10_s4_aa_lmb_lbp.json": "4deaeaa38f93a160ff023d2363c7271fee6e4d479e1f7b55713ba27a8068373a",
    "tests/fixtures/bench_n10_s4_ga_lmb_lbp.json": "6b8d54a2a2849dd73e75fcac60e19f84d0999dc639da8cc26e88cefad391342c",
    "tests/fixtures/bench_n10_s4_ic_lmb_lbp.json": "6168e95c3d579edb6fe1227b3d790d88477136ccc50f4393d8632c18406fa1ba",
    "tests/fixtures/bench_n10_s4_pu_lmb_lbp.json": "042ea8dd40e3c71527f40fba6d12038aa9c479b9f73c99e891a85c4c9ada7eb3",
    "tests/fixtures/bench_n5_s1_lmb_lbp.json": "2abf58c3c7ea095b88a53c67cf0494c42d15463cf879df62e434c5271666e2fe",
    "tests/fixtures/bench_n5_s2_aa_lmb_lbp.json": "46f7f27b6d782701cdce584354f1341a3ca2e3c97bb20155a42c3de492ea7a2f",
    "tests/fixtures/bench_n5_s2_ga_lmb_lbp.json": "42ea7e60b88d42f3eb7b3a0a4f91399cac7fb53824d9bf29d370ff58c14fe4fd",
    "tests/fixtures/bench_n5_s2_ic_lmb_lbp.json": "39d758e05c59be2c295aa2a9b8c8507d17cde495945f978664cda78a0cac2766",
    "tests/fixtures/bench_n5_s2_pu_lmb_lbp.json": "7eb61a852e1a696f7c7987a387f438ed55c0d5525f02aae66fd24ccf0da34bca",
    "tests/fixtures/scenario_n10_s1.json": "19263b3f5a27cd38b2efeb4c210f5e14dc00031ebc2dc6508b73c46b53b3ae34",
    "tests/fixtures/scenario_n10_s2.json": "df0728206e4e435d6b7e8acf1c145769c46b4d5a515b40f9dd8e38708e249716",
    "tests/fixtures/scenario_n10_s4.json": "0872e2a71924985a7b28bf92e06d3a4782a4308cacb2bfb78cbaa2f123b7845f",
    "tests/fixtures/scenario_n20_s1.json": "f59869556be146268d9b4b7aa0d3ff5feb969713b8d6df3224de9b8c732006de",
    "tests/fixtures/scenario_n20_s2.json": "c4146f2db82bc4307942d5b7d93914f89027d092572fd539d6a8d2a37ba0da58",
    "tests/fixtures/scenario_n20_s4.json": "8e85a59f67ae29f333259e43dfb94f16e99c570725889932ec7d4e34517f86bf",
    "tests/fixtures/scenario_n20_s8.json": "a00d66b8b41c91bbe2235acc065cda265e2ab66cc45a757798defb881730f6e0",
    "tests/fixtures/scenario_n50_s8.json": "ca0db5927a90fc473ff9d0d2db2c4e6c085d5f4b600e46950f31eea59bcb9375",
    "tests/fixtures/scenario_n5_s1.json": "53ec0fb776161321aafa1cb905a0c7eac6bff419c7c2de4d87e2130f3b4e496c",
    "tests/fixtures/scenario_n5_s2.json": "0b49e83c7f8b7ff5d3ad474514300e4f6d2c119a45b39bb4e1fa91e640c6dfeb",
    "tests/fixtures/trial_ms_clutter_quick_seed42.json": "12852fef1a17239890e9a68e3d47ad325232d2c95590785568524b064895b789",
    "tests/fixtures/trial_ms_detection_quick_seed42.json": "4af815974c74fcd51c464629d78b55ec4b2b3e3e640de3d197675a2cc25f0b4a",
    "tests/fixtures/trial_ms_seed42.json": "b89dc9ec04884082961d4a450185fbf524e10e30710d59e6855ab827e26c8ffa",
    "tests/fixtures/trial_ss_detection_quick_seed42.json": "abfa55cc9b35f5f3fda6be4447d7321c70b1c9a20063176c5dc086312d730d62",
    "tests/fixtures/trial_ss_seed42.json": "e284696e94b06fdc33764151337e1408165472559d18f87120c0b8c591fa5a97",
    "tests/fixtures/trial_ss_quick_seed42.json": "89238583fc5e7bb8ff744436159e77b1f126b60b0d7610e5282b13597535e176",
    "tests/fixtures/step_ms_aa_lmb_seed42.json": "73517b7efcdce709f3f1a822f2089786d380538cd2643778c19d7f138de540c1",
    "tests/fixtures/step_ms_ga_lmb_seed42.json": "25b07a534392ce3b9278a2e24debd6564f540d8f62f44c1fcb0c291f50df12f6",
    "tests/fixtures/step_ms_ic_lmb_seed42.json": "2c03b924b39f0baadf7b17bbf3dd121da2871676a7359490ad39a21254333579",
    "tests/fixtures/step_ss_lmb_seed42.json": "2d3cee2619955ffc1d18537d4cb076f24213de60cf133fde5f3fb7d5a76d666b",
    "tests/fixtures/step_ss_lmbm_seed42.json": "f9b2ac9a9974ddaa25355eb0e850148b632171199e135f0d2e1dd67a109d5548",
    "tests/fixtures/step_ms_lmb_seed42.json": "1285e690806702d5ffe3479f38ff2aa3813375b23c5d79e916d7b6a315e0f2a6",
    "tests/fixtures/step_ms_lmbm_seed42.json": "b96d3e9388eea25bfe281e2d269482ae3a5ac766911384e84d66f54013b3de5b",
    "tests/fixtures/step_ms_pu_lmb_seed42.json": "b967984080bf578015d2718d4374ea743655dd9319b3cd01b8837a4e4530c5af",
}


def compute_hash(filepath: Path) -> str:
    """Compute SHA256 hash of a file's contents."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TestFixtureHashedDeterminism:
    """Validate that fixtures match the expected hashes exactly."""

    def test_fixture_integrity(self):
        """Verify presence and hash integrity of all expected fixtures."""
        project_root = Path(__file__).parent.parent.parent

        failures = []

        # 1. Verify all expected files exist and match hash
        for rel_path, expected_hash in EXPECTED_HASHES.items():
            full_path = project_root / rel_path

            if not full_path.exists():
                failures.append(f"MISSING: {rel_path}")
                continue

            actual_hash = compute_hash(full_path)
            if actual_hash != expected_hash:
                failures.append(
                    f"HASH MISMATCH: {rel_path} (got {actual_hash[:8]}..., expected {expected_hash[:8]}...)"
                )

        # 2. Verify no unexpected files exist in target directories
        # We check specific directories recursively for ANY .json file
        scan_roots = ["benchmarks", "tests/fixtures"]

        for rel_root in scan_roots:
            full_root = project_root / rel_root
            if not full_root.exists():
                continue

            # Scan all .json files recursively
            for file_path in full_root.glob("**/*.json"):
                rel_file_path = file_path.relative_to(project_root)

                # Exclude results directory inside benchmarks if it exists
                if "results" in rel_file_path.parts:
                    continue

                if str(rel_file_path) not in EXPECTED_HASHES:
                    failures.append(f"UNEXPECTED FILE: {rel_file_path}")

        if failures:
            pytest.fail("\n".join(["Fixture integrity check failed:"] + failures))
