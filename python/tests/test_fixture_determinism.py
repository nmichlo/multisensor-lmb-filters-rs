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
    "benchmarks/fixtures/bouncing_n10_s1_LMB_LBP.json": "7a6b8a1e71331bbb435cfcf39ced6844a65517988ab010a83172a787d26f310e",
    "benchmarks/fixtures/bouncing_n10_s2_AA_LMB_LBP.json": "ba5922fd73d8d722aa1b5213e4012d08cc866039fe499d88d3bda2461477136c",
    "benchmarks/fixtures/bouncing_n10_s2_GA_LMB_LBP.json": "f31b3a1a238a97da1ed9514b497400044baf61c4d6712f7d2d270e30a6857d46",
    "benchmarks/fixtures/bouncing_n10_s2_IC_LMB_LBP.json": "94fac870d6a322db9d4857585039a5828139fc2149150ba1d812361b1d3b2b92",
    "benchmarks/fixtures/bouncing_n10_s2_PU_LMB_LBP.json": "265b5cb50a69184147e982f8fbf6e33b1927beba8de6681bcbc820bf98b4e0ef",
    "benchmarks/fixtures/bouncing_n10_s4_AA_LMB_LBP.json": "aa2476c5fcc007a92eceea26542bb8dab97eca624ab61fd6c5d56239286620f8",
    "benchmarks/fixtures/bouncing_n10_s4_GA_LMB_LBP.json": "acc98c77985b7aa33964bd873ae235844d1cfd4d95a6acaa4df428d3b9dc8cf2",
    "benchmarks/fixtures/bouncing_n10_s4_IC_LMB_LBP.json": "84628029dcbf664e753337a36fc584fb847ff39063ec8fbc395c585e5d0662ab",
    "benchmarks/fixtures/bouncing_n10_s4_PU_LMB_LBP.json": "ba148f88b411315403537d1b3e6d738873c4e7abd1c05031151264b5ac477a38",
    "benchmarks/fixtures/bouncing_n5_s1_LMB_LBP.json": "327838cfb6844db6bdff8f7fc08f1b0f3e739ac4afe97e5dd0dca49cd803575d",
    "benchmarks/fixtures/bouncing_n5_s2_AA_LMB_LBP.json": "64068a27afef32228fe37f7f7688ed54c746f2860464b2744e9252277a0fde23",
    "benchmarks/fixtures/bouncing_n5_s2_GA_LMB_LBP.json": "fff5cfc2858b562f802e0b0d4984130d2294e2d92092c685b86db1f965e889d2",
    "benchmarks/fixtures/bouncing_n5_s2_IC_LMB_LBP.json": "1730c00c4de3b107427ec4fa04ed97fc8b96ef0a574aee45708205c73a868b1b",
    "benchmarks/fixtures/bouncing_n5_s2_PU_LMB_LBP.json": "5d65d25c9f5c0ebdb46b3d2eb40876c8402fe67a0d6de3b8b132cb1d02fc3e80",
    "benchmarks/scenarios/bouncing_n10_s1.json": "19263b3f5a27cd38b2efeb4c210f5e14dc00031ebc2dc6508b73c46b53b3ae34",
    "benchmarks/scenarios/bouncing_n10_s2.json": "df0728206e4e435d6b7e8acf1c145769c46b4d5a515b40f9dd8e38708e249716",
    "benchmarks/scenarios/bouncing_n10_s4.json": "0872e2a71924985a7b28bf92e06d3a4782a4308cacb2bfb78cbaa2f123b7845f",
    "benchmarks/scenarios/bouncing_n20_s1.json": "f59869556be146268d9b4b7aa0d3ff5feb969713b8d6df3224de9b8c732006de",
    "benchmarks/scenarios/bouncing_n20_s2.json": "c4146f2db82bc4307942d5b7d93914f89027d092572fd539d6a8d2a37ba0da58",
    "benchmarks/scenarios/bouncing_n20_s4.json": "8e85a59f67ae29f333259e43dfb94f16e99c570725889932ec7d4e34517f86bf",
    "benchmarks/scenarios/bouncing_n20_s8.json": "a00d66b8b41c91bbe2235acc065cda265e2ab66cc45a757798defb881730f6e0",
    "benchmarks/scenarios/bouncing_n50_s8.json": "ca0db5927a90fc473ff9d0d2db2c4e6c085d5f4b600e46950f31eea59bcb9375",
    "benchmarks/scenarios/bouncing_n5_s1.json": "53ec0fb776161321aafa1cb905a0c7eac6bff419c7c2de4d87e2130f3b4e496c",
    "benchmarks/scenarios/bouncing_n5_s2.json": "0b49e83c7f8b7ff5d3ad474514300e4f6d2c119a45b39bb4e1fa91e640c6dfeb",
    "tests/data/multisensor_clutter_trial_42_quick.json": "12852fef1a17239890e9a68e3d47ad325232d2c95590785568524b064895b789",
    "tests/data/multisensor_detection_trial_42_quick.json": "4af815974c74fcd51c464629d78b55ec4b2b3e3e640de3d197675a2cc25f0b4a",
    "tests/data/multisensor_trial_42.json": "b89dc9ec04884082961d4a450185fbf524e10e30710d59e6855ab827e26c8ffa",
    "tests/data/single_detection_trial_42_quick.json": "abfa55cc9b35f5f3fda6be4447d7321c70b1c9a20063176c5dc086312d730d62",
    "tests/data/single_trial_42.json": "e284696e94b06fdc33764151337e1408165472559d18f87120c0b8c591fa5a97",
    "tests/data/single_trial_42_quick.json": "89238583fc5e7bb8ff744436159e77b1f126b60b0d7610e5282b13597535e176",
    "tests/data/step_by_step/aa_lmb_step_by_step_seed42.json": "73517b7efcdce709f3f1a822f2089786d380538cd2643778c19d7f138de540c1",
    "tests/data/step_by_step/ga_lmb_step_by_step_seed42.json": "25b07a534392ce3b9278a2e24debd6564f540d8f62f44c1fcb0c291f50df12f6",
    "tests/data/step_by_step/ic_lmb_step_by_step_seed42.json": "2c03b924b39f0baadf7b17bbf3dd121da2871676a7359490ad39a21254333579",
    "tests/data/step_by_step/lmb_step_by_step_seed42.json": "2d3cee2619955ffc1d18537d4cb076f24213de60cf133fde5f3fb7d5a76d666b",
    "tests/data/step_by_step/lmbm_step_by_step_seed42.json": "f9b2ac9a9974ddaa25355eb0e850148b632171199e135f0d2e1dd67a109d5548",
    "tests/data/step_by_step/multisensor_lmb_step_by_step_seed42.json": "1285e690806702d5ffe3479f38ff2aa3813375b23c5d79e916d7b6a315e0f2a6",
    "tests/data/step_by_step/multisensor_lmbm_step_by_step_seed42.json": "b96d3e9388eea25bfe281e2d269482ae3a5ac766911384e84d66f54013b3de5b",
    "tests/data/step_by_step/pu_lmb_step_by_step_seed42.json": "b967984080bf578015d2718d4374ea743655dd9319b3cd01b8837a4e4530c5af",
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
        scan_roots = ["benchmarks", "tests/data"]

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
