#!/usr/bin/env python3
"""Debug script to check Gibbs sampling output."""

import json

import numpy as np
from multisensor_lmb_filters_rs import FilterMultisensorLmbm

# Load fixture
with open("tests/data/step_by_step/multisensor_lmbm_step_by_step_seed42.json") as f:
    fixture = json.load(f)

print("=== MATLAB Results ===")
matlab_samples = np.array(fixture["step3_gibbs"]["output"]["A"])
print(f"Shape: {matlab_samples.shape}")
print(f"Number of unique samples: {len(matlab_samples)}")
print("First 3 samples:")
for i in range(min(3, len(matlab_samples))):
    print(f"  {matlab_samples[i]}")

# Create Rust filter
import sys

sys.path.append("tests")
from test_equivalence import (
    make_birth_model_empty,
    make_motion_model,
    make_multisensor_config,
    nested_measurements_to_numpy,
)

model = fixture["model"]
motion = make_motion_model(model)
sensor_config = make_multisensor_config(model)
birth = make_birth_model_empty()
measurements = nested_measurements_to_numpy(fixture["measurements"])

filter = FilterMultisensorLmbm(motion, sensor_config, birth, seed=fixture["seed"])

# Load prior tracks
from test_equivalence import make_track_from_fixture

prior_prediction = fixture["step1_prediction"]["output"]
tracks = [make_track_from_fixture(obj) for obj in prior_prediction["newobjects"]]
filter.set_tracks(tracks)

# Run step
output = filter.step_detailed(measurements, timestep=fixture["timestep"])

print("\n=== Rust Results ===")
if output.association_result and output.association_result.assignments:
    rust_samples = np.array(output.association_result.assignments)
    print(f"Shape: {rust_samples.shape}")
    print(f"Number of unique samples: {len(rust_samples)}")
    print("First 3 samples:")
    for i in range(min(3, len(rust_samples))):
        print(f"  {rust_samples[i]}")
    print("\nAll unique samples:")
    for i in range(len(rust_samples)):
        print(f"  Sample {i+1}: {rust_samples[i]}")
else:
    print("No association result!")

print("\n=== Comparison ===")
print(f"MATLAB: {len(matlab_samples)} unique samples")
print(
    f"Rust:   {len(rust_samples) if output.association_result and output.association_result.assignments else 0} unique samples"
)
