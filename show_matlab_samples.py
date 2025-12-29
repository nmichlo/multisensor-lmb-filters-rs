#!/usr/bin/env python3
"""Show MATLAB Gibbs samples."""

import json

import numpy as np

with open("tests/data/step_by_step/multisensor_lmbm_step_by_step_seed42.json") as f:
    fixture = json.load(f)

samples = np.array(fixture["step3_gibbs"]["output"]["A"])
n_sensors = fixture["numberOfSensors"]
n_objects = len(samples[0]) // n_sensors

print("MATLAB Gibbs Sampling (seed=42):")
print(f"  Sensors: {n_sensors}")
print(f"  Objects (tracks): {n_objects}")
print("  Total samples generated: 1000")
print(f"  Unique samples: {len(samples)}")
print(f"\nAll {len(samples)} unique samples:")
print("(Format: [obj0_s0, obj1_s0, obj2_s0, obj3_s0, obj0_s1, obj1_s1, obj2_s1, obj3_s1])")
print("where objX_sY = measurement assigned to object X from sensor Y (0=miss)")
for i, sample in enumerate(samples):
    print(f"  {i+1:2d}. {sample.tolist()}")
