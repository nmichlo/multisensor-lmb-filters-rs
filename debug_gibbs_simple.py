#!/usr/bin/env python3
"""Debug script to check Gibbs sampling implementation."""

import json

import numpy as np

# Load fixture
with open("tests/data/step_by_step/multisensor_lmbm_step_by_step_seed42.json") as f:
    fixture = json.load(f)

print("=== Fixture Analysis ===")
print(f"Seed: {fixture['seed']}")
print(f"Number of sensors: {fixture['numberOfSensors']}")

# Measurements
measurements = fixture["measurements"]
print("\nMeasurements per sensor:")
for i, sensor_meas in enumerate(measurements):
    print(f"  Sensor {i+1}: {len(sensor_meas)} measurements")

# We'll infer number of tracks from the Gibbs output later

# Gibbs input
gibbs_input = fixture["step3_gibbs"]["input"]
print(f"\nGibbs input log-likelihood dimensions: {gibbs_input['L_dims']}")
print(f"Gibbs number of samples: {gibbs_input['numberOfSamples']}")

# Gibbs output
gibbs_output = fixture["step3_gibbs"]["output"]
samples = np.array(gibbs_output["A"])
print("\n=== MATLAB Gibbs Output ===")
print(f"Shape: {samples.shape}  # (unique_samples, num_objects * num_sensors)")
print(f"Number of UNIQUE samples: {len(samples)}")
print(
    f"\nAll {len(samples)} unique samples (rows are samples, columns are [obj1_sensor1, obj2_sensor1, obj3_sensor1, obj4_sensor1, obj1_sensor2, obj2_sensor2, obj3_sensor2, obj4_sensor2]):"
)
for i, sample in enumerate(samples):
    # Reshape to show per-object per-sensor
    n_sensors = fixture["numberOfSensors"]
    n_objects = len(sample) // n_sensors
    reshaped = sample.reshape(n_sensors, n_objects).T  # Transpose to get objects as rows
    print(f"  Sample {i+1:2d}: {sample.tolist()} -> Objects×Sensors:\n{reshaped}")
print("\nKey: 0 = miss, k>0 = measurement k assigned to this object from this sensor")
