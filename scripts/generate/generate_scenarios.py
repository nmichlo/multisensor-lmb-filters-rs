"""
Benchmark scenario generator for multi-object tracking filters.

Generates JSON files containing bouncing-box tracking scenarios that can be
loaded by Rust, Python, and MATLAB benchmark runners.
"""

import json
import random
from pathlib import Path

# =============================================================================
# SCENARIO CONFIGURATIONS
# =============================================================================

# Scenario matrix: (num_objects, num_sensors)
CONFIGS = [
    (5, 1),
    (10, 1),
    (20, 1),
    (5, 2),
    (10, 2),
    (20, 2),
    (10, 4),
    (20, 4),
    (20, 8),
    (50, 8),
]

NUM_STEPS = 100
SEED = 42

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

# Spatial bounds
BOUNDS = (-100.0, 100.0, -100.0, 100.0)  # x_min, x_max, y_min, y_max
SPAWN_MARGIN = 10.0  # Margin from walls for initial spawn

# Object dynamics
INIT_VELOCITY_STD = 3.0  # Initial velocity standard deviation

# Sensor model
DETECTION_PROBABILITY = 0.98
MEASUREMENT_NOISE_STD = 1.0
CLUTTER_MIN = 0
CLUTTER_MAX = 2

# Motion model
DT = 1.0
PROCESS_NOISE_STD = 0.5
SURVIVAL_PROBABILITY = 0.99

# Birth model - dense grid covering the space
BIRTH_GRID_POINTS = [-75, -25, 25, 75]
BIRTH_LOCATIONS = [[x, y] for x in BIRTH_GRID_POINTS for y in BIRTH_GRID_POINTS]

# =============================================================================
# SCENARIO GENERATOR
# =============================================================================


def generate_bouncing_scenario(n_objects: int, n_sensors: int, n_steps: int, seed: int) -> dict:
    """Generate a bouncing-box tracking scenario with random object spawns."""
    random.seed(seed)

    # Spawn objects randomly across the space
    objects = []
    for _ in range(n_objects):
        x = random.uniform(BOUNDS[0] + SPAWN_MARGIN, BOUNDS[1] - SPAWN_MARGIN)
        y = random.uniform(BOUNDS[2] + SPAWN_MARGIN, BOUNDS[3] - SPAWN_MARGIN)
        vx = random.gauss(0, INIT_VELOCITY_STD)
        vy = random.gauss(0, INIT_VELOCITY_STD)
        objects.append([x, y, vx, vy])

    steps = []
    for t in range(n_steps):
        # Update positions (constant velocity motion)
        for obj in objects:
            obj[0] += obj[2]  # x += vx
            obj[1] += obj[3]  # y += vy

            # Bounce off walls
            if obj[0] < BOUNDS[0]:
                obj[0] = BOUNDS[0] + (BOUNDS[0] - obj[0])
                obj[2] *= -1
            elif obj[0] > BOUNDS[1]:
                obj[0] = BOUNDS[1] - (obj[0] - BOUNDS[1])
                obj[2] *= -1

            if obj[1] < BOUNDS[2]:
                obj[1] = BOUNDS[2] + (BOUNDS[2] - obj[1])
                obj[3] *= -1
            elif obj[1] > BOUNDS[3]:
                obj[1] = BOUNDS[3] - (obj[1] - BOUNDS[3])
                obj[3] *= -1

        # Ground truth positions (before noise)
        ground_truth = [[obj[0], obj[1]] for obj in objects]

        # Generate sensor readings
        sensor_readings = []
        for _ in range(n_sensors):
            readings = []

            # Detections with noise
            for obj in objects:
                if random.random() < DETECTION_PROBABILITY:
                    readings.append(
                        [
                            obj[0] + random.gauss(0, MEASUREMENT_NOISE_STD),
                            obj[1] + random.gauss(0, MEASUREMENT_NOISE_STD),
                        ]
                    )

            # Clutter
            n_clutter = random.randint(CLUTTER_MIN, CLUTTER_MAX)
            for _ in range(n_clutter):
                readings.append(
                    [
                        random.uniform(BOUNDS[0], BOUNDS[1]),
                        random.uniform(BOUNDS[2], BOUNDS[3]),
                    ]
                )

            sensor_readings.append(readings)

        steps.append(
            {
                "step": t,
                "object_ids": list(range(n_objects)),
                "ground_truth": ground_truth,
                "sensor_readings": sensor_readings,
            }
        )

    # Compute effective clutter rate (average clutter per frame)
    clutter_rate = (CLUTTER_MIN + CLUTTER_MAX) / 2.0

    return {
        "type": "bouncing",
        "measurement_format": "x_y",
        "seed": seed,
        "num_objects": n_objects,
        "num_sensors": n_sensors,
        "num_steps": n_steps,
        "bounds": list(BOUNDS),
        "spawn_margin": SPAWN_MARGIN,
        "init_velocity_std": INIT_VELOCITY_STD,
        "model": {
            "dt": DT,
            "process_noise_std": PROCESS_NOISE_STD,
            "measurement_noise_std": MEASUREMENT_NOISE_STD,
            "detection_probability": DETECTION_PROBABILITY,
            "survival_probability": SURVIVAL_PROBABILITY,
            "clutter_rate": clutter_rate,
            "clutter_min": CLUTTER_MIN,
            "clutter_max": CLUTTER_MAX,
            "birth_locations": [[b[0], b[1], 0.0, 0.0] for b in BIRTH_LOCATIONS],
        },
        "steps": steps,
    }


# =============================================================================
# OUTPUT UTILITIES
# =============================================================================


def round_floats(obj, decimals=3):
    """Recursively round floats in nested structure."""
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, list):
        return [round_floats(x, decimals) for x in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    return obj


def write_compact_json(scenario: dict, path: Path):
    """Write JSON with compact steps but readable metadata."""
    with open(path, "w") as f:
        f.write("{\n")
        f.write(f'  "type": "{scenario["type"]}",\n')
        f.write(f'  "measurement_format": "{scenario["measurement_format"]}",\n')
        f.write(f'  "seed": {scenario["seed"]},\n')
        f.write(f'  "num_objects": {scenario["num_objects"]},\n')
        f.write(f'  "num_sensors": {scenario["num_sensors"]},\n')
        f.write(f'  "num_steps": {scenario["num_steps"]},\n')
        f.write(f'  "bounds": {json.dumps(scenario["bounds"])},\n')
        f.write(f'  "spawn_margin": {scenario["spawn_margin"]},\n')
        f.write(f'  "init_velocity_std": {scenario["init_velocity_std"]},\n')
        f.write(f'  "model": {json.dumps(scenario["model"])},\n')
        # Steps: one line per step
        f.write('  "steps": [\n')
        for i, step in enumerate(scenario["steps"]):
            step_rounded = round_floats(step)
            step_json = json.dumps(step_rounded, separators=(",", ":"))
            comma = "," if i < len(scenario["steps"]) - 1 else ""
            f.write(f"    {step_json}{comma}\n")
        f.write("  ]\n")
        f.write("}\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    out_dir = Path(__file__).parent / "scenarios"
    out_dir.mkdir(parents=True, exist_ok=True)

    for n_obj, n_sens in CONFIGS:
        scenario = generate_bouncing_scenario(n_obj, n_sens, NUM_STEPS, SEED)
        path = out_dir / f"bouncing_n{n_obj}_s{n_sens}.json"
        write_compact_json(scenario, path)
        print(f"Generated {path}")

    print(f"\nGenerated {len(CONFIGS)} scenario files in {out_dir}")
