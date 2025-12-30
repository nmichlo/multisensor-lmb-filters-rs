"""
Benchmark scenario generator for multi-object tracking filters.

Generates JSON files containing bouncing-box tracking scenarios that can be
loaded by Rust, Python, and MATLAB benchmark runners.
"""

import json
import random
from pathlib import Path

# Hard-coded configs: (num_objects, num_sensors)
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
BOUNDS = (-100.0, 100.0, -100.0, 100.0)  # x_min, x_max, y_min, y_max
SEED = 42


def generate_bouncing_scenario(n_objects: int, n_sensors: int, n_steps: int, seed: int) -> dict:
    """Generate a bouncing-box tracking scenario."""
    random.seed(seed)

    # Initialize objects with random positions and velocities
    objects = []
    for _ in range(n_objects):
        x = random.uniform(BOUNDS[0], BOUNDS[1])
        y = random.uniform(BOUNDS[2], BOUNDS[3])
        vx = random.gauss(0, 5)
        vy = random.gauss(0, 5)
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

        # Generate sensor readings: [x, y, object_id] where object_id=-1 for clutter
        sensor_readings = []
        for _ in range(n_sensors):
            readings = []

            # True detections (with probability of detection)
            for obj_id, obj in enumerate(objects):
                if random.random() < 0.95:
                    readings.append(
                        [obj[0] + random.gauss(0, 3), obj[1] + random.gauss(0, 3), obj_id]
                    )

            # Clutter (approximately Poisson(2)), id=-1
            n_clutter = int(random.expovariate(0.5))  # ~Poisson(2)
            for _ in range(n_clutter):
                readings.append(
                    [random.uniform(BOUNDS[0], BOUNDS[1]), random.uniform(BOUNDS[2], BOUNDS[3]), -1]
                )

            sensor_readings.append(readings)

        steps.append({"step": t, "sensor_readings": sensor_readings})

    return {
        "type": "bouncing",
        "measurement_format": "x_y_id",
        "seed": seed,
        "num_objects": n_objects,
        "num_sensors": n_sensors,
        "num_steps": n_steps,
        "bounds": list(BOUNDS),
        "init_velocity_std": 5.0,
        "model": {
            "dt": 1.0,
            "process_noise_std": 1.0,
            "measurement_noise_std": 3.0,
            "detection_probability": 0.95,
            "survival_probability": 0.99,
            "clutter_rate": 2.0,
            "birth_locations": [
                [0.0, 0.0, 0.0, 0.0],
                [50.0, 50.0, 0.0, 0.0],
                [-50.0, -50.0, 0.0, 0.0],
                [50.0, -50.0, 0.0, 0.0],
            ],
        },
        "steps": steps,
    }


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


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "scenarios"
    out_dir.mkdir(parents=True, exist_ok=True)

    for n_obj, n_sens in CONFIGS:
        scenario = generate_bouncing_scenario(n_obj, n_sens, NUM_STEPS, SEED)
        path = out_dir / f"bouncing_n{n_obj}_s{n_sens}.json"
        write_compact_json(scenario, path)
        print(f"Generated {path}")

    print(f"\nGenerated {len(CONFIGS)} scenario files in {out_dir}")
