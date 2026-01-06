#!/usr/bin/env bash
#
# Unified benchmark runner for LMB filter performance comparison.
# Runs benchmarks across Octave, Rust (Native), and Python (Typical) implementations.
#
# Usage:
#   ./benchmarks/run_benchmarks.sh                    # Full suite
#   ./benchmarks/run_benchmarks.sh --quick            # Python only
#   ./benchmarks/run_benchmarks.sh --timeout 5        # Custom timeout
#   ./benchmarks/run_benchmarks.sh --lang rust        # Single language
#

set -e

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/results"
MATLAB_DIR="$PROJECT_ROOT/../multisensor-lmb-filters"

# Defaults
TIMEOUT=10
LANGUAGES="octave,rust,python"

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            LANGUAGES="python"
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --lang)
            LANGUAGES="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --quick           Run Python benchmarks only"
            echo "  --timeout N       Set timeout per benchmark in seconds (default: 10)"
            echo "  --lang LANGS      Comma-separated list of languages to run (octave, rust, python)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Setup
# =============================================================================

cd "$PROJECT_ROOT"
mkdir -p "$RESULTS_DIR"

echo "======================================="
echo "LMB Filter Benchmark Suite"
echo "======================================="
echo "Timeout: ${TIMEOUT}s"
echo "Languages: $LANGUAGES"
echo ""

# =============================================================================
# Helper: Octave Script (Embedded)
# =============================================================================
read -r -d '' OCTAVE_BENCHMARK_SCRIPT << 'EOF' || true
% Octave benchmark runner
% This script runs benchmarks using the MATLAB implementation in ../multisensor-lmb-filters

% =============================================================================
% HELPER FUNCTIONS (must be defined before use in Octave)
% =============================================================================

function model = buildMatlabModel(scenario, numSteps, thresholds)
    m = scenario.model;
    model.xDimension = 4; model.zDimension = 2; model.T = m.dt;
    model.survivalProbability = m.survival_probability; model.existenceThreshold = thresholds.existence;
    model.A = [eye(2), m.dt*eye(2); zeros(2), eye(2)]; model.u = zeros(4, 1);
    q = m.process_noise_std^2;
    model.R = q * [(1/3)*m.dt^3*eye(2), 0.5*m.dt^2*eye(2); 0.5*m.dt^2*eye(2), m.dt*eye(2)];
    model.observationSpaceLimits = [scenario.bounds(1), scenario.bounds(2); scenario.bounds(3), scenario.bounds(4)];
    model.observationSpaceVolume = prod(model.observationSpaceLimits(:,2) - model.observationSpaceLimits(:,1));
    model.C = [eye(2), zeros(2)]; model.Q = m.measurement_noise_std^2 * eye(2);
    model.detectionProbability = m.detection_probability; model.clutterRate = m.clutter_rate;
    model.clutterPerUnitVolume = m.clutter_rate / model.observationSpaceVolume;
    
    birthLocs = m.birth_locations;
    model.numberOfBirthLocations = size(birthLocs, 1);
    model.birthLocationLabels = 1:model.numberOfBirthLocations;
    model.rB = 0.01 * ones(model.numberOfBirthLocations, 1);
    model.muB = cell(model.numberOfBirthLocations, 1); model.SigmaB = cell(model.numberOfBirthLocations, 1);
    for i = 1:model.numberOfBirthLocations, model.muB{i} = birthLocs(i, :)'; model.SigmaB{i} = diag([2500, 2500, 100, 100]); end
    
    object.birthLocation = 0; object.birthTime = 0; object.r = 0; object.numberOfGmComponents = 0;
    object.w = zeros(0, 1); object.mu = {}; object.Sigma = {}; object.trajectoryLength = 0;
    object.trajectory = repmat(80 * ones(model.xDimension, 1), 1, 100); object.timestamps = zeros(1, 100);
    birthParameters = repmat(object, model.numberOfBirthLocations, 1);
    for i = 1:model.numberOfBirthLocations
        birthParameters(i).birthLocation = model.birthLocationLabels(i); birthParameters(i).mu = model.muB(i); birthParameters(i).Sigma = model.SigmaB(i); 
        birthParameters(i).r = model.rB(i); birthParameters(i).numberOfGmComponents=1; birthParameters(i).w=1;
    end
    model.birthParameters = birthParameters;
    model.lbpConvergenceTolerance = 1e-6; model.maximumNumberOfLbpIterations = 100;
    model.numberOfSamples = 1000; model.numberOfAssignments = 25;
    model.weightThreshold = thresholds.gm_weight; model.gmWeightThreshold = thresholds.gm_weight;
    model.maximumNumberOfGmComponents = thresholds.max_components; model.mahalanobisDistanceThreshold = thresholds.gm_merge;
    model.minimumTrajectoryLength = 3; model.object = repmat(object, 0, 1); model.simulationLength = numSteps;
endfunction

function z = convertToMeasCell(meas)
    if isempty(meas) z = {}; return; end
    if iscell(meas) z = meas; return; end
    if size(meas, 1) == 2 && size(meas, 2) ~= 2 meas = meas'; end
    z = cell(1, size(meas, 1)); for k=1:size(meas, 1) z{k} = meas(k, :)'; end
endfunction

function measurements = extractMeasurements(scenario, numSensors, numSteps)
    measurements = cell(1, numSteps);
    for t = 1:numSteps
        sr = scenario.steps(t).sensor_readings;
        if isempty(sr) || (isnumeric(sr) && numel(sr) == 0) measurements{t} = {};
        elseif iscell(sr) measurements{t} = convertToMeasCell(sr{1});
        else measurements{t} = convertToMeasCell(squeeze(sr)); end
    end
endfunction

% =============================================================================
% MAIN SCRIPT
% =============================================================================

clc;
scriptDir = fileparts(mfilename('fullpath'));
matlabDir = fullfile(scriptDir, '..', '..', 'multisensor-lmb-filters');
addpath(genpath(matlabDir));

scenariosDir = fullfile(scriptDir, 'scenarios');
files = dir(fullfile(scenariosDir, 'bouncing_*.json'));

fprintf('Scenario               | Filter             |  Time(ms) |    OSPA | Progress\n');
fprintf('---------------------------------------------------------------------------\n');

filterConfigs = {
    'LMB-LBP',   'LMB',  'LBP',   struct('max_iterations', 100, 'tolerance', 1e-6), '';
};

thresholds = struct('existence', 1e-3, 'gm_weight', 1e-4, 'max_components', 100, 'gm_merge', inf);
seed = 42;

for i = 1:numel(files)
    scenario_name = strrep(files(i).name, '.json', '');
    if ~isempty(strfind(scenario_name, '_s2')) || ~isempty(strfind(scenario_name, '_s4')) || ~isempty(strfind(scenario_name, '_s8')) || ~isempty(strfind(scenario_name, 'n50')) || ~isempty(strfind(scenario_name, 'n20'))
        continue;
    end
    
    scenarioPath = fullfile(scenariosDir, files(i).name);
    scenario = jsondecode(fileread(scenarioPath));
    numSteps = scenario.num_steps;
    
    model = buildMatlabModel(scenario, numSteps, thresholds);
    measurements = extractMeasurements(scenario, scenario.num_sensors, numSteps);
    
    for fIdx = 1:size(filterConfigs, 1)
        filterName = filterConfigs{fIdx, 1};
        filterType = filterConfigs{fIdx, 2};
        assocType = filterConfigs{fIdx, 3};
        
        try
            model.dataAssociationMethod = assocType;
            rng = SimpleRng(seed);
            tic;
            if strcmp(filterType, 'LMB'), [~, ~] = runLmbFilter(rng, model, measurements);
            else end
            elapsed = toc * 1000;
            fprintf('%-22s | %-18s | %9.1f |       - | %4d/%d\n', scenario_name, filterName, elapsed, numSteps, numSteps);
            fflush(stdout); 
        catch err
            fprintf('%-22s | %-18s |     ERROR |       - |        -\n', scenario_name, filterName);
            disp(err.message);
        end
    end
end
EOF

# =============================================================================
# Helper: Rust Script (Embedded)
# =============================================================================
read -r -d '' RUST_BENCHMARK_SCRIPT << 'EOF' || true
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::Instant;
use serde::Deserialize;
use multisensor_lmb_filters_rs::lmb::*;
use nalgebra::{DVector, DMatrix};
use rand::prelude::*;

// =============================================================================
// JSON Schema (matches scenario files)
// =============================================================================

#[derive(Deserialize)]
struct ScenarioJson {
    model: ModelJson,
    bounds: [f64; 4],
    num_sensors: usize,
    steps: Vec<StepJson>,
}

#[derive(Deserialize)]
struct ModelJson {
    dt: f64,
    process_noise_std: f64,
    measurement_noise_std: f64,
    detection_probability: f64,
    survival_probability: f64,
    clutter_rate: f64,
    birth_locations: Vec<[f64; 4]>,
}

#[derive(Deserialize)]
struct StepJson {
    #[allow(dead_code)]
    step: usize,
    sensor_readings: Option<Vec<Vec<[f64; 2]>>>,
}

// =============================================================================
// Filter Configs - MUST MATCH PYTHON EXACTLY
// =============================================================================

struct Config { name: &'static str, filter_type: &'static str, assoc: &'static str }
const CONFIGS: &[Config] = &[
    Config { name: "LMB-LBP", filter_type: "LMB", assoc: "LBP" },
    // Config { name: "LMB-Gibbs", filter_type: "LMB", assoc: "Gibbs" },
];

// THRESHOLDS - Match Python: FilterThresholds(existence=1e-3, gm_weight=1e-4, max_components=100, gm_merge=inf)
const EXISTENCE_THRESHOLD: f64 = 1e-3;
const GM_WEIGHT_THRESHOLD: f64 = 1e-4;
const MAX_GM_COMPONENTS: usize = 100;
const GM_MERGE_THRESHOLD: f64 = f64::INFINITY;

// =============================================================================
// Preprocessing - Match Python preprocess() exactly
// =============================================================================

struct PreprocessedScenario {
    motion: MotionModel,
    sensor: SensorModel,
    birth: BirthModel,
    steps: Vec<(usize, Vec<DVector<f64>>)>,  // (timestep, single_sensor_measurements)
}

fn preprocess(scenario: &ScenarioJson) -> PreprocessedScenario {
    // Motion model - matches Python: MotionModel.constant_velocity_2d(dt, process_noise_std, survival_probability)
    let motion = MotionModel::constant_velocity_2d(
        scenario.model.dt, 
        scenario.model.process_noise_std, 
        scenario.model.survival_probability
    );
    
    // Observation volume - matches Python: (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    let obs_vol = (scenario.bounds[1] - scenario.bounds[0]) * (scenario.bounds[3] - scenario.bounds[2]);
    
    // Sensor model - matches Python: SensorModel.position_2d(measurement_noise_std, detection_probability, clutter_rate, obs_vol)
    let sensor = SensorModel::position_sensor_2d(
        scenario.model.measurement_noise_std, 
        scenario.model.detection_probability, 
        scenario.model.clutter_rate, 
        obs_vol
    );
    
    // Birth locations - matches Python: BirthLocation(i, np.array(loc), np.diag([2500.0, 2500.0, 100.0, 100.0]))
    let birth_locs: Vec<_> = scenario.model.birth_locations.iter().enumerate().map(|(i, &loc)| {
        BirthLocation::new(
            i, 
            DVector::from_vec(vec![loc[0], loc[1], loc[2], loc[3]]), 
            DMatrix::from_diagonal(&DVector::from_vec(vec![2500.0, 2500.0, 100.0, 100.0]))
        )
    }).collect();
    
    // Birth model - matches Python: BirthModel(birth_locs, lmb_existence=0.01, lmbm_existence=0.001)
    let birth = BirthModel::new(birth_locs, 0.01, 0.001);
    
    // Steps - matches Python: steps.append((t, single_meas, multi_meas))
    // For single sensor, we use multi_meas[0]
    let steps: Vec<_> = scenario.steps.iter().enumerate().map(|(t, step)| {
        let readings = step.sensor_readings.as_ref();
        let single_meas: Vec<DVector<f64>> = if let Some(rss) = readings {
            if !rss.is_empty() && !rss[0].is_empty() {
                rss[0].iter().map(|m| DVector::from_vec(vec![m[0], m[1]])).collect()
            } else {
                vec![]
            }
        } else {
            vec![]
        };
        (t, single_meas)
    }).collect();
    
    PreprocessedScenario { motion, sensor, birth, steps }
}

// =============================================================================
// Filter runner - Match Python run_filter() exactly  
// =============================================================================

fn run_filter(
    filter: &mut LmbFilter<DynamicAssociator>, 
    steps: &[(usize, Vec<DVector<f64>>)],
    rng: &mut StdRng,
    timeout_secs: u64,
) -> (f64, usize, bool) {
    let start = Instant::now();
    
    for (t, single_meas) in steps.iter() {
        if start.elapsed().as_secs() >= timeout_secs {
            return (start.elapsed().as_micros() as f64 / 1000.0, *t, true);
        }
        // Match Python: result = filt.step(single_meas, t)
        let _result = filter.step(rng, single_meas, *t).unwrap();
    }
    
    let elapsed_ms = start.elapsed().as_micros() as f64 / 1000.0;
    (elapsed_ms, steps.len(), false)
}

// =============================================================================
// Main - Match Python main() structure exactly
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let timeout_secs = args.iter().position(|s| s == "--timeout")
        .map(|i| args[i+1].parse::<u64>().unwrap())
        .unwrap_or(10);
    
    println!("{:<22} | {:<18} | {:>9} | {:>7} | {:>8}", "Scenario", "Filter", "Time(ms)", "OSPA", "Progress");
    println!("{}", "-".repeat(75));

    // Sort scenarios by (n, s) - matches Python sort order
    let scenarios_dir = Path::new("benchmarks/scenarios");
    let mut entries: Vec<_> = fs::read_dir(scenarios_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
        .collect();
    
    entries.sort_by_key(|e| {
        let name = e.file_name().to_string_lossy().to_string();
        let n: u32 = name.split("n").nth(1).and_then(|s| s.split("_").next()).and_then(|s| s.parse().ok()).unwrap_or(999);
        let s: u32 = name.split("_s").nth(1).and_then(|s| s.split(".").next()).and_then(|s| s.parse().ok()).unwrap_or(999);
        (n, s)
    });

    let mut timed_out_filters: HashSet<&str> = HashSet::new();

    for entry in &entries {
        let path = entry.path();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();

        // Load and preprocess - matches Python: scenario = json.load(...); preprocess(scenario)
        let content = fs::read_to_string(&path)?;
        let scenario: ScenarioJson = serde_json::from_str(&content)?;
        let preprocessed = preprocess(&scenario);

        for config in CONFIGS {
            // Skip timed out filters - matches Python timed_out_filters logic
            if timed_out_filters.contains(config.name) {
                println!("{:<22} | {:<18} | {:>9} | {:>7} | {:>8}", name, config.name, "SKIP", "-", "-");
                continue;
            }

            // Association config - matches Python: AssociatorConfig.lbp(100, 1e-6)
            let assoc_config = match config.assoc {
                "LBP" => AssociationConfig::lbp(100, 1e-6),
                "Gibbs" => AssociationConfig::gibbs(1000),
                _ => AssociationConfig::default()
            };
            
            // Create filter with thresholds - match Python: filter_cls(motion, sensor, birth, assoc_fn(), THRESHOLDS)
            let associator = DynamicAssociator::from_config(&assoc_config);
            let mut filter = LmbFilter::with_associator_type(
                preprocessed.motion.clone(),
                preprocessed.sensor.clone(),
                preprocessed.birth.clone(),
                assoc_config,
                associator,
            )
            .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
            .with_gm_merge_threshold(GM_MERGE_THRESHOLD);
            
            // Set existence threshold (need builder method or direct access)
            // Note: existence threshold set via filter defaults matching Python's 1e-3
            
            let mut rng = StdRng::seed_from_u64(42);
            
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_filter(&mut filter, &preprocessed.steps, &mut rng, timeout_secs)
            }));
            
            match result {
                Ok((elapsed_ms, steps_done, is_timeout)) => {
                    if is_timeout {
                        println!("{:<22} | {:<18} | {:>9} | {:>7} | {:>8}", name, config.name, "TIMEOUT", "-", "?");
                        timed_out_filters.insert(config.name);
                    } else {
                        println!("{:<22} | {:<18} | {:>9.1} | {:>7} | {}/{}", 
                            name, config.name, elapsed_ms, "-", steps_done, preprocessed.steps.len());
                    }
                },
                Err(_) => {
                    println!("{:<22} | {:<18} | {:>9} | {:>7} | {:>8}", name, config.name, "ERROR", "-", "?");
                }
            }
        }
    }
    Ok(())
}
EOF

# =============================================================================
# Helper: Python Script (Embedded)
# =============================================================================
read -r -d '' PY_BENCHMARK_SCRIPT << 'EOF' || true
import argparse, json, signal, time, sys, re
from pathlib import Path
import numpy as np
from multisensor_lmb_filters_rs import (AssociatorConfig, BirthLocation, BirthModel, FilterAaLmb, FilterGaLmb, FilterIcLmb, FilterLmb, FilterLmbm, FilterMultisensorLmbm, FilterPuLmb, FilterThresholds, MotionModel, SensorConfigMulti, SensorModel)

SCENARIOS_DIR = Path("benchmarks/scenarios")
THRESHOLDS = FilterThresholds(existence=1e-3, gm_weight=1e-4, max_components=100, gm_merge=float("inf"))

CONFIGS = [
    ("LMB-LBP", FilterLmb, lambda: AssociatorConfig.lbp(100, 1e-6), False),
    #("LMB-Gibbs", FilterLmb, lambda: AssociatorConfig.gibbs(1000), False),
    #("AA-LMB-LBP", FilterAaLmb, lambda: AssociatorConfig.lbp(100, 1e-6), True),
    #("IC-LMB-LBP", FilterIcLmb, lambda: AssociatorConfig.lbp(100, 1e-6), True),
    #("PU-LMB-LBP", FilterPuLmb, lambda: AssociatorConfig.lbp(100, 1e-6), True),
    #("GA-LMB-LBP", FilterGaLmb, lambda: AssociatorConfig.lbp(100, 1e-6), True),
]

class TimeoutError(Exception): pass
def on_timeout(signum, frame): raise TimeoutError()

def preprocess(scenario):
    m = scenario["model"]; bounds = scenario["bounds"]; n_sensors = scenario["num_sensors"]
    motion = MotionModel.constant_velocity_2d(m["dt"], m["process_noise_std"], m["survival_probability"])
    obs_vol = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    sensor = SensorModel.position_2d(m["measurement_noise_std"], m["detection_probability"], m["clutter_rate"], obs_vol)
    birth_locs = [BirthLocation(i, np.array(loc), np.diag([2500.0, 2500.0, 100.0, 100.0])) for i, loc in enumerate(m["birth_locations"])]
    birth = BirthModel(birth_locs, lmb_existence=0.01, lmbm_existence=0.001)
    multi_sensor = SensorConfigMulti([sensor] * n_sensors); steps = []
    for step in scenario["steps"]:
        t = step["step"]; readings = step["sensor_readings"]
        multi_meas = [np.array(s) if s else np.empty((0, 2)) for s in readings]
        single_meas = multi_meas[0] if multi_meas else np.empty((0, 2))
        steps.append((t, single_meas, multi_meas))
    return motion, sensor, multi_sensor, birth, steps

def run_filter(filt, steps, is_multi, mode):
    start = time.perf_counter()
    for i, (t, single_meas, multi_meas) in enumerate(steps):
        result = filt.step(multi_meas if is_multi else single_meas, t)
        if mode == 'python' and result.tracks: _ = np.array([[tr.mean[0], tr.mean[2]] for tr in result.tracks])
    return (time.perf_counter() - start) * 1000, len(steps)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--mode", choices=["python"], required=True)
    parser.add_argument("--timeout", type=int, default=10); args = parser.parse_args()
    try: signal.signal(signal.SIGALRM, on_timeout); use_timeout = True
    except AttributeError: use_timeout = False
    print(f"{'Scenario':<22} | {'Filter':<18} | {'Time(ms)':>9} | {'OSPA':>7} | {'Progress':>8}")
    print("-" * 75)
    scenarios = sorted(SCENARIOS_DIR.glob("bouncing_*.json"))
    scenarios.sort(key=lambda p: (int(re.search(r"n(\d+)", p.stem).group(1)), int(re.search(r"s(\d+)", p.stem).group(1))))
    timed_out_filters = set()
    for path in scenarios:
        scenario = json.load(open(path)); name = path.stem;
        motion, sensor, multi_sensor, birth, steps = preprocess(scenario)
        for filter_name, filter_cls, assoc_fn, is_multi in CONFIGS:
            if filter_name in timed_out_filters: print(f"{name:<22} | {filter_name:<18} | {'SKIP':>9} | {'-':>7} | {'-':>8}"); continue
            try:
                if use_timeout: signal.alarm(args.timeout)
                filt = filter_cls(motion, multi_sensor if is_multi else sensor, birth, assoc_fn(), THRESHOLDS)
                elapsed_ms, steps_done = run_filter(filt, steps, is_multi, args.mode)
                if use_timeout: signal.alarm(0)
                print(f"{name:<22} | {filter_name:<18} | {elapsed_ms:>9.1f} | {'-':>7} | {steps_done}/{len(steps)}")
            except TimeoutError:
                if use_timeout: signal.alarm(0)
                print(f"{name:<22} | {filter_name:<18} | {'TIMEOUT':>9} | {'-':>7} | {'?':>8}")
                timed_out_filters.add(filter_name)
            except Exception:
                if use_timeout: signal.alarm(0)
                print(f"{name:<22} | {filter_name:<18} | {'ERROR':>9} | {'-':>7} | {'?':>8}")
if __name__ == "__main__": main()
EOF

# =============================================================================
# Benchmark Runners
# =============================================================================

run_octave_benchmarks() {
    echo "🔬 Running Octave benchmarks..."
    if [ ! -d "$MATLAB_DIR" ]; then echo "⚠ MATLAB dir not found"; return; fi
    TMP_SCRIPT="$SCRIPT_DIR/run_octave_temp.m"
    echo "$OCTAVE_BENCHMARK_SCRIPT" > "$TMP_SCRIPT"
    octave --no-gui "$TMP_SCRIPT" 2>&1 | tee "$RESULTS_DIR/octave_benchmarks.txt"
    rm "$TMP_SCRIPT"
    echo "✓ Octave benchmarks complete"
    echo ""
}

run_rust_benchmarks() {
    echo "🦀 Running Rust benchmarks (Native)..."
    mkdir -p src/bin
    echo "$RUST_BENCHMARK_SCRIPT" > src/bin/benchmark_runner.rs
    cargo build --release --bin benchmark_runner
    ./target/release/benchmark_runner --timeout "$TIMEOUT" 2>&1 | tee "$RESULTS_DIR/rust_benchmarks.txt"
    rm src/bin/benchmark_runner.rs
    echo "✓ Rust benchmarks complete"
    echo ""
}

run_python_benchmarks() {
    echo "🐍 Building Python bindings (maturin develop --release)..."
    uv run maturin develop --release
    echo "🐍 Running Python benchmarks (Typical application)..."
    uv run python -c "$PY_BENCHMARK_SCRIPT" --mode "python" --timeout "$TIMEOUT" 2>&1 | tee "$RESULTS_DIR/python_benchmarks.txt"
    echo "✓ Python benchmarks complete"
    echo ""
}

# =============================================================================
# Execution
# =============================================================================

IFS=',' read -ra LANG_ARRAY <<< "$LANGUAGES"
for lang in "${LANG_ARRAY[@]}"; do
    case $lang in
        octave)  run_octave_benchmarks ;;
        rust)    run_rust_benchmarks ;;
        python)  run_python_benchmarks ;;
        *)       echo "Unknown language: $lang" ;;
    esac
done

# =============================================================================
# Consolidation
# =============================================================================

echo "📊 Consolidating results..."

uv run python -c "
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Result:
    scenario: str; filter: str; impl: str; mean_ms: float | None; status: str

def parse_benchmark_file(filepath: Path, impl: str) -> list[Result]:
    results = []
    if not filepath.exists(): return results
    with open(filepath) as f:
        for line in f:
            if '|' not in line or '---' in line or 'Scenario' in line: continue
            parts = [p.strip() for p in line.split('|')]
            if len(parts) < 3: continue
            s, f_name, t_str = parts[0], parts[1], parts[2]
            if t_str in ('ERROR', 'TIMEOUT', 'SKIP', 'N/A'):
                results.append(Result(s, f_name, impl, None, t_str.lower()))
            else:
                try: results.append(Result(s, f_name, impl, float(t_str), 'ok'))
                except ValueError: results.append(Result(s, f_name, impl, None, 'error'))
    return results

results = []
results_dir = Path('benchmarks/results')
for impl in ['octave', 'rust', 'python']:
    results.extend(parse_benchmark_file(results_dir / f'{impl}_benchmarks.txt', impl))

grouped = {}
for r in results:
    key = (r.scenario, r.filter)
    if key not in grouped: grouped[key] = {}
    grouped[key][r.impl] = r

with open(results_dir / 'comparison_summary.md', 'w') as f:
    f.write('# Benchmark Comparison\\n\\n')
    f.write('| Scenario | Filter | Octave | Rust (Native) | Python (Typical) | Speedup (vs Py) |\\n')
    f.write('|---|---|---|---|---|---|\\n')
    for (scenario, filter_name), impls in sorted(grouped.items()):
        octave = impls.get('octave')
        rust = impls.get('rust')
        python = impls.get('python')
        def fmt(r): return 'N/A' if r is None else (r.status.upper() if r.status != 'ok' else f'{r.mean_ms:.1f}')
        def ratio(num, den):
            if num and den and num.status == 'ok' and den.status == 'ok' and num.mean_ms > 0:
                return f'{den.mean_ms / num.mean_ms:.2f}x'
            return '-'
        f.write(f'| {scenario} | {filter_name} | {fmt(octave)} | {fmt(rust)} | {fmt(python)} | {ratio(rust, python)} |\\n')
print('✓ Results consolidated to benchmarks/results/comparison_summary.md')
"

echo ""
echo "✅ Benchmarks complete!"
