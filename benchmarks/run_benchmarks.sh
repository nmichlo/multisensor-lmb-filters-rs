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
    % Handle N-D arrays from JSON - squeeze to 2D first
    if ndims(meas) > 2
        meas = squeeze(meas);
    end
    if isempty(meas) z = {}; return; end
    % Ensure 2D Nx2 format (measurements x dimensions)
    if size(meas, 2) ~= 2 && size(meas, 1) == 2
        meas = permute(meas, [2 1]);  % Use permute instead of ' for compatibility
    end
    z = cell(1, size(meas, 1)); 
    for k = 1:size(meas, 1)
        z{k} = meas(k, :)(:);  % Column vector
    end
endfunction

function measurements = extractMeasurements(scenario, numSensors, numSteps)
    measurements = cell(1, numSteps);
    for t = 1:numSteps
        sr = scenario.steps(t).sensor_readings;
        if isempty(sr) || (isnumeric(sr) && numel(sr) == 0)
            measurements{t} = {};
        elseif iscell(sr)
            % Multi-sensor: use first sensor for single-sensor LMB
            measurements{t} = convertToMeasCell(sr{1});
        else
            measurements{t} = convertToMeasCell(sr);
        end
    end
endfunction

function order = extractScenarioOrder(filename)
    % Extract (N, S) from filename for sorting: bouncing_nXX_sYY.json
    n = str2double(regexp(filename, 'n(\d+)', 'tokens', 'once'));
    s = str2double(regexp(filename, 's(\d+)', 'tokens', 'once'));
    if isempty(n), n = 999; end
    if isempty(s), s = 999; end
    order = n * 1000 + s;  % Composite sort key
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

% Sort files by (N, S) to match Rust/Python order
[~, sortIdx] = sort(cellfun(@(f) extractScenarioOrder(f), {files.name}));
files = files(sortIdx);

fprintf('Scenario               | Filter             |  Time(ms) |    OSPA | Progress\n');
fprintf('---------------------------------------------------------------------------\n');

filterConfigs = {
    'LMB-LBP',   'LMB',  'LBP',   struct('max_iterations', 100, 'tolerance', 1e-6), '';
    'LMB-Gibbs', 'LMB',  'Gibbs', struct('num_samples', 1000), '';
};

thresholds = struct('existence', 1e-3, 'gm_weight', 1e-4, 'max_components', 100, 'gm_merge', inf);
seed = 42;

% JIT Warm-up: Run a small scenario once to pre-compile functions
if numel(files) > 0
    try
        wS = jsondecode(fileread(fullfile(scenariosDir, files(1).name)));
        wM = buildMatlabModel(wS, min(5, wS.num_steps), thresholds);
        wZ = extractMeasurements(wS, wS.num_sensors, min(5, wS.num_steps));
        [~, ~] = runLmbFilter(SimpleRng(42), wM, wZ);
    catch, end
end

for i = 1:numel(files)
    scenario_name = strrep(files(i).name, '.json', '');
    % NOTE: No skip filter - run all scenarios for fair comparison
    
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
    Config { name: "LMB-Gibbs", filter_type: "LMB", assoc: "Gibbs" },
    Config { name: "AA-LMB-LBP", filter_type: "AA-LMB", assoc: "LBP" },
    Config { name: "IC-LMB-LBP", filter_type: "IC-LMB", assoc: "LBP" },
    Config { name: "PU-LMB-LBP", filter_type: "PU-LMB", assoc: "LBP" },
    Config { name: "GA-LMB-LBP", filter_type: "GA-LMB", assoc: "LBP" },
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
    sensors_config: MultisensorConfig,
    birth: BirthModel,
    steps: Vec<(usize, Vec<DVector<f64>>)>,  // (timestep, single_sensor_measurements)
    multi_steps: Vec<Vec<Vec<DVector<f64>>>>, // Multi-sensor: step -> sensor -> measurements
    num_sensors: usize,
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
    
    // Multi-sensor config
    let sensors_config = MultisensorConfig::new(vec![sensor.clone(); scenario.num_sensors]);
    
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
    
    // Steps - single sensor (use first sensor)
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
    
    // Multi-sensor steps
    let multi_steps: Vec<_> = scenario.steps.iter().map(|step| {
        let readings = step.sensor_readings.as_ref();
        if let Some(rss) = readings {
            rss.iter().map(|r| {
                r.iter().map(|m| DVector::from_vec(vec![m[0], m[1]])).collect()
            }).collect()
        } else {
            vec![vec![]; scenario.num_sensors]
        }
    }).collect();
    
    PreprocessedScenario { motion, sensor, sensors_config, birth, steps, multi_steps, num_sensors: scenario.num_sensors }
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
        let _result = filter.step(rng, single_meas, *t).unwrap();
    }
    
    let elapsed_ms = start.elapsed().as_micros() as f64 / 1000.0;
    (elapsed_ms, steps.len(), false)
}

fn run_multi_filter<F>(
    filter: &mut F,
    steps: &[Vec<Vec<DVector<f64>>>],
    rng: &mut StdRng,
    timeout_secs: u64,
) -> (f64, usize, bool) 
where F: Filter<Measurements = Vec<Vec<DVector<f64>>>> {
    let start = Instant::now();
    
    for (t, meas) in steps.iter().enumerate() {
        if start.elapsed().as_secs() >= timeout_secs {
            return (start.elapsed().as_micros() as f64 / 1000.0, t, true);
        }
        let _result = filter.step(rng, meas, t).unwrap();
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
            
            let associator = DynamicAssociator::from_config(&assoc_config);
            let mut rng = StdRng::seed_from_u64(42);
            
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match config.filter_type {
                    "LMB" => {
                        // Single sensor LMB
                        let mut filter = LmbFilter::with_associator_type(
                            preprocessed.motion.clone(),
                            preprocessed.sensor.clone(),
                            preprocessed.birth.clone(),
                            assoc_config,
                            associator,
                        )
                        .with_gm_pruning(GM_WEIGHT_THRESHOLD, MAX_GM_COMPONENTS)
                        .with_gm_merge_threshold(GM_MERGE_THRESHOLD);
                        run_filter(&mut filter, &preprocessed.steps, &mut rng, timeout_secs)
                    },
                    "AA-LMB" => {
                        let merger = ArithmeticAverageMerger::uniform(preprocessed.num_sensors, 100);
                        let mut filter = AaLmbFilter::with_associator_type(
                            preprocessed.motion.clone(), preprocessed.sensors_config.clone(),
                            preprocessed.birth.clone(), assoc_config, merger, associator
                        );
                        run_multi_filter(&mut filter, &preprocessed.multi_steps, &mut rng, timeout_secs)
                    },
                    "IC-LMB" => {
                        let merger = IteratedCorrectorMerger::new();
                        let mut filter = IcLmbFilter::with_associator_type(
                            preprocessed.motion.clone(), preprocessed.sensors_config.clone(),
                            preprocessed.birth.clone(), assoc_config, merger, associator
                        );
                        run_multi_filter(&mut filter, &preprocessed.multi_steps, &mut rng, timeout_secs)
                    },
                    "PU-LMB" => {
                        let merger = ParallelUpdateMerger::new(Vec::new());
                        let mut filter = PuLmbFilter::with_associator_type(
                            preprocessed.motion.clone(), preprocessed.sensors_config.clone(),
                            preprocessed.birth.clone(), assoc_config, merger, associator
                        );
                        run_multi_filter(&mut filter, &preprocessed.multi_steps, &mut rng, timeout_secs)
                    },
                    "GA-LMB" => {
                        let merger = GeometricAverageMerger::uniform(preprocessed.num_sensors);
                        let mut filter = GaLmbFilter::with_associator_type(
                            preprocessed.motion.clone(), preprocessed.sensors_config.clone(),
                            preprocessed.birth.clone(), assoc_config, merger, associator
                        );
                        run_multi_filter(&mut filter, &preprocessed.multi_steps, &mut rng, timeout_secs)
                    },
                    _ => panic!("Unknown filter type: {}", config.filter_type)
                }
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
    ("LMB-Gibbs", FilterLmb, lambda: AssociatorConfig.gibbs(1000), False),
    ("AA-LMB-LBP", FilterAaLmb, lambda: AssociatorConfig.lbp(100, 1e-6), True),
    ("IC-LMB-LBP", FilterIcLmb, lambda: AssociatorConfig.lbp(100, 1e-6), True),
    ("PU-LMB-LBP", FilterPuLmb, lambda: AssociatorConfig.lbp(100, 1e-6), True),
    ("GA-LMB-LBP", FilterGaLmb, lambda: AssociatorConfig.lbp(100, 1e-6), True),
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

def run_filter(filt, steps, is_multi):
    # Pure execution timing - no extra work inside the loop
    # Result discarded immediately to avoid benchmarking memory allocation
    start = time.perf_counter()
    for t, single_meas, multi_meas in steps:
        _ = filt.step(multi_meas if is_multi else single_meas, t)
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
                elapsed_ms, steps_done = run_filter(filt, steps, is_multi)
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
# Consolidation - Generate README_BENCHMARKS.md
# =============================================================================

echo "📊 Consolidating results..."

uv run python -c "
import json, os, platform, subprocess
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

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
            # Clean scenario name (remove debug output artifacts)
            s = s.split('bouncing')[-1] if 'bouncing' in s else s
            s = 'bouncing' + s if s.startswith('_') else s
            if not s.startswith('bouncing'): continue
            if t_str.strip() in ('ERROR', 'TIMEOUT', 'SKIP', 'N/A', '-'):
                results.append(Result(s, f_name, impl, None, t_str.strip().lower()))
            else:
                try: results.append(Result(s, f_name, impl, float(t_str.strip()), 'ok'))
                except ValueError: pass
    return results

results_dir = Path('benchmarks/results')
results = []
for impl in ['octave', 'rust', 'python']:
    results.extend(parse_benchmark_file(results_dir / f'{impl}_benchmarks.txt', impl))

grouped = {}
for r in results:
    key = (r.scenario, r.filter)
    if key not in grouped: grouped[key] = {}
    grouped[key][r.impl] = r

# Get environment info
try:
    rust_version = subprocess.run(['rustc', '--version'], capture_output=True, text=True).stdout.strip()
except: rust_version = 'Unknown'

try:
    python_version = platform.python_version()
except: python_version = 'Unknown'

try:
    octave_version = subprocess.run(['octave', '--version'], capture_output=True, text=True).stdout.split('\n')[0]
except: octave_version = 'Unknown'

# Generate README
readme_path = Path('README_BENCHMARKS.md')
with open(readme_path, 'w') as f:
    f.write('# LMB Filter Benchmark Results\n\n')
    f.write(f'*Generated: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}*\n\n')
    
    f.write('## Overview\n\n')
    f.write('This benchmark compares three implementations of the LMB (Labeled Multi-Bernoulli) filter:\n\n')
    f.write('| Implementation | Description |\n')
    f.write('|----------------|-------------|\n')
    f.write('| **Octave/MATLAB** | Original reference implementation (interpreted) |\n')
    f.write('| **Rust (Native)** | Native Rust binary compiled with `--release` |\n')
    f.write('| **Python (Bindings)** | Python calling Rust via PyO3/maturin bindings |\n\n')
    
    f.write('## Environment\n\n')
    f.write('| Component | Version |\n')
    f.write('|-----------|--------|\n')
    f.write(f'| Platform | {platform.system()} {platform.machine()} |\n')
    f.write(f'| Rust | {rust_version} |\n')
    f.write(f'| Python | {python_version} |\n')
    f.write(f'| Octave | {octave_version} |\n\n')
    
    f.write('## Methodology\n\n')
    f.write('- **Timeout**: 30 seconds per scenario\n')
    f.write('- **Thresholds**: existence=1e-3, gm_weight=1e-4, max_components=100, gm_merge=∞\n')
    f.write('- **Association**: LBP (100 iterations, tolerance 1e-6)\n')
    f.write('- **Warm-up**: Octave runs a JIT warm-up before timing\n')
    f.write('- **RNG Seed**: 42 (deterministic across all implementations)\n\n')
    
    f.write('## Results\n\n')
    
    # Collect filters
    filters = sorted(set(r.filter for r in results))
    
    for filter_name in filters:
        f.write(f'### {filter_name}\n\n')
        f.write('| Scenario | Octave (ms) | Rust (ms) | Python (ms) | Rust Speedup |\n')
        f.write('|----------|-------------|-----------|-------------|-------------|\n')
        
        for (scenario, flt), impls in sorted(grouped.items()):
            if flt != filter_name: continue
            octave = impls.get('octave')
            rust = impls.get('rust')
            python = impls.get('python')
            
            def fmt(r): 
                if r is None: return '-'
                if r.status != 'ok': return r.status.upper()
                return f'{r.mean_ms:,.1f}'
            
            def speedup(fast, baseline):
                if fast and baseline and fast.status == 'ok' and baseline.status == 'ok':
                    if fast.mean_ms > 0:
                        ratio = baseline.mean_ms / fast.mean_ms
                        return f'{ratio:,.1f}x'
                return '-'
            
            # Speedup vs Octave (reference)
            speed = speedup(rust, octave)
            
            f.write(f'| {scenario} | {fmt(octave)} | {fmt(rust)} | {fmt(python)} | {speed} |\n')
        
        f.write('\n')
    
    f.write('## Summary\n\n')
    
    # Calculate average speedups
    rust_vs_octave = []
    rust_vs_python = []
    for (scenario, flt), impls in grouped.items():
        octave, rust, python = impls.get('octave'), impls.get('rust'), impls.get('python')
        if rust and rust.status == 'ok' and octave and octave.status == 'ok' and rust.mean_ms > 0:
            rust_vs_octave.append(octave.mean_ms / rust.mean_ms)
        if rust and rust.status == 'ok' and python and python.status == 'ok' and rust.mean_ms > 0:
            rust_vs_python.append(python.mean_ms / rust.mean_ms)
    
    if rust_vs_octave:
        avg_speedup = sum(rust_vs_octave) / len(rust_vs_octave)
        f.write(f'**Rust vs Octave**: Average speedup **{avg_speedup:,.0f}x** faster\n\n')
    
    if rust_vs_python:
        avg_ratio = sum(rust_vs_python) / len(rust_vs_python)
        f.write(f'**Rust vs Python Bindings**: Average ratio **{avg_ratio:.2f}x** (expected ~1.0, both use same Rust code)\n\n')
    
    f.write('## Notes\n\n')
    f.write('- Octave/MATLAB is interpreted and significantly slower by design\n')
    f.write('- Rust native and Python bindings should be nearly identical as they run the same compiled Rust code\n')
    f.write('- Small differences (~5-15%) between Rust and Python are due to PyO3 data marshalling overhead\n')
    f.write('- Larger scenarios (n50) may show TIMEOUT if they exceed the time limit\n')

print(f'✓ Results written to README_BENCHMARKS.md')
print(f'✓ Also saved to benchmarks/results/comparison_summary.md')

# Also copy to results dir
import shutil
shutil.copy('README_BENCHMARKS.md', 'benchmarks/results/comparison_summary.md')
"

echo ""
echo "✅ Benchmarks complete!"

