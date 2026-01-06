function run_octave(scenario_path, filter_name)
% RUN_OCTAVE - Minimal benchmark runner for Octave/MATLAB
%
% Usage:
%   octave --no-gui --eval "run_octave('path/to/scenario.json', 'LMB-LBP')"
%
% Output:
%   Prints elapsed time in milliseconds as a single number.
%   Exit 0 on success, non-zero on error.
%
% Supported filters:
%   Single-sensor: LMB-LBP, LMB-Gibbs, LMB-Murty, LMBM-Gibbs, LMBM-Murty
%   Multi-sensor: AA-LMB-LBP, IC-LMB-LBP, PU-LMB-LBP, GA-LMB-LBP

% =============================================================================
% Setup paths
% =============================================================================

scriptDir = fileparts(mfilename('fullpath'));
matlabDir = fullfile(scriptDir, '..', '..', '..', 'multisensor-lmb-filters');
addpath(genpath(matlabDir));

% =============================================================================
% Load scenario
% =============================================================================

scenario = jsondecode(fileread(scenario_path));

% =============================================================================
% Parse filter name
% =============================================================================

% Parse filter name to get filter type and associator
% Examples: LMB-LBP, LMBM-Gibbs, AA-LMB-LBP, IC-LMB-LBP
filterParts = strsplit(filter_name, '-');

% Determine if multi-sensor
isMultiSensor = any(strcmp(filterParts{1}, {'AA', 'IC', 'PU', 'GA', 'MS'}));

if isMultiSensor
    fusionType = filterParts{1};     % AA, IC, PU, GA
    filterType = filterParts{2};     % LMB
    assocType = filterParts{3};      % LBP, Gibbs, Murty
else
    fusionType = '';
    filterType = filterParts{1};     % LMB or LMBM
    assocType = filterParts{end};    % LBP, Gibbs, Murty
end

% =============================================================================
% Build model
% =============================================================================

thresholds = struct('existence', 1e-3, 'gm_weight', 1e-4, 'max_components', 100, 'gm_merge', inf);
numSensors = scenario.num_sensors;

m = scenario.model;
model.xDimension = 4;
model.zDimension = 2;
model.T = m.dt;
model.survivalProbability = m.survival_probability;
model.existenceThreshold = thresholds.existence;

% Motion model: constant velocity 2D
model.A = [eye(2), m.dt*eye(2); zeros(2), eye(2)];
model.u = zeros(4, 1);

% Process noise
q = m.process_noise_std^2;
model.R = q * [(1/3)*m.dt^3*eye(2), 0.5*m.dt^2*eye(2); 0.5*m.dt^2*eye(2), m.dt*eye(2)];

% Observation space
model.observationSpaceLimits = [scenario.bounds(1), scenario.bounds(2); scenario.bounds(3), scenario.bounds(4)];
model.observationSpaceVolume = prod(model.observationSpaceLimits(:,2) - model.observationSpaceLimits(:,1));

% Sensor model - handle single vs multi-sensor
if isMultiSensor
    model.numberOfSensors = numSensors;
    % Per-sensor parameters (cell arrays for multi-sensor)
    model.C = repmat({[eye(2), zeros(2)]}, 1, numSensors);
    model.Q = repmat({m.measurement_noise_std^2 * eye(2)}, 1, numSensors);
    model.detectionProbability = m.detection_probability * ones(1, numSensors);
    model.clutterRate = m.clutter_rate * ones(1, numSensors);
    model.clutterPerUnitVolume = model.clutterRate / model.observationSpaceVolume;
    model.lmbParallelUpdateMode = fusionType;
    % AA and GA fusion need sensor weights (uniform by default)
    model.aaSensorWeights = ones(1, numSensors) / numSensors;
    model.gaSensorWeights = ones(1, numSensors) / numSensors;
else
    model.C = [eye(2), zeros(2)];
    model.Q = m.measurement_noise_std^2 * eye(2);
    model.detectionProbability = m.detection_probability;
    model.clutterRate = m.clutter_rate;
    model.clutterPerUnitVolume = m.clutter_rate / model.observationSpaceVolume;
end

% Birth model
birthLocs = m.birth_locations;
model.numberOfBirthLocations = size(birthLocs, 1);
model.birthLocationLabels = 1:model.numberOfBirthLocations;
model.rB = 0.01 * ones(model.numberOfBirthLocations, 1);
model.muB = cell(model.numberOfBirthLocations, 1);
model.SigmaB = cell(model.numberOfBirthLocations, 1);
for i = 1:model.numberOfBirthLocations
    model.muB{i} = birthLocs(i, :)';
    model.SigmaB{i} = diag([2500, 2500, 100, 100]);
end

% Build birth parameters
object.birthLocation = 0; object.birthTime = 0; object.r = 0; object.numberOfGmComponents = 0;
object.w = zeros(0, 1); object.mu = {}; object.Sigma = {}; object.trajectoryLength = 0;
object.trajectory = repmat(80 * ones(model.xDimension, 1), 1, 100); object.timestamps = zeros(1, 100);
birthParameters = repmat(object, model.numberOfBirthLocations, 1);
for i = 1:model.numberOfBirthLocations
    birthParameters(i).birthLocation = model.birthLocationLabels(i);
    birthParameters(i).mu = model.muB(i);
    birthParameters(i).Sigma = model.SigmaB(i);
    birthParameters(i).r = model.rB(i);
    birthParameters(i).numberOfGmComponents = 1;
    birthParameters(i).w = 1;
end
model.birthParameters = birthParameters;

% Association parameters
model.dataAssociationMethod = assocType;
model.lbpConvergenceTolerance = 1e-6;
model.maximumNumberOfLbpIterations = 100;
model.numberOfSamples = 1000;
model.numberOfAssignments = 25;

% Pruning thresholds
model.weightThreshold = thresholds.gm_weight;
model.gmWeightThreshold = thresholds.gm_weight;
model.maximumNumberOfGmComponents = thresholds.max_components;
model.mahalanobisDistanceThreshold = thresholds.gm_merge;
model.minimumTrajectoryLength = 3;
model.object = repmat(object, 0, 1);
model.simulationLength = scenario.num_steps;

% LMBM-specific model fields (hypotheses, trajectory, birthTrajectory)
% These are required by runLmbmFilter and runMultisensorLmbmFilter
if strcmp(filterType, 'LMBM')
    % LMBM uses a separate birth existence probability
    model.rBLmbm = 0.001 * ones(model.numberOfBirthLocations, 1);

    % Hypothesis struct (empty initial hypothesis)
    hypotheses.birthLocation = zeros(0, 1);
    hypotheses.birthTime = zeros(0, 1);
    hypotheses.w = 1;  % Hypothesis weight
    hypotheses.r = zeros(0, 1);
    hypotheses.mu = repmat({}, 0, 1);
    hypotheses.Sigma = repmat({}, 0, 1);
    model.hypotheses = hypotheses;

    % Trajectory template struct
    trajectory.birthLocation = 0;
    trajectory.birthTime = 0;
    trajectory.trajectory = [];
    trajectory.trajectoryLength = 0;
    trajectory.timestamps = zeros(1, 0);
    model.trajectory = repmat(trajectory, 1, 0);

    % Birth trajectory (one per birth location)
    birthTrajectory = repmat(trajectory, 1, model.numberOfBirthLocations);
    for i = 1:model.numberOfBirthLocations
        birthTrajectory(i).birthLocation = model.birthLocationLabels(i);
        birthTrajectory(i).trajectoryLength = 0;
        birthTrajectory(i).trajectory = repmat(80 * ones(model.xDimension, 1), 1, 100);
        birthTrajectory(i).timestamps = zeros(1, 100);
    end
    model.birthTrajectory = birthTrajectory;

    % LMBM hypothesis pruning parameters
    model.maximumNumberOfPosteriorHypotheses = 25;
    model.posteriorHypothesisWeightThreshold = 1e-3;
end

% =============================================================================
% Extract measurements
% =============================================================================

numSteps = scenario.num_steps;

if isMultiSensor
    % Multi-sensor: measurements{sensor, timestep}
    measurements = cell(numSensors, numSteps);
    for t = 1:numSteps
        sr = scenario.steps(t).sensor_readings;
        for s = 1:numSensors
            if iscell(sr) && numel(sr) >= s
                measurements{s, t} = convertToMeasCell(sr{s});
            else
                measurements{s, t} = {};
            end
        end
    end
else
    % Single-sensor: measurements{timestep}
    measurements = cell(1, numSteps);
    for t = 1:numSteps
        sr = scenario.steps(t).sensor_readings;
        if isempty(sr) || (isnumeric(sr) && numel(sr) == 0)
            measurements{t} = {};
        elseif iscell(sr)
            % Multi-sensor data but single-sensor filter: use first sensor
            measurements{t} = convertToMeasCell(sr{1});
        else
            measurements{t} = convertToMeasCell(sr);
        end
    end
end

% =============================================================================
% Run benchmark
% =============================================================================

rng_obj = SimpleRng(42);
tic;

if isMultiSensor
    switch fusionType
        case 'IC'
            stateEstimates = runIcLmbFilter(model, measurements);
        case {'AA', 'GA', 'PU'}
            stateEstimates = runParallelUpdateLmbFilter(model, measurements);
        case 'MS'
            % Multi-sensor LMBM (only Gibbs sampling supported)
            [~, stateEstimates] = runMultisensorLmbmFilter(rng_obj, model, measurements);
        otherwise
            error('Unknown fusion type: %s', fusionType);
    end
else
    switch filterType
        case 'LMB'
            [~, ~] = runLmbFilter(rng_obj, model, measurements);
        case 'LMBM'
            [~, ~] = runLmbmFilter(rng_obj, model, measurements);
        otherwise
            error('Unknown filter type: %s', filterType);
    end
end

elapsed_ms = toc * 1000;

% Output only the timing
fprintf('%.3f\n', elapsed_ms);

end

% =============================================================================
% Helper functions
% =============================================================================

function z = convertToMeasCell(meas)
    if isempty(meas)
        z = {};
        return;
    end
    if iscell(meas)
        z = meas;
        return;
    end
    % Handle N-D arrays from JSON - reshape to 2D
    meas = squeeze(meas);
    if isempty(meas)
        z = {};
        return;
    end
    % Force into Nx2 matrix (N measurements, 2 dimensions each)
    meas = reshape(meas, [], 2);
    z = cell(1, size(meas, 1));
    for k = 1:size(meas, 1)
        z{k} = meas(k, :)(:);  % Column vector
    end
end
