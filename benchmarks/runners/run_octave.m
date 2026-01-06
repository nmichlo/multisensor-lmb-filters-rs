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
%   - LMB-LBP, LMB-Gibbs, LMB-Murty (single-sensor LMB)
%   - LMBM-Gibbs, LMBM-Murty (single-sensor LMBM)

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
% Build model
% =============================================================================

thresholds = struct('existence', 1e-3, 'gm_weight', 1e-4, 'max_components', 100, 'gm_merge', inf);

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

% Sensor model
model.C = [eye(2), zeros(2)];
model.Q = m.measurement_noise_std^2 * eye(2);
model.detectionProbability = m.detection_probability;
model.clutterRate = m.clutter_rate;
model.clutterPerUnitVolume = m.clutter_rate / model.observationSpaceVolume;

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

% =============================================================================
% Extract measurements (single-sensor: use first sensor)
% =============================================================================

numSteps = scenario.num_steps;
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

% =============================================================================
% Configure association method
% =============================================================================

% Parse filter name to get filter type and associator
filterParts = strsplit(filter_name, '-');
filterType = filterParts{1};  % LMB or LMBM
assocType = filterParts{end}; % LBP, Gibbs, or Murty

model.dataAssociationMethod = assocType;

% =============================================================================
% Run benchmark
% =============================================================================

rng_obj = SimpleRng(42);
tic;

switch filterType
    case 'LMB'
        [~, ~] = runLmbFilter(rng_obj, model, measurements);
    case 'LMBM'
        [~, ~] = runLmbmFilter(rng_obj, model, measurements);
    otherwise
        error('Unknown filter type: %s', filterType);
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
    % Handle N-D arrays from JSON - squeeze to 2D first
    if ndims(meas) > 2
        meas = squeeze(meas);
    end
    if isempty(meas)
        z = {};
        return;
    end
    % Ensure 2D Nx2 format (measurements x dimensions)
    if size(meas, 2) ~= 2 && size(meas, 1) == 2
        meas = permute(meas, [2 1]);  % Use permute instead of ' for compatibility
    end
    z = cell(1, size(meas, 1));
    for k = 1:size(meas, 1)
        z{k} = meas(k, :)(:);  % Column vector
    end
end
