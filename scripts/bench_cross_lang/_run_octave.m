function run_octave(scenario_path, filter_name, get_config, skip_run)
% RUN_OCTAVE - Minimal benchmark runner for Octave/MATLAB
%
% Usage:
%   octave --no-gui --eval "run_octave('path/to/scenario.json', 'LMB-LBP')"
%   octave --no-gui --eval "run_octave('path/to/scenario.json', 'LMB-LBP', true)"  % get config
%   octave --no-gui --eval "run_octave('path/to/scenario.json', 'LMB-LBP', true, true)"  % config only
%
% Output:
%   Prints elapsed time in milliseconds as a single number.
%   Exit 0 on success, non-zero on error.
%
% Supported filters:
%   Single-sensor: LMB-LBP, LMB-Gibbs, LMB-Murty, LMBM-Gibbs, LMBM-Murty
%   Multi-sensor: AA-LMB-LBP, IC-LMB-LBP, PU-LMB-LBP, GA-LMB-LBP

% Handle optional arguments
if nargin < 3
    get_config = false;
end
if nargin < 4
    skip_run = false;
end

% =============================================================================
% Setup paths
% =============================================================================

scriptDir = fileparts(mfilename('fullpath'));
matlabDir = fullfile(scriptDir, '..', '..', 'vendor', 'multisensor-lmb-filters');
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
% Handle --get-config
% =============================================================================

if get_config
    % Output JSON config and exit
    fprintf('{\n');
    fprintf('  "filter_type": "%s",\n', getFilterType(filterType, fusionType, isMultiSensor));
    fprintf('  "motion": {\n');
    fprintf('    "x_dim": %d,\n', model.xDimension);
    fprintf('    "survival_probability": %s,\n', formatFloat(model.survivalProbability));
    fprintf('    "transition_matrix": ');
    printJsonArray(model.A(:)');
    fprintf(',\n');
    fprintf('    "process_noise": ');
    printJsonArray(model.R(:)');
    fprintf('\n');
    fprintf('  },\n');
    fprintf('  "sensor": {\n');
    fprintf('    "z_dim": %d,\n', model.zDimension);
    fprintf('    "x_dim": %d,\n', model.xDimension);
    if isMultiSensor
        fprintf('    "detection_probability": %s,\n', formatFloat(model.detectionProbability(1)));
        fprintf('    "clutter_rate": %s,\n', formatFloat(model.clutterRate(1)));
    else
        fprintf('    "detection_probability": %s,\n', formatFloat(model.detectionProbability));
        fprintf('    "clutter_rate": %s,\n', formatFloat(model.clutterRate));
    end
    fprintf('    "observation_volume": %s,\n', formatFloat(model.observationSpaceVolume));
    if isMultiSensor
        fprintf('    "clutter_density": %s,\n', formatFloat(model.clutterPerUnitVolume(1)));
        % Use first sensor's C and Q matrices
        fprintf('    "observation_matrix": ');
        printJsonArray(model.C{1}(:)');
        fprintf(',\n');
        fprintf('    "measurement_noise": ');
        printJsonArray(model.Q{1}(:)');
        fprintf('\n');
    else
        fprintf('    "clutter_density": %s,\n', formatFloat(model.clutterPerUnitVolume));
        fprintf('    "observation_matrix": ');
        printJsonArray(model.C(:)');
        fprintf(',\n');
        fprintf('    "measurement_noise": ');
        printJsonArray(model.Q(:)');
        fprintf('\n');
    end
    fprintf('  },\n');
    fprintf('  "num_sensors": %d,\n', numSensors);
    fprintf('  "birth": {\n');
    fprintf('    "num_locations": %d,\n', model.numberOfBirthLocations);
    fprintf('    "lmb_existence": %s,\n', formatFloat(model.rB(1)));
    if isfield(model, 'rBLmbm')
        fprintf('    "lmbm_existence": %s,\n', formatFloat(model.rBLmbm(1)));
    else
        fprintf('    "lmbm_existence": 0.001,\n');
    end
    fprintf('    "locations": [\n');
    for i = 1:model.numberOfBirthLocations
        fprintf('      {\n');
        fprintf('        "label": %d,\n', model.birthLocationLabels(i) - 1);  % 0-indexed
        fprintf('        "mean": ');
        printJsonArray(model.muB{i}', 8);
        fprintf(',\n');
        fprintf('        "covariance_diag": ');
        printJsonArray(diag(model.SigmaB{i})', 8);
        fprintf('\n');
        if i < model.numberOfBirthLocations
            fprintf('      },\n');
        else
            fprintf('      }\n');
        end
    end
    fprintf('    ]\n');
    fprintf('  },\n');
    fprintf('  "association": {\n');
    fprintf('    "method": "%s",\n', assocType);
    % Output 0 for non-applicable association parameters (matching Rust)
    if strcmp(assocType, 'LBP')
        fprintf('    "lbp_max_iterations": %d,\n', model.maximumNumberOfLbpIterations);
        fprintf('    "lbp_tolerance": %s,\n', formatFloat(model.lbpConvergenceTolerance));
        fprintf('    "gibbs_samples": 0,\n');
        fprintf('    "murty_assignments": 0\n');
    elseif strcmp(assocType, 'Gibbs')
        fprintf('    "lbp_max_iterations": 0,\n');
        fprintf('    "lbp_tolerance": 0.0,\n');
        fprintf('    "gibbs_samples": %d,\n', model.numberOfSamples);
        fprintf('    "murty_assignments": 0\n');
    else  % Murty
        fprintf('    "lbp_max_iterations": 0,\n');
        fprintf('    "lbp_tolerance": 0.0,\n');
        fprintf('    "gibbs_samples": 0,\n');
        fprintf('    "murty_assignments": %d\n', model.numberOfAssignments);
    end
    fprintf('  },\n');
    fprintf('  "thresholds": {\n');
    fprintf('    "existence_threshold": %s,\n', formatFloat(model.existenceThreshold));
    % LMBM doesn't use GM pruning - output 0.0/0 to match Rust/Python
    if strcmp(filterType, 'LMBM')
        fprintf('    "gm_weight_threshold": 0,\n');
        fprintf('    "max_gm_components": 0,\n');
    else
        fprintf('    "gm_weight_threshold": %s,\n', formatFloat(model.gmWeightThreshold));
        fprintf('    "max_gm_components": %d,\n', model.maximumNumberOfGmComponents);
    end
    fprintf('    "min_trajectory_length": %d,\n', model.minimumTrajectoryLength);
    if isinf(model.mahalanobisDistanceThreshold) || strcmp(filterType, 'LMBM')
        fprintf('    "gm_merge_threshold": null\n');
    else
        fprintf('    "gm_merge_threshold": %s\n', formatFloat(model.mahalanobisDistanceThreshold));
    end
    fprintf('  }');
    if strcmp(filterType, 'LMBM')
        fprintf(',\n');
        fprintf('  "lmbm_config": {\n');
        fprintf('    "max_hypotheses": %d,\n', model.maximumNumberOfPosteriorHypotheses);
        fprintf('    "hypothesis_weight_threshold": %s,\n', formatFloat(model.posteriorHypothesisWeightThreshold));
        fprintf('    "use_eap": false\n');
        fprintf('  }\n');
    else
        fprintf('\n');
    end
    fprintf('}\n');
end

% =============================================================================
% Skip run if requested
% =============================================================================

if skip_run
    return;
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

% Run filter and capture per-step execution times
if isMultiSensor
    switch fusionType
        case 'IC'
            [stateEstimates, executionTimes] = runIcLmbFilter(model, measurements);
        case {'AA', 'GA', 'PU'}
            [stateEstimates, executionTimes] = runParallelUpdateLmbFilter(model, measurements);
        case 'MS'
            % Multi-sensor LMBM (only Gibbs sampling supported)
            [~, stateEstimates, executionTimes] = runMultisensorLmbmFilter(rng_obj, model, measurements);
        otherwise
            error('Unknown fusion type: %s', fusionType);
    end
else
    switch filterType
        case 'LMB'
            [~, ~, executionTimes] = runLmbFilter(rng_obj, model, measurements);
        case 'LMBM'
            [~, ~, executionTimes] = runLmbmFilter(rng_obj, model, measurements);
        otherwise
            error('Unknown filter type: %s', filterType);
    end
end

% Output the timing (avg,std format)
% executionTimes is in seconds, convert to ms
executionTimes_ms = executionTimes * 1000;
avg_time_ms = mean(executionTimes_ms);
std_time_ms = std(executionTimes_ms);
fprintf('%.4f,%.4f\n', avg_time_ms, std_time_ms);

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

function printJsonArray(arr, indent)
    % Print array as pretty-printed JSON format (matching Rust serde_json output)
    % Default: elements at 6 spaces, closing bracket at 4 spaces (matches level 2 nesting)
    if nargin < 2
        indent = 4;  % base indent for the array (property level)
    end
    elemIndent = repmat(' ', 1, indent + 2);
    closeIndent = repmat(' ', 1, indent);
    
    fprintf('[\n');
    for i = 1:length(arr)
        fprintf('%s', elemIndent);
        fprintf('%s', formatFloat(arr(i)));
        if i < length(arr)
            fprintf(',\n');
        else
            fprintf('\n');
        end
    end
    fprintf('%s]', closeIndent);
end

function s = formatFloat(val)
    % Format a float exactly like Rust serde_json:
    % - Whole numbers get .0 suffix
    % - Adaptive precision (15-17 digits) to find shortest round-trip representation
    % - Decimal notation for 1e-5 <= |val| < 1e16 approx
    % - Scientific notation otherwise
    % - Exponent format: 1e-6 not 1e-06
    
    if val == 0
        s = '0.0';
        return;
    end
    
    if ~isfinite(val)
        s = 'null'; % Or whatever is appropriate, though config usually has finite numbers
        return;
    end
    
    absVal = abs(val);
    
    % 1. Determine shortest precision (15, 16, or 17) that round-trips
    % Use exact equality check because we want the shortest string that
    % parses back to the exact same double value (bitwise identical).
    s = sprintf('%.15g', val);
    if str2double(s) ~= val
        s = sprintf('%.16g', val);
        if str2double(s) ~= val
            s = sprintf('%.17g', val);
        end
    end
    
    % 2. Enforce Notation Style matching Rust serde_json
    % Rust tends to use decimal for >= 1e-5 (0.00001) and < 1e16??
    % We know 2.5e-5 (0.000025) is decimal. 1e-6 is scientific.
    
    useDecimal = (absVal >= 1e-5 && absVal < 1e16);
    
    % If format chose scientific but we want decimal (e.g. 2.5e-5)
    if useDecimal && (~isempty(strfind(s, 'e')) || ~isempty(strfind(s, 'E')))
        % Force decimal. precision needed?
        % We can use a large fixed precision and trim.
        % Or use %.16f / %.17f depending on what we found above.
        % Actually, just converting the current 's' to decimal is likely safest
        % if s matches val sufficiently.
        s = sprintf('%.20f', val); % Overshoot precision
        % Remove trailing zeros, but keep at least one digit if it's .0?
        % No, %.20f will not output scientific.
        s = regexprep(s, '0+$', ''); % Trim trailing zeros
        if s(end) == '.'
            s = [s '0'];
        end
    % If format chose decimal but we want scientific (e.g. very small or very large)
    elseif ~useDecimal && ~(~isempty(strfind(s, 'e')) || ~isempty(strfind(s, 'E')))
         s = sprintf('%.17g', val); % Re-format with general to force scientific if needed? 
    end

    % 3. Cleanup
    % Fix exponent format: e-06 -> e-6
    if ~isempty(strfind(s, 'e')) || ~isempty(strfind(s, 'E'))
        s = regexprep(s, 'e([+-])0+(\d)', 'e$1$2');
    else
        % Ensure decimal point for floats
        if isempty(strfind(s, '.'))
            s = [s '.0'];
        end
    end
end


function name = getFilterType(filterType, fusionType, isMultiSensor)
    % Return filter type name for config output (match Rust naming)
    if isMultiSensor
        switch fusionType
            case 'AA'
                name = 'MultisensorLmbFilter<ArithmeticAverage>';
            case 'IC'
                name = 'MultisensorLmbFilter<IteratedCorrector>';
            case 'PU'
                name = 'MultisensorLmbFilter<ParallelUpdate>';
            case 'GA'
                name = 'MultisensorLmbFilter<GeometricAverage>';
            case 'MS'
                name = 'MultisensorLmbmFilter';
            otherwise
                name = ['MultisensorLmbFilter<' fusionType '>'];
        end
    else
        switch filterType
            case 'LMB'
                name = 'LmbFilter';
            case 'LMBM'
                name = 'LmbmFilter';
            otherwise
                name = filterType;
        end
    end
end
