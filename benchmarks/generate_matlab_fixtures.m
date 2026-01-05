function generate_matlab_fixtures()
% GENERATE_MATLAB_FIXTURES - Generate self-describing fixtures from benchmark scenarios
%
% Each fixture contains ALL configuration needed to reproduce the test:
% - Model parameters
% - Filter type and settings
% - Per-step outputs (track counts, existences, means)
%
% Usage (from multisensor-lmb-filters directory):
%   octave --eval "run('../multisensor-lmb-filters-rs/benchmarks/generate_matlab_fixtures.m')"

%% Setup
clc;
pkg load statistics;

% Add MATLAB library to path
scriptDir = fileparts(mfilename('fullpath'));
matlabDir = fullfile(scriptDir, '..', '..', 'multisensor-lmb-filters');
addpath(genpath(matlabDir));

%% Configuration
scenariosDir = fullfile(scriptDir, 'scenarios');
fixturesDir = fullfile(scriptDir, 'fixtures');

% Scenarios to process
scenarios = {
    'bouncing_n5_s1.json';  % Single-sensor
    'bouncing_n5_s2.json';  % Multi-sensor
};

% Filter configurations (will be stored in fixture)
% Format: {name, type, associator, assoc_params, update_mode}
% NOTE: Gibbs and Murty require MEX files that may fail with code signing on macOS
filterConfigs = {
    % Single-sensor filters
    'LMB-LBP',   'LMB',  'LBP',   struct('max_iterations', 100, 'tolerance', 1e-6), '';
    'LMB-Gibbs', 'LMB',  'Gibbs', struct('num_samples', 1000), '';
    'LMB-Murty', 'LMB',  'Murty', struct('num_assignments', 25), '';
    % Multi-sensor filters
    'AA-LMB-LBP', 'AA-LMB', 'LBP', struct('max_iterations', 100, 'tolerance', 1e-6), 'AA';
    'GA-LMB-LBP', 'GA-LMB', 'LBP', struct('max_iterations', 100, 'tolerance', 1e-6), 'GA';
    'PU-LMB-LBP', 'PU-LMB', 'LBP', struct('max_iterations', 100, 'tolerance', 1e-6), 'PU';
    'IC-LMB-LBP', 'IC-LMB', 'LBP', struct('max_iterations', 100, 'tolerance', 1e-6), ''
};

% Common thresholds (stored in fixture)
thresholds = struct(...
    'existence', 1e-3, ...
    'gm_weight', 1e-4, ...
    'max_components', 100, ...
    'gm_merge', inf ...
);

% Run only first N steps for speed
maxSteps = 10;
seed = 42;

%% Create fixtures directory
if ~exist(fixturesDir, 'dir')
    mkdir(fixturesDir);
end

%% Process each scenario
for sIdx = 1:numel(scenarios)
    scenarioFile = scenarios{sIdx};
    scenarioPath = fullfile(scenariosDir, scenarioFile);

    if ~exist(scenarioPath, 'file')
        fprintf('Skipping %s (not found)\n', scenarioFile);
        continue;
    end

    fprintf('\n=== %s ===\n', scenarioFile);

    % Load scenario
    scenario = jsondecode(fileread(scenarioPath));
    numSensors = scenario.num_sensors;
    numSteps = min(scenario.num_steps, maxSteps);
    isSingleSensor = (numSensors == 1);

    % Build model config (will be stored in fixture)
    modelConfig = struct(...
        'dt', scenario.model.dt, ...
        'process_noise_std', scenario.model.process_noise_std, ...
        'measurement_noise_std', scenario.model.measurement_noise_std, ...
        'detection_probability', scenario.model.detection_probability, ...
        'survival_probability', scenario.model.survival_probability, ...
        'clutter_rate', scenario.model.clutter_rate, ...
        'bounds', scenario.bounds, ...
        'birth_locations', scenario.model.birth_locations, ...
        'birth_existence', 0.01, ...
        'birth_covariance', [2500, 2500, 100, 100] ...
    );

    % Build internal MATLAB model
    model = buildMatlabModel(scenario, numSteps, thresholds);

    % Extract measurements
    measurements = extractMeasurements(scenario, numSensors, numSteps);

    % Run each applicable filter
    for fIdx = 1:size(filterConfigs, 1)
        filterName = filterConfigs{fIdx, 1};
        filterType = filterConfigs{fIdx, 2};
        assocType = filterConfigs{fIdx, 3};
        assocParams = filterConfigs{fIdx, 4};
        updateMode = filterConfigs{fIdx, 5};

        % Skip incompatible filter/scenario combinations
        isMultiFilter = ~strcmp(filterType, 'LMB') && ~strcmp(filterType, 'LMBM');
        if isSingleSensor && isMultiFilter
            continue;
        end
        if ~isSingleSensor && (strcmp(filterType, 'LMB') || strcmp(filterType, 'LMBM'))
            continue;
        end

        fprintf('  Running %s... ', filterName);
        fflush(stdout);

        try
            % Configure model for this filter
            model.dataAssociationMethod = assocType;
            if ~isempty(updateMode)
                model.lmbParallelUpdateMode = updateMode;
            end

            % Run filter
            rng = SimpleRng(seed);
            if strcmp(filterType, 'LMB')
                [~, stateEstimates] = runLmbFilter(rng, model, measurements);
            elseif strcmp(filterType, 'IC-LMB')
                stateEstimates = runIcLmbFilter(model, measurements);
            else
                stateEstimates = runParallelUpdateLmbFilter(model, measurements);
            end

            % Build complete fixture
            fixture = struct();
            fixture.scenario_file = scenarioFile;
            fixture.num_sensors = numSensors;
            fixture.num_steps = numSteps;
            fixture.seed = seed;

            % Filter config (everything needed to instantiate)
            fixture.filter = struct(...
                'name', filterName, ...
                'type', filterType, ...
                'associator', struct('type', assocType, 'params', assocParams), ...
                'update_mode', updateMode ...
            );

            % Model parameters
            fixture.model = modelConfig;

            % Thresholds
            fixture.thresholds = thresholds;

            % Per-step results
            fixture.steps = extractStepResults(stateEstimates, numSteps);

            % Save fixture
            fixtureName = sprintf('%s_%s.json', ...
                strrep(scenarioFile, '.json', ''), ...
                strrep(filterName, '-', '_'));
            savePath = fullfile(fixturesDir, fixtureName);
            saveJson(fixture, savePath);

            lastStep = fixture.steps{end};
            fprintf('done (%d tracks at step %d)\n', lastStep.num_tracks, numSteps - 1);

        catch err
            fprintf('ERROR: %s\n', err.message);
        end
    end
end

fprintf('\n=== Complete ===\n');
fprintf('Fixtures saved to: %s\n', fixturesDir);
end

%% =========================================================================
%% Helper Functions
%% =========================================================================

function model = buildMatlabModel(scenario, numSteps, thresholds)
    m = scenario.model;

    model.xDimension = 4;
    model.zDimension = 2;
    model.T = m.dt;
    model.survivalProbability = m.survival_probability;
    model.existenceThreshold = thresholds.existence;

    % Dynamics
    model.A = [eye(2), m.dt*eye(2); zeros(2), eye(2)];
    model.u = zeros(4, 1);
    q = m.process_noise_std^2;
    model.R = q * [(1/3)*m.dt^3*eye(2), 0.5*m.dt^2*eye(2);
                   0.5*m.dt^2*eye(2), m.dt*eye(2)];

    % Observation space limits
    model.observationSpaceLimits = [scenario.bounds(1), scenario.bounds(2);
                                     scenario.bounds(3), scenario.bounds(4)];
    model.observationSpaceVolume = prod(model.observationSpaceLimits(:,2) - model.observationSpaceLimits(:,1));

    % Multi-sensor setup (observation model and detection/clutter)
    model.numberOfSensors = scenario.num_sensors;
    baseC = [eye(2), zeros(2)];
    baseQ = m.measurement_noise_std^2 * eye(2);

    if scenario.num_sensors == 1
        % Single sensor - use simple matrices
        model.C = baseC;
        model.Q = baseQ;
        model.detectionProbability = m.detection_probability;
        model.clutterRate = m.clutter_rate;
        model.clutterPerUnitVolume = m.clutter_rate / model.observationSpaceVolume;
    else
        % Multi-sensor - use cell arrays
        model.C = repmat({baseC}, 1, scenario.num_sensors);
        model.Q = repmat({baseQ}, 1, scenario.num_sensors);
        model.detectionProbability = m.detection_probability * ones(scenario.num_sensors, 1);
        model.clutterRate = m.clutter_rate * ones(1, scenario.num_sensors);
        model.clutterPerUnitVolume = m.clutter_rate / model.observationSpaceVolume * ones(1, scenario.num_sensors);
        model.sensorMeasurementNoiseCovariance = model.Q;
        model.gaSensorWeights = ones(1, scenario.num_sensors) / scenario.num_sensors;
        model.aaSensorWeights = ones(1, scenario.num_sensors) / scenario.num_sensors;
    end

    % Birth
    birthLocs = m.birth_locations;
    model.numberOfBirthLocations = size(birthLocs, 1);
    model.birthLocationLabels = 1:model.numberOfBirthLocations;
    model.rB = 0.01 * ones(model.numberOfBirthLocations, 1);
    model.rBLmbm = 0.001 * ones(model.numberOfBirthLocations, 1);
    model.muB = cell(model.numberOfBirthLocations, 1);
    model.SigmaB = cell(model.numberOfBirthLocations, 1);
    for i = 1:model.numberOfBirthLocations
        model.muB{i} = birthLocs(i, :)';
        model.SigmaB{i} = diag([2500, 2500, 100, 100]);
    end

    % Birth parameters (required by lmbPredictionStep)
    object.birthLocation = 0;
    object.birthTime = 0;
    object.r = 0;
    object.numberOfGmComponents = 0;
    object.w = zeros(0, 1);
    object.mu = {};
    object.Sigma = {};
    object.trajectoryLength = 0;
    object.trajectory = repmat(80 * ones(model.xDimension, 1), 1, 100);
    object.timestamps = zeros(1, 100);

    birthParameters = repmat(object, model.numberOfBirthLocations, 1);
    for i = 1:model.numberOfBirthLocations
        birthParameters(i).birthLocation = model.birthLocationLabels(i);
        birthParameters(i).birthTime = 0;
        birthParameters(i).r = model.rB(i);
        birthParameters(i).numberOfGmComponents = 1;
        birthParameters(i).w = ones(1, 1);
        birthParameters(i).mu = model.muB(i);
        birthParameters(i).Sigma = model.SigmaB(i);
        birthParameters(i).trajectoryLength = 0;
        birthParameters(i).trajectory = repmat(80 * ones(model.xDimension, 1), 1, 100);
        birthParameters(i).timestamps = zeros(1, 100);
    end
    model.birthParameters = birthParameters;

    % LBP/Gibbs/Murty params
    model.lbpConvergenceTolerance = 1e-6;
    model.maximumNumberOfLbpIterations = 100;
    model.numberOfSamples = 1000;
    model.numberOfAssignments = 25;

    % GM pruning
    model.weightThreshold = thresholds.gm_weight;
    model.gmWeightThreshold = thresholds.gm_weight;  % Another name for the same thing
    model.maximumNumberOfGmComponents = thresholds.max_components;
    model.mahalanobisDistanceThreshold = thresholds.gm_merge;
    model.minimumTrajectoryLength = 3;

    % Object template
    object.birthLocation = 0;
    object.birthTime = 0;
    object.r = 0;
    object.numberOfGmComponents = 0;
    object.w = zeros(0, 1);
    object.mu = {};
    object.Sigma = {};
    object.trajectoryLength = 0;
    object.trajectory = [];
    object.timestamps = zeros(1, 0);
    model.object = repmat(object, 0, 1);
    model.simulationLength = numSteps;
end

function measurements = extractMeasurements(scenario, numSensors, numSteps)
    % Handle both single and multi-sensor scenarios
    % MATLAB expects measurements as cell array of column vectors!
    % Octave decodes JSON differently than MATLAB:
    % - When measurements differ in count across sensors: cell array
    % - When measurements have same count: 3D numeric array

    if numSensors == 1
        measurements = cell(1, numSteps);
        for t = 1:numSteps
            sr = scenario.steps(t).sensor_readings;
            if isempty(sr) || (isnumeric(sr) && numel(sr) == 0)
                measurements{t} = {};
            elseif iscell(sr)
                meas = sr{1};
                measurements{t} = convertToMeasCell(meas);
            else
                % 3D array (1 x num_meas x 2)
                meas = squeeze(sr);
                measurements{t} = convertToMeasCell(meas);
            end
        end
    else
        measurements = cell(numSensors, numSteps);
        for t = 1:numSteps
            sr = scenario.steps(t).sensor_readings;
            if isempty(sr) || (isnumeric(sr) && numel(sr) == 0)
                for s = 1:numSensors
                    measurements{s, t} = {};
                end
            elseif iscell(sr)
                % Cell array - each cell is one sensor's data
                for s = 1:numSensors
                    meas = sr{s};
                    measurements{s, t} = convertToMeasCell(meas);
                end
            else
                % 3D numeric array (num_sensors x num_meas x 2)
                for s = 1:numSensors
                    if ndims(sr) == 3
                        meas = squeeze(sr(s, :, :));  % (num_meas x 2)
                        measurements{s, t} = convertToMeasCell(meas);
                    else
                        measurements{s, t} = {};
                    end
                end
            end
        end
    end
end

function z = convertToMeasCell(meas)
    % Convert measurement matrix to cell array of column vectors
    if isempty(meas)
        z = {};
        return;
    end
    if iscell(meas)
        z = meas;  % Already a cell array
        return;
    end
    % Ensure (num_meas x 2) orientation
    if size(meas, 1) == 2 && size(meas, 2) ~= 2
        meas = meas';
    end
    numMeas = size(meas, 1);
    z = cell(1, numMeas);
    for k = 1:numMeas
        z{k} = meas(k, :)';  % Column vector (2 x 1)
    end
end

function steps = extractStepResults(stateEstimates, numSteps)
    steps = cell(numSteps, 1);
    for t = 1:numSteps
        step = struct();
        step.step = t - 1;  % 0-indexed

        % labels is a 2 x nTracks matrix where each column is [birthTime; birthLocation]
        labels = stateEstimates.labels{t};
        if isempty(labels)
            step.num_tracks = 0;
        else
            step.num_tracks = size(labels, 2);
        end

        if step.num_tracks > 0
            step.tracks = cell(step.num_tracks, 1);
            for i = 1:step.num_tracks
                track = struct();
                % Create label as birthTime * 1000 + birthLocation for unique ID
                track.label = labels(1, i) * 1000 + labels(2, i);
                track.mean = stateEstimates.mu{t}{i}';
                step.tracks{i} = track;
            end
        else
            step.tracks = {};
        end
        steps{t} = step;
    end
end

function saveJson(data, filepath)
    jsonStr = jsonencode(data);
    fid = fopen(filepath, 'w');
    fprintf(fid, '%s', jsonStr);
    fclose(fid);
end
