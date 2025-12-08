"""Tests for filter implementations."""

import numpy as np


class TestLmbFilter:
    """Test single-sensor LMB filter."""

    def test_create(self, motion_model_2d, sensor_model_2d, birth_model_2d):
        from multisensor_lmb_filters_rs import LmbFilter

        filter = LmbFilter(
            motion=motion_model_2d,
            sensor=sensor_model_2d,
            birth=birth_model_2d,
        )
        assert filter.num_tracks == 0

    def test_step_empty(self, lmb_filter):
        """Test step with no measurements."""
        estimate = lmb_filter.step(measurements=[], timestep=0)
        assert estimate.timestamp == 0
        # With no measurements, tracks may or may not be created from birth
        # (depends on existence threshold)

    def test_step_with_measurements(self, lmb_filter):
        """Test step with measurements."""
        measurements = [np.array([1.0, 2.0])]
        estimate = lmb_filter.step(measurements=measurements, timestep=0)
        assert estimate.timestamp == 0

    def test_multiple_steps(self, lmb_filter):
        """Test multiple filter steps."""
        for t in range(5):
            meas = [np.array([t * 0.1, t * 0.2])]
            estimate = lmb_filter.step(measurements=meas, timestep=t)
            assert estimate.timestamp == t

    def test_reset(self, lmb_filter):
        """Test filter reset."""
        # Run a few steps
        for t in range(3):
            lmb_filter.step(measurements=[np.array([1.0, 2.0])], timestep=t)

        # Reset
        lmb_filter.reset()
        assert lmb_filter.num_tracks == 0

    def test_reproducibility_with_seed(self, motion_model_2d, sensor_model_2d, birth_model_2d):
        """Test that filters with same seed produce same results."""
        from multisensor_lmb_filters_rs import LmbFilter

        filter1 = LmbFilter(
            motion=motion_model_2d,
            sensor=sensor_model_2d,
            birth=birth_model_2d,
            seed=42,
        )
        filter2 = LmbFilter(
            motion=motion_model_2d,
            sensor=sensor_model_2d,
            birth=birth_model_2d,
            seed=42,
        )

        measurements = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]

        est1 = filter1.step(measurements=measurements, timestep=0)
        est2 = filter2.step(measurements=measurements, timestep=0)

        assert est1.num_tracks == est2.num_tracks
        for t1, t2 in zip(est1.tracks, est2.tracks):
            np.testing.assert_array_almost_equal(t1.mean, t2.mean)

    def test_different_association_methods(self, motion_model_2d, sensor_model_2d, birth_model_2d):
        """Test different association methods produce results."""
        from multisensor_lmb_filters_rs import AssociationConfig, LmbFilter

        methods = [
            AssociationConfig.lbp(),
            AssociationConfig.gibbs(samples=100),
            AssociationConfig.murty(assignments=50),
        ]

        measurements = [np.array([1.0, 2.0])]

        for config in methods:
            filter = LmbFilter(
                motion=motion_model_2d,
                sensor=sensor_model_2d,
                birth=birth_model_2d,
                association=config,
                seed=42,
            )
            estimate = filter.step(measurements=measurements, timestep=0)
            assert estimate.timestamp == 0


class TestLmbmFilter:
    """Test single-sensor LMBM filter."""

    def test_create(self, motion_model_2d, sensor_model_2d, birth_model_2d):
        from multisensor_lmb_filters_rs import LmbmFilter

        filter = LmbmFilter(
            motion=motion_model_2d,
            sensor=sensor_model_2d,
            birth=birth_model_2d,
        )
        # LMBM starts with at least 1 hypothesis (the empty hypothesis)
        assert filter.num_hypotheses >= 1

    def test_step(self, lmbm_filter):
        """Test LMBM filter step."""
        measurements = [np.array([1.0, 2.0])]
        estimate = lmbm_filter.step(measurements=measurements, timestep=0)
        assert estimate.timestamp == 0

    def test_custom_config(self, motion_model_2d, sensor_model_2d, birth_model_2d):
        """Test LMBM with custom config."""
        from multisensor_lmb_filters_rs import LmbmConfig, LmbmFilter

        config = LmbmConfig(max_hypotheses=100, hypothesis_weight_threshold=1e-4, use_eap=True)
        filter = LmbmFilter(
            motion=motion_model_2d,
            sensor=sensor_model_2d,
            birth=birth_model_2d,
            lmbm_config=config,
            seed=42,
        )
        estimate = filter.step(measurements=[np.array([1.0, 2.0])], timestep=0)
        assert estimate.timestamp == 0


class TestMultisensorFilters:
    """Test multi-sensor filter variants."""

    def test_aa_lmb_filter(self, motion_model_2d, multisensor_config_2d, birth_model_2d):
        """Test AA-LMB filter."""
        from multisensor_lmb_filters_rs import AaLmbFilter

        filter = AaLmbFilter(
            motion=motion_model_2d,
            sensors=multisensor_config_2d,
            birth=birth_model_2d,
            seed=42,
        )
        assert filter.num_tracks == 0

        # Measurements for 2 sensors
        measurements = [
            [np.array([1.0, 2.0])],  # sensor 0
            [np.array([1.1, 2.1])],  # sensor 1
        ]
        estimate = filter.step(measurements=measurements, timestep=0)
        assert estimate.timestamp == 0

    def test_ga_lmb_filter(self, motion_model_2d, multisensor_config_2d, birth_model_2d):
        """Test GA-LMB filter."""
        from multisensor_lmb_filters_rs import GaLmbFilter

        filter = GaLmbFilter(
            motion=motion_model_2d,
            sensors=multisensor_config_2d,
            birth=birth_model_2d,
            seed=42,
        )
        measurements = [
            [np.array([1.0, 2.0])],
            [np.array([1.1, 2.1])],
        ]
        estimate = filter.step(measurements=measurements, timestep=0)
        assert estimate.timestamp == 0

    def test_pu_lmb_filter(self, motion_model_2d, multisensor_config_2d, birth_model_2d):
        """Test PU-LMB filter."""
        from multisensor_lmb_filters_rs import PuLmbFilter

        filter = PuLmbFilter(
            motion=motion_model_2d,
            sensors=multisensor_config_2d,
            birth=birth_model_2d,
            seed=42,
        )
        measurements = [
            [np.array([1.0, 2.0])],
            [np.array([1.1, 2.1])],
        ]
        estimate = filter.step(measurements=measurements, timestep=0)
        assert estimate.timestamp == 0

    def test_ic_lmb_filter(self, motion_model_2d, multisensor_config_2d, birth_model_2d):
        """Test IC-LMB filter."""
        from multisensor_lmb_filters_rs import IcLmbFilter

        filter = IcLmbFilter(
            motion=motion_model_2d,
            sensors=multisensor_config_2d,
            birth=birth_model_2d,
            seed=42,
        )
        measurements = [
            [np.array([1.0, 2.0])],
            [np.array([1.1, 2.1])],
        ]
        estimate = filter.step(measurements=measurements, timestep=0)
        assert estimate.timestamp == 0

    def test_multisensor_lmbm_filter(self, motion_model_2d, multisensor_config_2d, birth_model_2d):
        """Test multi-sensor LMBM filter."""
        from multisensor_lmb_filters_rs import MultisensorLmbmFilter

        filter = MultisensorLmbmFilter(
            motion=motion_model_2d,
            sensors=multisensor_config_2d,
            birth=birth_model_2d,
            seed=42,
        )
        measurements = [
            [np.array([1.0, 2.0])],
            [np.array([1.1, 2.1])],
        ]
        estimate = filter.step(measurements=measurements, timestep=0)
        assert estimate.timestamp == 0


class TestStateEstimate:
    """Test StateEstimate output class."""

    def test_state_estimate_properties(self, lmb_filter):
        """Test StateEstimate properties."""
        # Run with high birth probability to ensure tracks
        estimate = lmb_filter.step(measurements=[np.array([1.0, 2.0])], timestep=5)

        assert estimate.timestamp == 5
        assert estimate.num_tracks >= 0
        assert len(estimate) == estimate.num_tracks

    def test_estimated_track_properties(self, motion_model_2d, sensor_model_2d, birth_model_2d):
        """Test EstimatedTrack properties."""
        from multisensor_lmb_filters_rs import BirthLocation, BirthModel, LmbFilter

        # High birth probability to ensure tracks are created
        birth_loc = BirthLocation(
            label=0,
            mean=np.array([1.0, 0.0, 2.0, 0.0]),  # Near measurement
            covariance=np.eye(4) * 10.0,
        )
        birth = BirthModel(
            locations=[birth_loc],
            lmb_existence=0.9,  # High birth probability
            lmbm_existence=0.1,
        )

        filter = LmbFilter(motion=motion_model_2d, sensor=sensor_model_2d, birth=birth, seed=42)

        # Measurement near birth location
        estimate = filter.step(measurements=[np.array([1.0, 2.0])], timestep=0)

        if estimate.num_tracks > 0:
            track = estimate.tracks[0]
            assert track.label is not None
            assert track.mean.shape == (4,)
            assert track.covariance.shape == (4, 4)
            assert track.x_dim == 4


class TestProtocolCompliance:
    """Test that classes comply with defined protocols."""

    def test_has_label_protocol(self, motion_model_2d, sensor_model_2d, birth_model_2d):
        """Test HasLabel protocol compliance."""
        from multisensor_lmb_filters_rs import Track, TrackLabel

        label = TrackLabel(birth_time=0, birth_location=0)
        track = Track(
            label=label,
            existence=0.8,
            mean=np.zeros(4),
            covariance=np.eye(4),
        )

        # Protocol check
        assert hasattr(track, "label")
        assert track.label == label

    def test_has_mean_protocol(self):
        """Test HasMean protocol compliance."""
        from multisensor_lmb_filters_rs import BirthLocation, GaussianComponent

        # GaussianComponent has mean
        comp = GaussianComponent(weight=1.0, mean=np.array([1.0, 2.0]), covariance=np.eye(2))
        assert hasattr(comp, "mean")

        # BirthLocation has mean
        loc = BirthLocation(label=0, mean=np.zeros(4), covariance=np.eye(4))
        assert hasattr(loc, "mean")

    def test_has_state_dimension_protocol(self):
        """Test HasStateDimension protocol compliance."""
        from multisensor_lmb_filters_rs import (
            GaussianComponent,
            MotionModel,
            SensorModel,
            Track,
            TrackLabel,
        )

        # All these should have x_dim
        motion = MotionModel.constant_velocity_2d(1.0, 0.1, 0.99)
        assert hasattr(motion, "x_dim")
        assert motion.x_dim == 4

        sensor = SensorModel.position_sensor_2d(1.0, 0.9, 10.0, 100.0)
        assert hasattr(sensor, "x_dim")
        assert sensor.x_dim == 4

        comp = GaussianComponent(weight=1.0, mean=np.zeros(4), covariance=np.eye(4))
        assert hasattr(comp, "x_dim")
        assert comp.x_dim == 4

        label = TrackLabel(birth_time=0, birth_location=0)
        track = Track(
            label=label,
            existence=0.8,
            mean=np.zeros(4),
            covariance=np.eye(4),
        )
        assert hasattr(track, "x_dim")
        assert track.x_dim == 4

    def test_single_sensor_filter_protocol(self, motion_model_2d, sensor_model_2d, birth_model_2d):
        """Test SingleSensorFilter protocol compliance."""
        from multisensor_lmb_filters_rs import LmbFilter

        filter = LmbFilter(
            motion=motion_model_2d,
            sensor=sensor_model_2d,
            birth=birth_model_2d,
            seed=42,
        )

        # Protocol methods
        assert hasattr(filter, "step")
        assert hasattr(filter, "reset")
        assert hasattr(filter, "num_tracks")

        # Can call protocol methods
        estimate = filter.step(measurements=[], timestep=0)
        assert estimate is not None
        filter.reset()
        assert filter.num_tracks == 0

    def test_multisensor_filter_protocol(
        self, motion_model_2d, multisensor_config_2d, birth_model_2d
    ):
        """Test MultisensorFilter protocol compliance."""
        from multisensor_lmb_filters_rs import AaLmbFilter

        filter = AaLmbFilter(
            motion=motion_model_2d,
            sensors=multisensor_config_2d,
            birth=birth_model_2d,
            seed=42,
        )

        # Protocol methods
        assert hasattr(filter, "step")
        assert hasattr(filter, "reset")
        assert hasattr(filter, "num_tracks")

        # Can call protocol methods with nested measurements
        estimate = filter.step(measurements=[[], []], timestep=0)
        assert estimate is not None
