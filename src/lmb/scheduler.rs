//! Update scheduling strategies for LMB filters.
//!
//! This module provides the [`UpdateScheduler`] trait and implementations that control
//! how sensors are processed during the measurement update phase. This replaces the
//! `is_sequential()` boolean flag pattern with polymorphic schedulers.
//!
//! # Scheduler Types
//!
//! - [`SequentialScheduler`]: Processes sensors sequentially (output of sensor A becomes
//!   input to sensor B). Used by IC-LMB and single-sensor LMB filters.
//!
//! - [`ParallelScheduler`]: Processes all sensors independently from the same prior,
//!   then fuses results using a [`Merger`]. Used by AA-LMB, GA-LMB, and PU-LMB filters.
//!
//! # Design Rationale
//!
//! The scheduler pattern inverts control flow: instead of the filter core checking
//! `if merger.is_sequential()`, the scheduler owns the iteration logic. This makes
//! the behavior explicit in types rather than runtime flags.
//!
//! # Future Integration
//!
//! These schedulers are designed to be used in the unified `LmbFilterCore` (Phase 8).
//! Currently, they serve as standalone abstractions that can be tested independently.
//!
//! # Example (Future API)
//!
//! ```ignore
//! use multisensor_lmb_filters_rs::lmb::{SequentialScheduler, ParallelScheduler, ArithmeticAverageMerger};
//!
//! // Single-sensor or IC-LMB: sequential processing
//! let scheduler = SequentialScheduler;
//!
//! // AA-LMB: parallel processing with arithmetic average fusion (2 sensors, 100 max components)
//! let scheduler = ParallelScheduler::new(ArithmeticAverageMerger::uniform(2, 100));
//! ```

use crate::lmb::traits::Merger;

// ============================================================================
// UpdateScheduler Trait
// ============================================================================

/// Trait controlling how sensors are processed during measurement update.
///
/// This trait abstracts the sensor processing strategy, allowing filters to be
/// parameterized by their update scheduling behavior rather than using runtime
/// flags.
///
/// # Type Parameters
///
/// The scheduler is designed to work with the filter's associator type, though
/// the current standalone implementation doesn't require it. The `Associator`
/// bound will be added when integrating with `LmbFilterCore` in Phase 8.
///
/// # Implementors
///
/// - [`SequentialScheduler`]: Sequential sensor processing (IC-LMB, single-sensor)
/// - [`ParallelScheduler<M>`]: Parallel processing with fusion (AA, GA, PU-LMB)
///
/// # Thread Safety
///
/// Schedulers are `Send + Sync` to enable filters to be used across threads.
pub trait UpdateScheduler: Send + Sync {
    /// Returns a human-readable name for this scheduler.
    fn name(&self) -> &'static str;

    /// Returns true if this scheduler processes sensors sequentially.
    ///
    /// Sequential schedulers pass the output of sensor N as input to sensor N+1.
    /// Parallel schedulers use the same prior tracks for all sensors, then fuse.
    fn is_sequential(&self) -> bool;

    /// Returns the number of sensors this scheduler expects.
    ///
    /// Returns `None` if the scheduler works with any number of sensors.
    fn expected_sensors(&self) -> Option<usize> {
        None
    }
}

// ============================================================================
// SequentialScheduler
// ============================================================================

/// Sequential sensor processing scheduler.
///
/// In sequential processing, the output of sensor N becomes the input to sensor N+1.
/// This is used by:
/// - **IC-LMB (Iterated Corrector)**: Multi-sensor filter that processes sensors in sequence
/// - **Single-sensor LMB**: Only one sensor, so sequential by default
///
/// # Properties
///
/// - Order-dependent: sensor ordering affects results
/// - No fusion step: final result is output of last sensor
/// - Simpler computation: no need to maintain per-sensor copies
///
/// # Example
///
/// ```
/// use multisensor_lmb_filters_rs::lmb::SequentialScheduler;
/// use multisensor_lmb_filters_rs::lmb::UpdateScheduler;
///
/// let scheduler = SequentialScheduler;
/// assert!(scheduler.is_sequential());
/// assert_eq!(scheduler.name(), "Sequential");
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SequentialScheduler;

impl SequentialScheduler {
    /// Create a new sequential scheduler.
    pub fn new() -> Self {
        Self
    }
}

impl UpdateScheduler for SequentialScheduler {
    fn name(&self) -> &'static str {
        "Sequential"
    }

    fn is_sequential(&self) -> bool {
        true
    }
}

// ============================================================================
// ParallelScheduler
// ============================================================================

/// Parallel sensor processing scheduler with track fusion.
///
/// In parallel processing, all sensors receive the same prior tracks and produce
/// independent posteriors. These posteriors are then fused using a [`Merger`].
///
/// This is used by:
/// - **AA-LMB (Arithmetic Average)**: Fast fusion via weighted averaging
/// - **GA-LMB (Geometric Average)**: Conservative fusion for unknown correlations
/// - **PU-LMB (Parallel Update)**: Optimal fusion for independent sensors
///
/// # Type Parameters
///
/// * `M` - The merger type used to fuse per-sensor track posteriors
///
/// # Properties
///
/// - Order-independent: sensor ordering doesn't affect results (for most mergers)
/// - Requires fusion step: maintains per-sensor copies, then merges
/// - Higher memory: stores S copies of track set during update
///
/// # Example
///
/// ```
/// use multisensor_lmb_filters_rs::lmb::{ParallelScheduler, MergerAverageArithmetic, UpdateScheduler};
///
/// // Create merger with 2 sensors and max 100 components
/// let merger = MergerAverageArithmetic::uniform(2, 100);
/// let scheduler = ParallelScheduler::new(merger);
/// assert!(!scheduler.is_sequential());
/// assert_eq!(scheduler.name(), "Parallel(ArithmeticAverage)");
/// ```
#[derive(Debug, Clone)]
pub struct ParallelScheduler<M: Merger> {
    merger: M,
}

impl<M: Merger> ParallelScheduler<M> {
    /// Create a new parallel scheduler with the given merger.
    pub fn new(merger: M) -> Self {
        Self { merger }
    }

    /// Returns a reference to the underlying merger.
    pub fn merger(&self) -> &M {
        &self.merger
    }

    /// Returns a mutable reference to the underlying merger.
    ///
    /// This is needed for mergers like PU-LMB that require `set_prior()` calls.
    pub fn merger_mut(&mut self) -> &mut M {
        &mut self.merger
    }

    /// Consumes the scheduler and returns the underlying merger.
    pub fn into_merger(self) -> M {
        self.merger
    }
}

impl<M: Merger> UpdateScheduler for ParallelScheduler<M> {
    fn name(&self) -> &'static str {
        // We can't create dynamic strings, so we use a fixed format
        // The merger name provides specificity
        match self.merger.name() {
            "ArithmeticAverage" => "Parallel(ArithmeticAverage)",
            "GeometricAverage" => "Parallel(GeometricAverage)",
            "ParallelUpdate" => "Parallel(ParallelUpdate)",
            "IteratedCorrector" => "Parallel(IteratedCorrector)", // Unusual but valid
            _ => "Parallel(Custom)",
        }
    }

    fn is_sequential(&self) -> bool {
        // ParallelScheduler with IC merger is still parallel processing
        // (the merger just returns the last sensor's result)
        false
    }
}

// ============================================================================
// SingleSensorScheduler
// ============================================================================

/// Scheduler for single-sensor filters.
///
/// This is a specialized scheduler that wraps [`SequentialScheduler`] but
/// enforces that exactly one sensor is used. It provides clearer intent
/// and better error messages for single-sensor filter configurations.
///
/// # Example
///
/// ```
/// use multisensor_lmb_filters_rs::lmb::{SingleSensorScheduler, UpdateScheduler};
///
/// let scheduler = SingleSensorScheduler::new();
/// assert!(scheduler.is_sequential());
/// assert_eq!(scheduler.expected_sensors(), Some(1));
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SingleSensorScheduler;

impl SingleSensorScheduler {
    /// Create a new single-sensor scheduler.
    pub fn new() -> Self {
        Self
    }
}

impl UpdateScheduler for SingleSensorScheduler {
    fn name(&self) -> &'static str {
        "SingleSensor"
    }

    fn is_sequential(&self) -> bool {
        true
    }

    fn expected_sensors(&self) -> Option<usize> {
        Some(1)
    }
}

// ============================================================================
// DynamicScheduler (for Python bindings)
// ============================================================================

/// Dynamic scheduler selection for runtime configuration.
///
/// This enum allows scheduler selection at runtime, which is useful for
/// Python bindings and configuration-driven filter creation.
///
/// # Example
///
/// ```
/// use multisensor_lmb_filters_rs::lmb::{DynamicScheduler, UpdateScheduler};
///
/// let scheduler = DynamicScheduler::Sequential;
/// assert!(scheduler.is_sequential());
///
/// let scheduler = DynamicScheduler::SingleSensor;
/// assert_eq!(scheduler.expected_sensors(), Some(1));
/// ```
#[derive(Debug, Clone, Copy)]
pub enum DynamicScheduler {
    /// Sequential processing (IC-LMB style).
    Sequential,
    /// Single-sensor processing.
    SingleSensor,
    /// Parallel with Arithmetic Average fusion.
    ParallelAA,
    /// Parallel with Geometric Average fusion.
    ParallelGA,
    /// Parallel with Parallel Update fusion.
    ParallelPU,
}

impl UpdateScheduler for DynamicScheduler {
    fn name(&self) -> &'static str {
        match self {
            DynamicScheduler::Sequential => "Sequential",
            DynamicScheduler::SingleSensor => "SingleSensor",
            DynamicScheduler::ParallelAA => "Parallel(ArithmeticAverage)",
            DynamicScheduler::ParallelGA => "Parallel(GeometricAverage)",
            DynamicScheduler::ParallelPU => "Parallel(ParallelUpdate)",
        }
    }

    fn is_sequential(&self) -> bool {
        match self {
            DynamicScheduler::Sequential | DynamicScheduler::SingleSensor => true,
            DynamicScheduler::ParallelAA
            | DynamicScheduler::ParallelGA
            | DynamicScheduler::ParallelPU => false,
        }
    }

    fn expected_sensors(&self) -> Option<usize> {
        match self {
            DynamicScheduler::SingleSensor => Some(1),
            _ => None,
        }
    }
}

// ============================================================================
// Type Aliases for Common Configurations
// ============================================================================

/// Type alias for scheduler used by single-sensor LMB filters.
pub type LmbScheduler = SingleSensorScheduler;

/// Type alias for scheduler used by IC-LMB (Iterated Corrector) filters.
pub type IcLmbScheduler = SequentialScheduler;

// Note: AA, GA, PU scheduler aliases would require the specific merger types,
// which creates a circular dependency. We define them in the multisensor module.

// ============================================================================
// Marker Traits for Compile-Time Guarantees
// ============================================================================

/// Marker trait for schedulers that support multi-sensor configurations.
///
/// This trait is automatically implemented for schedulers that can handle
/// more than one sensor.
pub trait MultisensorCapable: UpdateScheduler {}

impl MultisensorCapable for SequentialScheduler {}
impl<M: Merger> MultisensorCapable for ParallelScheduler<M> {}

// SingleSensorScheduler does NOT implement MultisensorCapable

/// Marker trait for schedulers that perform track fusion.
///
/// This trait is automatically implemented for parallel schedulers that
/// merge per-sensor track posteriors.
pub trait FusionCapable: UpdateScheduler {
    /// The merger type used for fusion.
    type Merger: Merger;

    /// Returns a reference to the merger.
    fn get_merger(&self) -> &Self::Merger;
}

impl<M: Merger> FusionCapable for ParallelScheduler<M> {
    type Merger = M;

    fn get_merger(&self) -> &Self::Merger {
        &self.merger
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmb::multisensor::{
        MergerAverageArithmetic, MergerAverageGeometric, MergerIteratedCorrector,
        MergerParallelUpdate,
    };

    #[test]
    fn test_sequential_scheduler() {
        let scheduler = SequentialScheduler::new();
        assert!(scheduler.is_sequential());
        assert_eq!(scheduler.name(), "Sequential");
        assert_eq!(scheduler.expected_sensors(), None);
    }

    #[test]
    fn test_single_sensor_scheduler() {
        let scheduler = SingleSensorScheduler::new();
        assert!(scheduler.is_sequential());
        assert_eq!(scheduler.name(), "SingleSensor");
        assert_eq!(scheduler.expected_sensors(), Some(1));
    }

    #[test]
    fn test_parallel_scheduler_aa() {
        // ArithmeticAverageMerger requires num_sensors and max_components
        let merger = MergerAverageArithmetic::uniform(2, 100);
        let scheduler = ParallelScheduler::new(merger);
        assert!(!scheduler.is_sequential());
        assert_eq!(scheduler.name(), "Parallel(ArithmeticAverage)");
        assert_eq!(scheduler.expected_sensors(), None);
    }

    #[test]
    fn test_parallel_scheduler_ga() {
        // GeometricAverageMerger requires num_sensors
        let merger = MergerAverageGeometric::uniform(2);
        let scheduler = ParallelScheduler::new(merger);
        assert!(!scheduler.is_sequential());
        assert_eq!(scheduler.name(), "Parallel(GeometricAverage)");
    }

    #[test]
    fn test_parallel_scheduler_pu() {
        // ParallelUpdateMerger requires prior tracks (empty initially)
        let merger = MergerParallelUpdate::new(Vec::new());
        let scheduler = ParallelScheduler::new(merger);
        assert!(!scheduler.is_sequential());
        assert_eq!(scheduler.name(), "Parallel(ParallelUpdate)");
    }

    #[test]
    fn test_parallel_scheduler_ic() {
        // IC can be wrapped in ParallelScheduler (unusual but valid)
        let merger = MergerIteratedCorrector::new();
        let scheduler = ParallelScheduler::new(merger);
        // ParallelScheduler is always parallel (merger just returns last result)
        assert!(!scheduler.is_sequential());
        assert_eq!(scheduler.name(), "Parallel(IteratedCorrector)");
    }

    #[test]
    fn test_parallel_scheduler_merger_access() {
        let merger = MergerAverageArithmetic::uniform(2, 100);
        let mut scheduler = ParallelScheduler::new(merger);

        // Can access merger
        assert_eq!(scheduler.merger().name(), "ArithmeticAverage");

        // Can mutably access merger (needed for PU-LMB set_prior)
        let _ = scheduler.merger_mut();

        // Can consume and get merger back
        let recovered = scheduler.into_merger();
        assert_eq!(recovered.name(), "ArithmeticAverage");
    }

    #[test]
    fn test_dynamic_scheduler() {
        let sequential = DynamicScheduler::Sequential;
        assert!(sequential.is_sequential());
        assert_eq!(sequential.name(), "Sequential");

        let single = DynamicScheduler::SingleSensor;
        assert!(single.is_sequential());
        assert_eq!(single.expected_sensors(), Some(1));

        let aa = DynamicScheduler::ParallelAA;
        assert!(!aa.is_sequential());
        assert_eq!(aa.name(), "Parallel(ArithmeticAverage)");

        let ga = DynamicScheduler::ParallelGA;
        assert!(!ga.is_sequential());

        let pu = DynamicScheduler::ParallelPU;
        assert!(!pu.is_sequential());
    }

    #[test]
    fn test_multisensor_capable_marker() {
        // SequentialScheduler is multisensor capable
        fn accepts_multisensor<S: MultisensorCapable>(_: &S) {}
        accepts_multisensor(&SequentialScheduler::new());

        let aa = ParallelScheduler::new(MergerAverageArithmetic::uniform(2, 100));
        accepts_multisensor(&aa);

        // SingleSensorScheduler is NOT multisensor capable (won't compile)
        // accepts_multisensor(&SingleSensorScheduler::new()); // Compile error
    }

    #[test]
    fn test_fusion_capable_marker() {
        fn accepts_fusion<S: FusionCapable>(s: &S) -> &'static str {
            s.get_merger().name()
        }

        let aa = ParallelScheduler::new(MergerAverageArithmetic::uniform(2, 100));
        assert_eq!(accepts_fusion(&aa), "ArithmeticAverage");

        let ga = ParallelScheduler::new(MergerAverageGeometric::uniform(2));
        assert_eq!(accepts_fusion(&ga), "GeometricAverage");

        // SequentialScheduler does NOT implement FusionCapable (won't compile)
        // accepts_fusion(&SequentialScheduler::new()); // Compile error
    }

    #[test]
    fn test_scheduler_send_sync() {
        // Verify all schedulers are Send + Sync (required for multi-threaded use)
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<SequentialScheduler>();
        assert_send_sync::<SingleSensorScheduler>();
        assert_send_sync::<ParallelScheduler<MergerAverageArithmetic>>();
        assert_send_sync::<ParallelScheduler<MergerAverageGeometric>>();
        assert_send_sync::<ParallelScheduler<MergerParallelUpdate>>();
        assert_send_sync::<ParallelScheduler<MergerIteratedCorrector>>();
        assert_send_sync::<DynamicScheduler>();
    }

    #[test]
    fn test_type_aliases() {
        let _lmb: LmbScheduler = SingleSensorScheduler::new();
        let _ic: IcLmbScheduler = SequentialScheduler::new();
    }
}
