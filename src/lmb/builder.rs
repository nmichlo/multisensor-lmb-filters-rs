//! Builder pattern traits for filter configuration.
//!
//! This module provides traits with default implementations for common
//! filter configuration methods, reducing code duplication across filter types.
//!
//! # Traits
//!
//! - [`FilterBuilder`] - Base trait for all filters (existence threshold, trajectory length)
//! - [`LmbFilterBuilder`] - Extended trait for LMB filters (GM pruning parameters)

/// Base builder trait for all filter types.
///
/// Provides common configuration methods with default implementations.
/// Filters implement the `*_mut()` accessor methods and get the builder
/// methods for free.
pub trait FilterBuilder: Sized {
    /// Mutable access to the existence threshold.
    fn existence_threshold_mut(&mut self) -> &mut f64;

    /// Mutable access to the minimum trajectory length.
    fn min_trajectory_length_mut(&mut self) -> &mut usize;

    /// Set the existence probability threshold for track pruning.
    ///
    /// Tracks with existence probability below this threshold are removed.
    /// Default is typically 1e-3.
    fn with_existence_threshold(mut self, threshold: f64) -> Self {
        *self.existence_threshold_mut() = threshold;
        self
    }

    /// Set the minimum trajectory length to save when pruning.
    ///
    /// Tracks shorter than this are discarded without saving their trajectory.
    /// Default is typically 3.
    fn with_min_trajectory_length(mut self, length: usize) -> Self {
        *self.min_trajectory_length_mut() = length;
        self
    }
}

/// Extended builder trait for LMB filters with Gaussian mixture components.
///
/// LMB filters (as opposed to LMBM) maintain Gaussian mixture posteriors
/// and need additional pruning parameters for the GM components.
pub trait LmbFilterBuilder: FilterBuilder {
    /// Mutable access to the GM weight threshold.
    fn gm_weight_threshold_mut(&mut self) -> &mut f64;

    /// Mutable access to the maximum GM components.
    fn max_gm_components_mut(&mut self) -> &mut usize;

    /// Set Gaussian mixture pruning parameters.
    ///
    /// - `threshold`: Components with weight below this are pruned (default 1e-4)
    /// - `max_components`: Maximum components to keep per track (default 100)
    fn with_gm_pruning(mut self, threshold: f64, max_components: usize) -> Self {
        *self.gm_weight_threshold_mut() = threshold;
        *self.max_gm_components_mut() = max_components;
        self
    }
}
