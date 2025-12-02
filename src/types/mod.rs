//! Core types for the tracking library
//!
//! This module provides the unified type system used throughout the library.
//!
//! # Types
//!
//! - [`Track`] - Core track type with Gaussian mixture components
//! - [`GaussianComponent`] - Single Gaussian component
//! - [`TrackLabel`] - Unique track identifier
//! - [`LmbmHypothesis`] - LMBM hypothesis
//! - [`StateEstimate`] - Output estimates at a timestep
//! - [`FilterOutput`] - Complete filter run output

pub mod track;
pub mod config;
pub mod output;

// Re-export all public types
pub use track::{
    GaussianComponent,
    LmbmHypothesis,
    Track,
    TrackLabel,
    TrajectoryHistory,
};

pub use config::{
    AssociationConfig,
    BirthLocation,
    BirthModel,
    DataAssociationMethod,
    FilterParams,
    FilterParamsBuilder,
    FilterThresholds,
    LmbmConfig,
    MotionModel,
    MultisensorConfig,
    SensorModel,
    SensorVariant,
};

pub use output::{
    EstimatedTrack,
    FilterOutput,
    StateEstimate,
    Trajectory,
};
