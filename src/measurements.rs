//! Zero-copy measurement abstractions for LMB filters.
//!
//! This module provides the [`MeasurementSource`] trait for flexible, zero-copy
//! measurement input to filters. It avoids forcing `Vec<Vec<...>>` allocations
//! on users with pre-allocated or memory-mapped measurement data.
//!
//! # Design Philosophy
//!
//! The key insight is that filters only need to **iterate** over measurements, not own them.
//! By accepting anything that can produce iterators over measurement references, we support:
//!
//! - Borrowed slices from existing allocations
//! - Memory-mapped arrays from disk
//! - Pre-allocated sensor buffers
//! - Zero-copy views into packed data structures
//!
//! # Example Usage
//!
//! ```ignore
//! use multisensor_lmb_filters_rs::lmb::{MeasurementSource, SingleSensorMeasurements};
//!
//! // Single-sensor: slice of measurements
//! let measurements: Vec<DVector<f64>> = vec![/* ... */];
//! let source = SingleSensorMeasurements::from(&measurements[..]);
//!
//! // Multi-sensor: slice of slices
//! let sensor_0: Vec<DVector<f64>> = vec![/* ... */];
//! let sensor_1: Vec<DVector<f64>> = vec![/* ... */];
//! let multi = vec![&sensor_0[..], &sensor_1[..]];
//! let source = SliceOfSlicesMeasurements::from(&multi[..]);
//! ```
//!
//! # Implementation Notes
//!
//! This trait uses Generic Associated Types (GATs) to express the lifetime
//! relationship between the source and its iterators. This enables truly
//! zero-copy iteration without requiring `Box<dyn Iterator>` or `impl Trait`.

use nalgebra::DVector;
use std::iter::{Copied, Map};
use std::slice::Iter;

// ============================================================================
// MeasurementSource Trait
// ============================================================================

/// Zero-copy abstraction over measurement data sources.
///
/// This trait enables filters to accept measurements from various sources
/// without requiring ownership or specific container types. Implementations
/// provide iterators over sensors and measurements without copying data.
///
/// # Type Parameters
///
/// The trait uses Generic Associated Types (GATs) to express the relationship
/// between the source lifetime and iterator lifetimes:
///
/// - `SensorIter<'a>`: Iterator over sensors, each yielding a `MeasIter<'a>`
/// - `MeasIter<'a>`: Iterator over measurements for a single sensor
///
/// # Zero-Copy Guarantee
///
/// All implementations in this module are guaranteed to be zero-copy:
/// - No measurement data is cloned or moved
/// - Iterators yield references to the original data
/// - The only allocations are for the iterator state itself (typically stack-allocated)
pub trait MeasurementSource {
    /// Iterator type over sensors. Each item is itself an iterator over measurements.
    type SensorIter<'a>: Iterator<Item = Self::MeasIter<'a>>
    where
        Self: 'a;

    /// Iterator type over measurements for a single sensor.
    type MeasIter<'a>: Iterator<Item = &'a DVector<f64>>
    where
        Self: 'a;

    /// Returns the number of sensors in this measurement source.
    ///
    /// For single-sensor sources, this always returns 1.
    /// For multi-sensor sources, this returns the number of sensor measurement sets.
    fn num_sensors(&self) -> usize;

    /// Returns an iterator over sensors, where each sensor yields an iterator over its measurements.
    ///
    /// The returned iterator yields `Self::MeasIter` items, each representing
    /// the measurements from one sensor at the current timestep.
    fn sensors(&self) -> Self::SensorIter<'_>;

    /// Returns true if there are no measurements from any sensor.
    fn is_empty(&self) -> bool {
        self.sensors().all(|mut meas| meas.next().is_none())
    }

    /// Returns true if any sensor has at least one measurement.
    fn has_any_measurements(&self) -> bool {
        self.sensors().any(|mut meas| meas.next().is_some())
    }

    /// Counts the total number of measurements across all sensors.
    fn total_measurements(&self) -> usize {
        self.sensors().map(|meas| meas.count()).sum()
    }

    /// Returns the number of measurements for a specific sensor.
    ///
    /// Returns `None` if the sensor index is out of bounds.
    fn measurements_for_sensor(&self, sensor_idx: usize) -> Option<usize> {
        self.sensors().nth(sensor_idx).map(|meas| meas.count())
    }
}

// ============================================================================
// Single-Sensor Wrapper: &[DVector<f64>]
// ============================================================================

/// Zero-copy wrapper for single-sensor measurements.
///
/// Wraps a slice of measurements and presents them as a single-sensor source.
/// This is the most utils case for traditional single-sensor tracking.
///
/// # Example
///
/// ```ignore
/// let measurements: Vec<DVector<f64>> = vec![/* ... */];
/// let source = SingleSensorMeasurements::new(&measurements);
/// assert_eq!(source.num_sensors(), 1);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SingleSensorMeasurements<'a> {
    measurements: &'a [DVector<f64>],
}

impl<'a> SingleSensorMeasurements<'a> {
    /// Creates a new single-sensor measurement source from a slice.
    #[inline]
    pub const fn new(measurements: &'a [DVector<f64>]) -> Self {
        Self { measurements }
    }

    /// Returns the underlying slice of measurements.
    #[inline]
    pub const fn as_slice(&self) -> &'a [DVector<f64>] {
        self.measurements
    }
}

impl<'a> From<&'a [DVector<f64>]> for SingleSensorMeasurements<'a> {
    #[inline]
    fn from(measurements: &'a [DVector<f64>]) -> Self {
        Self::new(measurements)
    }
}

impl<'a> From<&'a Vec<DVector<f64>>> for SingleSensorMeasurements<'a> {
    #[inline]
    fn from(measurements: &'a Vec<DVector<f64>>) -> Self {
        Self::new(measurements.as_slice())
    }
}

/// Iterator that yields exactly one measurement iterator (for single-sensor).
#[derive(Debug)]
pub struct SingleSensorIter<'a> {
    inner: std::iter::Once<Iter<'a, DVector<f64>>>,
}

impl<'a> Iterator for SingleSensorIter<'a> {
    type Item = Iter<'a, DVector<f64>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> ExactSizeIterator for SingleSensorIter<'a> {}

impl<'a> MeasurementSource for SingleSensorMeasurements<'a> {
    type SensorIter<'b>
        = SingleSensorIter<'b>
    where
        Self: 'b;
    type MeasIter<'b>
        = Iter<'b, DVector<f64>>
    where
        Self: 'b;

    #[inline]
    fn num_sensors(&self) -> usize {
        1
    }

    #[inline]
    fn sensors(&self) -> Self::SensorIter<'_> {
        SingleSensorIter {
            inner: std::iter::once(self.measurements.iter()),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.measurements.is_empty()
    }

    #[inline]
    fn has_any_measurements(&self) -> bool {
        !self.measurements.is_empty()
    }

    #[inline]
    fn total_measurements(&self) -> usize {
        self.measurements.len()
    }

    #[inline]
    fn measurements_for_sensor(&self, sensor_idx: usize) -> Option<usize> {
        if sensor_idx == 0 {
            Some(self.measurements.len())
        } else {
            None
        }
    }
}

// ============================================================================
// Multi-Sensor Wrapper: &[Vec<DVector<f64>>]
// ============================================================================

/// Zero-copy wrapper for multi-sensor measurements stored as `&[Vec<DVector<f64>>]`.
///
/// This is the standard layout for multi-sensor tracking where each sensor's
/// measurements are stored in a separate `Vec`.
///
/// # Example
///
/// ```ignore
/// let measurements: Vec<Vec<DVector<f64>>> = vec![vec![/* sensor 0 */], vec![/* sensor 1 */]];
/// let source = VecOfVecsMeasurements::new(&measurements);
/// assert_eq!(source.num_sensors(), 2);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct VecOfVecsMeasurements<'a> {
    sensors: &'a [Vec<DVector<f64>>],
}

impl<'a> VecOfVecsMeasurements<'a> {
    /// Creates a new multi-sensor measurement source.
    #[inline]
    pub const fn new(sensors: &'a [Vec<DVector<f64>>]) -> Self {
        Self { sensors }
    }

    /// Returns the underlying slice of sensor measurements.
    #[inline]
    pub const fn as_slice(&self) -> &'a [Vec<DVector<f64>>] {
        self.sensors
    }
}

impl<'a> From<&'a [Vec<DVector<f64>>]> for VecOfVecsMeasurements<'a> {
    #[inline]
    fn from(sensors: &'a [Vec<DVector<f64>>]) -> Self {
        Self::new(sensors)
    }
}

impl<'a> From<&'a Vec<Vec<DVector<f64>>>> for VecOfVecsMeasurements<'a> {
    #[inline]
    fn from(sensors: &'a Vec<Vec<DVector<f64>>>) -> Self {
        Self::new(sensors.as_slice())
    }
}

/// Iterator over sensors in a `VecOfVecsMeasurements`.
#[derive(Debug)]
#[allow(clippy::type_complexity)]
pub struct VecOfVecsIter<'a> {
    inner: Map<Iter<'a, Vec<DVector<f64>>>, fn(&Vec<DVector<f64>>) -> Iter<'_, DVector<f64>>>,
}

impl<'a> Iterator for VecOfVecsIter<'a> {
    type Item = Iter<'a, DVector<f64>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> ExactSizeIterator for VecOfVecsIter<'a> {}

impl<'a> MeasurementSource for VecOfVecsMeasurements<'a> {
    type SensorIter<'b>
        = VecOfVecsIter<'b>
    where
        Self: 'b;
    type MeasIter<'b>
        = Iter<'b, DVector<f64>>
    where
        Self: 'b;

    #[inline]
    fn num_sensors(&self) -> usize {
        self.sensors.len()
    }

    #[inline]
    #[allow(clippy::ptr_arg)]
    fn sensors(&self) -> Self::SensorIter<'_> {
        fn as_iter(v: &Vec<DVector<f64>>) -> Iter<'_, DVector<f64>> {
            v.iter()
        }
        VecOfVecsIter {
            inner: self
                .sensors
                .iter()
                .map(as_iter as fn(&Vec<DVector<f64>>) -> Iter<'_, DVector<f64>>),
        }
    }
}

// ============================================================================
// Multi-Sensor Wrapper: &[&[DVector<f64>]]
// ============================================================================

/// Zero-copy wrapper for multi-sensor measurements stored as `&[&[DVector<f64>]]`.
///
/// This layout is useful when measurements come from memory-mapped files,
/// pre-allocated buffers, or other sources that provide slices directly.
///
/// # Example
///
/// ```ignore
/// let sensor_0: Vec<DVector<f64>> = vec![/* ... */];
/// let sensor_1: Vec<DVector<f64>> = vec![/* ... */];
/// let slices: Vec<&[DVector<f64>]> = vec![&sensor_0, &sensor_1];
/// let source = SliceOfSlicesMeasurements::new(&slices);
/// assert_eq!(source.num_sensors(), 2);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct SliceOfSlicesMeasurements<'a> {
    sensors: &'a [&'a [DVector<f64>]],
}

impl<'a> SliceOfSlicesMeasurements<'a> {
    /// Creates a new multi-sensor measurement source from a slice of slices.
    #[inline]
    pub const fn new(sensors: &'a [&'a [DVector<f64>]]) -> Self {
        Self { sensors }
    }

    /// Returns the underlying slice of sensor measurement slices.
    #[inline]
    pub const fn as_slice(&self) -> &'a [&'a [DVector<f64>]] {
        self.sensors
    }
}

impl<'a> From<&'a [&'a [DVector<f64>]]> for SliceOfSlicesMeasurements<'a> {
    #[inline]
    fn from(sensors: &'a [&'a [DVector<f64>]]) -> Self {
        Self::new(sensors)
    }
}

/// Iterator over sensors in a `SliceOfSlicesMeasurements`.
#[derive(Debug)]
pub struct SliceOfSlicesIter<'a> {
    #[allow(clippy::type_complexity)]
    inner:
        Map<Copied<Iter<'a, &'a [DVector<f64>]>>, fn(&'a [DVector<f64>]) -> Iter<'a, DVector<f64>>>,
}

impl<'a> Iterator for SliceOfSlicesIter<'a> {
    type Item = Iter<'a, DVector<f64>>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> ExactSizeIterator for SliceOfSlicesIter<'a> {}

impl<'a> MeasurementSource for SliceOfSlicesMeasurements<'a> {
    type SensorIter<'b>
        = SliceOfSlicesIter<'b>
    where
        Self: 'b;
    type MeasIter<'b>
        = Iter<'b, DVector<f64>>
    where
        Self: 'b;

    #[inline]
    fn num_sensors(&self) -> usize {
        self.sensors.len()
    }

    #[inline]
    fn sensors(&self) -> Self::SensorIter<'_> {
        fn as_iter(slice: &[DVector<f64>]) -> Iter<'_, DVector<f64>> {
            slice.iter()
        }
        SliceOfSlicesIter {
            inner: self
                .sensors
                .iter()
                .copied()
                .map(as_iter as fn(&[DVector<f64>]) -> Iter<'_, DVector<f64>>),
        }
    }
}

// ============================================================================
// Convenience: Implement MeasurementSource for reference types
// ============================================================================

// &[DVector<f64>] implements MeasurementSource as single-sensor
impl MeasurementSource for &[DVector<f64>] {
    type SensorIter<'b>
        = SingleSensorIter<'b>
    where
        Self: 'b;
    type MeasIter<'b>
        = Iter<'b, DVector<f64>>
    where
        Self: 'b;

    #[inline]
    fn num_sensors(&self) -> usize {
        1
    }

    #[inline]
    fn sensors(&self) -> Self::SensorIter<'_> {
        SingleSensorIter {
            inner: std::iter::once((*self).iter()),
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        <[DVector<f64>]>::is_empty(self)
    }

    #[inline]
    fn has_any_measurements(&self) -> bool {
        !<[DVector<f64>]>::is_empty(self)
    }

    #[inline]
    fn total_measurements(&self) -> usize {
        self.len()
    }

    #[inline]
    fn measurements_for_sensor(&self, sensor_idx: usize) -> Option<usize> {
        if sensor_idx == 0 {
            Some(self.len())
        } else {
            None
        }
    }
}

// &[Vec<DVector<f64>>] implements MeasurementSource as multi-sensor
impl MeasurementSource for &[Vec<DVector<f64>>] {
    type SensorIter<'b>
        = VecOfVecsIter<'b>
    where
        Self: 'b;
    type MeasIter<'b>
        = Iter<'b, DVector<f64>>
    where
        Self: 'b;

    #[inline]
    fn num_sensors(&self) -> usize {
        self.len()
    }

    #[inline]
    #[allow(clippy::ptr_arg)]
    fn sensors(&self) -> Self::SensorIter<'_> {
        fn as_iter(v: &Vec<DVector<f64>>) -> Iter<'_, DVector<f64>> {
            v.iter()
        }
        VecOfVecsIter {
            inner: (*self)
                .iter()
                .map(as_iter as fn(&Vec<DVector<f64>>) -> Iter<'_, DVector<f64>>),
        }
    }
}

// &[&[DVector<f64>]] implements MeasurementSource as multi-sensor
impl<'a> MeasurementSource for &'a [&'a [DVector<f64>]] {
    type SensorIter<'b>
        = SliceOfSlicesIter<'b>
    where
        Self: 'b;
    type MeasIter<'b>
        = Iter<'b, DVector<f64>>
    where
        Self: 'b;

    #[inline]
    fn num_sensors(&self) -> usize {
        self.len()
    }

    #[inline]
    fn sensors(&self) -> Self::SensorIter<'_> {
        fn as_iter(slice: &[DVector<f64>]) -> Iter<'_, DVector<f64>> {
            slice.iter()
        }
        SliceOfSlicesIter {
            inner: (*self)
                .iter()
                .copied()
                .map(as_iter as fn(&[DVector<f64>]) -> Iter<'_, DVector<f64>>),
        }
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::dvector;

    fn make_measurement(x: f64, y: f64) -> DVector<f64> {
        dvector![x, y]
    }

    #[test]
    fn test_single_sensor_measurements() {
        let measurements = vec![make_measurement(1.0, 2.0), make_measurement(3.0, 4.0)];
        let source = SingleSensorMeasurements::new(&measurements);

        assert_eq!(source.num_sensors(), 1);
        assert!(!source.is_empty());
        assert!(source.has_any_measurements());
        assert_eq!(source.total_measurements(), 2);
        assert_eq!(source.measurements_for_sensor(0), Some(2));
        assert_eq!(source.measurements_for_sensor(1), None);

        // Test iteration
        let collected: Vec<Vec<&DVector<f64>>> = source
            .sensors()
            .map(|meas_iter| meas_iter.collect())
            .collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].len(), 2);
        assert_eq!(collected[0][0][0], 1.0);
        assert_eq!(collected[0][1][0], 3.0);
    }

    #[test]
    fn test_single_sensor_empty() {
        let measurements: Vec<DVector<f64>> = vec![];
        let source = SingleSensorMeasurements::new(&measurements);

        assert_eq!(source.num_sensors(), 1);
        assert!(source.is_empty());
        assert!(!source.has_any_measurements());
        assert_eq!(source.total_measurements(), 0);
    }

    #[test]
    fn test_vec_of_vecs_measurements() {
        let sensor_0 = vec![make_measurement(1.0, 2.0), make_measurement(3.0, 4.0)];
        let sensor_1 = vec![make_measurement(5.0, 6.0)];
        let sensors = vec![sensor_0, sensor_1];
        let source = VecOfVecsMeasurements::new(&sensors);

        assert_eq!(source.num_sensors(), 2);
        assert!(!source.is_empty());
        assert!(source.has_any_measurements());
        assert_eq!(source.total_measurements(), 3);

        // Test iteration
        let collected: Vec<Vec<&DVector<f64>>> = source
            .sensors()
            .map(|meas_iter| meas_iter.collect())
            .collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].len(), 2);
        assert_eq!(collected[1].len(), 1);
        assert_eq!(collected[0][0][0], 1.0);
        assert_eq!(collected[1][0][0], 5.0);
    }

    #[test]
    fn test_vec_of_vecs_with_empty_sensor() {
        let sensor_0 = vec![make_measurement(1.0, 2.0)];
        let sensor_1: Vec<DVector<f64>> = vec![];
        let sensor_2 = vec![make_measurement(3.0, 4.0)];
        let sensors = vec![sensor_0, sensor_1, sensor_2];
        let source = VecOfVecsMeasurements::new(&sensors);

        assert_eq!(source.num_sensors(), 3);
        assert!(!source.is_empty()); // has_any_measurements
        assert!(source.has_any_measurements());
        assert_eq!(source.total_measurements(), 2);
    }

    #[test]
    fn test_slice_of_slices_measurements() {
        let sensor_0 = vec![make_measurement(1.0, 2.0)];
        let sensor_1 = vec![make_measurement(3.0, 4.0), make_measurement(5.0, 6.0)];
        let slices: Vec<&[DVector<f64>]> = vec![&sensor_0, &sensor_1];
        let source = SliceOfSlicesMeasurements::new(&slices);

        assert_eq!(source.num_sensors(), 2);
        assert!(!source.is_empty());
        assert_eq!(source.total_measurements(), 3);

        // Test iteration
        let collected: Vec<Vec<&DVector<f64>>> = source
            .sensors()
            .map(|meas_iter| meas_iter.collect())
            .collect();
        assert_eq!(collected.len(), 2);
        assert_eq!(collected[0].len(), 1);
        assert_eq!(collected[1].len(), 2);
    }

    #[test]
    fn test_direct_slice_impl() {
        let measurements = vec![make_measurement(1.0, 2.0), make_measurement(3.0, 4.0)];
        let source: &[DVector<f64>] = &measurements;

        assert_eq!(source.num_sensors(), 1);
        assert_eq!(source.total_measurements(), 2);

        let collected: Vec<Vec<&DVector<f64>>> = source
            .sensors()
            .map(|meas_iter| meas_iter.collect())
            .collect();
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].len(), 2);
    }

    #[test]
    fn test_direct_vec_of_vecs_impl() {
        let sensor_0 = vec![make_measurement(1.0, 2.0)];
        let sensor_1 = vec![make_measurement(3.0, 4.0)];
        let sensors = vec![sensor_0, sensor_1];
        let source: &[Vec<DVector<f64>>] = &sensors;

        assert_eq!(source.num_sensors(), 2);
        assert_eq!(source.total_measurements(), 2);
    }

    #[test]
    fn test_from_conversions() {
        let measurements = vec![make_measurement(1.0, 2.0)];

        // From slice
        let _: SingleSensorMeasurements = (&measurements[..]).into();
        // From &Vec
        let _: SingleSensorMeasurements = (&measurements).into();

        let sensors = vec![vec![make_measurement(1.0, 2.0)]];
        // From slice
        let _: VecOfVecsMeasurements = (&sensors[..]).into();
        // From &Vec
        let _: VecOfVecsMeasurements = (&sensors).into();
    }

    #[test]
    fn test_zero_copy_guarantee() {
        // Verify that measurements are not copied by checking pointer equality
        let measurements = vec![make_measurement(1.0, 2.0), make_measurement(3.0, 4.0)];
        let source = SingleSensorMeasurements::new(&measurements);

        let collected: Vec<&DVector<f64>> = source.sensors().next().unwrap().collect();

        // The references should point to the same memory as the original
        assert!(std::ptr::eq(collected[0], &measurements[0]));
        assert!(std::ptr::eq(collected[1], &measurements[1]));
    }
}
