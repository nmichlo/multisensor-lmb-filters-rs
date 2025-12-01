//! Workspace buffers for multi-sensor LMBM likelihood computation
//!
//! Pre-allocated buffers to avoid repeated allocations in hot loops.

use nalgebra::{DMatrix, DVector};

/// Pre-allocated workspace buffers for likelihood computation
///
/// Reusing these buffers across iterations eliminates ~12.5K allocations per thread
/// in the parallel association matrix generation.
#[derive(Debug, Clone)]
pub struct LmbmLikelihoodWorkspace {
    /// Assignment flags per sensor (number_of_sensors)
    pub assignments: Vec<bool>,
    /// Stacked measurement vector (max: z_dimension * number_of_sensors)
    pub z: DVector<f64>,
    /// Stacked observation matrix (max: z_dim_total × x_dimension)
    pub c: DMatrix<f64>,
    /// Block diagonal noise covariance (max: z_dim_total × z_dim_total)
    pub q: DMatrix<f64>,
    /// Innovation vector (max: z_dim_total)
    pub nu: DVector<f64>,
    /// Innovation covariance (max: z_dim_total × z_dim_total)
    pub z_matrix: DMatrix<f64>,
    /// Inverse of innovation covariance (max: z_dim_total × z_dim_total)
    pub z_inv: DMatrix<f64>,
    /// Kalman gain (x_dimension × z_dim_total)
    pub k: DMatrix<f64>,
    /// Posterior mean (x_dimension)
    pub mu_posterior: DVector<f64>,
    /// Posterior covariance (x_dimension × x_dimension)
    pub sigma_posterior: DMatrix<f64>,
    /// Identity matrix for Kalman update (x_dimension × x_dimension)
    pub identity: DMatrix<f64>,
    /// Temporary for C * Sigma (z_dim_total × x_dimension)
    pub temp_c_sigma: DMatrix<f64>,
    /// Q block references indices - which sensors detected
    pub q_block_indices: Vec<usize>,
    /// Maximum z dimension (for bounds checking)
    max_z_dim: usize,
}

impl LmbmLikelihoodWorkspace {
    /// Create a new workspace with pre-allocated buffers
    ///
    /// # Arguments
    /// * `number_of_sensors` - Number of sensors
    /// * `x_dimension` - State dimension
    /// * `z_dimension` - Single-sensor measurement dimension
    pub fn new(number_of_sensors: usize, x_dimension: usize, z_dimension: usize) -> Self {
        let max_z_dim = z_dimension * number_of_sensors;
        Self {
            assignments: vec![false; number_of_sensors],
            z: DVector::zeros(max_z_dim),
            c: DMatrix::zeros(max_z_dim, x_dimension),
            q: DMatrix::zeros(max_z_dim, max_z_dim),
            nu: DVector::zeros(max_z_dim),
            z_matrix: DMatrix::zeros(max_z_dim, max_z_dim),
            z_inv: DMatrix::zeros(max_z_dim, max_z_dim),
            k: DMatrix::zeros(x_dimension, max_z_dim),
            mu_posterior: DVector::zeros(x_dimension),
            sigma_posterior: DMatrix::zeros(x_dimension, x_dimension),
            identity: DMatrix::identity(x_dimension, x_dimension),
            temp_c_sigma: DMatrix::zeros(max_z_dim, x_dimension),
            q_block_indices: Vec::with_capacity(number_of_sensors),
            max_z_dim,
        }
    }

    /// Reset workspace for reuse with a new association
    ///
    /// Zeros out matrices that will be partially filled
    #[inline]
    pub fn reset(&mut self, z_dim_total: usize) {
        // Only zero the portion that will be used
        if z_dim_total > 0 {
            self.z.rows_mut(0, z_dim_total).fill(0.0);
            self.c.view_mut((0, 0), (z_dim_total, self.c.ncols())).fill(0.0);
            self.q.view_mut((0, 0), (z_dim_total, z_dim_total)).fill(0.0);
        }
        self.q_block_indices.clear();
    }

    /// Get the maximum z dimension this workspace can handle
    #[inline]
    pub fn max_z_dim(&self) -> usize {
        self.max_z_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_creation() {
        let ws = LmbmLikelihoodWorkspace::new(3, 4, 2);

        // max_z_dim = 2 * 3 = 6
        assert_eq!(ws.max_z_dim(), 6);
        assert_eq!(ws.assignments.len(), 3);
        assert_eq!(ws.z.len(), 6);
        assert_eq!(ws.c.nrows(), 6);
        assert_eq!(ws.c.ncols(), 4);
        assert_eq!(ws.q.nrows(), 6);
        assert_eq!(ws.q.ncols(), 6);
        assert_eq!(ws.identity.nrows(), 4);
    }

    #[test]
    fn test_workspace_reset() {
        let mut ws = LmbmLikelihoodWorkspace::new(3, 4, 2);

        // Modify some values
        ws.z[0] = 1.0;
        ws.c[(0, 0)] = 2.0;
        ws.q_block_indices.push(0);

        // Reset for z_dim_total = 4
        ws.reset(4);

        assert_eq!(ws.z[0], 0.0);
        assert_eq!(ws.c[(0, 0)], 0.0);
        assert!(ws.q_block_indices.is_empty());
    }
}
