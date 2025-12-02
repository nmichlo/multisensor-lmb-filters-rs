//! Error types for filters and components
//!
//! This module provides proper error handling instead of panics.

use std::fmt;

/// Errors that can occur during filtering
#[derive(Debug, Clone)]
pub enum FilterError {
    /// Matrix inversion failed (singular matrix)
    SingularMatrix {
        /// Description of which matrix failed
        context: String,
    },

    /// Dimension mismatch between expected and actual
    DimensionMismatch {
        /// What was expected
        expected: usize,
        /// What was received
        actual: usize,
        /// Context (e.g., "state dimension", "measurement dimension")
        context: String,
    },

    /// Data association failed
    Association(AssociationError),

    /// Numerical instability detected
    NumericalInstability {
        /// Description of the issue
        description: String,
    },

    /// Configuration error
    Configuration {
        /// Description of the configuration issue
        description: String,
    },

    /// No valid tracks/hypotheses remain
    NoValidTracks,
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterError::SingularMatrix { context } => {
                write!(f, "Matrix inversion failed: {}", context)
            }
            FilterError::DimensionMismatch {
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "Dimension mismatch for {}: expected {}, got {}",
                    context, expected, actual
                )
            }
            FilterError::Association(e) => write!(f, "Association failed: {}", e),
            FilterError::NumericalInstability { description } => {
                write!(f, "Numerical instability: {}", description)
            }
            FilterError::Configuration { description } => {
                write!(f, "Configuration error: {}", description)
            }
            FilterError::NoValidTracks => write!(f, "No valid tracks or hypotheses remain"),
        }
    }
}

impl std::error::Error for FilterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FilterError::Association(e) => Some(e),
            _ => None,
        }
    }
}

impl From<AssociationError> for FilterError {
    fn from(e: AssociationError) -> Self {
        FilterError::Association(e)
    }
}

/// Errors that can occur during data association
#[derive(Debug, Clone)]
pub enum AssociationError {
    /// LBP did not converge
    LbpNoConvergence {
        /// Number of iterations run
        iterations: usize,
        /// Final residual
        residual: f64,
    },

    /// Gibbs sampling failed
    GibbsFailed {
        /// Description of the failure
        description: String,
    },

    /// Murty's algorithm failed
    MurtyFailed {
        /// Description of the failure
        description: String,
    },

    /// No valid assignments found
    NoValidAssignments,

    /// Invalid cost matrix (e.g., all infinities)
    InvalidCostMatrix,
}

impl fmt::Display for AssociationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssociationError::LbpNoConvergence {
                iterations,
                residual,
            } => {
                write!(
                    f,
                    "LBP did not converge after {} iterations (residual: {:.2e})",
                    iterations, residual
                )
            }
            AssociationError::GibbsFailed { description } => {
                write!(f, "Gibbs sampling failed: {}", description)
            }
            AssociationError::MurtyFailed { description } => {
                write!(f, "Murty's algorithm failed: {}", description)
            }
            AssociationError::NoValidAssignments => write!(f, "No valid assignments found"),
            AssociationError::InvalidCostMatrix => write!(f, "Invalid cost matrix"),
        }
    }
}

impl std::error::Error for AssociationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_error_display() {
        let err = FilterError::SingularMatrix {
            context: "innovation covariance".to_string(),
        };
        assert!(err.to_string().contains("innovation covariance"));

        let err = FilterError::DimensionMismatch {
            expected: 4,
            actual: 6,
            context: "state".to_string(),
        };
        assert!(err.to_string().contains("4"));
        assert!(err.to_string().contains("6"));
    }

    #[test]
    fn test_association_error_display() {
        let err = AssociationError::LbpNoConvergence {
            iterations: 50,
            residual: 0.01,
        };
        assert!(err.to_string().contains("50"));

        let err = AssociationError::NoValidAssignments;
        assert!(err.to_string().contains("No valid assignments"));
    }

    #[test]
    fn test_error_conversion() {
        let assoc_err = AssociationError::NoValidAssignments;
        let filter_err: FilterError = assoc_err.into();
        assert!(matches!(filter_err, FilterError::Association(_)));
    }
}
