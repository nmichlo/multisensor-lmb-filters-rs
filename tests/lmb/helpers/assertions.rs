//! Generic assertion functions for numerical comparisons with tolerance
//!
//! These functions eliminate duplicate comparison code across test files.

use nalgebra::{DMatrix, DVector};

/// Compare scalar values with tolerance
pub fn assert_scalar_close(actual: f64, expected: f64, tolerance: f64, field_name: &str) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= tolerance,
        "{}: expected {}, got {} (diff: {}, tolerance: {})",
        field_name,
        expected,
        actual,
        diff,
        tolerance
    );
}

/// Compare vector slices element-wise with tolerance
pub fn assert_vec_close(actual: &[f64], expected: &[f64], tolerance: f64, field_name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch (actual: {}, expected: {})",
        field_name,
        actual.len(),
        expected.len()
    );

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tolerance,
            "{}[{}]: expected {}, got {} (diff: {}, tolerance: {})",
            field_name,
            i,
            e,
            a,
            diff,
            tolerance
        );
    }
}

/// Compare DVector with tolerance
pub fn assert_dvector_close(
    actual: &DVector<f64>,
    expected: &DVector<f64>,
    tolerance: f64,
    field_name: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: dimension mismatch (actual: {}, expected: {})",
        field_name,
        actual.len(),
        expected.len()
    );

    for i in 0..actual.len() {
        let diff = (actual[i] - expected[i]).abs();
        assert!(
            diff <= tolerance,
            "{}[{}]: expected {}, got {} (diff: {}, tolerance: {})",
            field_name,
            i,
            expected[i],
            actual[i],
            diff,
            tolerance
        );
    }
}

/// Compare DMatrix with tolerance
pub fn assert_dmatrix_close(
    actual: &DMatrix<f64>,
    expected: &DMatrix<f64>,
    tolerance: f64,
    field_name: &str,
) {
    assert_eq!(
        actual.nrows(),
        expected.nrows(),
        "{}: row count mismatch (actual: {}, expected: {})",
        field_name,
        actual.nrows(),
        expected.nrows()
    );
    assert_eq!(
        actual.ncols(),
        expected.ncols(),
        "{}: column count mismatch (actual: {}, expected: {})",
        field_name,
        actual.ncols(),
        expected.ncols()
    );

    for i in 0..actual.nrows() {
        for j in 0..actual.ncols() {
            let diff = (actual[(i, j)] - expected[(i, j)]).abs();
            assert!(
                diff <= tolerance,
                "{}[{},{}]: expected {}, got {} (diff: {}, tolerance: {})",
                field_name,
                i,
                j,
                expected[(i, j)],
                actual[(i, j)],
                diff,
                tolerance
            );
        }
    }
}

/// Compare 2D nested Vec (MATLAB serialization format) with tolerance
pub fn assert_matrix_close(
    actual: &[Vec<f64>],
    expected: &[Vec<f64>],
    tolerance: f64,
    field_name: &str,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: row count mismatch (actual: {}, expected: {})",
        field_name,
        actual.len(),
        expected.len()
    );

    for (i, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual_row.len(),
            expected_row.len(),
            "{}[{}]: column count mismatch (actual: {}, expected: {})",
            field_name,
            i,
            actual_row.len(),
            expected_row.len()
        );

        for (j, (&a, &e)) in actual_row.iter().zip(expected_row.iter()).enumerate() {
            let diff = (a - e).abs();
            assert!(
                diff <= tolerance,
                "{}[{},{}]: expected {}, got {} (diff: {}, tolerance: {})",
                field_name,
                i,
                j,
                e,
                a,
                diff,
                tolerance
            );
        }
    }
}

/// Compare integer vectors with exact equality (TOLERANCE=0)
pub fn assert_ivec_exact(actual: &[i32], expected: &[i32], field_name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch (actual: {}, expected: {})",
        field_name,
        actual.len(),
        expected.len()
    );

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "{}[{}]: expected {}, got {}", field_name, i, e, a);
    }
}

/// Compare 2D integer matrices with exact equality (TOLERANCE=0)
pub fn assert_imatrix_exact(actual: &[Vec<i32>], expected: &[Vec<i32>], field_name: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: row count mismatch (actual: {}, expected: {})",
        field_name,
        actual.len(),
        expected.len()
    );

    for (i, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            actual_row.len(),
            expected_row.len(),
            "{}[{}]: column count mismatch (actual: {}, expected: {})",
            field_name,
            i,
            actual_row.len(),
            expected_row.len()
        );

        for (j, (&a, &e)) in actual_row.iter().zip(expected_row.iter()).enumerate() {
            assert_eq!(
                a, e,
                "{}[{},{}]: expected {}, got {}",
                field_name, i, j, e, a
            );
        }
    }
}
