use prak::common::types::{DMatrix, MatrixExt};
use prak::common::association::gibbs::{lmb_gibbs_frequency_sampling, GibbsAssociationMatrices};
use prak::common::rng::SimpleRng;

#[test]
fn test_gibbs_frequency_cross_language_equivalence() {
    // Test cross-language equivalence with Octave testGibbsFrequency.m
    // Simple 2 objects, 2 measurements matching Octave test
    let p = DMatrix::from_row_slice(
        2,
        2,
        &[0.7, 0.3, 0.4, 0.6],
    );

    let l = DMatrix::from_row_slice(
        2,
        3,
        &[0.05, 0.8, 0.15, 0.05, 0.3, 0.65],
    );

    let r = DMatrix::from_row_slice(
        2,
        3,
        &[0.05, 1.0, 1.0, 0.05, 1.0, 1.0],
    );

    let c = DMatrix::from_row_slice(
        2,
        2,
        &[1.0, 3.0, 3.0, 1.0],
    );

    let matrices = GibbsAssociationMatrices { p, l, r, c };

    // Use same seed as Octave
    let mut rng = SimpleRng::new(42);

    // Debug: manually run first 5 iterations to compare with Octave
    use prak::common::association::gibbs::initialize_gibbs_association_vectors;
    use prak::common::association::gibbs::generate_gibbs_sample;

    let (mut v_debug, mut w_debug) = initialize_gibbs_association_vectors(&matrices.c);
    println!("Initial v: [{}, {}]", v_debug[0], v_debug[1]);

    for i in 0..5 {
        print!("Iter {}: v = [{}, {}], ", i + 1, v_debug[0], v_debug[1]);

        // Tally (just print, don't actually update)
        println!("tallying to columns [{}, {}]", v_debug[0], v_debug[1]);

        // Generate new sample
        (v_debug, w_debug) = generate_gibbs_sample(&mut rng, &matrices.p, v_debug, w_debug);
    }
    println!();

    // Reset RNG for actual test
    let mut rng = SimpleRng::new(42);
    let result = lmb_gibbs_frequency_sampling(&mut rng, &matrices, 1000);

    // Expected values from Octave with SimpleRng(42) and 1000 samples:
    // r = [0.6922, 0.6399500]
    // W = [[0.023404, 0.869691, 0.106906],
    //      [0.029612, 0.165638, 0.804750]]

    println!("Cross-language equivalence test:");
    println!("Rust r: [{}, {}]", result.r[0], result.r[1]);
    println!(
        "Rust W: [[{}, {}, {}], [{}, {}, {}]]",
        result.w[(0, 0)],
        result.w[(0, 1)],
        result.w[(0, 2)],
        result.w[(1, 0)],
        result.w[(1, 1)],
        result.w[(1, 2)]
    );

    // Verify exact match (deterministic with SimpleRng)
    assert!(
        (result.r[0] - 0.6922).abs() < 1e-10,
        "r[0] mismatch: {}",
        result.r[0]
    );
    assert!(
        (result.r[1] - 0.6399500000).abs() < 1e-10,
        "r[1] mismatch: {}",
        result.r[1]
    );

    assert!(
        (result.w[(0, 0)] - 0.023404).abs() < 1e-6,
        "W[0,0] mismatch: {}",
        result.w[(0, 0)]
    );
    assert!(
        (result.w[(0, 1)] - 0.869691).abs() < 1e-6,
        "W[0,1] mismatch: {}",
        result.w[(0, 1)]
    );
    assert!(
        (result.w[(0, 2)] - 0.106906).abs() < 1e-6,
        "W[0,2] mismatch: {}",
        result.w[(0, 2)]
    );

    assert!(
        (result.w[(1, 0)] - 0.029612).abs() < 1e-6,
        "W[1,0] mismatch: {}",
        result.w[(1, 0)]
    );
    assert!(
        (result.w[(1, 1)] - 0.165638).abs() < 1e-6,
        "W[1,1] mismatch: {}",
        result.w[(1, 1)]
    );
    assert!(
        (result.w[(1, 2)] - 0.804750).abs() < 1e-6,
        "W[1,2] mismatch: {}",
        result.w[(1, 2)]
    );

    println!("✓ Cross-language equivalence verified!");
}
