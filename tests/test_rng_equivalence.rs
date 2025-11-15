/// Cross-language RNG equivalence test
///
/// This test verifies that SimpleRng produces identical output in both Rust and MATLAB/Octave.
/// To verify equivalence, run the corresponding MATLAB/Octave test script and compare outputs.
///
/// MATLAB/Octave verification script (testSimpleRng.m):
/// ```matlab
/// rng = SimpleRng(42);
/// fprintf('Testing SimpleRng cross-language equivalence\n');
/// fprintf('Seed: 42\n\n');
///
/// fprintf('First 10 next_u64() values:\n');
/// for i = 1:10
///     [rng, val] = rng.next_u64();
///     fprintf('  %d: %s\n', i-1, num2str(val, '%.0f'));
/// end
///
/// rng = SimpleRng(42);
/// fprintf('\nFirst 10 rand() values:\n');
/// for i = 1:10
///     [rng, val] = rng.rand();
///     fprintf('  %d: %.17e\n', i-1, val);
/// end
///
/// rng = SimpleRng(42);
/// fprintf('\nFirst 10 randn() values:\n');
/// for i = 1:10
///     [rng, val] = rng.randn();
///     fprintf('  %d: %.17e\n', i-1, val);
/// end
///
/// rng = SimpleRng(42);
/// fprintf('\nFirst 10 poissrnd(5.0) values:\n');
/// for i = 1:10
///     [rng, val] = rng.poissrnd(5.0);
///     fprintf('  %d: %d\n', i-1, val);
/// end
/// ```

use prak::common::rng::{Rng, SimpleRng};

#[test]
fn test_rng_next_u64_seed_42() {
    let mut rng = SimpleRng::new(42);

    println!("Testing SimpleRng cross-language equivalence");
    println!("Seed: 42\n");

    println!("First 10 next_u64() values:");
    for i in 0..10 {
        let val = rng.next_u64();
        println!("  {}: {}", i, val);
    }
}

#[test]
fn test_rng_rand_seed_42() {
    let mut rng = SimpleRng::new(42);

    println!("\nFirst 10 rand() values:");
    for i in 0..10 {
        let val = rng.rand();
        println!("  {}: {:.17e}", i, val);
    }
}

#[test]
fn test_rng_randn_seed_42() {
    let mut rng = SimpleRng::new(42);

    println!("\nFirst 10 randn() values:");
    for i in 0..10 {
        let val = rng.randn();
        println!("  {}: {:.17e}", i, val);
    }
}

#[test]
fn test_rng_poissrnd_seed_42() {
    let mut rng = SimpleRng::new(42);

    println!("\nFirst 10 poissrnd(5.0) values:");
    for i in 0..10 {
        let val = rng.poissrnd(5.0);
        println!("  {}: {}", i, val);
    }
}

#[test]
fn test_rng_multiple_seeds() {
    let seeds = vec![0, 1, 42, 12345, u32::MAX as u64, u64::MAX >> 1];

    println!("\nTesting multiple seeds:");
    for seed in seeds {
        let mut rng = SimpleRng::new(seed);
        let val1 = rng.next_u64();
        let val2 = rng.next_u64();
        let val3 = rng.next_u64();
        println!("Seed {}: first 3 values = {}, {}, {}", seed, val1, val2, val3);
    }
}

#[test]
fn test_rng_deterministic() {
    // Verify that same seed produces identical sequences
    let mut rng1 = SimpleRng::new(42);
    let mut rng2 = SimpleRng::new(42);

    for _ in 0..10000 {
        assert_eq!(rng1.next_u64(), rng2.next_u64(), "Same seed should produce identical sequences");
    }
}

#[test]
fn test_rng_seed_zero_handling() {
    // Seed 0 should be converted to 1
    let mut rng0 = SimpleRng::new(0);
    let mut rng1 = SimpleRng::new(1);

    for _ in 0..100 {
        assert_eq!(rng0.next_u64(), rng1.next_u64(), "Seed 0 should be equivalent to seed 1");
    }
}
