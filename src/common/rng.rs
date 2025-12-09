/// Random number generator trait for deterministic cross-language testing.
///
/// This trait provides a minimal interface for random number generation
/// that can be implemented identically in both Rust and MATLAB/Octave,
/// enabling 100% deterministic testing and exact numerical equivalence.
pub trait Rng {
    /// Generate the next uint64 value
    fn next_u64(&mut self) -> u64;

    /// Generate a random f64 in [0, 1)
    fn rand(&mut self) -> f64 {
        self.next_u64() as f64 / (u64::MAX as f64 + 1.0)
    }

    /// Generate a random f64 from standard normal distribution N(0, 1)
    /// Using Box-Muller transform
    fn randn(&mut self) -> f64 {
        let u1 = self.rand();
        let u2 = self.rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Generate a random sample from Poisson distribution
    /// Using Knuth algorithm
    fn poissrnd(&mut self, lambda: f64) -> usize {
        let l = (-lambda).exp();
        let mut k = 0;
        let mut p = 1.0;
        loop {
            p *= self.rand();
            if p <= l {
                break;
            }
            k += 1;
        }
        k
    }
}

/// Simple deterministic random number generator using Xorshift64.
///
/// This PRNG is:
/// - Minimal (~5 lines of bit operations)
/// - Fast (no lookup tables, no heavy math)
/// - Deterministic (identical output for same seed across Rust/MATLAB/Octave)
/// - Good enough quality for testing (passes basic randomness tests)
///
/// The implementation matches MATLAB/Octave's SimpleRng class exactly,
/// enabling bit-for-bit identical random sequences across languages.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Create a new SimpleRng with the given seed.
    /// If seed is 0, uses 1 instead to avoid degenerate state.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }
}

impl Rng for SimpleRng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

// Implement rand::RngCore to enable use with rand::Rng trait bound
impl rand::RngCore for SimpleRng {
    fn next_u32(&mut self) -> u32 {
        Rng::next_u64(self) as u32
    }

    fn next_u64(&mut self) -> u64 {
        Rng::next_u64(self)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        let mut i = 0;
        let len = dest.len();
        while i + 8 <= len {
            let bytes = Rng::next_u64(self).to_le_bytes();
            dest[i..i + 8].copy_from_slice(&bytes);
            i += 8;
        }
        if i < len {
            let bytes = Rng::next_u64(self).to_le_bytes();
            let remaining = len - i;
            dest[i..].copy_from_slice(&bytes[..remaining]);
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rng_seed_zero() {
        let mut rng = SimpleRng::new(0);
        // Should use state = 1 when seed is 0
        assert_eq!(rng.state, 1);
        let val = rng.next_u64();
        assert_ne!(val, 0); // Should produce non-zero output
    }

    #[test]
    fn test_simple_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        // Same seed should produce identical sequences
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_simple_rng_different_seeds() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(43);

        // Different seeds should produce different sequences
        let val1 = rng1.next_u64();
        let val2 = rng2.next_u64();
        assert_ne!(val1, val2);
    }

    #[test]
    fn test_rand_range() {
        let mut rng = SimpleRng::new(42);

        for _ in 0..100 {
            let val = rng.rand();
            assert!(val >= 0.0 && val < 1.0, "rand() should return [0, 1)");
        }
    }

    #[test]
    fn test_randn_distribution() {
        let mut rng = SimpleRng::new(42);
        let mut sum = 0.0;
        let n = 10000;

        for _ in 0..n {
            sum += rng.randn();
        }

        let mean = sum / n as f64;
        // Mean should be close to 0 for standard normal
        assert!(mean.abs() < 0.1, "randn() mean should be close to 0");
    }

    #[test]
    fn test_poissrnd() {
        let mut rng = SimpleRng::new(42);
        let lambda = 5.0;

        // Just verify it produces reasonable values
        for _ in 0..100 {
            let val = rng.poissrnd(lambda);
            assert!(val < 100, "poissrnd should produce reasonable values");
        }
    }

    #[test]
    fn test_cross_language_compatibility_setup() {
        // This test documents the expected first few values from SimpleRng(42)
        // to enable cross-language verification with MATLAB/Octave
        let mut rng = SimpleRng::new(42);

        // First 10 next_u64() values from seed 42
        let expected_u64 = vec![
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
            rng.next_u64(),
        ];

        // Document these for comparison with MATLAB/Octave
        println!("First 5 u64 values from SimpleRng(42):");
        for (i, val) in expected_u64.iter().enumerate() {
            println!("  {}: {}", i, val);
        }
    }
}
