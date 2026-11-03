//! Random number generation.

use keanu_tensor::{DoubleTensor, IntTensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// The random number generator used by Keanu's samplers: ChaCha8, so seeded runs reproduce on every platform.
///
/// There is no global generator; pass one explicitly. `Clone` it to give independent chains or particles a
/// copy of the same stream, or create several from different seeds.
#[derive(Debug, Clone)]
pub struct KeanuRng {
    rng: ChaCha8Rng,
    /// The second value from the last polar Box–Muller draw.
    spare_gaussian: Option<f64>,
}

impl KeanuRng {
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            spare_gaussian: None,
        }
    }

    /// Uniform on [0, 1), with 53 random bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform on (0, 1).
    pub fn next_f64_non_zero(&mut self) -> f64 {
        loop {
            let value = self.next_f64();
            if value != 0.0 {
                return value;
            }
        }
    }

    /// Uniform on [min, max).
    pub fn next_f64_between(&mut self, min: f64, max: f64) -> f64 {
        self.next_f64() * (max - min) + min
    }

    /// Standard normal, by the polar Box–Muller (Marsaglia) method.
    pub fn next_gaussian(&mut self) -> f64 {
        if let Some(spare) = self.spare_gaussian.take() {
            return spare;
        }
        loop {
            let u = 2.0 * self.next_f64() - 1.0;
            let v = 2.0 * self.next_f64() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let factor = (-2.0 * s.ln() / s).sqrt();
                self.spare_gaussian = Some(v * factor);
                return u * factor;
            }
        }
    }

    pub fn next_bool(&mut self) -> bool {
        self.rng.next_u64() >> 63 == 1
    }

    /// Uniform integer in [0, bound), without modulo bias. Panics if bound is 0.
    pub fn next_below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "bound must be positive");
        let zone = u64::MAX - (u64::MAX % bound + 1) % bound;
        loop {
            let value = self.rng.next_u64();
            if value <= zone {
                return value % bound;
            }
        }
    }

    /// A tensor of uniform [0, 1) values.
    pub fn uniform(&mut self, shape: &[usize]) -> DoubleTensor {
        let len = shape.iter().product();
        DoubleTensor::new(shape, (0..len).map(|_| self.next_f64()).collect())
            .expect("length matches shape")
    }

    /// A tensor of standard normal values.
    pub fn gaussian(&mut self, shape: &[usize]) -> DoubleTensor {
        let len = shape.iter().product();
        DoubleTensor::new(shape, (0..len).map(|_| self.next_gaussian()).collect())
            .expect("length matches shape")
    }

    /// A tensor of uniform integers in [0, bound).
    pub fn ints_below(&mut self, shape: &[usize], bound: u64) -> IntTensor {
        let len = shape.iter().product();
        IntTensor::new(
            shape,
            (0..len).map(|_| self.next_below(bound) as i64).collect(),
        )
        .expect("length matches shape")
    }
}
