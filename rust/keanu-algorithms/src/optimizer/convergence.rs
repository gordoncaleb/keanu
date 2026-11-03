use keanu_tensor::DoubleTensor;

/// How a vector of deltas is reduced to one number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Norm {
    /// The Euclidean norm over every element of every tensor.
    L2,
    /// The largest element over every tensor.
    Max,
}

impl Norm {
    fn of(self, deltas: &[DoubleTensor]) -> f64 {
        match self {
            Norm::L2 => deltas
                .iter()
                .map(|d| d.as_slice().iter().map(|x| x * x).sum::<f64>())
                .sum::<f64>()
                .sqrt(),
            Norm::Max => deltas
                .iter()
                .filter_map(|d| d.max())
                .fold(-f64::MAX, f64::max),
        }
    }
}

/// When an iterative optimizer has converged, from the change in its position between iterations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceChecker {
    /// The norm of |a - b| / (|a| + epsilon) is below the threshold.
    Relative {
        /// How the change is measured.
        norm: Norm,
        /// Converged when the change is below this.
        threshold: f64,
        /// Added to the denominator so values near zero don't blow up the relative change.
        epsilon: f64,
    },
    /// The norm of |a - b| is below the threshold.
    Absolute {
        /// How the change is measured.
        norm: Norm,
        /// Converged when the change is below this.
        threshold: f64,
    },
}

impl ConvergenceChecker {
    /// A relative checker with epsilon 1e-6.
    pub fn relative(norm: Norm, threshold: f64) -> Self {
        Self::Relative {
            norm,
            threshold,
            epsilon: 1e-6,
        }
    }

    /// A convergence checker on the absolute change.
    pub fn absolute(norm: Norm, threshold: f64) -> Self {
        Self::Absolute { norm, threshold }
    }

    /// Whether the move from position to next_position counts as converged.
    pub fn has_converged(&self, position: &[DoubleTensor], next: &[DoubleTensor]) -> bool {
        let deltas = position.iter().zip(next).map(|(a, b)| {
            a.zip_with(b, |a, b| match self {
                // epsilon only guards the denominator, so a position at zero can converge
                Self::Relative { epsilon, .. } => (a - b).abs() / (a.abs() + epsilon),
                Self::Absolute { .. } => (a - b).abs(),
            })
        });
        let deltas: Vec<DoubleTensor> = match deltas.collect() {
            Ok(d) => d,
            Err(_) => return false,
        };
        let (Self::Relative {
            norm, threshold, ..
        }
        | Self::Absolute { norm, threshold }) = *self;
        norm.of(&deltas) < threshold
    }
}

impl Default for ConvergenceChecker {
    /// Relative, L2 norm, threshold 1e-6 (Adam's default).
    fn default() -> Self {
        Self::relative(Norm::L2, 1e-6)
    }
}
