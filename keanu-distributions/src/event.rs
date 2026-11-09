//! Batches of independent events for the multivariate families whose gradients are written by hand over [`Real`]
//! numbers: the layout of a value `[batch..., event...]` and of each parameter `[batch'..., parameter event...]`
//! (its leading axes broadcasting to the value's batch), the total log density, the gradient summed back to each
//! input's shape, and Hessian-vector products from the same gradient code on [`Dual`] numbers.

use crate::error::{DistributionError, Result, invalid_parameter};
use crate::multivariate::sum_over_batch;
use crate::real::{Dual, Real};
use crate::rng::KeanuRng;
use keanu_tensor::{DoubleTensor, broadcast_shape};

/// One event's log density and its gradient with respect to the value and each parameter (flattened row-major,
/// each the length of its event).
pub(crate) struct EventGradient<S> {
    pub(crate) log_prob: S,
    pub(crate) x: Vec<S>,
    pub(crate) parameters: Vec<Vec<S>>,
}

/// A family of distributions over events of a fixed rank.
pub(crate) trait EventFamily {
    /// The rank of the value's event (1 for vectors, 2 for matrices).
    const EVENT_RANK: usize;

    /// An error unless the value's event shape is one the family is defined over (a square matrix, say).
    fn validate_event(&self, event: &[usize]) -> Result<()> {
        let _ = event;
        Ok(())
    }

    /// The event shape of each parameter given the value's event shape.
    fn parameter_events(&self, event: &[usize]) -> Vec<Vec<usize>>;

    /// None if the parameters are valid and the value is in the support, else the log density to report: NaN for
    /// invalid parameters, −∞ for a value outside the support.
    fn check(&self, event: &[usize], x: &[f64], parameters: &[&[f64]]) -> Option<f64>;

    /// The log density of one event and its gradient. Called only where [`Self::check`] passes, for the log
    /// density; the gradient is computed wherever it is asked for (and is NaN where it doesn't exist).
    fn gradient<S: Real>(&self, event: &[usize], x: &[S], parameters: &[&[S]]) -> EventGradient<S>;
}

/// A batch of events of a family with the given parameters.
pub(crate) struct Events<'p, F> {
    pub(crate) family: F,
    pub(crate) what: &'static str,
    pub(crate) parameters: Vec<(&'static str, &'p DoubleTensor)>,
}

/// The layout of a batch: its shape, the value's event shape and every parameter spread to the batch.
pub(crate) struct Layout {
    pub(crate) batch: Vec<usize>,
    pub(crate) count: usize,
    pub(crate) event: Vec<usize>,
    pub(crate) event_len: usize,
    /// Each parameter's values at every batch member, flattened [batch, event], and its event length.
    pub(crate) parameters: Vec<(Vec<f64>, usize)>,
    /// Each parameter's event shape.
    pub(crate) events: Vec<Vec<usize>>,
}

impl Layout {
    /// The parameters of batch member `i`.
    pub(crate) fn at(&self, i: usize) -> Vec<&[f64]> {
        self.parameters
            .iter()
            .map(|(values, e)| &values[i * e..(i + 1) * e])
            .collect()
    }

    /// The full shape of parameter `k`'s spread values, [batch..., event...].
    fn full_shape(&self, k: usize) -> Vec<usize> {
        self.batch.iter().chain(&self.events[k]).copied().collect()
    }
}

impl<F: EventFamily> Events<'_, F> {
    /// The layout of a value of `shape`. An error if the shape has fewer axes than the event, if a parameter's
    /// trailing axes are not its event shape, or if its leading axes don't broadcast to the value's batch.
    pub(crate) fn layout(&self, shape: &[usize]) -> Result<Layout> {
        let rank = F::EVENT_RANK;
        if shape.len() < rank {
            return Err(DistributionError::InvalidValue(format!(
                "a {} value needs at least {rank} axes, not shape {shape:?}",
                self.what
            )));
        }
        let (batch, event) = shape.split_at(shape.len() - rank);
        self.family.validate_event(event)?;
        let events = self.family.parameter_events(event);
        let mut spread = Vec::with_capacity(self.parameters.len());
        for (&(name, p), expected) in self.parameters.iter().zip(&events) {
            let fits = p.rank() >= expected.len()
                && p.shape()[p.rank() - expected.len()..] == expected[..];
            let p_batch = &p.shape()[..p.rank() - expected.len().min(p.rank())];
            if !fits || broadcast_shape(batch, p_batch).ok().as_deref() != Some(batch) {
                return invalid_parameter(format!(
                    "{name} of shape {:?} must have trailing axes {expected:?} and leading axes broadcasting to \
                     the batch shape {batch:?} of a {} value of shape {shape:?}",
                    p.shape(),
                    self.what
                ));
            }
            let full: Vec<usize> = batch.iter().chain(expected).copied().collect();
            spread.push((p.broadcast_to(&full)?.into_vec(), expected.iter().product()));
        }
        Ok(Layout {
            batch: batch.to_vec(),
            count: batch.iter().product(),
            event: event.to_vec(),
            event_len: event.iter().product(),
            parameters: spread,
            events,
        })
    }

    /// The total log density of every event in x, summed over the batch: NaN if any member's parameters are
    /// invalid, else −∞ if any value is outside the support.
    pub(crate) fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
        let layout = self.layout(x.shape())?;
        let mut total = 0.0;
        let mut impossible = false;
        for i in 0..layout.count {
            let value = &x.as_slice()[i * layout.event_len..(i + 1) * layout.event_len];
            let parameters = layout.at(i);
            match self.family.check(&layout.event, value, &parameters) {
                Some(v) if v.is_nan() => return Ok(f64::NAN),
                Some(_) => impossible = true,
                None if !impossible => {
                    total += self
                        .family
                        .gradient::<f64>(&layout.event, value, &parameters)
                        .log_prob;
                }
                None => {}
            }
        }
        Ok(if impossible { f64::NEG_INFINITY } else { total })
    }

    /// One log density per independent event, shaped like the batch.
    pub(crate) fn log_prob_terms(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
        let layout = self.layout(x.shape())?;
        let mut terms = Vec::with_capacity(layout.count);
        for i in 0..layout.count {
            let value = &x.as_slice()[i * layout.event_len..(i + 1) * layout.event_len];
            let parameters = layout.at(i);
            let term = self
                .family
                .check(&layout.event, value, &parameters)
                .unwrap_or_else(|| {
                    self.family
                        .gradient::<f64>(&layout.event, value, &parameters)
                        .log_prob
                });
            terms.push(term);
        }
        Ok(DoubleTensor::new(&layout.batch, terms)?)
    }

    /// The gradient with respect to x and each parameter, each shaped like it (the parameters' summed over the
    /// batch axes they broadcast along).
    pub(crate) fn d_log_prob(&self, x: &DoubleTensor) -> Result<(DoubleTensor, Vec<DoubleTensor>)> {
        let layout = self.layout(x.shape())?;
        let mut dx = Vec::with_capacity(x.len());
        let mut dp: Vec<Vec<f64>> = layout
            .parameters
            .iter()
            .map(|(v, _)| Vec::with_capacity(v.len()))
            .collect();
        for i in 0..layout.count {
            let value = &x.as_slice()[i * layout.event_len..(i + 1) * layout.event_len];
            let g = self.family.gradient(&layout.event, value, &layout.at(i));
            dx.extend(g.x);
            for (d, g) in dp.iter_mut().zip(g.parameters) {
                d.extend(g);
            }
        }
        let gradients = dp
            .into_iter()
            .enumerate()
            .map(|(k, d)| {
                sum_over_batch(
                    &DoubleTensor::new(&layout.full_shape(k), d)?,
                    self.parameters[k].1.shape(),
                )
            })
            .collect::<Result<_>>()?;
        Ok((DoubleTensor::new(x.shape(), dx)?, gradients))
    }

    /// Hessian-vector products along `b` directions given as tangents of x ([b, x shape]) and of each parameter
    /// ([b, parameter shape]), None for zero: each output shaped [b, input shape].
    pub(crate) fn hvp(
        &self,
        x: &DoubleTensor,
        x_tangent: Option<&DoubleTensor>,
        parameter_tangents: &[Option<&DoubleTensor>],
        b: usize,
    ) -> Result<(DoubleTensor, Vec<DoubleTensor>)> {
        let layout = self.layout(x.shape())?;
        if parameter_tangents.len() != self.parameters.len() {
            return invalid_parameter(format!(
                "a {} has {} parameters, not {} parameter tangents",
                self.what,
                self.parameters.len(),
                parameter_tangents.len()
            ));
        }
        let tx = tangent(x_tangent, b, x.shape(), x.shape(), "the value")?;
        let tp: Vec<Option<Vec<f64>>> = parameter_tangents
            .iter()
            .enumerate()
            .map(|(k, t)| {
                let (name, p) = self.parameters[k];
                tangent(*t, b, p.shape(), &layout.full_shape(k), name)
            })
            .collect::<Result<_>>()?;
        let (event_len, len) = (layout.event_len, x.len());
        let mut hx = vec![0.0; b * len];
        let mut hp: Vec<Vec<f64>> = layout
            .parameters
            .iter()
            .map(|(v, _)| vec![0.0; b * v.len()])
            .collect();
        for i in 0..layout.count {
            let value = &x.as_slice()[i * event_len..(i + 1) * event_len];
            let parameters = layout.at(i);
            for k in 0..b {
                let xs: Vec<Dual> = value
                    .iter()
                    .enumerate()
                    .map(|(t, &v)| {
                        Dual::new(
                            v,
                            tx.as_ref()
                                .map_or(0.0, |tx| tx[k * len + i * event_len + t]),
                        )
                    })
                    .collect();
                let ps: Vec<Vec<Dual>> = parameters
                    .iter()
                    .enumerate()
                    .map(|(q, values)| {
                        let (all, e) = (&layout.parameters[q].0, layout.parameters[q].1);
                        values
                            .iter()
                            .enumerate()
                            .map(|(j, &v)| {
                                Dual::new(
                                    v,
                                    tp[q].as_ref().map_or(0.0, |t| t[k * all.len() + i * e + j]),
                                )
                            })
                            .collect()
                    })
                    .collect();
                let refs: Vec<&[Dual]> = ps.iter().map(Vec::as_slice).collect();
                let g = self.family.gradient(&layout.event, &xs, &refs);
                for (t, d) in g.x.iter().enumerate() {
                    hx[k * len + i * event_len + t] = d.d;
                }
                for (q, d) in g.parameters.iter().enumerate() {
                    let (all, e) = (layout.parameters[q].0.len(), layout.parameters[q].1);
                    for (j, d) in d.iter().enumerate() {
                        hp[q][k * all + i * e + j] = d.d;
                    }
                }
            }
        }
        let mut x_shape = vec![b];
        x_shape.extend_from_slice(x.shape());
        let products = hp
            .into_iter()
            .enumerate()
            .map(|(q, h)| {
                let full = layout.full_shape(q);
                let target = self.parameters[q].1.shape();
                let per = h.len() / b.max(1);
                let mut out = Vec::new();
                for k in 0..b {
                    let one = DoubleTensor::new(&full, h[k * per..(k + 1) * per].to_vec())?;
                    out.extend_from_slice(sum_over_batch(&one, target)?.as_slice());
                }
                let mut shape = vec![b];
                shape.extend_from_slice(target);
                Ok(DoubleTensor::new(&shape, out)?)
            })
            .collect::<Result<_>>()?;
        Ok((DoubleTensor::new(&x_shape, hx)?, products))
    }

    /// Draws a value of `shape`, one event per batch member from `draw(event, parameters, out, rng)`.
    pub(crate) fn sample(
        &self,
        shape: &[usize],
        rng: &mut KeanuRng,
        mut draw: impl FnMut(&[usize], &[&[f64]], &mut [f64], &mut KeanuRng) -> Result<()>,
    ) -> Result<DoubleTensor> {
        let layout = self.layout(shape)?;
        let mut values = vec![0.0; layout.count * layout.event_len];
        for i in 0..layout.count {
            let out = &mut values[i * layout.event_len..(i + 1) * layout.event_len];
            draw(&layout.event, &layout.at(i), out, rng)?;
        }
        Ok(DoubleTensor::new(shape, values)?)
    }
}

/// A tangent of an input of shape `shape`, checked to be [b, shape...], broadcast to [b, full...] and flattened.
fn tangent(
    t: Option<&DoubleTensor>,
    b: usize,
    shape: &[usize],
    full: &[usize],
    what: &str,
) -> Result<Option<Vec<f64>>> {
    let Some(t) = t else {
        return Ok(None);
    };
    let mut expected = vec![b];
    expected.extend_from_slice(shape);
    if t.shape() != expected {
        return invalid_parameter(format!(
            "the tangent of {what} must have shape {expected:?}, not {:?}",
            t.shape()
        ));
    }
    let per: usize = shape.iter().product();
    let mut out = Vec::with_capacity(b * full.iter().product::<usize>());
    for k in 0..b {
        let one = DoubleTensor::new(shape, t.as_slice()[k * per..(k + 1) * per].to_vec())?;
        out.extend(one.broadcast_to(full)?.into_vec());
    }
    Ok(Some(out))
}

/// Defines a gradient struct `$name` with an `x` field and one per parameter, built from [`Events::d_log_prob`]'s
/// or [`Events::hvp`]'s parts.
macro_rules! event_gradient {
    ($(#[$doc:meta])* $name:ident { $($field:ident: $field_doc:expr),+ $(,)? }) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name {
            /// With respect to the value.
            pub x: DoubleTensor,
            $(#[doc = $field_doc] pub $field: DoubleTensor,)+
        }

        impl $name {
            fn from_parts((x, parameters): (DoubleTensor, Vec<DoubleTensor>)) -> Self {
                let mut parameters = parameters.into_iter();
                Self {
                    x,
                    $($field: parameters.next().unwrap_or_else(|| DoubleTensor::scalar(0.0)),)+
                }
            }
        }
    };
}
pub(crate) use event_gradient;

/// Defines a distribution of f64 parameters over f64 events from an [`EventFamily`] kernel: the struct, its
/// gradient struct, `new`, `log_prob`, `d_log_prob` and `hvp` (the sampler is written separately, with `events`).
macro_rules! event_distribution {
    (
        $(#[$doc:meta])*
        $name:ident, $gradient:ident, kernel = $kernel:expr, what = $what:literal,
        ($($(#[$pdoc:meta])* $param:ident),+ $(,)?)
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $name<'a> {
            $($(#[$pdoc])* pub $param: Param<'a>,)+
        }

        $crate::event::event_gradient!(
            #[doc = concat!(
                "Derivatives of [`", stringify!($name), "::log_prob`] (each shaped like its input), or its ",
                "Hessian-vector products along `b` directions (each shaped [b, input shape...])."
            )]
            $gradient { $($param: concat!("With respect to `", stringify!($param), "`.")),+ }
        );

        impl<'a> $name<'a> {
            /// Creates the distribution from its parameters.
            pub fn new($($param: impl IntoParam<'a, f64>),+) -> Self {
                Self { $($param: $param.into_param()),+ }
            }

            #[allow(clippy::redundant_closure_call)]
            fn events(&self) -> $crate::event::Events<'_, impl $crate::event::EventFamily> {
                $crate::event::Events {
                    family: $kernel,
                    what: $what,
                    parameters: vec![$((stringify!($param), &*self.$param)),+],
                }
            }

            /// The total log density of every event in x (summed over the batch): NaN where the parameters are
            /// invalid, −∞ for a value outside the support. An error if the shapes don't fit.
            pub fn log_prob(&self, x: &DoubleTensor) -> Result<f64> {
                self.events().log_prob(x)
            }

            /// One log density per independent event, shaped like the batch.
            pub fn log_prob_terms(&self, x: &DoubleTensor) -> Result<DoubleTensor> {
                self.events().log_prob_terms(x)
            }

            /// The derivatives of [`Self::log_prob`] with respect to x and every parameter, each summed to its
            /// input's shape (the support is not checked).
            pub fn d_log_prob(&self, x: &DoubleTensor) -> Result<$gradient> {
                Ok($gradient::from_parts(self.events().d_log_prob(x)?))
            }

            /// Hessian-vector products of [`Self::log_prob`] along `b` directions, given as tangents of x
            /// ([b, x shape]) and of each parameter (in the order of [`Self::new`], each [b, parameter shape]),
            /// None for zero: for each input u, Σᵥ ∂² log p / ∂u ∂v · v̇, shaped [b, input shape]. Matrix tangents
            /// of symmetric inputs are taken to be symmetric.
            ///
            /// # Errors
            ///
            /// An error if the number of parameter tangents is wrong or a tangent has the wrong shape.
            pub fn hvp(
                &self,
                x: &DoubleTensor,
                x_tangent: Option<&DoubleTensor>,
                parameter_tangents: &[Option<&DoubleTensor>],
                b: usize,
            ) -> Result<$gradient> {
                Ok($gradient::from_parts(self.events().hvp(x, x_tangent, parameter_tangents, b)?))
            }
        }
    };
}
pub(crate) use event_distribution;
