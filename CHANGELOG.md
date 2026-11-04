# Changelog

All notable changes to Keanu. The Rust rewrite replaces the Java library; entries below "Rust rewrite" describe
the Java releases, kept for history.

## Unreleased: Rust rewrite

Keanu is now a pure Rust library: a Cargo workspace of the `keanu` facade crate and the `keanu-tensor`,
`keanu-distributions`, `keanu-graph`, `keanu-algorithms` and `keanu-io` crates. The Java implementation, its Gradle
build, the examples, benchmarks and documentation site were removed; the last Java version is at the `java-final`
tag. `RUST_MIGRATION_PLAN.md` describes the rewrite and `book/src/from-java.md` the differences for Java users.

- Same model semantics as the final Java version, checked against frozen golden vectors recorded from it
  (`keanu-parity/fixtures/golden-vectors`). Models saved by the final Java version load in Rust, and models saved by
  Rust use the same protobuf and JSON formats.
- Immutable, thread-safe graphs with a separate, cloneable `State`, so chains and particles can run in parallel.
- No global state and no native dependencies. BOBYQA is a translation of PRIMA's (BSD-3-Clause); everything else
  is MIT.
- The "Unreleased: Java" changes below were made to the Java implementation during the rewrite, mostly bug fixes
  the Rust port found, so that the Rust version is a faithful conversion.

### Code review fixes

Fixes from the 2026-11-04 whole-codebase review. Where a fix changes a result recorded from Java, the golden
vector was edited by hand and the entry says so.

- keanu-graph: using a handle from one `GraphBuilder` with another now fails `build()` with the new
  `GraphError::WrongBuilder`; it used to silently wire the other builder's node with the same id (or panic).
- keanu-graph: auto diff now passes gradients through `set_with_mask` (the operand's where the mask isn't 1, the set
  value's where it is), `replace_nan` (where the input wasn't NaN) and `multiplexer` (the selected option's); they
  were treated as non-differentiable, silently dropping those paths.
- keanu-graph: a failing `add_distribution` (or other untyped builder call) no longer leaves a placeholder node
  behind, and returns its error even when an earlier deferred error is pending, which still fails `build()`.
- keanu-graph: `add_node` checks each op's number of parents and returns an error instead of panicking (e.g.
  `Op::Proxy` or `Op::Not` with no parents) or silently accepting extra parents.
- keanu-graph: building a graph's child lists is linear in fan-out (it was quadratic).
- keanu-graph: auto diff of `product`/`product_axes`, `cumsum`, `cumprod` and `tensor_multiply` uses closed-form
  forward and reverse rules (exclusive products, reverse scans, contraction with the other operand) instead of a
  dense Jacobian built by evaluating the op twice per input element; still exact where elements are zero. Reverse
  gradients of a 2000-element cumsum, product or cumprod drop from 12-40 ms to ~20-30 µs, and of a 30×30
  `tensor_multiply` from 31 ms to 90 µs.
- `Multinomial::log_prob` validates x: a count vector with a negative count or counts not summing to n has log probability -infinity, and an x whose last dimension is not the number of categories is an error.
- `Logistic` log probability and gradients use a stable softplus and sigmoid, so they no longer overflow to -infinity / NaN far in the tails.
- `MultivariateGaussian::d_log_prob` reduces each gradient to its parameter's own shape, summing size-1 broadcast dimensions as well as extra leading ones.
- Samplers whose requested shape is smaller than the parameters' broadcast shape (Gaussian, HalfGaussian, LogNormal, Cauchy, HalfCauchy, Beta, Exponential, Logistic, Pareto, StudentT, Triangular, Uniform, Bernoulli, Geometric, UniformInt, Categorical) draw one value per broadcast element instead of reusing draws. Sampling at the full broadcast shape uses the same random stream as before.
- `Kde` log probability and gradient use log-sum-exp, so they stay finite far from the samples. `Kde::new` rejects a bandwidth that is not positive and finite, and `Kde::with_scotts_bandwidth` returns an error for samples with no spread.
- `Dirichlet::log_prob` takes (c - 1) ln x as 0 where c = 1 (no NaN at x = 0) and is -infinity for negative x.
- `Poisson::log_prob` at mu = 0, k = 0 is 0 rather than NaN.
- The `Binomial` gradient takes 0 / p as 0, so it is finite at p = 0, k = 0 and p = 1, k = n. Golden vector `Binomial/p=0.0,n=4` dLogProb/P at k = 0 edited by hand from NaN to -4.
- `Pareto` and `Geometric` log probabilities treat invalid parameters per element (and return the broadcast shape) instead of making every element -infinity.
- `Pareto` support includes x = location. Golden vectors `Pareto/location=1.0,scale=2.0` (x = 1) and `Pareto/location=0.5,scale=1.0` (x = 0.5) logProb edited by hand from -infinity to ln 2.
- Multinomial and Categorical sampling never return a category with zero probability when rounding leaves the cumulative probability below the uniform draw; they return the last category with non-zero probability.
- keanu-io: a `SliceVertex` whose `dimension` is not a dimension of its input is an `InvalidParameter` error
  instead of a capacity overflow panic.
- keanu-io: shapes whose element count overflows (tensor shapes, vertex shapes, `BroadcastVertex` `toShape`) are
  `Decode` errors instead of panicking (debug) or wrapping (release).
- keanu-io: an unlabelled proxy is saved with an empty label instead of an invented `proxy_<n>`, which could collide
  with another node's label (failing to load) and persisted after loading; invented labels in older files are
  ignored (proxies are labelled from the vertex label, which Java always wrote equal to the parameter).
- keanu-io: saving and loading repeatedly no longer grows the graph: loading folds back the zero location constant
  of half Gaussian / half Cauchy vertices (including Java's) and the subgraph saved for a Rust cholesky inverse.
- keanu-io: duplicate or missing vertex ids are `Decode` errors instead of being accepted silently.
- keanu-io: DOT node labels escape `"`, `\` and line breaks.
- keanu-io: JSON serialisation failures in `save_json` are the new `IoError::Encode`, not `Decode`.
- keanu-io: generic (object) stored values are the new `IoError::UnsupportedValue`, not `UnsupportedVertex`.
- keanu-io: values stored by label are matched with a label map built once, instead of a scan of every node each.
- keanu-io: `prost` is only a dependency with the `protobuf` or `json` feature; keanu-parity drops its unused
  `serde` dependency and inherits the workspace package fields.
- keanu-io: integer parameters loaded as axes, offsets or reshape dimensions are converted with one checked
  conversion (a `Decode` error where they don't fit an `isize`), and a `TriLowerVertex` `k` of `i64::MIN` is a
  `Decode` error instead of overflowing when negated (a panic in debug builds).
- keanu-io: saving borrows node labels, shapes, parents and constant / KDE data instead of copying them, and maps
  nodes to saved vertices with a vector instead of a hash map; DOT export names a distribution without formatting
  its data (it formatted a KDE's whole sample array to get the name "Kde").
- keanu-tensor: slice strings allow spaces inside items (`"1 : 3"`) and a trailing comma (`"1,"`), as numpy does.
- keanu-tensor: the owned scalar operators (`tensor + 1.0`, `-tensor`, ...) update the tensor's buffer in place
  instead of allocating a new one (four chained ops on 100k elements: 318 → 123 µs); integer `div`, `rem` and
  `pow` take the same-shape fast path and stop calling the element function after the first error (100k-element
  division: 433 → 129 µs; a broadcast remainder 437 → 89 µs); `tensor_multiply` no longer copies its permuted
  operands again to reshape them (20×30×40 · 40×30×10: 49 → 40 µs), and small matrix products zero their output
  once.
- `keanu-algorithms`: the particle filter treats a NaN log probability as impossible (−∞) when weighting, ranking
  and resampling; one NaN particle used to make every resampling draw pick the last particle.
- `keanu-algorithms`: `Samples::variance` uses a two-pass computation (still with an n − 1 denominator) instead of
  E[x²] − E[x]², which lost all precision for a large mean (samples of N(1e9, 1) had variance −896).
- `keanu-algorithms`: NUTS rejects a potential whose `initial_weight` or `initial_variance` is not finite and
  greater than 0 (`InvalidSetting`); a zero gave a zero mass matrix entry and a chain that never moved.
- `keanu-algorithms`: with `calculate_log_prob`, `ForwardSampler` recomputes the deterministic nodes downstream of
  the resampled latents before computing the log probability, which could otherwise use stale values. The docs of
  `ForwardSampler::new` no longer claim it records log probabilities by default.
- `keanu-algorithms`: `NetworkSamples::record` checks every value's type before recording any, so a mismatch no
  longer leaves the columns with different lengths.
- `keanu-algorithms`: simulated annealing rejects a schedule with zero iterations or a non-finite or non-positive
  start or end temperature (`InvalidSetting`, from the new `AnnealingSchedule::validate`).
- `keanu-algorithms`: conjugate gradient stops, without converging and at the best point found, when the fitness
  is impossible or the gradient NaN or infinite; it used to loop forever (with unlimited evaluations) and leave the
  latents at NaN. The initial-condition check now reports a NaN or infinite starting gradient as the new
  `AlgorithmError::NonFiniteGradient` (it used to pass, as `f64::max` skips NaN).
- `keanu-algorithms`: conjugate gradient keeps to `max_evaluations` inside its line search too
  (`with_max_evaluations(5)` used 19 evaluations).
- `keanu-algorithms`: BOBYQA no longer reports `converged: true` for an optimum with impossible (non-finite)
  fitness.
- `keanu-algorithms`: BOBYQA bound cleaning matches PRIMA: only a NaN or very low lower bound (or a NaN or very
  high upper bound) means "no bound", so a lower bound of +∞ or an upper bound of −∞ is now an error like any other
  lower bound above its upper bound. `NonGradientOptimizer` rejects NaN bounds (`InvalidArgument`).
- `keanu-algorithms`: BOBYQA no longer overflows checking a huge `interpolation_points`; it falls back to the
  default, as for any other out-of-range value.
- keanu-tensor: `digamma`'s asymptotic series had the wrong sign on its 1/(12x²) term (inherited from Apache Commons), so results were off by up to ~3e-9 (ψ(1) = -0.5772156677920679). Golden vectors edited by hand to mpmath values: `unary/digamma` and `unary/digamma/scalar` (tensors.json), `unary/digamma` and the `unary/logGamma` partials (vertices.json), the catalog model's `digamma` node (models.json), and the ψ-dependent gradients of `Beta/alpha=0.5,beta=0.5`, `Dirichlet/concentration=[0.5, 2.0, 3.0]` (both cases), the three `Gamma` cases and the two `InverseGamma` cases (distributions.json). The Rust-written `keanu-io` fixtures were regenerated, and the custom distribution example test uses seed 9, as its NUTS chain changed with the Beta gradient.
- keanu-tensor: `trigamma`'s asymptotic series subtracted its 1/(42z⁷) term instead of adding it.
- keanu-tensor: `digamma` and `trigamma` of negative arguments use the reflection formulas, so they no longer loop |x| times (or for ever from about -1e16). Non-positive integers are still poles (-∞ and +∞).
- keanu-tensor: shape lengths are computed with checked arithmetic. `Tensor::new` returns `LengthMismatch`, `reshape` returns `InvalidReshape`, and broadcasting and `matmul`/`tensor_multiply` return `IncompatibleShapes` when the length overflows `usize`, instead of panicking or wrapping. A shape with a zero dimension has length 0 whatever its other dimensions. `full`, `zeros`, `ones` and `from_fn` stay infallible and panic on overflow, as `vec!` does.
- keanu-tensor: slicing with a huge step (e.g. `"0:10:9223372036854775807"`) no longer overflows; it takes one element.
- keanu-tensor: `argmax_axis`/`argmin_axis` along an empty axis return an error instead of index 0.
- keanu-tensor (breaking): `DoubleTensor::arange` and `DoubleTensor::eye` return `Result`. `arange` rejects non-finite bounds and ranges too long to allocate, and `eye` rejects n where n × n overflows, instead of panicking.
- keanu-tensor: `tensor_multiply` over zero-length contracted axes returns zeros (e.g. [2, 0]·[0, 3] gives a [2, 3] zero tensor, as `matmul` does) instead of failing to reshape.
- Internal refactor, no behaviour or file format change: keanu-graph describes each distribution once, in a table that generates its name, parameter types, sampling, log probability and gradient; keanu-io keeps one table of distribution and simple-operation Java classes and parameter names, shared by saving, loading and DOT export (with hashed / binary search lookups instead of a scan per vertex), and saving parses slices with keanu-tensor's parser, now public as `keanu_tensor::ops::shape_ops::{parse_slice, SliceItem}`.

## Unreleased: Java

#### Breaking changes
- Removed the ND4J tensor implementations and the ND4J dependency. Keanu's tensors are now pure JVM.
- Removed the OpenBLAS (JavaCPP) dependency. Matrix multiply, LU, inverse, determinant and Cholesky are implemented in pure Java (`KeanuLapack`).
- Removed the Python package (`keanu-python`), the Python code generator (`codegen`), the Py4j byte array converter and the `@ExportVertexToPythonBindings` annotation.
- `ExponentialVertex` now treats its parameter as a rate, as documented (pdf = rate * exp(-rate * x)). Previously it was treated as the mean (scale). To keep old models unchanged pass `1 / oldValue`.
- Element-wise floating point equality now uses IEEE 754 semantics everywhere (`NaN != NaN`, `0.0 == -0.0`).
- `argMax`/`argMin` on floating point tensors now return the index of the first NaN, as numpy does. Use `nanArgMax`/`nanArgMin` to ignore NaNs.
- Integer and long `pow` with a negative exponent now truncates `1 / base^|exponent|` (0 raised to a negative power throws), consistently across all code paths.
- `reshape` with `-1` now throws if the length doesn't divide evenly.
- Slicing now follows numpy for negative steps and steps greater than 1 (e.g. `"::-1"` reverses, `"3:3:2"` is empty).
- The build requires Gradle 9 (wrapper included) and compiles with a JDK 17 toolchain.

#### Bug fixes
- `RegressionModel` treated rank 1 training data as a single sample with many features; a vector is now samples
  of one feature (and is reshaped the same way for `predict`).
- The Laplace distribution's gradient with respect to x and mu was NaN at the mode (0 / 0); it is now the
  subgradient 0.
- Gamma sampling (and Beta, Dirichlet, ChiSquared and StudentT, which use it) was biased when the shape k < 1 because the wrong constant (Euler-Mascheroni instead of e) was used.
- Multinomial gradient with respect to p was wrong.
- UniformInt sampling was biased, and could return the exclusive upper bound, for ranges including negative numbers.
- Triangular logProb was -infinity at the mode and NaN when the mode equalled xMin or xMax. Triangular now also has a gradient.
- Poisson sampling did not broadcast mu correctly. Binomial and Poisson logProb were NaN instead of -infinity outside of their support.
- MultivariateGaussian logProbGraph used the wrong dimension when mu was batched, and compiled log prob graphs failed for non-scalar latents of most distributions.
- Metropolis Hastings with the prior proposal and several dependent variables proposed at once used the wrong Hastings correction.
- Particle filter: latents were sampled using another particle's parent values, resampling weights underflowed to zero, and building one failed on Java 9+.
- NUTS could not be reused for a second sampling run (all steps diverged). The first post-adaptation sample now uses the averaged step size.
- Adam's default convergence check never converged when the optimum was near zero, and the initial "flat gradient" check could reject steep negative gradients.
- `NetworkSamples.downSample` reported the wrong size.
- Forward mode auto diff with several requested vertices only returned the first one reached.
- Simulated annealing ignored the configured variable selector.
- Tensors: `atan2` with a scalar argument had its arguments swapped, `max`/`min` ignored infinities, `diagPart` was wrong for batches of non-square matrices, `matrixDeterminant` returned a single 0 for a whole batch with one singular matrix, `tensorMultiply` failed with negative dimensions and mutated its arguments, `logAddExp`/`logAddExp2` returned NaN for two matching infinities, `argMax`/`argMin` rejected negative axes.

- Gamma, ChiSquared, InverseGamma, LogNormal and Beta logProb returned NaN instead of -infinity outside of their support, and Gamma with k = 1 was NaN at x = 0.
- The Beta distribution's logProb and dLogProb ignored its xMin / xMax bounds (sampling honoured them). `BetaVertex` always uses [0, 1] so it is unaffected apart from the support fix.
- Loading a saved network (protobuf or JSON) dropped the values, including observations, of integer vertices.
- Networks containing a full reduction (`sum()`, `product()` or `mean()` over all dimensions) could not be saved.
- `RoundVertex` claimed to be differentiable and reported an identity gradient. floor, ceil, round and sign are now consistently non-differentiable, and asking for a partial derivative through a non-differentiable vertex throws an `IllegalArgumentException` instead of a `NullPointerException`.
- Unary `max()` / `min()` vertices reported their input's shape instead of a scalar, and had no auto diff. They now pass the gradient to the selected element (split equally between ties).
- Binary `max` / `min` vertices did not broadcast their operands, unlike the tensor operations.
- `logGamma(+infinity)` returned NaN; it now returns +infinity.
- Reverse mode autodiff lost one contribution when the same vertex was both operands of a binary op (e.g. `x.times(x)`), halving the gradient.
- Forward mode autodiff through a full `sum()` or `mean()` summed the wrong dimensions when the with-respect-to vertex's rank differed from the operand's, giving wrongly shaped partials.
- The gradient of `product()` was NaN where an element is zero; it is now exact.
- Forward mode through `where` (and so binary `max` / `min`), and both modes through a double CPT, reported no partial when only an unselected branch depended on the input; the partial is now zero.
- `Diffs.get` returned the gradient of the nearest smaller parameter name instead of failing when asked for a parameter that is not present.
- `Categorical.logProb` returned 0 (probability 1) for a value that is not one of the categories; it now returns -infinity.
- `Binomial`, `Gamma`, `Laplace` and `Poisson` sampling failed when the parameters had a larger shape than the requested sample shape; they now return the broadcast shape.
- The forward sampler rejected any model with two or more observed vertices, even when no random variable was upstream of them.
- Adam threw a `NullPointerException` when maximizing the likelihood of a model with a latent the likelihood doesn't depend on; that latent now has a zero gradient (as with conjugate gradient).
- Sequence labels used an identity hash code, so they changed every time a program ran. They now use a counter (`SequenceBuilder.resetSequenceIdentifiers()` restarts it).
- Networks could not be saved if any vertex had an array parameter: reshape, permute, broadcast, take, tensor multiply, sum / product / mean over given dimensions (`int[]` / `long[]`), or a string slice (`boolean[]`, now saved as a boolean tensor and converted back on load).
- `argMax()` / `argMin()` / `nanArgMax()` / `nanArgMin()` over every element, and string slices without an ellipsis, could not be saved (their null parameters were not nullable).
- `KDEVertex` failed to evaluate its log prob or gradient at its own (scalar) shape.
- `UniformIntVertex` threw when asked for its gradient, so a model containing one could not use gradient based algorithms (NUTS, gradient optimizers); its gradient contribution is now empty.
- The "No value returned from Save Function" error names the vertex type and parameter.

#### New
- Golden test vectors (`golden-vectors/`, generated by the `keanu-golden` module) recording the Java implementation's results as the parity reference for the Rust port. `./gradlew check` fails if they change.
- Auto diff for `cumSum`, `cumProd`, `tensorMultiply`, `choleskyDecomposition`, `choleskyInverse`, `standardize`, `standardDeviation`, `trigamma`, `logAddExp` and `logAddExp2`.

## Version 0.0.27

### Java

#### Breaking changes
- Moved `io.improbable.keanu.vertices.dbl` classes to `io.improbable.keanu.vertices.tensor.number.floating.dbl`
- Moved `io.improbable.keanu.vertices.int` classes to `io.improbable.keanu.vertices.tensor.number.fixed.int`
- Moved `io.improbable.keanu.vertices.bool` classes to `io.improbable.keanu.vertices.tensor.bool`
- renamed DoubleTensor `determinant()` method `matrixDeterminant()`
- renamed DoubleTensor `average()` to `mean()` and allowed it to be done on a given dimension

### Common

- Previously some operations were available on the tensor classes but not the vertex classes. All operations
on the tensor classes have been implemented with vertex operations now. A few operations have been added in order
to accommodate autodiff for the new vertex operations.

- All Tensor operations now have a corresponding Vertex. This brings the DoubleTensor, IntegerTensor and BooleanTensor
  in line with the DoubleVertex, IntegerVertex and BooleanVertex.   

- Along with this change, the vertices package has been changed:
io.improbable.keanu.vertices.* -> io.improbable.keanu.vertices.tensor.*

- There is now an experimental LongTensor implementation.
- There is now a native JVM implementation of the IntegerTensor. This will bring a significant performance increase 
to models with lots of small integer operations.

#### New operations

- sign
- strided slice
- trigamma
- triLower
- triUpper
- fillTriangular
- batch matrix multiply
- batch matrix inverse
- batch matrix determinant
- batch cholesky decomposition
- cholesky inverse (with batch)


## Version 0.0.26

### Common

* Made constructing the children set of a vertex more efficient by not constantly reconstructing the set.
* Upgrade to nd4j 1.0.0-beta4

### Java

* Fixed an error where a space was present in one of the dependencies, which broke some build systems fetching dependencies.

- Fixed issues with MultinomialVertex. The issues with it before were:
    - It was extremely strict on checking that p summed to 1.0
    - It disallowed scalar n. This is probably the most common use case.
    - It didn't support the usual batch sampling nor batch logProb (see MultinomialVertex.java for docs)

- MultinomialVertex Functional changes:
    - The shape of p expected k to be the far left dimension. It is now the far right in order to allow for broadcasting semantics.
    - n and p parameter validation was semi-controllable but is now completely toggleable with `vertex.setValidationEnabled(true);`

- Add more generic tensor slice that allows start stop and interval slicing.

- Change name on DoubleTensor compare methods by removing get (e.g. `getGreaterThanMask(...))` -> `greaterThanMask(...)`)

- BooleanTensor now uses boolean arrays instead of Nd4j doubles

- Added custom distribution example to examples

- Added to DoubleTensor operations inspired by numpy: 
    - cumSum
    - cumProd
    - product
    - logAddExp
    - logAddExp2
    - log10
    - log2
    - log1p
    - exp2
    - expM1
    - tanh
    - atanh
    - sinh
    - asinh
    - cosh
    - acosh

- Fixed bug in NUTS where NaN gradients do not cause a step to be divergent

### Python

* Faster tensor creation in Python


## Version 0.0.25

## Version 0.0.24

### Common

* Sequences save / load

## Version 0.0.23

### Common

* You can now build a `Sequence` using more than one factory, whose vertices may be dependent on another.
  * In Java: you can call `SequenceBuilder#withFactories` instead of `SequenceBuilder#withFactory`
  * In Python: the parameter has been renamed from `factory` to `factories` (BREAKING CHANGE). You can pass in one or a list of multiple factories. 

## Version 0.0.22

### Common

* Generalised `GaussianProposalDistribution` to take different sigmas for each latent.
  * Example in Python:
    * `MetropolisHastingsSampler(proposal_distribution='gaussian', latents=[v1, v2], proposal_distribution_sigma=[1., np.array([[1., 2.], [3., 4.]]))`
* `DoubleVertex#matrixMultiply` now performs dot product when given two vectors, and matrix-vector product when given a matrix and a vector.
* Autodiff for Slice Vertices has now been fixed (caused issues with NUTs when run on a graph containing Slices)
* `MultivariateGaussianVertex` now requires that mu is a vector instead of a matrix.
* `DoubleTensor.scalar` will throw an exception if the DoubleTensor is not actually a scalar. Please use .getValue(0) if you just
want the first element in a DoubleTensor.

### Python

* Improved performance of getting samples by coercing them into a `Tensor` in Java (instead of iterating through an `ArrayList`).

## Version 0.0.21 

### Common

* Using Buildkite for CI
* Can save dot files with disconnected vertices in BayesianNetwork. You can also just pass a list of vertices to DotSaver.
* Plates have been renamed
  * This was done primarily because we were using the term incorrectly, since Dynamic Bayes Nets (e.g. Hidden Markov) are not Plates.
  * `Plates` --> `Sequence`
  * `Plate` --> `SequenceItem`
  * `plate` package --> `template`
* `ForwardSampler`
  * Samples from the prior of a bayesian network. The network cannot have observations that are dependent on random variables.
  * It's accessible from the `Keanu` factory like so: `Keanu.Sampling.Forward.withDefaultConfig()`
  * There is a togglabe option to record the log prob of each sample. This is disabled by default for performance reasons.
  
### Python

* Expose `unobserve`
* Added `iter_all_vertices`
* Rename methods returning generators to `iter_*` from `get_*`
* Split the ND4J dependencies into their own pip package.

## Version 0.0.20

### Common

* Saving a network as a DOT file includes labels on constant vertices.

### Python

* Improved performance of getting samples by using byte streams.
* Added Python docstrings for sampling

## Version 0.0.19

* Added `get_vertex_by_label` to `BayesNet`
* Added optional param `label` for all vertices in Python (e.g. `Gaussian(0., 1., label="gaussian")`). Now you must label vertices you are sampling from, otherwise Keanu will throw an exception.
* Improved the structure of the Python sample dictionary
  * Scalar networks
    * If all sampled vertices are scalar then you are returned a dictionary keyed on vertex ID with the values containing a list of primitive values.
    * An example of a network of size 2 with three samples: `{"gaussian": [0., 1., 2.], "exponential": [0.5, 1.5, 3.5]}`
    * `samples["gaussian"] = [0., 1., 2.]`
  * Tensor networks
    * If any of the sampled vertices are tensors then you are returned a dictionary keyed on a tuple of vertex ID and tensor index. 
    * This makes it much easier to extract all the samples for a given index and for creating a multi-index dataframe
    * An example of a scalar exponential and a `2x2` tensor gaussian with three samples:
    ```
            exp  gaussian                               
            (0)    (0, 0)    (0, 1)    (1, 0)     (1, 1)
    0  4.231334  5.017627  5.734130  3.904472   9.909033
    1  4.231334  5.017627  5.734130  3.904472   9.909033
    2  4.676046  4.308018  5.035550  6.240894  10.112683
    ```
    * `samples[("gaussian", (0, 1))] = [5.734130, 5.734130, 5.035550] `
  * As a result, whenever keying into a sample dictionary, you are guaranteed to receive a list of primitives.
  * This greatly reduces the complexity of the `autocorrelation` and `traceplot` API's as they now simply expect a list of values.  
* Added Adam optimizer
* GradientOptimizer and NonGradientOptimizer now takes an algorithm argument that by default will use Conjugate Gradient and BOBYQA respectively.
* GradientOptimizer and NonGradientOptimizer return a OptimizedResult object instead of just the optimized fitness as a double
* Reorganised the factory methods for building `PosteriorSamplingAlgorithm` objects. The following are available and give you access to either a default implementation or, when you need more control over the configuration, a Builder object:
  * `Keanu.Sampling.MetropolisHastings`
  * `Keanu.Sampling.NUTS`
  * `Keanu.Sampling.SimulatedAnnealing`
  * `Keanu.Sampling.MCMC` (to automatically choose between `MetropolisHastings` and `NUTS`)
* The `PosteriorSamplingAlgorithm` objects are now immutable: you cannot set their values after construction.
* When you choose a custom configuration of `MetropolisHastings` or `SimulatedAnnealing`, you can specify:
  * the proposal distribution. Options are `PriorProposalDistribution` and `GaussianProposalDistribution`
  * a proposal rejection strategy - options are `RollBackToCachedValuesOnRejection` and `RollbackAndCascadeOnRejection`
    * (in Python, this is done for you)
* `Hamiltonian` Monte Carlo has been removed: use `NUTS` instead which is an auto-tuning implementation of Hamiltonian Monte Carlo.
* The `PosteriorSamplingAlgorithm` and `FitnessFunction` interfaces requires a `ProbabilisticModel` as its argument instead of a `BayesianNetwork`
  * You can create a `ProbabilisticModel` from a `BayesianNetwork`:
    * For `MetropolisHastings`: `new KeanuProbabilisticModel(bayesNet)`
    * For `NUTS`: `new KeanuProbabilisticModelWithGradient(bayesNet)`
* The `ProposalDistribution` interface uses `Variable` instead of `Vertex`
  * A `Variable` is an abstraction that does not assume any graph-like structure of your model.
* `KeanuRandom` has been moved to package `io.improbable.keanu`
* The `ProposalListener` interface has been changed: `onProposalApplied` is now called `onProposalCreated`.
  * This is because the `Proposal` interface no longer has `apply` and `reject` methods.
* Reversed the ordering of namespace for `VertexLabel` (e.g. from `new VertexLabel("keanu", "improbable", "io")` to `new VertexLabel("io", "improbable", "keanu"))
* Added better Python operator overloading support.
  - Includes automatically casting to the correct datatype where possible.
* Added Geometric Distribution vertex
* Fixed issue with certain vertices not taking a list properly in Kotlin
* Added `__version__` attribute to Python
* Added a permute vertex
* Added the MIRSaver/MIRLoader + the proposed MIR proto format
* Added the release notes text file to the repo
