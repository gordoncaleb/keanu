package io.improbable.keanu.benchmarks;

import io.improbable.keanu.Keanu;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.VariableReference;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.KeanuProbabilisticModel;
import io.improbable.keanu.network.KeanuProbabilisticModelWithGradient;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.util.io.ProtobufLoader;
import io.improbable.keanu.util.status.StatusBar;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

import static java.util.concurrent.TimeUnit.MILLISECONDS;

/**
 * Log prob, log prob gradient and Metropolis–Hastings on the golden models (golden-vectors/models), which the Rust
 * benchmarks (rust/keanu/benches/models.rs) run on the same files. Run from the repository root.
 */
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1000, timeUnit = MILLISECONDS)
@Measurement(iterations = 5, time = 1000, timeUnit = MILLISECONDS)
@Fork(3)
public class ModelBenchmark {

    @Param({"thermometers", "hierarchicalGaussian", "linearRegression", "gammaPoisson"})
    public String model;

    private KeanuProbabilisticModel probabilisticModel;
    private KeanuProbabilisticModelWithGradient gradientModel;

    @Setup
    public void setup() throws IOException {
        StatusBar.disable();
        BayesianNetwork network;
        try (InputStream input = new FileInputStream("golden-vectors/models/" + model + ".pb")) {
            network = new ProtobufLoader().loadNetwork(input);
        }
        probabilisticModel = new KeanuProbabilisticModel(network);
        gradientModel = new KeanuProbabilisticModelWithGradient(network);
    }

    @Benchmark
    public double logProb() {
        return probabilisticModel.logProb();
    }

    @Benchmark
    public Map<VariableReference, DoubleTensor> logProbGradient() {
        return gradientModel.logProbGradients();
    }

    @Benchmark
    public NetworkSamples metropolisHastings1000() {
        return Keanu.Sampling.MetropolisHastings.withDefaultConfig()
            .getPosteriorSamples(probabilisticModel, probabilisticModel.getLatentVariables(), 1000);
    }
}
