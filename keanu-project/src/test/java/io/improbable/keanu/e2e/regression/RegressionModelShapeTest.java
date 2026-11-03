package io.improbable.keanu.e2e.regression;

import io.improbable.keanu.model.regression.RegressionModel;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

/**
 * Rank 1 training data is a column of samples of a single feature, not a single sample of many features.
 */
public class RegressionModelShapeTest {

    @Test
    public void rankOneTrainingDataIsTreatedAsSamplesOfOneFeature() {
        double[] x = new double[]{0, 1, 2, 3, 4, 5, 6, 7};
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = 2.0 * x[i] + 1.0;
        }

        RegressionModel<DoubleTensor> model = RegressionModel
            .withTrainingData(DoubleTensor.create(x), DoubleTensor.create(y))
            .build();

        assertArrayEquals(new long[]{1, 1}, model.getWeightVertex().getShape());
        assertEquals(2.0, model.getWeightVertex().getValue().scalar(), 1e-3);
        assertEquals(1.0, model.getInterceptVertex().getValue().scalar(), 1e-3);

        DoubleTensor prediction = model.predict(DoubleTensor.create(10.0, 20.0));
        assertArrayEquals(new double[]{21.0, 41.0}, prediction.asFlatDoubleArray(), 1e-2);
    }
}
