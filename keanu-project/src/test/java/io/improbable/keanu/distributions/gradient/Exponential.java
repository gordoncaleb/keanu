package io.improbable.keanu.distributions.gradient;

/**
 * Computer Generation of Statistical Distributions
 * by Richard Saucier
 * ARL-TR-2168 March 2000
 * 5.1.8 page 20
 */

public class Exponential {

    private Exponential() {
    }

    public static Diff dlnPdf(double lambda, double x) {
        // logPdf = ln(lambda) - lambda * x where lambda is the rate
        double dPdlambda = 1 / lambda - x;
        double dPdx = -lambda;
        return new Diff(dPdlambda, dPdx);
    }

    public static class Diff {
        public final double dPdlambda;
        public final double dPdx;

        public Diff(double dPdlambda, double dPdx) {
            this.dPdlambda = dPdlambda;
            this.dPdx = dPdx;
        }
    }
}
