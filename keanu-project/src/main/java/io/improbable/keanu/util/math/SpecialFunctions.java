package io.improbable.keanu.util.math;

public final class SpecialFunctions {

    private SpecialFunctions() {
    }

    /**
     * The tetragamma function, the second derivative of the digamma function (polygamma of order 2).
     * It is the derivative of the trigamma function.
     *
     * @param x point to evaluate at
     * @return psi''(x), NaN at the poles (non-positive integers)
     */
    public static double tetragamma(double x) {
        if (Double.isNaN(x) || (x <= 0 && x == Math.floor(x))) {
            return Double.NaN;
        }

        if (x < 0) {
            // reflection: psi''(1 - x) - psi''(x) = 2 pi^3 cot(pi x) / sin^2(pi x)
            final double sin = Math.sin(Math.PI * x);
            final double cot = Math.cos(Math.PI * x) / sin;
            return tetragamma(1 - x) - 2 * Math.pow(Math.PI, 3) * cot / (sin * sin);
        }

        // recurrence psi''(x) = psi''(x + 1) - 2 / x^3 until x is large enough for the asymptotic series
        double result = 0;
        double z = x;
        while (z < 10) {
            result -= 2 / (z * z * z);
            z += 1;
        }

        final double z2 = 1 / (z * z);
        final double series = -z2 * (1 + 1 / z + z2 * (0.5 + z2 * (-1.0 / 6 + z2 * (1.0 / 6 + z2 * (-3.0 / 10 + z2 * (5.0 / 6))))));
        return result + series;
    }
}
