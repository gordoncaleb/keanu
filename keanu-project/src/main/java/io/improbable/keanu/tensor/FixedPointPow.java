package io.improbable.keanu.tensor;

/**
 * Exact integer exponentiation shared by every int/long code path so they all agree.
 * <p>
 * Overflow wraps around as in ordinary Java int/long arithmetic. A negative exponent gives the result of
 * 1 / base^|exponent| truncated towards zero (so 1 for base 1, +/-1 for base -1 and 0 otherwise), and
 * 0 raised to a negative power throws an ArithmeticException in the same way as integer division by zero.
 */
public final class FixedPointPow {

    private FixedPointPow() {
    }

    public static int pow(int base, int exponent) {
        if (exponent < 0) {
            return (int) negativePow(base, exponent);
        }

        int result = 1;
        int b = base;
        int e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) {
                result *= b;
            }
            b *= b;
            e >>= 1;
        }
        return result;
    }

    public static long pow(long base, long exponent) {
        if (exponent < 0) {
            return negativePow(base, exponent);
        }

        long result = 1;
        long b = base;
        long e = exponent;
        while (e > 0) {
            if ((e & 1) == 1) {
                result *= b;
            }
            b *= b;
            e >>= 1;
        }
        return result;
    }

    private static long negativePow(long base, long exponent) {
        if (base == 0) {
            throw new ArithmeticException("0 cannot be raised to a negative power");
        } else if (base == 1) {
            return 1;
        } else if (base == -1) {
            return (exponent & 1) == 0 ? 1 : -1;
        } else {
            return 0;
        }
    }
}
