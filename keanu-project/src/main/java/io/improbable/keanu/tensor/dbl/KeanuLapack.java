package io.improbable.keanu.tensor.dbl;

/**
 * Pure Java implementations of the handful of BLAS/LAPACK routines that Keanu uses.
 * <p>
 * All matrices are dense, row-major and addressed as (array, offset). The semantics follow the
 * LAPACKE row-major conventions of the routines they are named after, so results (including which
 * triangle is written and the 1-based pivot indices) match what the previous native OpenBLAS
 * backed implementation produced.
 */
public class KeanuLapack {

    public enum Triangular {
        UPPER, LOWER
    }

    private KeanuLapack() {
    }

    /**
     * C[M,N] = op(A)[M,K] * op(B)[K,N] where op is an optional transpose. C is overwritten.
     *
     * @param transposeA whether A is stored transposed (i.e. as K x M)
     * @param transposeB whether B is stored transposed (i.e. as N x K)
     * @param M          rows of op(A) and C
     * @param N          columns of op(B) and C
     * @param K          columns of op(A) and rows of op(B)
     * @param A          buffer containing A
     * @param aOffset    offset into A
     * @param lda        row stride of A as stored
     * @param B          buffer containing B
     * @param bOffset    offset into B
     * @param ldb        row stride of B as stored
     * @param C          output buffer
     * @param cOffset    offset into C
     * @param ldc        row stride of C
     */
    public static void dgemm(boolean transposeA, boolean transposeB,
                             int M, int N, int K,
                             double[] A, int aOffset, int lda,
                             double[] B, int bOffset, int ldb,
                             double[] C, int cOffset, int ldc) {

        for (int i = 0; i < M; i++) {
            final int cRow = cOffset + i * ldc;
            for (int j = 0; j < N; j++) {
                C[cRow + j] = 0.0;
            }

            for (int k = 0; k < K; k++) {
                final double a = transposeA ? A[aOffset + k * lda + i] : A[aOffset + i * lda + k];
                if (a == 0.0) {
                    continue;
                }

                if (transposeB) {
                    for (int j = 0; j < N; j++) {
                        C[cRow + j] += a * B[bOffset + j * ldb + k];
                    }
                } else {
                    final int bRow = bOffset + k * ldb;
                    for (int j = 0; j < N; j++) {
                        C[cRow + j] += a * B[bRow + j];
                    }
                }
            }
        }
    }

    /**
     * Computes the Cholesky factorization of a symmetric positive-definite N x N matrix in place.
     * Only the requested triangle is read and written; the other triangle is left untouched.
     *
     * @param upperLower LOWER computes A = L * L^T, UPPER computes A = U^T * U
     * @param N          length of one dimension in the square matrix to operate on
     * @param buffer     matrix to decompose and buffer to store the result
     * @param offset     offset of the matrix in buffer
     * @return 0 if successful, or i (1-based) if the leading minor of order i is not positive-definite
     */
    public static int dpotrf(Triangular upperLower, int N, double[] buffer, int offset) {
        final boolean lower = upperLower == Triangular.LOWER;

        for (int j = 0; j < N; j++) {
            double diagonal = buffer[offset + j * N + j];
            for (int k = 0; k < j; k++) {
                final double l = buffer[index(lower, offset, N, j, k)];
                diagonal -= l * l;
            }

            if (!(diagonal > 0.0)) {
                return j + 1;
            }

            final double ljj = Math.sqrt(diagonal);
            buffer[offset + j * N + j] = ljj;

            for (int i = j + 1; i < N; i++) {
                double value = buffer[index(lower, offset, N, i, j)];
                for (int k = 0; k < j; k++) {
                    value -= buffer[index(lower, offset, N, i, k)] * buffer[index(lower, offset, N, j, k)];
                }
                buffer[index(lower, offset, N, i, j)] = value / ljj;
            }
        }

        return 0;
    }

    /**
     * Computes the inverse of a symmetric positive-definite matrix from its Cholesky factorization.
     * Only the requested triangle of the result is written; the other triangle is left untouched.
     *
     * @param upperLower which triangle holds the factorization (and receives the result)
     * @param N          length of one dimension in the square matrix to operate on
     * @param buffer     the Cholesky factorization and buffer to store the result
     * @param offset     offset of the matrix in buffer
     * @return 0 if successful, or i (1-based) if the i-th diagonal element of the factor is zero
     */
    public static int dpotri(Triangular upperLower, int N, double[] buffer, int offset) {
        final boolean lower = upperLower == Triangular.LOWER;

        // L[i][k] for lower, or U^T[i][k] for upper (U^T is lower triangular)
        final double[] L = new double[N * N];
        for (int i = 0; i < N; i++) {
            for (int k = 0; k <= i; k++) {
                L[i * N + k] = buffer[index(lower, offset, N, i, k)];
            }
            if (L[i * N + i] == 0.0) {
                return i + 1;
            }
        }

        // Linv = inverse of lower triangular L by forward substitution
        final double[] Linv = new double[N * N];
        for (int j = 0; j < N; j++) {
            Linv[j * N + j] = 1.0 / L[j * N + j];
            for (int i = j + 1; i < N; i++) {
                double sum = 0.0;
                for (int k = j; k < i; k++) {
                    sum += L[i * N + k] * Linv[k * N + j];
                }
                Linv[i * N + j] = -sum / L[i * N + i];
            }
        }

        // A^-1 = Linv^T * Linv, symmetric so only compute i >= j
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0.0;
                for (int k = i; k < N; k++) {
                    sum += Linv[k * N + i] * Linv[k * N + j];
                }
                buffer[index(lower, offset, N, i, j)] = sum;
            }
        }

        return 0;
    }

    /**
     * Computes the LU factorization with partial pivoting (P * A = L * U) of an m x n row-major matrix in place.
     * L has a unit diagonal that is not stored.
     *
     * @param m      row dimension of matrix
     * @param n      column dimension of matrix
     * @param buffer buffer that represents the matrix for LU factorization
     * @param offset offset of the matrix in buffer
     * @param ipiv   receives the 1-based pivot indices: row i was interchanged with row ipiv[i]
     * @return 0 if successful, or i (1-based) if U(i,i) is exactly zero (the matrix is singular)
     */
    public static int dgetrf(int m, int n, double[] buffer, int offset, int[] ipiv) {
        int info = 0;
        final int steps = Math.min(m, n);

        for (int k = 0; k < steps; k++) {
            int pivot = k;
            double max = Math.abs(buffer[offset + k * n + k]);
            for (int i = k + 1; i < m; i++) {
                final double candidate = Math.abs(buffer[offset + i * n + k]);
                if (candidate > max) {
                    max = candidate;
                    pivot = i;
                }
            }
            ipiv[k] = pivot + 1;

            if (pivot != k) {
                for (int j = 0; j < n; j++) {
                    final double tmp = buffer[offset + k * n + j];
                    buffer[offset + k * n + j] = buffer[offset + pivot * n + j];
                    buffer[offset + pivot * n + j] = tmp;
                }
            }

            final double ukk = buffer[offset + k * n + k];
            if (ukk == 0.0) {
                if (info == 0) {
                    info = k + 1;
                }
                continue;
            }

            for (int i = k + 1; i < m; i++) {
                final double lik = buffer[offset + i * n + k] / ukk;
                buffer[offset + i * n + k] = lik;
                if (lik != 0.0) {
                    for (int j = k + 1; j < n; j++) {
                        buffer[offset + i * n + j] -= lik * buffer[offset + k * n + j];
                    }
                }
            }
        }

        return info;
    }

    /**
     * Computes the inverse of an N x N matrix from its LU factorization produced by {@link #dgetrf}.
     *
     * @param N      length of one dimension in the square matrix to operate on
     * @param buffer the LU factorization and buffer to store the inverse
     * @param offset offset of the matrix in buffer
     * @param ipiv   the 1-based pivot indices from dgetrf
     * @return 0 if successful, or i (1-based) if U(i,i) is exactly zero (the matrix is singular)
     */
    public static int dgetri(int N, double[] buffer, int offset, int[] ipiv) {
        for (int i = 0; i < N; i++) {
            if (buffer[offset + i * N + i] == 0.0) {
                return i + 1;
            }
        }

        final double[] inverse = new double[N * N];
        final double[] column = new double[N];

        for (int c = 0; c < N; c++) {

            // column = P * e_c
            java.util.Arrays.fill(column, 0.0);
            column[c] = 1.0;
            for (int k = 0; k < N; k++) {
                final int p = ipiv[k] - 1;
                if (p != k) {
                    final double tmp = column[k];
                    column[k] = column[p];
                    column[p] = tmp;
                }
            }

            // forward solve L y = P e_c (unit diagonal)
            for (int i = 0; i < N; i++) {
                double sum = column[i];
                for (int k = 0; k < i; k++) {
                    sum -= buffer[offset + i * N + k] * column[k];
                }
                column[i] = sum;
            }

            // back solve U x = y
            for (int i = N - 1; i >= 0; i--) {
                double sum = column[i];
                for (int k = i + 1; k < N; k++) {
                    sum -= buffer[offset + i * N + k] * column[k];
                }
                column[i] = sum / buffer[offset + i * N + i];
            }

            for (int r = 0; r < N; r++) {
                inverse[r * N + c] = column[r];
            }
        }

        System.arraycopy(inverse, 0, buffer, offset, N * N);
        return 0;
    }

    /**
     * Index of element (i, j) with i >= j of the lower triangle, or of the mirrored element (j, i) of the upper
     * triangle when operating on the upper triangle.
     */
    private static int index(boolean lower, int offset, int N, int i, int j) {
        return lower ? offset + i * N + j : offset + j * N + i;
    }
}
