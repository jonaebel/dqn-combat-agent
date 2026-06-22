package nn.math;

import java.util.function.IntConsumer;
import java.util.stream.IntStream;

public class MatrixMultiplication {
    /*
     Optimized matrix multiplication:
       - Inner loop is column-major friendly via pre-transposition
       - Parallel execution kicks in when op count exceeds PARALLEL_THRESHOLD
       - Supports C = A@B, C = A@BT (BT already transposed), C = AT@B (no extra transpose)
    */

    private static final long PARALLEL_THRESHOLD = 500_000L;

    public static Matrix multiply(Matrix a, Matrix b) {
        double[][] av = a.values, bv = b.values;
        int n = av.length, k = av[0].length, m = bv[0].length;
        require(k == bv.length, "A is " + n + "×" + k + " but B is " + bv.length + "×" + m);

        double[][] result = new double[n][m];
        dotDispatch(av, transpose(bv, k, m), result, n, k, m);
        return new Matrix(result);
    }

    public static Matrix multiplyBT(Matrix a, Matrix bt) {
        double[][] av = a.values, btv = bt.values;
        int n = av.length, k = av[0].length, m = btv.length;
        require(k == btv[0].length, "A cols " + k + " ≠ BT cols " + btv[0].length);

        double[][] result = new double[n][m];
        dotDispatch(av, btv, result, n, k, m);
        return new Matrix(result);
    }

    public static Matrix multiplyATB(Matrix a, Matrix b) {
        double[][] av = a.values, bv = b.values;
        int k = av.length, n = av[0].length, m = bv[0].length;
        require(k == bv.length, "A^T rows " + k + " ≠ B rows " + bv.length);

        double[][] result = new double[n][m];
        dispatch(n, n * (long) k * m, i -> atbRow(i, av, bv, result[i], k, m));
        return new Matrix(result);
    }

    private static void dotDispatch(double[][] av, double[][] bt, double[][] result,
                                    int n, int k, int m) {
        dispatch(n, n * (long) k * m, i -> dotRow(av[i], bt, result[i], k, m));
    }

    private static void dispatch(int rows, long ops, IntConsumer body) {
        if (ops >= PARALLEL_THRESHOLD)
            IntStream.range(0, rows).parallel().forEach(body);
        else
            for (int i = 0; i < rows; i++) body.accept(i);
    }

    private static void dotRow(double[] aRow, double[][] bt, double[] dst, int k, int m) {
        for (int j = 0; j < m; j++) {
            double[] btj = bt[j];
            double sum = 0;
            for (int l = 0; l < k; l++) sum += aRow[l] * btj[l];
            dst[j] = sum;
        }
    }

    private static void atbRow(int i, double[][] av, double[][] bv, double[] dst, int k, int m) {
        for (int l = 0; l < k; l++) {
            double ali = av[l][i];
            double[] bl = bv[l];
            for (int j = 0; j < m; j++) dst[j] += ali * bl[j];
        }
    }

    private static double[][] transpose(double[][] src, int rows, int cols) {
        double[][] t = new double[cols][rows];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                t[c][r] = src[r][c];
        return t;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Dimension mismatch: " + message);
    }
}
