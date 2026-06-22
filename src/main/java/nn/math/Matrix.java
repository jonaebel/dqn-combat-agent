package nn.math;

import java.util.Arrays;

public class Matrix {

    double[][] values;

    public Matrix(int rows, int cols) {
        values = new double[rows][cols];
    }

    public Matrix(double[][] values) {
        this.values = values;
    }

    public int getRows() {
        return values.length;
    }

    public int getColumns() {
        return values[0].length;
    }

    public double get(int row, int col) {
        return values[row][col];
    }

    public double[][] getValues() {
        return values;
    }

    public Matrix fill(double value) {
        double[][] result = new double[getRows()][getColumns()];
        for (double[] row : result)
            Arrays.fill(row, value);
        return new Matrix(result);
    }

    public Matrix transpose() {
        double[][] result = new double[getColumns()][getRows()];
        for (int i = 0; i < getRows(); i++)
            for (int j = 0; j < getColumns(); j++)
                result[j][i] = values[i][j];
        return new Matrix(result);
    }

    public Matrix multiply(double scalar) {
        return map(v -> v * scalar);
    }

    public Matrix add(Matrix other) {
        requireSameShape(other);
        return zip(other, Double::sum);
    }

    public Matrix subtract(Matrix other) {
        requireSameShape(other);
        return zip(other, (a, b) -> a - b);
    }

    public Matrix hadamard(Matrix other) {
        requireSameShape(other);
        return zip(other, (a, b) -> a * b);
    }

    public Matrix addVector(double[] vector) {
        if (vector.length != getColumns())
            throw new IllegalArgumentException(
                    "Dimension mismatch: vector length " + vector.length +
                            " does not match column count " + getColumns());
        double[][] result = new double[getRows()][getColumns()];
        for (int i = 0; i < getRows(); i++)
            for (int j = 0; j < getColumns(); j++)
                result[i][j] = values[i][j] + vector[j];
        return new Matrix(result);
    }

    public Matrix matmul(Matrix other) {
        return MatrixMultiplication.multiply(this, other);
    }

    private interface DoubleOp {
        double apply(double value);
    }

    private interface DoubleBinaryOp {
        double apply(double a, double b);
    }

    private Matrix map(DoubleOp op) {
        double[][] result = new double[getRows()][getColumns()];
        for (int i = 0; i < getRows(); i++)
            for (int j = 0; j < getColumns(); j++)
                result[i][j] = op.apply(values[i][j]);
        return new Matrix(result);
    }

    private Matrix zip(Matrix other, DoubleBinaryOp op) {
        double[][] result = new double[getRows()][getColumns()];
        for (int i = 0; i < getRows(); i++)
            for (int j = 0; j < getColumns(); j++)
                result[i][j] = op.apply(values[i][j], other.values[i][j]);
        return new Matrix(result);
    }

    private void requireSameShape(Matrix other) {
        if (getRows() != other.getRows() || getColumns() != other.getColumns())
            throw new IllegalArgumentException(
                    "Dimension mismatch: " + getRows() + "x" + getColumns() +
                            " vs " + other.getRows() + "x" + other.getColumns());
    }

    @Override
    public String toString() {
        return getRows() + "x" + getColumns();
    }
}
