package nn.math.lossFunctions;

import nn.math.Matrix;

/*
    Huber Loss
        Let r = y_hat - y, delta the threshold.
        Per element:
            |r| <= delta:  0.5 * r^2              (quadratic, like MSE)
            |r| >  delta:  delta*(|r| - 0.5*delta) (linear, like MAE)
        Gradient:
            |r| <= delta:  r / N
            |r| >  delta:  delta * sign(r) / N
        Combines MAE robustness against outliers with MSE smoothness near 0.
 */
public class Huber implements LossFunction {

    private final double delta;

    public Huber(double delta) {
        this.delta = delta;
    }

    public Huber() {
        this(1.0);
    }

    @Override
    public double compute(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        double sum = 0;
        int N = p.length * p[0].length;
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++) {
                double r = Math.abs(p[i][j] - t[i][j]);
                sum += r <= delta ? 0.5 * r * r : delta * (r - 0.5 * delta);
            }
        return sum / N;
    }

    @Override
    public Matrix gradient(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        int N = p.length * p[0].length;
        double[][] grad = new double[p.length][p[0].length];
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++) {
                double r = p[i][j] - t[i][j];
                grad[i][j] = Math.abs(r) <= delta ? r / N : delta * Math.signum(r) / N;
            }
        return new Matrix(grad);
    }

    @Override
    public LossFunctionType getType() {
        return LossFunctionType.HUBER;
    }
}
