package nn.math.lossFunctions;

import nn.math.Matrix;

/*
    Combined Softmax + Categorical Cross-Entropy Loss.

    Why combined?
        Softmax.derivative() only returns the diagonal s*(1-s) of the true Jacobian
        (diag(s) - s*s^T). Running CCE and Softmax separately through the chain rule
        zeros out gradients for incorrect classes — the model learns nothing.

    Correct combined derivative (d(CCE∘Softmax)/dz):
        grad_i = (softmax(z)_i - target_i) / N

    Usage: output layer with Linear activation + this loss.
 */
public class SoftmaxCCE implements LossFunction {

    private static final double EPS = 1e-12;

    @Override
    public double compute(Matrix logits, Matrix targets) {
        Matrix probs = softmax(logits);
        double[][] p = probs.getValues();
        double[][] t = targets.getValues();
        int N = p.length;
        double sum = 0;
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++)
                sum += t[i][j] * Math.log(Math.max(EPS, p[i][j]));
        return -sum / N;
    }

    @Override
    public Matrix gradient(Matrix logits, Matrix targets) {
        Matrix probs = softmax(logits);
        double[][] p = probs.getValues();
        double[][] t = targets.getValues();
        int N = p.length;
        double[][] grad = new double[p.length][p[0].length];
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++)
                grad[i][j] = (p[i][j] - t[i][j]) / N;
        return new Matrix(grad);
    }

    public static Matrix softmax(Matrix m) {
        double[][] in = m.getValues();
        double[][] out = new double[in.length][in[0].length];
        for (int i = 0; i < in.length; i++) {
            double max = in[i][0];
            for (double v : in[i]) if (v > max) max = v;
            double sum = 0;
            for (int j = 0; j < in[i].length; j++) {
                out[i][j] = Math.exp(in[i][j] - max);
                sum += out[i][j];
            }
            for (int j = 0; j < in[i].length; j++) out[i][j] /= sum;
        }
        return new Matrix(out);
    }

    @Override
    public LossFunctionType getType() {
        return LossFunctionType.SOFTMAX_CCE;
    }
}
