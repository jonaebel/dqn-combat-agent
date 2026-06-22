package nn.math.lossFunctions;

import nn.math.Matrix;

/*
    Focal Loss (Lin et al., 2017)
        Extension of BCE for heavily imbalanced datasets.
        The focus term (1-p_t)^gamma down-weights easy examples
        so training concentrates on hard-to-classify cases.
        alpha weights the classes, gamma (>= 0) controls the focus.
 */
public class FocalLoss implements LossFunction {

    private static final double EPS = 1e-12;

    private final double alpha;
    private final double gamma;

    public FocalLoss(double alpha, double gamma) {
        this.alpha = alpha;
        this.gamma = gamma;
    }

    public FocalLoss() {
        this(0.25, 2.0);
    }

    @Override
    public double compute(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        double sum = 0;
        int N = p.length * p[0].length;
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++) {
                double pC = Math.max(EPS, Math.min(1 - EPS, p[i][j]));
                double posLoss = alpha * Math.pow(1 - pC, gamma) * t[i][j] * Math.log(pC);
                double negLoss = (1 - alpha) * Math.pow(pC, gamma) * (1 - t[i][j]) * Math.log(1 - pC);
                sum += posLoss + negLoss;
            }
        return -sum / N;
    }

    @Override
    public Matrix gradient(Matrix predictions, Matrix targets) {
        double[][] p = predictions.getValues();
        double[][] t = targets.getValues();
        int N = p.length * p[0].length;
        double[][] grad = new double[p.length][p[0].length];
        for (int i = 0; i < p.length; i++)
            for (int j = 0; j < p[0].length; j++) {
                double pC = Math.max(EPS, Math.min(1 - EPS, p[i][j]));
                double posGrad = alpha * (Math.pow(1 - pC, gamma) * (-t[i][j] / pC)
                        + gamma * Math.pow(1 - pC, gamma - 1) * t[i][j] * Math.log(pC));
                double negGrad = (1 - alpha) * (Math.pow(pC, gamma) * (-(1 - t[i][j]) / (1 - pC))
                        + gamma * Math.pow(pC, gamma - 1) * (1 - t[i][j]) * Math.log(1 - pC));
                grad[i][j] = -(posGrad + negGrad) / N;
            }
        return new Matrix(grad);
    }

    @Override
    public LossFunctionType getType() {
        return LossFunctionType.FOCAL_LOSS;
    }
}
