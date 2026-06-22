package nn.math.lossFunctions;

import nn.math.Matrix;

public interface LossFunction {
    double compute(Matrix predictions, Matrix targets);
    Matrix gradient(Matrix predictions, Matrix targets);
    LossFunctionType getType();
}
