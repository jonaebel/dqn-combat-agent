package nn.math.activationFunctions;

import nn.math.Matrix;

public interface ActivationFunction {
    Matrix apply(Matrix m);
    Matrix derivative(Matrix m);
    ActivationFunctionType getType();
}
