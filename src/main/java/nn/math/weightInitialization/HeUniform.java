package nn.math.weightInitialization;

import nn.math.Matrix;
import java.util.Random;

/*
    He Uniform: weights ~ U(-sqrt(6 / fanIn), sqrt(6 / fanIn))
    Designed for ReLU / LeakyReLU activations.
 */
public class HeUniform implements WeightInitialization {

    private final Random random = new Random();

    @Override
    public Matrix compute(int fanIn, int fanOut) {
        double limit = Math.sqrt(6.0 / fanIn);
        double[][] values = new double[fanIn][fanOut];
        for (int i = 0; i < fanIn; i++)
            for (int j = 0; j < fanOut; j++)
                values[i][j] = random.nextDouble() * 2 * limit - limit;
        return new Matrix(values);
    }

    @Override
    public WeightInitializationTyp getTyp() {
        return WeightInitializationTyp.HE_UNIFORM;
    }
}
