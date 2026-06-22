package nn.core;

import java.util.Random;

/*
 Single-neuron implementation — useful for understanding and experimentation.
 For production use, prefer NeuronalNetwork + Layer which operate on batched matrices.
*/
public class Neuron {

    double [] weights;
    double bias;

    public static final Random RANDOM = new Random();

    public Neuron(int inputsize) {
        weights = new double[inputsize];
        for(int i = 0; i < inputsize; i++) weights[i] = RANDOM.nextDouble(-0.5,0.5);
        bias = RANDOM.nextDouble(-0.5,0.5);
    }

    public double sigmoid(double z) {
        return 1.0/(1.0+Math.exp(-z));
    }
    public double ReLu(double z) {
        return Math.max(0, z);
    }
    public double LeakyReLu(double z) {
        return z > 0 ? z : 0.01 * z;
    }
    public double LeakyReLuDerivative(double z) {
        return z > 0 ? 1.0 : 0.01;
    }

    public double feed_forward(double [] z) {
        return (LeakyReLu(compute_z(z)));
    }

    public double compute_z(double [] z) {
        double result = 0;
        for(int i = 0; i < weights.length; i++) {
            result += z[i] * getWeights()[i];
        }
        return result+=bias;
    }

    public double loss(double predicted, double actual) {
        double diff = predicted - actual;
        return  (diff * diff);
    }
    public double lossderivative(double predicted, double actual) {
        return 2* (predicted - actual);
    }
    // TODO: Cross Entropy
    // TODO: Softmax


    // 1 Neuronen Trainierer
    public void train(double[] input, double actual, double learningrate) {
        // Forward Pass
        double predicted = feed_forward(input);
        double z = compute_z(input);
        // Gradient Descent
        double delta = lossderivative(predicted, actual) * LeakyReLuDerivative(z);
        // Update
        for(int i = 0; i < weights.length; i++) {
            weights[i] -= learningrate * delta * input[i];
        }
        //bias
        bias -= learningrate * delta;
    }


    public double[] getWeights() {
        return weights;
    }

    public double getBias() {
        return bias;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}