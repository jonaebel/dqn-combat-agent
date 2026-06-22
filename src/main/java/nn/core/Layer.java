package nn.core;

import nn.math.Matrix;
import nn.math.MatrixMultiplication;
import nn.math.activationFunctions.ActivationFunction;
import nn.math.activationFunctions.LeakyReLu;
import nn.math.activationFunctions.ReLu;
import nn.math.weightInitialization.HeNormal;
import nn.math.weightInitialization.WeightInitialization;
import nn.math.weightInitialization.XavierNormal;

public class Layer {


    private double[] biases; // 1D
    private Matrix weights; // Neuron * Weights -> 2D

    private Matrix last_input; // 2D
    private Matrix last_z; // 2D
    private ActivationFunction activation_function;


    public Layer(int count, int inputs_per_neuron, ActivationFunction activation_function, WeightInitialization weight_init) { //  Ein Layer braucht neuronenzahl, inputs pro neuron -> Matrix


        this.activation_function = activation_function;
        this.weights = weight_init.compute(inputs_per_neuron, count).transpose(); //
        this.biases = new double[count];
        this.last_input = new Matrix(1, inputs_per_neuron);
        this.last_z = new Matrix(1,count);


    }

    public Layer(int count, int inputs_per_neuron, ActivationFunction activation_function) {
        this(count, inputs_per_neuron, activation_function, defaultInit(activation_function));
    }

    private static WeightInitialization defaultInit(ActivationFunction activation) {
        if (activation instanceof LeakyReLu || activation instanceof ReLu)
            return new HeNormal();
        return new XavierNormal();
    }

    public Matrix feed_forward(Matrix batch) { // Batching


        last_input = batch;
        last_z = MatrixMultiplication.multiplyBT(batch, weights).addVector(biases);
        return activation_function.apply(last_z);

    }



    public Matrix backpropagation(Matrix grad_output, double lr) {

        int batch_size = last_input.getRows();
        Matrix delta = grad_output.hadamard(activation_function.derivative(last_z)); // Delta = grad * activation'(z)
        Matrix grad_weights = MatrixMultiplication.multiplyATB(delta, last_input).multiply(1.0 / batch_size); //     // 2. grad_weights = delta^T @ last_input / batch    [count × inputs]

        double[] grad_biases = new double[biases.length];
        double[][] dv = delta.getValues();
        for(int i = 0; i < batch_size; i++) {
            for(int j = 0; j < biases.length; j++) {
                grad_biases[j]  += dv[i][j] / batch_size;

            }
        }
        // zurückgeben
        Matrix grad_input = delta.matmul(weights);

        // Weights und biases updaten

        weights = weights.subtract(grad_weights.multiply(lr));
        for(int i = 0; i < biases.length; i++) {
            biases[i] -= lr*grad_biases[i];
        }
        return grad_input;


    }
    // Wahrscheinlichkeiten zwischen 0 und 1
    public static double[] softmax(double[] z) {
        double[] sum = new double[z.length];
        double denominator = 0.0;
        // Über alle rüber
        for(int j = 0; j < z.length; j++) {
            denominator += Math.pow(Math.E, z[j]);
        }
        for(int i = 0; i < z.length; i++) {
            sum[i] = Math.pow(Math.E, z[i]) / denominator;
        }
        return sum;

    }
    public void copy_weights_from(Layer source) {
        double[][] src = source.weights.getValues();
        double[][] cpy = new double[src.length][src[0].length];
        for(int i = 0; i < src.length; i++) {
            cpy[i] = src[i].clone();
        }
        this.weights = new Matrix(cpy);
        this.biases = source.biases.clone();

    }




}
