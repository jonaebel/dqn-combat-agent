package nn.core;

import nn.math.Matrix;
import nn.math.activationFunctions.ActivationFunction;
import nn.math.lossFunctions.LossFunction;

public class NeuronalNetwork {

    private Layer[] network;

    public NeuronalNetwork(int[] structure, int inputs, ActivationFunction[] activation_functions) { // Inputs sind hier die Werte des ersten Layers
        network = new Layer[structure.length];
        int dimensions = inputs;
        for(int i = 0; i < structure.length; i++) { // Initialisierung von allen Layern
            network[i] = new Layer(structure[i],dimensions,activation_functions[i]);
            dimensions = structure[i];
        }
    }

    public double[] feed_forward(double [] inputs) {
        Matrix result = new Matrix(new double[][]{inputs}); // 1 * inputs

        //    double[] results = inputs;
        for(Layer layer : network) result = layer.feed_forward(result);
        return result.getValues()[0]; // 1 * inputs
    }

    public void train(double[] inputs, double[] targets, double lr, LossFunction loss_function) {

        // 1. Forward pass
        //double[] prediction = feed_forward(inputs);
        Matrix prediction = new Matrix(new double[][]{inputs});
        for(Layer layer : network) prediction = layer.feed_forward(prediction);
        // richtiges Ergebnis
        Matrix actualMatrix = new Matrix(new double[][]{targets});

        // 2. Gradientenberechnung sind ableitungen von Losses
        // Jedes Neuron hat ein eingen Gradient

        Matrix gradients = loss_function.gradient(prediction, actualMatrix);

        // double[] gradients = crossEntropyLossDerivative(prediction, actual);

        // 3. Gewichte updaten:
        for(int i = network.length-1; i >= 0; i--) {
            gradients = network[i].backpropagation(gradients, lr);
        }


    }
    // Loss berechnung
    public double crossEntropyLoss(double[] predicted, double[] actual) {
        // Summe berechnen
        double cel = 0.0;

        for(int i = 0; i < predicted.length; i++) {
            cel += actual[i] * Math.log(predicted[i]);
        }
        return -cel;
    }

    public double[] crossEntropyLossDerivative(double[] predicted, double[] actual) {
        double[] sum = new double[predicted.length];
        for(int i = 0; i < sum.length; i++) sum[i] = predicted[i] - actual[i];
        return sum;
    }
    public void copy_weights_from(NeuronalNetwork source) {
        for(int i = 0; i < network.length; i++) {
            network[i].copy_weights_from(source.network[i]);
        }

    }


}
