package nn.math.weightInitialization;

import nn.math.Matrix;

public interface WeightInitialization {
    /*
     fanIn  – number of inputs a neuron receives (neurons in the previous layer)
     fanOut – number of outputs a neuron sends to (neurons in the next layer)
    */
    Matrix compute(int fanIn, int fanOut);
    WeightInitializationTyp getTyp();
}
