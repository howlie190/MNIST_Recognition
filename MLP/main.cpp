#include <iostream>
#include <opencv2/opencv.hpp>
#include "BmpDigitRecognition.h"

int main(int argc, char **argv) {
    BmpDigitRecognition BDR;
    BDR.InitNeuralNet({INPUT_LAYER_SIZE, 64, 64, OUTPUT_LAYER_SIZE});
    BDR.InitWeights(DISTRIBUTION::NORMAL, 0.0, 0.05);
    BDR.InitBias(cv::Scalar(0.0));
    BDR.Load("D:", "test");
    BDR.SetActivationFunction(ACTIVATION_FUNCTION::SIGMOID);
    BDR.SetOutputActivationFunction(ACTIVATION_FUNCTION::SOFTMAX);
    BDR.SetLossFunction(LOSS_FUNCTION::MEAN_SQUARED_ERROR);
    BDR.SetDerivativeOutputFunction(DERIVATIVE_FUNCTION::SOFTMAX_MSE);
    BDR.SetDerivativeFunction(DERIVATIVE_FUNCTION::SIGMOID);
    BDR.SetLearningRate(0.8);
    BDR.SetLambda(1.0e-12);
    BDR.SetThreshold(30);
    BDR.SetLoopCount(1);
    BDR.Train(argv[1]);
    BDR.Save("", "test", true);
//    BDR.Test(argv[2]);
}
