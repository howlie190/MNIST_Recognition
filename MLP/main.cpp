#include <iostream>
#include <opencv2/opencv.hpp>
#include "BmpDigitRecognition.h"

int main(int argc, char **argv) {
    BmpDigitRecognition BDR;
    BDR.InitNeuralNet({INPUT_LAYER_SIZE, 256, 128, OUTPUT_LAYER_SIZE});
    BDR.InitWeights(DISTRIBUTION::NORMAL, 0.0, 0.1);
    BDR.InitBias(cv::Scalar(0.5));
    BDR.SetActivationFunction(ACTIVATION_FUNCTION::SIGMOID);
    BDR.SetOutputActivationFunction(ACTIVATION_FUNCTION::SOFTMAX);
    BDR.SetLossFunction(LOSS_FUNCTION::MEAN_SQUARED_ERROR);
    BDR.SetDerivativeOutputFunction(DERIVATIVE_FUNCTION::SOFTMAX_MSE);
    BDR.SetDerivativeFunction(DERIVATIVE_FUNCTION::SIGMOID);
    BDR.SetLearningRate(0.03);
    BDR.SetLambda(1.0e-12);
    BDR.SetThreshold(0.00001);
    BDR.SetLoopCount(300);
    BDR.Train(argv[1]);
    BDR.Test(argv[2]);
}
