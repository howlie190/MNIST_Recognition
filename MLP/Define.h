//
// Created by howli on 2023/1/2.
//

#ifndef MLP_DEFINE_H
#define MLP_DEFINE_H

#include <opencv2/core/core.hpp>

#define INPUT_LAYER_SIZE    28 * 28
#define HIDDEN_LAYER_SIZE   256
#define OUTPUT_LAYER_SIZE   10

typedef cv::Mat (*pActivationFunction)(cv::Mat);
typedef double  (*pLossFunction)(cv::Mat, cv::Mat);
typedef void    (*pDerivativeOutputFunction)(cv::Mat, cv::Mat, cv::Mat&);
typedef void    (*pDerivativeFunction)(cv::Mat, cv::Mat&);

cv::Mat Sigmoid(cv::Mat mat);
cv::Mat Tanh(cv::Mat mat);
cv::Mat ReLU(cv::Mat mat);
cv::Mat Softmax(cv::Mat mat);

double  MeanSquaredError(cv::Mat output, cv::Mat target);
void    DerivativeSoftmaxMSE(cv::Mat input, cv::Mat target, cv::Mat& output);
void    DerivativeSigmoidMSE(cv::Mat input, cv::Mat target, cv::Mat& output);
void    DerivativeSigmoid(cv::Mat input, cv::Mat& output);

enum class DISTRIBUTION {
    NORMAL,
    UNIFORM
};

enum class ACTIVATION_FUNCTION {
    SIGMOID,
    TANH,
    RELU,
    SOFTMAX
};

enum class LOSS_FUNCTION {
    MEAN_SQUARED_ERROR
};

enum class DERIVATIVE_FUNCTION {
    SIGMOID,
    SIGMOID_MSE,
    SOFTMAX_MSE
};

#endif //MLP_DEFINE_H
