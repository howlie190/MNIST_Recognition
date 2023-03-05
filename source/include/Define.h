//
// Created by howli on 2023/1/2.
//

#ifndef MLP_DEFINE_H
#define MLP_DEFINE_H

#include <opencv2/core/core.hpp>

#ifdef MNIST_Recognition_Library_EXPORTS
#define DEFINE_API __declspec(dllexport)
#else
#define DEFINE_API __declspec(dllimport)
#endif

#define INPUT_LAYER_SIZE    28 * 28
#define HIDDEN_LAYER_SIZE   256
#define OUTPUT_LAYER_SIZE   10

#ifdef __cplusplus
extern "C" {
#endif

typedef cv::Mat (*pActivationFunction)(cv::Mat);

typedef double  (*pLossFunction)(cv::Mat, cv::Mat);

typedef void    (*pDerivativeOutputFunction)(cv::Mat, cv::Mat, cv::Mat &);

typedef void    (*pDerivativeFunction)(cv::Mat, cv::Mat &);

DEFINE_API cv::Mat Sigmoid(cv::Mat mat);

DEFINE_API cv::Mat Tanh(cv::Mat mat);

DEFINE_API cv::Mat ReLU(cv::Mat mat);

DEFINE_API cv::Mat Softmax(cv::Mat mat);

DEFINE_API double  MeanSquaredError(cv::Mat output, cv::Mat target);

DEFINE_API void    DerivativeSoftmaxMSE(cv::Mat input, cv::Mat target, cv::Mat &output);

DEFINE_API void    DerivativeSigmoidMSE(cv::Mat input, cv::Mat target, cv::Mat &output);

DEFINE_API void    DerivativeSigmoid(cv::Mat input, cv::Mat &output);

enum class DISTRIBUTION {
    NORMAL,
    UNIFORM
};

enum class ACTIVATION_FUNCTION {
    SIGMOID = 0,
    TANH,
    RELU,
    SOFTMAX
};

enum class LOSS_FUNCTION {
    MEAN_SQUARED_ERROR = 0,
};

enum class DERIVATIVE_FUNCTION {
    SIGMOID,
    SIGMOID_MSE,
    SOFTMAX_MSE
};

#ifdef __cplusplus
}
#endif

#endif //MLP_DEFINE_H
