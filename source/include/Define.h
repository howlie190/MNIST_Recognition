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

typedef cv::Mat     (*pActivationFunction)(const cv::Mat&);
typedef double      (*pLossFunction)(const cv::Mat&, const cv::Mat&);
typedef void        (*pDerivativeOutputFunction)(const cv::Mat&, const cv::Mat&, cv::Mat*);
typedef void        (*pDerivativeFunction)(const cv::Mat&, cv::Mat*);

DEFINE_API cv::Mat  Sigmoid(const cv::Mat&);
DEFINE_API cv::Mat  Tanh(const cv::Mat&);
DEFINE_API cv::Mat  ReLU(const cv::Mat&);
DEFINE_API cv::Mat  Softmax(const cv::Mat&);

DEFINE_API double   MeanSquaredError(const cv::Mat&, const cv::Mat&);
DEFINE_API double   CrossEntropy(const cv::Mat&, const cv::Mat&);
DEFINE_API void     DerivativeSoftmaxMSE(const cv::Mat&, const cv::Mat&, cv::Mat*);
DEFINE_API void     DerivativeSoftmaxCrossEntropy(const cv::Mat&, const cv::Mat&, cv::Mat*);
DEFINE_API void     DerivativeSigmoidMSE(const cv::Mat&, const cv::Mat&, cv::Mat*);
DEFINE_API void     DerivativeSigmoid(const cv::Mat&, cv::Mat*);
DEFINE_API void     DerivativeSigmoidCrossEntropy(const cv::Mat&, const cv::Mat&, cv::Mat*);
DEFINE_API void     DerivativeTanh(const cv::Mat&, cv::Mat*);
DEFINE_API void     DerivativeReLU(const cv::Mat&, cv::Mat*);
DEFINE_API void     CalMatTotal(const std::vector<cv::Mat>&, cv::Mat*, int, int);
DEFINE_API void     CalMatAvg(const cv::Mat&, int number, cv::Mat*);
DEFINE_API void     CalVecMatProduct(const std::vector<std::vector<cv::Mat>>&,
                                    const std::vector<std::vector<cv::Mat>>&,
                                    std::vector<cv::Mat>*,
                                    int,
                                    int,
                                    int,
                                    int);
DEFINE_API void     CalWeightLayerProduct(const std::vector<cv::Mat>&,
                                          std::vector<std::vector<cv::Mat>>*,
                                          int,
                                          int,
                                          int,
                                          int);
DEFINE_API void     ShowMat(cv::Mat);

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
    CROSS_ENTROPY
};

enum class DERIVATIVE_FUNCTION {
    SIGMOID,
    SIGMOID_MSE,
    SOFTMAX_MSE,
    SOFTMAX_CROSS_ENTROPY,
    SIGMOID_CROSS_ENTROPY,
    TANH,
    RELU
};

enum class OPTIMIZER {
    NONE = 0,
    ADAM
};

#ifdef __cplusplus
}
#endif

#endif //MLP_DEFINE_H
