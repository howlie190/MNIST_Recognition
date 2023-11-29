//
// Created by LG on 2023/11/15.
//

#ifndef MACHINELEARNING_MACHINE_LEARNING_MATH_H
#define MACHINELEARNING_MACHINE_LEARNING_MATH_H

#include <opencv2/core/core.hpp>

#ifdef MachineLearning_EXPORTS
#define MACHINE_LEARNING_API __declspec(dllexport)
#else
#define MACHINE_LEARNING_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

class MACHINE_LEARNING_API Machine_Learning_Math {
public:
    //激活函數
    static cv::Mat  Sigmoid(const cv::Mat&);
    static cv::Mat  Tanh(const cv::Mat&);
    static cv::Mat  ReLU(const cv::Mat&);
    static cv::Mat  Softmax(const cv::Mat&);
    static cv::Mat  None(const cv::Mat&);
//============================================================================================================
    //損失函數
    static double   MeanSquaredError(const cv::Mat&, const cv::Mat&);
    static double   CrossEntropy(const cv::Mat&, const cv::Mat&);
    static double   BinaryCrossEntropy(const cv::Mat&, const cv::Mat&);
//============================================================================================================
    //激活函數導數
    static void     DerivativeSigmoid(const cv::Mat&, cv::Mat&);
    static void     DerivativeReLU(const cv::Mat&, cv::Mat&);
    static void     DerivativeTanh(const cv::Mat&, cv::Mat&);
    static void     DerivativeNone(const cv::Mat&, cv::Mat&);
//============================================================================================================
    //輸出層導數
    static void     DerivativeSoftmaxCrossEntropy(const cv::Mat&, const cv::Mat&, cv::Mat*);
    static void     DerivativeSigmoidBinaryCrossEntropy(const cv::Mat&, const cv::Mat&, cv::Mat*);
    static void     DerivativeTanhMeanSquaredError(const cv::Mat&, const cv::Mat&, cv::Mat*);
    static void     DerivativeMeanSquaredError(const cv::Mat&, const cv::Mat&, cv::Mat*);
    static void     DerivativeReLUMeanSquaredError(const cv::Mat&, const cv::Mat&, cv::Mat*);
//============================================================================================================
    //正則化
    static double   L2Regression(const std::vector<cv::Mat>&, double);
    static cv::Mat  DerivativeL2Regression(const cv::Mat&, const double&);
//============================================================================================================
};



#ifdef __cplusplus
}
#endif


#endif //MACHINELEARNING_MACHINE_LEARNING_MATH_H
