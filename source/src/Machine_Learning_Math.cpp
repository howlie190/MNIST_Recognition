//
// Created by LG on 2023/11/15.
//

#include "Machine_Learning_Math.h"
#include <opencv2/opencv.hpp>

cv::Mat Machine_Learning_Math::Sigmoid(const cv::Mat &input) {
    cv::Mat exp_x;

    cv::exp(-input, exp_x);
    exp_x += 1.0;
    cv::Mat output = 1.0 / exp_x;
#ifdef USE_PATCH_NANS
    cv::patchNaNs(output, 1.0);
#endif
    return output;
}
//============================================================================================================
cv::Mat Machine_Learning_Math::Tanh(const cv::Mat &input) {
    cv::Mat exp_x, exp_x_n;

    cv::exp(input, exp_x);
    cv::exp(-input, exp_x_n);

    cv::Mat numerator   = exp_x - exp_x_n;
    cv::Mat denominator = exp_x + exp_x_n;

    cv::divide(numerator, denominator, numerator);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(numerator, 1.0);
#endif
    return numerator;
}
//============================================================================================================
cv::Mat Machine_Learning_Math::ReLU(const cv::Mat &input) {
    cv::Mat output;

    cv::threshold(input, output, 0, 0, cv::THRESH_TOZERO);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(output, 1.0);
#endif
    return output;
}
//============================================================================================================
cv::Mat Machine_Learning_Math::Softmax(const cv::Mat &input) {
    cv::Mat exp;
    double  min, max;

    cv::minMaxLoc(input, &min, &max);
    cv::Mat adjusted;
    cv::subtract(input, max, adjusted);

    cv::exp(adjusted, exp);

    double sum = cv::sum(exp)[0];
    cv::Mat output = exp / sum;
#ifdef USE_PATCH_NANS
    cv::patchNaNs(output, 1.0);
#endif
    return output;
}
//============================================================================================================
double Machine_Learning_Math::MeanSquaredError(const cv::Mat &output, const cv::Mat &label) {
    cv::Mat result;
    cv::pow(label - output, 2.0, result);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(result, 1.0);
#endif
    cv::Scalar mse = cv::mean(result);

    return mse[0];
}
//============================================================================================================
double Machine_Learning_Math::CrossEntropy(const cv::Mat &output, const cv::Mat &label) {
    cv::Mat result;
    cv::log(output, result);

    cv::Mat elementsProduct = label.mul(result);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(elementsProduct, 1.0);
#endif
    double sum = cv::sum(elementsProduct)[0];
    return -sum / result.rows;
}
//============================================================================================================
double Machine_Learning_Math::BinaryCrossEntropy(const cv::Mat &output, const cv::Mat &label) {
    cv::Mat firstLog, secondLog;
    cv::log(output, firstLog);
    cv::log(1.0 - output, secondLog);

    cv::Mat first = label.mul(firstLog);
    cv::Mat second = (1.0 - label).mul(secondLog);

    cv::Scalar sum = cv::sum(first + second);
    double loss = -sum[0] / label.rows;

    return loss;
}
//============================================================================================================
void Machine_Learning_Math::DerivativeSigmoid(const cv::Mat &source, cv::Mat &destination) {
    destination = source.mul(1.0f - source);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(destination, 1.0);
#endif
}
//============================================================================================================
void Machine_Learning_Math::DerivativeReLU(const cv::Mat &source, cv::Mat &destination) {
    destination = source.clone();
    cv::threshold(destination, destination, 0, 1, cv::THRESH_BINARY);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(destination, 1.0);
#endif
}
//============================================================================================================
void Machine_Learning_Math::DerivativeTanh(const cv::Mat &source, cv::Mat &destination) {
    cv::Mat pow;
    cv::pow(source, 2, pow);
    cv::add(pow, -1.0, destination);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(destination, 1.0);
#endif
}
//============================================================================================================
void Machine_Learning_Math::DerivativeSoftmaxCrossEntropy(const cv::Mat &output, const cv::Mat &label, cv::Mat *destination) {
    cv::subtract(output, label, *destination);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(*destination, 1.0);
#endif
}
//============================================================================================================
void Machine_Learning_Math::DerivativeSigmoidBinaryCrossEntropy(const cv::Mat &output, const cv::Mat &label, cv::Mat *destination) {
    cv::subtract(output, label, *destination);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(*destination, 1.0);
#endif
}
//============================================================================================================
void Machine_Learning_Math::DerivativeTanhMeanSquaredError(const cv::Mat &output, const cv::Mat &label, cv::Mat *destination) {
    cv::Mat pow;
    cv::pow(output, 2.0, pow);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(pow, 1.0);
#endif
    cv::Mat first = 2.0 * (output - label);
    cv::Mat second = 1.0 - pow;

    *destination = first.mul(second);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(*destination, 1.0);
#endif
}
//============================================================================================================
void Machine_Learning_Math::DerivativeMeanSquaredError(const cv::Mat &output, const cv::Mat &label, cv::Mat *destination) {
    cv::subtract(output, label, *destination);
    *destination *= 2.0;
#ifdef USE_PATCH_NANS
    cv::patchNaNs(*destination, 1.0);
#endif
}
//============================================================================================================
cv::Mat Machine_Learning_Math::None(const cv::Mat &input) {
    return input.clone();
}
//============================================================================================================
void Machine_Learning_Math::DerivativeNone(const cv::Mat &source, cv::Mat &destination) {
    destination     = source.clone();
}
//============================================================================================================
void Machine_Learning_Math::DerivativeReLUMeanSquaredError(const cv::Mat &output, const cv::Mat &label, cv::Mat *destination) {
    cv::Mat dMSE = 2.0 * (output - label);
    cv::threshold(dMSE, *destination, 0, 0, cv::THRESH_TOZERO);
#ifdef USE_PATCH_NANS
    cv::patchNaNs(*destination, 1.0);
#endif
}
//============================================================================================================
double Machine_Learning_Math::L2Regression(const std::vector<cv::Mat> &weight, double lambda) {
    cv::Mat     squared;
    cv::Scalar  sum = 0;

    for (const auto & i : weight) {
        cv::pow(i, 2, squared);
        sum += cv::sum(squared);
    }

    return lambda * sum[0];
}
//============================================================================================================
