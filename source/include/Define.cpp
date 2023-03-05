//
// Created by howli on 2023/1/11.
//

#include "Define.h"

cv::Mat Sigmoid(cv::Mat mat) {
    cv::Mat exp_x, fx;
    cv::exp(-mat, exp_x);
    fx = 1.0 / (1.0 + exp_x);
    return fx;
}

cv::Mat Tanh(cv::Mat mat) {
    cv::Mat exp_x, exp_x_n, fx;
    cv::exp(mat, exp_x);
    cv::exp(-mat, exp_x_n);
    fx = (exp_x - exp_x_n) / (exp_x + exp_x_n);
    return fx;
}

cv::Mat ReLU(cv::Mat mat) {
    cv::Mat fx = mat;
    for (int i = 0; i < fx.rows; i++) {
        for (int j = 0; j < fx.cols; j++) {
            if (fx.at<float>(i, j) < 0)
                fx.at<float>(i, j) = 0;
        }
    }
    return fx;
}

cv::Mat Softmax(cv::Mat mat) {
    cv::Mat exp;
    cv::exp(mat, exp);
    cv::Scalar sum = cv::sum(exp);
    return exp / sum[0];
}

double MeanSquaredError(cv::Mat output, cv::Mat target) {
    cv::Mat deltaSquared;
    cv::pow(target - output, 2.0, deltaSquared);
    cv::Scalar deltaSquaredSum = cv::sum(deltaSquared);
    return deltaSquaredSum[0] / 2.0;
}

void DerivativeSoftmaxMSE(cv::Mat input, cv::Mat target, cv::Mat &output) {
    cv::Mat subtraction = input - target;
    for (int i = 0; i < output.rows; i++) {
        cv::Mat ret = cv::Mat::zeros(1, 1, CV_32FC1);
        for (int j = 0; j < output.rows; j++) {
            if (i == j) {
                cv::Mat yi_1_yi = input.row(j).mul((float) 1 - input.row(j));
                ret = ret + (subtraction.row(j).mul(yi_1_yi));
            } else {
                cv::Mat _yj_yi = ((float) -1 * input.row(i)).mul(input.row(j));
                ret = ret + (subtraction.row(j).mul(_yj_yi));
            }
        }
        output.at<float>(i, 0) = ret.at<float>(0, 0);
    }
}

void DerivativeSigmoid(cv::Mat input, cv::Mat &output) {
    output = input.mul((float) 1 - input);
}

void DerivativeSigmoidMSE(cv::Mat input, cv::Mat target, cv::Mat &output) {
    cv::Mat subtraction = input - target;
    cv::Mat dSigmoid = input.mul((float) 1 - input);
    output = subtraction.mul(dSigmoid);
}