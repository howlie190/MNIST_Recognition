//
// Created by howli on 2023/1/11.
//

#include "Define.h"
#include <mutex>
#include <iostream>

std::mutex mtxCalVecMatProduct;

cv::Mat Sigmoid(const cv::Mat& mat) {
    cv::Mat exp_x, fx;
    cv::exp(-mat, exp_x);
    fx = 1.0 / (1.0 + exp_x);
    return fx;
}
//============================================================================================================
cv::Mat Tanh(const cv::Mat& mat) {
    cv::Mat exp_x, exp_x_n, fx;
    cv::exp(mat, exp_x);
    cv::exp(-mat, exp_x_n);
    fx = (exp_x - exp_x_n) / (exp_x + exp_x_n);
    return fx;
}
//============================================================================================================
cv::Mat ReLU(const cv::Mat& mat) {
    cv::Mat fx = mat;
    for (int i = 0; i < fx.rows; i++) {
        for (int j = 0; j < fx.cols; j++) {
            if (fx.at<float>(i, j) < 0)
                fx.at<float>(i, j) = 0;
        }
    }
    return fx;
}
//============================================================================================================
cv::Mat Softmax(const cv::Mat& mat) {
    cv::Mat exp;
    cv::exp(mat, exp);
    cv::Scalar sum = cv::sum(exp);
    return exp / sum[0];
}
//============================================================================================================
double MeanSquaredError(const cv::Mat& result, const cv::Mat& target) {
    cv::Mat deltaSquared;
    cv::pow(target - result, 2.0, deltaSquared);
    cv::Scalar deltaSquaredSum = cv::sum(deltaSquared);
    return deltaSquaredSum[0] / 2.0;
}
//============================================================================================================
void DerivativeSoftmaxMSE(const cv::Mat& result, const cv::Mat& target, cv::Mat *destination) {
    cv::Mat subtraction = result - target;

    for (int i = 0; i < destination->rows; i++) {
        cv::Mat ret = cv::Mat::zeros(1, 1, CV_32FC1);

        for (int j = 0; j < destination->rows; j++) {
            if (i == j) {
                cv::Mat yi_1_yi = result.row(j).mul((float) 1 - result.row(j));
                ret = ret + (subtraction.row(j).mul(yi_1_yi));
            } else {
                cv::Mat _yj_yi = ((float) -1 * result.row(i)).mul(result.row(j));
                ret = ret + (subtraction.row(j).mul(_yj_yi));
            }
        }

        destination->at<float>(i, 0) = ret.at<float>(0, 0);
    }
}
//============================================================================================================
void DerivativeSigmoid(const cv::Mat& source, cv::Mat *destination) {
    *destination = source.mul((float) 1 - source);
}
//============================================================================================================
void DerivativeSigmoidMSE(const cv::Mat& result, const cv::Mat& target, cv::Mat *destination) {
    cv::Mat subtraction = result - target;
    cv::Mat dSigmoid = result.mul((float) 1 - result);
    *destination = subtraction.mul(dSigmoid);
}
//============================================================================================================
double CrossEntropy(const cv::Mat& result, const cv::Mat& target) {
    double sum = 0;
    cv::Mat log_e;
    cv::log(result, log_e);
    for(int i = 0; i < log_e.rows; i++) {
        sum += target.at<float>(i, 0) * log_e.at<float>(i, 0);
    }
    return -1 * sum / log_e.rows;
}
//============================================================================================================
void DerivativeSoftmaxCrossEntropy(const cv::Mat& result, const cv::Mat& target, cv::Mat *destination) {
    cv::Mat subtraction = result - target;

    for (int i = 0; i < destination->rows; i++) {
        cv::Mat ret = cv::Mat::zeros(1, 1, CV_32FC1);

        for (int j = 0; j < destination->rows; j++) {
            if (i == j) {
                cv::Mat yi_1_yi = result.row(j).mul((float) 1 - result.row(j));
                ret = ret + (subtraction.row(j).mul(yi_1_yi));
            } else {
                cv::Mat _yj_yi = ((float) -1 * result.row(i)).mul(result.row(j));
                ret = ret + (subtraction.row(j).mul(_yj_yi));
            }
        }

        destination->at<float>(i, 0) = ret.at<float>(0, 0);
    }
}
//============================================================================================================
void DerivativeSigmoidCrossEntropy(const cv::Mat& result, const cv::Mat& target, cv::Mat *destination) {
    cv::Mat subtraction = result - target;
    cv::Mat dSigmoid = result.mul((float) 1 - result);
    *destination = subtraction.mul(dSigmoid);
}
//============================================================================================================
void DerivativeTanh(const cv::Mat& source, cv::Mat *destination) {
    cv::Mat pow2, tanh;
    cv::pow(source, 2, pow2);
    tanh = Tanh(pow2);
    *destination = 1 - tanh;
}
//============================================================================================================
void DerivativeReLU(const cv::Mat& source, cv::Mat *destination) {
    *destination = cv::Mat::zeros(source.rows, source.cols, CV_32FC1);

    for(int i = 0; i < source.rows; i++) {
        if(source.at<float>(i, 0) > 0) {
            destination->at<float>(i, 0) = 1;
        } else {
            destination->at<float>(i, 0) = 0;
        }
    }
}
//============================================================================================================
void CalMatTotal(const std::vector<cv::Mat>& source, cv::Mat* destination, int begin, int end) {
    *destination = cv::Mat::zeros(source[0].rows, source[0].cols, CV_32FC1);

    for(int i = begin; i <= end || i < source.size(); i++) {
        *destination += source[i];
    }
}
//============================================================================================================
void CalMatAvg(const cv::Mat& source, int number, cv::Mat* destination) {
    *destination = source / number;
}
//============================================================================================================
void CalVecMatProduct(const std::vector<std::vector<cv::Mat>>& vec1,
                      const std::vector<std::vector<cv::Mat>>& vec2,
                      std::vector<cv::Mat>* destination,
                      int begin,
                      int end,
                      int idx1,
                      int idx2) {
    for(int i = begin; i <= end || i < vec1.size(); i++) {
        cv::Mat temp = vec1[i][idx1] * std::move(vec2[i][idx2].t());

        std::unique_lock<std::mutex> lock(mtxCalVecMatProduct);
        destination->at(i) = temp;
        lock.unlock();
    }
}
//============================================================================================================
void CalWeightLayerProduct(const std::vector<cv::Mat>& weight,
                           std::vector<std::vector<cv::Mat>>* layer,
                           int begin,
                           int end,
                           int idx1,
                           int idx2) {
    for(int i = begin; i <= end || i < layer->size(); i++) {
        layer->at(i).at(idx1) = weight[idx1].t() * layer->at(i).at(idx2);
    }
}
//============================================================================================================
void ShowMat(cv::Mat mat) {
    for(int i = 0; i < mat.rows; i++) {
        for(int j = 0; j < mat.cols; j++) {
            std::cout << mat.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }
    getchar();
}
//============================================================================================================