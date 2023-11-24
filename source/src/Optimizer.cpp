//
// Created by LG on 2023/11/15.
//

#include "Optimizer.h"
#include "MLP.h"

void Optimizer::Init() {
    type        = TYPE::NONE;

    beta1       = 0.8;
    beta2       = 0.9;

    ADAM_count  = 1;

    mt_weight.clear();
    vt_weight.clear();
    mt_bias.clear();
    vt_bias.clear();
}
//============================================================================================================
void Optimizer::SetADAM(const std::vector<int>& nnl, double b1, double b2) {
    neural_number_of_layer  = nnl;
    beta1                   = b1;
    beta2                   = b2;
}
//============================================================================================================
void Optimizer::SetType(Optimizer::TYPE t) {
    type    = t;
}
//============================================================================================================
void Optimizer::SetOptimizer() {
    switch (type) {
        case TYPE::NONE:
            break;
        case TYPE::ADAM:
            size_t      vec_size    = neural_number_of_layer.size();

            mt_weight.resize(vec_size - 1);
            vt_weight.resize(vec_size - 1);
            mt_bias.resize(vec_size - 1);
            vt_bias.resize(vec_size - 1);

            for(int i = 0; i < vec_size - 1; i++) {
                int layer_size      = neural_number_of_layer[i + 1];
                int pre_layer_size  = neural_number_of_layer[i];

                mt_weight[i]    = cv::Mat::zeros(layer_size, pre_layer_size, CV_32FC1);
                vt_weight[i]    = cv::Mat::zeros(layer_size, pre_layer_size, CV_32FC1);
                mt_bias[i]      = cv::Mat::zeros(layer_size, 1, CV_32FC1);
                vt_bias[i]      = cv::Mat::zeros(layer_size, 1, CV_32FC1);
            }
            break;
    }
}
//============================================================================================================
void Optimizer::UpdateADAM(const std::vector<cv::Mat> &weight, const std::vector<cv::Mat> &bias, const size_t &idx) {
    cv::pow(weight[idx], 2, squared);
    mt_weight[idx]  = (beta1 * mt_weight[idx]) + ((1.0 - beta1) * weight[idx]);
    vt_weight[idx]  = (beta2 * vt_weight[idx]) + ((1.0 - beta2) * squared);

    cv::pow(bias[idx], 2, squared);
    mt_bias[idx]    = (beta1 * mt_bias[idx]) + ((1.0 - beta1) * bias[idx]);
    vt_bias[idx]    = (beta2 * vt_bias[idx]) + ((1.0 - beta2) * squared);

    mt_weight[idx]  = mt_weight[idx] / (1.0 - std::pow(beta1, ADAM_count));
    vt_weight[idx]  = vt_weight[idx] / (1.0 - std::pow(beta2, ADAM_count));

    mt_bias[idx]    = mt_bias[idx] / (1.0 - std::pow(beta1, ADAM_count));
    vt_bias[idx]    = vt_bias[idx] / (1.0 - std::pow(beta2, ADAM_count));

    cv::sqrt(vt_weight[idx], sqrt);
    sqrt    += epsilon;
    cv::divide(mt_weight[idx], sqrt, weight_result);

    cv::sqrt(vt_bias[idx], sqrt);
    sqrt    += epsilon;
    cv::divide(mt_bias[idx], sqrt, bias_result);
}
//============================================================================================================
cv::Mat Optimizer::GetWeightResult() const {
    return weight_result;
}
//============================================================================================================
cv::Mat Optimizer::GetBiasResult() const {
    return bias_result;
}
//============================================================================================================
void Optimizer::PlusADAMCount() {
    ADAM_count++;
}
//============================================================================================================