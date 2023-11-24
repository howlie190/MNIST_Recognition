//
// Created by LG on 2023/11/3.
//

#include "MLP.h"
#include <random>
#include <iostream>

MLP::~MLP() {
    for(const auto& thd : vec_thread) {
        if(thd && thd->joinable())
            thd->join();
    }
}
//============================================================================================================
void MLP::Init() {
    neural_number_of_layer.clear();
    mlp_layer_values.clear();
    mlp_layer_weight_values.clear();
    mlp_layer_bias_values.clear();
    update_layer_values.clear();
    update_layer_weight_values.clear();
    update_layer_bias_values.clear();
    mlp_layer_value_predict.clear();

    for(auto & t : vec_thread) {
        t->interrupt();
    }

    vec_thread.clear();

    output_loss_value.clear();
    label_value.clear();
    output_value.clear();

    thread_size                     = 1;
    batch_size                      = 1;
    learning_rate                   = 0.01;
    section_size                    = 0;

    lambda                          = 1e-8;

    enable_optimizer                = false;

    optimizer                       = nullptr;

    ActivationFunction              = nullptr;
    OutputFunction                  = nullptr;
    LossFunction                    = nullptr;
    DerivativeActivationFunction    = nullptr;
    DerivativeOutputFunction        = nullptr;

    optimizer_type                  = Optimizer::TYPE::NONE;

    optimizer                       = std::make_unique<Optimizer>();
    optimizer->Init();
}
//============================================================================================================
void MLP::InitNeuralNet(std::vector<int> &&number) {
    neural_number_of_layer      = number;

    InitNeuralNetLayer(mlp_layer_value_predict);
    InitNeuralNetWeightBias(mlp_layer_weight_values, mlp_layer_bias_values);
}
//============================================================================================================
void MLP::InitNeuralNetLayer(std::vector<std::vector<cv::Mat>>& layer) {
    layer.resize(batch_size);

    for(int i = 0; i < batch_size; i++) {
        layer[i].resize(neural_number_of_layer.size());

        for(int j = 0; j < layer[i].size(); j++) {
            layer[i][j].create(neural_number_of_layer[j], 1, CV_32FC1);
        }
    }
}
//============================================================================================================
void MLP::InitNeuralNetWeightBias(std::vector<cv::Mat> &weight, std::vector<cv::Mat> &bias) {
    weight.resize(neural_number_of_layer.size() - 1);
    bias.resize(neural_number_of_layer.size() - 1);

    for (int i = 0; i < neural_number_of_layer.size() - 1; i++) {
        weight[i].create(neural_number_of_layer[i + 1], neural_number_of_layer[i], CV_32FC1);
        bias[i] = cv::Mat::zeros(neural_number_of_layer[i + 1], 1, CV_32FC1);
    }
}
//============================================================================================================
void MLP::SetInitWeightValue(DISTRIBUTION distribution, double mean, double standard_deviation) {
    std::random_device  rd;
    cv::RNG             rng(rd());

    for(auto & mlp_layer_weight_value : mlp_layer_weight_values) {
        switch (distribution) {
            case DISTRIBUTION::NORMAL:
                cv::randn(mlp_layer_weight_value, mean, standard_deviation);
                break;
            case DISTRIBUTION::UNIFORM:
                cv::randu(mlp_layer_weight_value, mean, standard_deviation);
                break;
            default:
                cv::randn(mlp_layer_weight_value, 0.0, 0.5);
        }
    }
}
//============================================================================================================
void MLP::SetInitBiasValue(cv::Scalar &&scalar) {
    for(auto &bias: mlp_layer_bias_values)
        bias = scalar;
}
//============================================================================================================
void MLP::SetActivationFunction(ACTIVATION_FUNCTION af) {
    switch (af) {
        case ACTIVATION_FUNCTION::SIGMOID:
            ActivationFunction      = Machine_Learning_Math::Sigmoid;
            break;
        case ACTIVATION_FUNCTION::TANH:
            ActivationFunction      = Machine_Learning_Math::Tanh;
            break;
        case ACTIVATION_FUNCTION::RELU:
            ActivationFunction      = Machine_Learning_Math::ReLU;
            break;
        case ACTIVATION_FUNCTION::NONE:

        default:
            ActivationFunction      = Machine_Learning_Math::None;
    }
}
//============================================================================================================
void MLP::SetOutputFunction(OUTPUT_FUNCTION of) {
    switch (of) {
        case OUTPUT_FUNCTION::SIGMOID:
            OutputFunction          = Machine_Learning_Math::Sigmoid;
            break;
        case OUTPUT_FUNCTION::TANH:
            OutputFunction          = Machine_Learning_Math::Tanh;
            break;
        case OUTPUT_FUNCTION::RELU:
            OutputFunction          = Machine_Learning_Math::ReLU;
            break;
        case OUTPUT_FUNCTION::SOFTMAX:
            OutputFunction          = Machine_Learning_Math::Softmax;
            break;
        case OUTPUT_FUNCTION::NONE:

        default:
            OutputFunction          = Machine_Learning_Math::None;
    }
}
//============================================================================================================
void MLP::SetLossFunction(LOSS_FUNCTION lf) {
    switch (lf) {
        case LOSS_FUNCTION::MEAN_SQUARED_ERROR:
            LossFunction            = Machine_Learning_Math::MeanSquaredError;
            break;
        case LOSS_FUNCTION::CROSS_ENTROPY:
            LossFunction            = Machine_Learning_Math::CrossEntropy;
            break;
        case LOSS_FUNCTION::BINARY_CROSS_ENTROPY:
            LossFunction            = Machine_Learning_Math::BinaryCrossEntropy;
            break;
    }
}
//============================================================================================================
void MLP::SetDerivativeActivationFunction(DERIVATIVE_ACTIVATION_FUNCTION daf) {
    switch (daf) {
        case DERIVATIVE_ACTIVATION_FUNCTION::SIGMOID:
            DerivativeActivationFunction        = Machine_Learning_Math::DerivativeSigmoid;
            break;
        case DERIVATIVE_ACTIVATION_FUNCTION::TANH:
            DerivativeActivationFunction        = Machine_Learning_Math::DerivativeTanh;
            break;
        case DERIVATIVE_ACTIVATION_FUNCTION::RELU:
            DerivativeActivationFunction        = Machine_Learning_Math::DerivativeReLU;
            break;
        case DERIVATIVE_ACTIVATION_FUNCTION::NONE:
            DerivativeActivationFunction        = Machine_Learning_Math::DerivativeNone;
            break;
    }
}
//============================================================================================================
void MLP::SetDerivativeOutputFunction(DERIVATIVE_OUTPUT_FUNCTION dof) {
    switch (dof) {
        case DERIVATIVE_OUTPUT_FUNCTION::SOFTMAX_CROSS_ENTROPY:
            DerivativeOutputFunction        = Machine_Learning_Math::DerivativeSoftmaxCrossEntropy;
            break;
        case DERIVATIVE_OUTPUT_FUNCTION::SIGMOID_BINARY_CROSS_ENTROPY:
            DerivativeOutputFunction        = Machine_Learning_Math::DerivativeSigmoidBinaryCrossEntropy;
            break;
        case DERIVATIVE_OUTPUT_FUNCTION::TANH_MEAN_SQUARED_ERROR:
            DerivativeOutputFunction        = Machine_Learning_Math::DerivativeTanhMeanSquaredError;
            break;
        case DERIVATIVE_OUTPUT_FUNCTION::RELU_MEAN_SQUARED_ERROR:
            DerivativeOutputFunction        = Machine_Learning_Math::DerivativeReLUMeanSquaredError;
            break;
        case DERIVATIVE_OUTPUT_FUNCTION::NONE_MEAN_SQUARED_ERROR:
            DerivativeOutputFunction        = Machine_Learning_Math::DerivativeMeanSquaredError;
            break;
    }
}
//============================================================================================================
void MLP::ForwardBatch() {
    for(int i = 0; i < thread_size; i++) {
        size_t  begin   = i * section_size;
        size_t  end     = (i + 1) * section_size - 1;

        end         = end > batch_size ? batch_size - 1 : end;

        vec_thread.emplace_back(std::make_unique<boost::thread>(&MLP::ForwardBatchPropagation, this, begin, end));
    }

    for(const auto &thd : vec_thread) {
        if(thd && thd->joinable())
            thd->join();
    }

    vec_thread.clear();
}
//============================================================================================================
void MLP::SetFunctions(ACTIVATION_FUNCTION af, OUTPUT_FUNCTION of, LOSS_FUNCTION lf) {
    SetActivationFunction(af);
    SetOutputFunction(of);
    SetLossFunction(lf);

    SetDerivativeActivationFunction(static_cast<DERIVATIVE_ACTIVATION_FUNCTION>(af));

    auto derivative = derivative_map.find({of, lf});

    if(derivative != derivative_map.end()) {
        SetDerivativeOutputFunction(derivative->second);
    } else {
        SetDerivativeOutputFunction(DERIVATIVE_OUTPUT_FUNCTION::NONE_MEAN_SQUARED_ERROR);
    }
}
//============================================================================================================
void MLP::SetParameter(int thd_size, double lr, double lamb, Optimizer::TYPE opt_type, double b1 = 0.8, double b2 = 0.9) {
    thread_size     = thd_size;
    learning_rate   = lr;
    lambda          = lamb;

    enable_optimizer    = opt_type != Optimizer::TYPE::NONE;
    optimizer_type      = opt_type;

    switch (optimizer_type) {
        case Optimizer::TYPE::NONE:
            break;
        case Optimizer::TYPE::ADAM:
            SetADAM(b1, b2);
    }

    if(thread_size) {
        section_size    = batch_size % thread_size == 0 ? batch_size / thread_size : batch_size / thread_size +1;
    }
}
//============================================================================================================
void MLP::ForwardBatchPropagation(const size_t &begin, const size_t &end) {
    for(size_t i = begin; i <= end; i++) {
        for(size_t j = 0; j < neural_number_of_layer.size() - 1; j++) {
            cv::Mat product = mlp_layer_weight_values[j] * mlp_layer_values[i][j] + mlp_layer_bias_values[j];

            if(j == neural_number_of_layer.size() - 2) {
                mlp_layer_values[i][j + 1]      = product;
            } else {
                mlp_layer_values[i][j + 1]      = ActivationFunction(product);
            }
        }

        output_value[i]         = OutputFunction(mlp_layer_values[i][mlp_layer_values[i].size() - 1]);
        output_loss_value[i]    = LossFunction(output_value[i], label_value[i]) + Machine_Learning_Math::L2Regression(mlp_layer_weight_values, lambda);
    }
}
//============================================================================================================
void MLP::Backward() {
    UpdateNeuralNet();
    optimizer->PlusADAMCount();
}
//============================================================================================================
void MLP::SetADAM(double b1, double b2) const {
    if(optimizer) {
        optimizer->SetType(Optimizer::TYPE::ADAM);
        optimizer->SetADAM(neural_number_of_layer, b1, b2);
        optimizer->SetOptimizer();
    }
}
//============================================================================================================
void MLP::UpdateNeuralNet() {
    const int   vec_size    = static_cast<int>(neural_number_of_layer.size());

    for(int i = vec_size - 2; i >= 0; i--) {
        if(i == vec_size - 2) {
            DerivativeOutputLayer();
        } else {
            DerivativeHiddenLayer(i + 1);
        }

//        boost::thread   thd_weight(&MLP::DerivativeCalWeight, this, i);
//        boost::thread   thd_bias(&MLP::DerivativeCalBias, this, i);
        DerivativeCalWeight(i);
        DerivativeCalBias(i);
        boost::thread   thd_layer(&MLP::DerivativeCalLayer, this, i);

//        thd_weight.join();
//        thd_bias.join();

        switch (optimizer_type) {
            case Optimizer::TYPE::NONE:
                mlp_layer_weight_values[i]  -= learning_rate * update_layer_weight_values[i];
                mlp_layer_bias_values[i]    -= learning_rate * update_layer_bias_values[i];
                break;
            case Optimizer::TYPE::ADAM:
                optimizer->UpdateADAM(update_layer_weight_values, update_layer_bias_values, i);
                cv::Mat     weight_result   = optimizer->GetWeightResult();
                cv::Mat     bias_result     = optimizer->GetBiasResult();
                mlp_layer_weight_values[i]  -= learning_rate * weight_result;
                mlp_layer_bias_values[i]    -= learning_rate * bias_result;
                break;
        }

        thd_layer.join();
    }
}
//============================================================================================================
void MLP::DerivativeOutputLayer() {
    for(int i = 0; i < thread_size; i++) {
        size_t  begin   = i * section_size;
        size_t  end     = (i + 1) * section_size - 1;

        end         = end > batch_size ? batch_size - 1 : end;

        vec_thread.emplace_back(std::make_unique<boost::thread>(&MLP::DerivativeOutputLayerBatch, this, begin, end));
    }

    for(const auto &thd : vec_thread) {
        if(thd && thd->joinable())
            thd->join();
    }

    vec_thread.clear();
}
//============================================================================================================
void MLP::DerivativeOutputLayerBatch(const size_t &begin, const size_t &end) {
    const size_t    layer_idx   = neural_number_of_layer.size() - 1;

    for(size_t i = begin; i <= end; i++) {
        DerivativeOutputFunction(output_value[i], label_value[i], &update_layer_values[i][layer_idx]);
    }
}
//============================================================================================================
void MLP::DerivativeHiddenLayer(const size_t &idx) {
    for(int i = 0; i < thread_size; i++) {
        size_t  begin   = i * section_size;
        size_t  end     = (i + 1) * section_size - 1;

        end         = end > batch_size ? batch_size - 1 : end;

        vec_thread.emplace_back(std::make_unique<boost::thread>(&MLP::DerivativeHiddenLayerBatch,
                                                                this,
                                                                idx,
                                                                begin,
                                                                end));
    }

    for(const auto &thd : vec_thread) {
        if(thd && thd->joinable())
            thd->join();
    }

    vec_thread.clear();
}
//============================================================================================================
void MLP::DerivativeHiddenLayerBatch(const size_t &idx, const size_t &begin, const size_t &end) {
    for(size_t i = begin; i <= end; i++) {
        cv::Mat mat;
        DerivativeActivationFunction(mlp_layer_values[i][idx], mat);
        update_layer_values[i][idx] = mat.mul(update_layer_values[i][idx]);
    }
}
//============================================================================================================
void MLP::DerivativeCalWeight(const size_t &idx) {
    std::vector<cv::Mat>    temp_mat(batch_size);
    std::vector<cv::Mat>    temp_sum(thread_size);
    cv::Mat                 sum;

    for(int i = 0; i < thread_size; i++) {
        size_t  begin   = i * section_size;
        size_t  end     = (i + 1) * section_size - 1;

        end         = end > batch_size ? batch_size - 1 : end;

        vec_thread.emplace_back(std::make_unique<boost::thread>(
                [this, &temp_mat, begin, end, capture0 = idx + 1, idx] {
                    CalVecMatProduct(temp_mat, begin, end, capture0, idx);
                }));
    }

    for(const auto &thd : vec_thread) {
        if(thd && thd->joinable())
            thd->join();
    }

    vec_thread.clear();

    for(int i = 0; i < thread_size; i++) {
        size_t begin = i * section_size;
        size_t end = (i + 1) * section_size - 1;

        end = end > batch_size ? batch_size - 1 : end;

        vec_thread.emplace_back(std::make_unique<boost::thread>(&MLP::CalVecMatSum,
                                                                this,
                                                                temp_mat,
                                                                &temp_sum[i],
                                                                begin,
                                                                end));
    }

    for(const auto &thd : vec_thread) {
        if(thd && thd->joinable())
            thd->join();
    }

    vec_thread.clear();

    CalVecMatSum(temp_sum, &sum, 0, thread_size - 1);
    CalMatAverage(sum, &update_layer_weight_values[idx], batch_size);

    update_layer_weight_values[idx] += DerivativeL2Regression(mlp_layer_weight_values[idx]);
}
//============================================================================================================
void MLP::CalVecMatProduct(std::vector<cv::Mat> &temp, const size_t &begin, const size_t &end, const size_t &idx1, const size_t &idx2) {
    for(size_t i = begin; i <= end; i++) {
        std::unique_lock<std::mutex>    lock(mtx_cal_vec_mat_product);
        cv::Mat     product = update_layer_values[i][idx1] * mlp_layer_values[i][idx2].t();
        temp[i]             = std::move(product);
        lock.unlock();
    }
}
//============================================================================================================
void MLP::CalVecMatSum(const std::vector<cv::Mat> &source, cv::Mat *destination, const size_t &begin, const size_t &end) {
    *destination     = cv::Mat::zeros(source[0].rows, source[0].cols, CV_32FC1);

    for(size_t i = begin; i <= end; i++) {
        *destination += source[i];
    }
}
//============================================================================================================
void MLP::CalMatAverage(const cv::Mat &source, cv::Mat *destination, const size_t &size) {
    if(size) {
        *destination = source / static_cast<double>(size);
    }
}
//============================================================================================================
cv::Mat MLP::DerivativeL2Regression(const cv::Mat &weight) const {
    return 2.0 * lambda * weight;
}
//============================================================================================================
void MLP::DerivativeCalBias(const size_t &idx) {
    std::vector<cv::Mat> temp_sum(thread_size);
    std::vector<cv::Mat> temp_layer(batch_size);
    cv::Mat sum;

    for (int i = 0; i < batch_size; i++) {
        update_layer_values[i][idx + 1].copyTo(temp_layer[i]);
    }

    for (int i = 0; i < thread_size; i++) {
        size_t begin = i * section_size;
        size_t end = (i + 1) * section_size - 1;

        end = end > batch_size ? batch_size - 1 : end;

        vec_thread.emplace_back(std::make_unique<boost::thread>(&MLP::CalVecMatSum,
                                                                this,
                                                                temp_layer,
                                                                &temp_sum[i],
                                                                begin,
                                                                end));
    }

    for (const auto &thd: vec_thread) {
        if (thd && thd->joinable())
            thd->join();
    }

    vec_thread.clear();

    CalVecMatSum(temp_sum, &sum, 0, thread_size - 1);
    CalMatAverage(sum, &update_layer_bias_values[idx], batch_size);
}
//============================================================================================================
void MLP::DerivativeCalLayer(const size_t &idx) {
    std::vector<cv::Mat>    temp_sum(thread_size);
    cv::Mat                 sum;

    for (int i = 0; i < thread_size; i++) {
        size_t begin = i * section_size;
        size_t end = (i + 1) * section_size - 1;

        end = end > batch_size ? batch_size - 1 : end;

        vec_thread.emplace_back(std::make_unique<boost::thread>(&MLP::CalWeightLayerProduct,
                                                                this,
                                                                begin,
                                                                end,
                                                                idx,
                                                                idx + 1));
    }
}
//============================================================================================================
void MLP::CalWeightLayerProduct(const size_t &begin, const size_t &end, const size_t &idx1, const size_t &idx2) {
    for(size_t i = begin; i <= end; i++) {
        update_layer_values[i][idx1]    = mlp_layer_weight_values[idx1].t() * update_layer_values[i][idx2];
    }
}
//============================================================================================================
std::vector<int> MLP::GetNeuralNumberOfLayer() const {
    return neural_number_of_layer;
}
//============================================================================================================
std::vector<cv::Mat> MLP::GetMLPLayerWeight() const {
    return mlp_layer_weight_values;
}
//============================================================================================================
std::vector<cv::Mat> MLP::GetMLPLayerBias() const {
    return mlp_layer_bias_values;
}
//============================================================================================================
void MLP::InitNeuralNetLayer(std::vector<cv::Mat> &layer) {
    layer.resize(neural_number_of_layer.size());

    for (int i = 0; i < layer.size(); i++) {
        layer[i].create(neural_number_of_layer[i], 1, CV_32FC1);
    }
}
//============================================================================================================
void MLP::AssignWeight(const std::vector<cv::Mat> &weight) {
    mlp_layer_weight_values     = weight;
}
//============================================================================================================
void MLP::AssignBias(const std::vector<cv::Mat> &bias) {
    mlp_layer_bias_values       = bias;
}
//============================================================================================================
void MLP::AssignLayer(const std::vector<cv::Mat> &layer) {
    mlp_layer_value_predict     = layer;
}
//============================================================================================================
void MLP::AssignNeuralNumberOfLayer(const std::vector<int> &number) {
    neural_number_of_layer      = number;
}
//============================================================================================================
size_t MLP::GetThreadSize() const {
    return thread_size;
}
//============================================================================================================
size_t MLP::GetSectionSize() const {
    return section_size;
}
//============================================================================================================
size_t MLP::GetBatchSize() const {
    return batch_size;
}
//============================================================================================================
size_t MLP::GetNumberOfInputLayer() const {
    return neural_number_of_layer[0];
}
//============================================================================================================
void MLP::SetTrainingInputLayer(const size_t &idx, cv::Mat &&input) {
    mlp_layer_values[idx][0]    = input;
}
//============================================================================================================
size_t MLP::GetNumberOfOutputLayer() const {
    return neural_number_of_layer[neural_number_of_layer.size() - 1];
}
//============================================================================================================
void MLP::SetTrainingLabel(const size_t &idx, cv::Mat &&input) {
    label_value[idx]            = input;
}
//============================================================================================================
double MLP::GetLossValue() const {
    double      sum  = 0;
    for(auto value : output_loss_value) {
        sum += value;
    }
    return sum / static_cast<double>(output_loss_value.size());
}
//============================================================================================================
void MLP::SetInputLayer(cv::Mat &&input) {
    mlp_layer_value_predict[0]  = input;
}
//============================================================================================================
void MLP::Forward() {
    for(size_t i = 0; i < neural_number_of_layer.size() - 1; i++) {
        cv::Mat product = mlp_layer_weight_values[i] * mlp_layer_value_predict[i] + mlp_layer_bias_values[i];

        if(i == neural_number_of_layer.size() - 2) {
            mlp_layer_value_predict[i + 1]  = product;
        } else {
            mlp_layer_value_predict[i + 1]  = ActivationFunction(product);
        }
    }

    output_predict  = OutputFunction(mlp_layer_value_predict[mlp_layer_value_predict.size() - 1]);
}
//============================================================================================================
cv::Mat MLP::GetPredictOutput() const {
    return output_predict;
}
//============================================================================================================
void MLP::InitTrainNeuralNet(const size_t &size) {
    batch_size      = size;

    label_value.clear();
    output_value.clear();
    output_loss_value.clear();

    InitNeuralNetLayer(mlp_layer_values);
    label_value.resize(batch_size);
    output_value.resize(batch_size);
    output_loss_value.resize(batch_size);
    InitNeuralNetLayer(update_layer_values);
    InitNeuralNetWeightBias(update_layer_weight_values, update_layer_bias_values);
}
//============================================================================================================
void MLP::ShowMat(const cv::Mat &input) {
    for(int i = 0; i < input.rows; i++) {
        for(int j = 0; j < input.cols; j++) {
            std::cout << input.at<float>(i, j) << " ";
        }
        std::cout << std::endl;
    }
}
//============================================================================================================