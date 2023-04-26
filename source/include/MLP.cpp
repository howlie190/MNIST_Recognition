//
// Created by howlie190 on 2023/1/2.
//

#include "MLP.h"
#include <opencv2/opencv.hpp>
#include <random>

MLP::MLP() {
    _pActivationFunction        = nullptr;
    _pOutputActivationFunction  = nullptr;
    _pLossFunction              = nullptr;
    _pDerivativeOutputFunction  = nullptr;
    _learningRate               = 0.0;
    _lambda                     = 0.0;
    _miniBatchSize              = 1;
}
//============================================================================================================
MLP::~MLP() {
    for(int i = 0; i < _vecThread.size(); i++) {
        _vecThread[i].interrupt();
    }

    _vecThread.clear();
}
//============================================================================================================
void MLP::InitNeuralNet(std::vector<int> layerNeuralNumber) {
    _layerNeuralNumber = layerNeuralNumber;
    InitNeuralNetHelper(_weights, _bias);
}
//============================================================================================================
void MLP::InitWeights(DISTRIBUTION distribution, double mean, double standardDeviation) {
    for (int i = 0; i < _weights.size(); i++) {
        InitWeightHelper(distribution, _weights[i], mean, standardDeviation);
    }
}
//============================================================================================================
void MLP::InitBias(cv::Scalar &&scalar) {
    for (int i = 0; i < _bias.size(); i++) {
        _bias[i] = scalar;
    }
}
//============================================================================================================
void MLP::SetActivationFunction(ACTIVATION_FUNCTION activationFunction) {
    switch (activationFunction) {
        case ACTIVATION_FUNCTION::RELU:
            _pActivationFunction = ReLU;
            break;

        case ACTIVATION_FUNCTION::TANH:
            _pActivationFunction = Tanh;
            break;

        case ACTIVATION_FUNCTION::SIGMOID:
            _pActivationFunction = Sigmoid;
            break;
    }
}
//============================================================================================================
void MLP::SetOutputActivationFunction(ACTIVATION_FUNCTION outputActivationFunction) {
    switch (outputActivationFunction) {
        case ACTIVATION_FUNCTION::RELU:
            _pOutputActivationFunction = ReLU;
            break;

        case ACTIVATION_FUNCTION::SIGMOID:
            _pOutputActivationFunction = Sigmoid;
            break;

        case ACTIVATION_FUNCTION::TANH:
            _pOutputActivationFunction = Tanh;
            break;

        case ACTIVATION_FUNCTION::SOFTMAX:
            _pOutputActivationFunction = Softmax;
            break;
    }
}
//============================================================================================================
void MLP::SetLossFunction(LOSS_FUNCTION lossFunction) {
    switch (lossFunction) {
        case LOSS_FUNCTION::MEAN_SQUARED_ERROR:
            _pLossFunction = MeanSquaredError;
            break;
        case LOSS_FUNCTION::CROSS_ENTROPY:
            _pLossFunction = CrossEntropy;
            break;
    }
}
//============================================================================================================
void MLP::SetDerivativeOutputFunction(DERIVATIVE_FUNCTION derivativeOutputFunction) {
    switch (derivativeOutputFunction) {
        case DERIVATIVE_FUNCTION::SIGMOID_MSE:
            _pDerivativeOutputFunction = DerivativeSigmoidMSE;
            break;
        case DERIVATIVE_FUNCTION::SOFTMAX_MSE:
            _pDerivativeOutputFunction = DerivativeSoftmaxMSE;
            break;
        case DERIVATIVE_FUNCTION::SOFTMAX_CROSS_ENTROPY:
            _pDerivativeOutputFunction = DerivativeSoftmaxCrossEntropy;
            break;
        case DERIVATIVE_FUNCTION::SIGMOID_CROSS_ENTROPY:
            _pDerivativeOutputFunction = DerivativeSigmoidCrossEntropy;
            break;
    }
}
//============================================================================================================
void MLP::SetDerivativeFunction(DERIVATIVE_FUNCTION derivativeFunction) {
    switch (derivativeFunction) {
        case DERIVATIVE_FUNCTION::SIGMOID:
            _pDerivativeFunction = DerivativeSigmoid;
            break;
        case DERIVATIVE_FUNCTION::TANH:
            _pDerivativeFunction = DerivativeTanh;
            break;
        case DERIVATIVE_FUNCTION::RELU:
            _pDerivativeFunction = DerivativeReLU;
            break;
    }
}
//============================================================================================================
void MLP::SetLearningRate(double rate) {
    _learningRate = rate;
}
//============================================================================================================
void MLP::SetLambda(double lambda) {
    _lambda = lambda;
}
//============================================================================================================
void MLP::SetTargetTraining(int idx, const cv::Mat& target) {
    _targetThreading[idx] = target;
}
//============================================================================================================
void MLP::SetTarget(const cv::Mat& target) {
    _target = target;
}
//============================================================================================================
void MLP::SetMiniBatchSize(size_t size) {
    for(int i = 0; i < _vecThread.size(); i++) {
        _vecThread[i].interrupt();
    }

    _miniBatchSize = size;

    _vecThread.clear();
    _lossThreading.clear();
    _outputThreading.clear();
    _layerThreading.clear();
    _targetThreading.clear();

    _vecThread.resize(_miniBatchSize);
    _lossThreading.resize(_miniBatchSize);
    _outputThreading.resize(_miniBatchSize);
    _targetThreading.resize(_miniBatchSize);

    InitNeuralLayerHelper(_layerThreading);
}
//============================================================================================================
double MLP::GetLoss() {
    double total = 0;
    for(auto value : _lossThreading) {
        total += value;
    }
    return total / _lossThreading.size();
}
//============================================================================================================
void MLP::InitNeuralNetHelper(std::vector<cv::Mat>& weights, std::vector<cv::Mat>& bias) {
    InitNeuralLayerHelper(_layer);

    weights.resize(_layerNeuralNumber.size() - 1);
    bias.resize(_layerNeuralNumber.size() - 1);

    for (int i = 0; i < _layerNeuralNumber.size() - 1; i++) {
        weights[i].create(_layerNeuralNumber[i + 1], _layerNeuralNumber[i], CV_32FC1);
        bias[i] = cv::Mat::zeros(_layerNeuralNumber[i + 1], 1, CV_32FC1);
    }
}
//============================================================================================================
void MLP::InitNeuralLayerHelper(std::vector<std::vector<cv::Mat>>& layer) {
    layer.resize(_miniBatchSize);

    for(int i = 0; i < _miniBatchSize; i++) {
        layer[i].resize(_layerNeuralNumber.size());

        for(int j = 0; j < layer[i].size(); j++) {
            layer[i][j].create(_layerNeuralNumber[j], 1, CV_32FC1);
        }
    }
}
//============================================================================================================
void MLP::InitNeuralLayerHelper(std::vector<cv::Mat>& layer) {
    layer.resize(_layerNeuralNumber.size());

    for (int i = 0; i < layer.size(); i++) {
        layer[i].create(_layerNeuralNumber[i], 1, CV_32FC1);
    }
}
//============================================================================================================
void MLP::InitWeightHelper(DISTRIBUTION distribution, cv::Mat& destination, double mean, double standardDeviation) {
    std::random_device rd;
    cv::theRNG().state = rd();

    switch (distribution) {
        case DISTRIBUTION::NORMAL:
            cv::randn(destination, mean, standardDeviation);
            break;
        case DISTRIBUTION::UNIFORM:
            cv::randu(destination, mean, standardDeviation);
            break;
    }
}
//============================================================================================================
void MLP::ForwardPropagation() {
    for (int i = 0; i < _layerNeuralNumber.size() - 1; i++) {
        cv::Mat product = _weights[i] * _layer[i] + _bias[i];

        if (i == _layerNeuralNumber.size() - 2) {
            _layer[i + 1] = product;
        } else {
            _layer[i + 1] = _pActivationFunction(product);
        }
    }

    _output = _pOutputActivationFunction(_layer[_layer.size() - 1]);
}
//============================================================================================================
void MLP::TrainingForwardPropagation() {
    for(int i = 0; i < _miniBatchSize; i++) {
        _vecThread[i]   = boost::thread(&MLP::TrainingForwardPropagationHelper, this, i);
    }

    for(int i = 0; i < _miniBatchSize; i++) {
        _vecThread[i].join();
    }
}
//============================================================================================================
void MLP::TrainingForwardPropagationHelper(const int idx) {
    for (int i = 0; i < _layerNeuralNumber.size() - 1; i++) {
        cv::Mat product = _weights[i] * _layerThreading[idx][i] + _bias[i];

        if (i == _layerNeuralNumber.size() - 2) {
            _layerThreading[idx][i + 1] = product;
        } else {
            _layerThreading[idx][i + 1] = _pActivationFunction(product);
        }
    }

    _outputThreading[idx]   = _pOutputActivationFunction(_layerThreading[idx][_layerThreading[idx].size() - 1]);

    _lossThreading[idx]     = _pLossFunction(_outputThreading[idx], _targetThreading[idx]) + L2Regression(_weights);
}
//============================================================================================================
void MLP::BackPropagation() {
    UpdateWeights();
}
//============================================================================================================
void MLP::UpdateWeights() {
    std::vector<std::vector<cv::Mat>>   layer;
    std::vector<cv::Mat>                weights;
    std::vector<cv::Mat>                bias;

    InitNeuralNetHelper(weights, bias);
    InitNeuralLayerHelper(layer);

    for (int i = _layerNeuralNumber.size() - 2; i >= 0; i--) {
        if (i == _layerNeuralNumber.size() - 2) {
            DerivativeOutputFunction(layer);
        } else {
            DerivativeFunction(i + 1, layer);
        }

        CalWeights(i, layer, weights[i]);
        CalBias(i + 1, layer, bias[i]);
        CalLayer(i, &layer);

        _weights[i] -= (_learningRate * weights[i]);
        _bias[i] -= (_learningRate * bias[i]);
    }
}
//============================================================================================================
void MLP::DerivativeOutputFunction(std::vector<std::vector<cv::Mat>>& destination) {
    std::vector<boost::thread>  vecThread(_miniBatchSize);

    for(int i = 0; i < _miniBatchSize; i++) {
        vecThread[i] = boost::thread(_pDerivativeOutputFunction,
                                     _outputThreading[i],
                                     _targetThreading[i],
                                     &destination[i][destination[i].size() - 1]);
    }

    for(int i = 0; i < vecThread.size(); i++) {
        vecThread[i].join();
    }
}
//============================================================================================================
void MLP::DerivativeFunction(const int idx, std::vector<std::vector<cv::Mat>>& destination) {
    std::vector<boost::thread>  vecThread(_miniBatchSize);
    std::vector<cv::Mat>        vecMat(_miniBatchSize);

    for(int i = 0; i < _miniBatchSize; i++) {
        vecThread[i] = boost::thread(_pDerivativeFunction,
                                     _layerThreading[i][idx],
                                     &vecMat[i]);
    }

    for(int i = 0; i < _miniBatchSize; i++) {
        vecThread[i].join();
    }

    for(int i = 0; i < _miniBatchSize; i++) {
        destination[i][idx] = vecMat[i].mul(destination[i][idx]);
    }
}
//============================================================================================================
void MLP::CalWeights(const int idx,
                     const std::vector<std::vector<cv::Mat>>& layer,
                     cv::Mat& destination) {
    std::vector<boost::thread>  vecThread;
    std::vector<cv::Mat>        vecMat;
    std::vector<cv::Mat>        threadTotal;
    cv::Mat                     total;
    int                         calSize =   _miniBatchSize % 10 == 0
                                            ? _miniBatchSize / 10
                                            : _miniBatchSize / 10 + 1;

    vecMat.resize(_miniBatchSize);
    vecThread.resize(calSize);
    threadTotal.resize(calSize);

    for(int i = 0; i < calSize; i++) {
        int begin   = i * 10;
        int end     = (i + 1) * 10 - 1;

        end = end > _miniBatchSize ? _miniBatchSize % 10 - 1 : end;

        boost::thread::attributes attrs;
        attrs.set_stack_size(256 * 1024 * 1024);

        vecThread[i] = boost::thread(boost::bind(CalVecMatProduct,
                                     layer,
                                     _layerThreading,
                                     &vecMat,
                                     begin,
                                     end,
                                     idx + 1,
                                     idx));
    }

    for(int i = 0; i < calSize; i++) {
        vecThread[i].join();
    }

    for(int i = 0; i < calSize; i++) {
        int begin   = i * 10;
        int end     = (i + 1) * 10 - 1;

        end = end > _miniBatchSize ? _miniBatchSize % 10 - 1 : end;

        vecThread[i] = boost::thread(CalMatTotal,
                                     vecMat,
                                     &threadTotal[i],
                                     begin,
                                     end);
    }

    for(int i = 0; i < calSize; i++) {
        vecThread[i].join();
    }

    CalMatTotal(threadTotal, &total, 0, calSize - 1);
    CalMatAvg(total, _miniBatchSize, &destination);

    destination += DerivativeL2Regression(_weights[idx]);
}
//============================================================================================================
void MLP::CalBias(const int idx,
                  const std::vector<std::vector<cv::Mat>>& layer,
                  cv::Mat& destination) {
    std::vector<boost::thread>  vecThread;
    std::vector<cv::Mat>        vecMat(_miniBatchSize);
    std::vector<cv::Mat>        layerTotal(_miniBatchSize);
    std::vector<cv::Mat>        threadTotal;
    cv::Mat                     total;
    cv::Mat                     avg;
    int                         calSize =   _miniBatchSize % 10 == 0
                                            ? _miniBatchSize / 10
                                            : _miniBatchSize / 10 + 1;
    vecThread.resize(calSize);
    threadTotal.resize(calSize);

    for(int i = 0; i < _miniBatchSize; i++) {
        layerTotal[i] = layer[i][idx];
    }

    for(int i = 0; i < calSize; i++) {
        int begin   = i * 10;
        int end     = (i + 1) * 10 - 1;

        end = end > _miniBatchSize ? _miniBatchSize % 10 - 1 : end;

        vecThread[i] = boost::thread(CalMatTotal,
                                     layerTotal,
                                     &threadTotal[i],
                                     begin,
                                     end);
    }

    for(int i = 0; i < calSize; i++) {
        vecThread[i].join();
    }

    CalMatTotal(threadTotal, &total, 0, calSize - 1);
    CalMatAvg(total, _miniBatchSize, &destination);
}
//============================================================================================================
void MLP::CalLayer(const int idx, std::vector<std::vector<cv::Mat>>* layer) {
    std::vector<boost::thread>          vecThread;
    std::vector<cv::Mat>                threadTotal;
    cv::Mat                             total;
    int                                 calSize =   _miniBatchSize % 10 == 0
                                                    ? _miniBatchSize / 10
                                                    : _miniBatchSize / 10 + 1;

    vecThread.resize(calSize);
    threadTotal.resize(calSize);

    for(int i = 0; i < calSize; i++) {
        int begin   = i * 10;
        int end     = (i + 1) * 10 - 1;

        end = end > _miniBatchSize ? _miniBatchSize % 10 - 1 : end;

        vecThread[i] = boost::thread(CalWeightLayerProduct,
                                     _weights,
                                     layer,
                                     begin,
                                     end,
                                     idx,
                                     idx + 1);
    }

    for(int i = 0; i < calSize; i++) {
        vecThread[i].join();
    }
}
//============================================================================================================
void MLP::SetInputLayerTraining(const int idx, const cv::Mat& mat) {
    _layerThreading[idx][0] = mat;
}
//============================================================================================================
void MLP::SetInputLayer(const cv::Mat& mat) {
    _layer[0] = mat;
}
//============================================================================================================
double MLP::L2Regression(const std::vector<cv::Mat>& weights) {
    cv::Mat squared;
    cv::Scalar sum = 0;

    for (int i = 0; i < weights.size(); i++) {
        cv::pow(weights[i], 2, squared);
        sum += cv::sum(squared);
    }

    return _lambda * sum[0];
}
//============================================================================================================
cv::Mat MLP::DerivativeL2Regression(const cv::Mat& weights) {
    return 2 * _lambda * weights;
}
//============================================================================================================
cv::Mat MLP::GetOutputLayer() {
    return _output;
}
//============================================================================================================
std::vector<cv::Mat> MLP::GetWeights() {
    return _weights;
}
//============================================================================================================
std::vector<cv::Mat> MLP::GetBias() {
    return _bias;
}
//============================================================================================================
std::vector<int> MLP::GetLayerNeuralNumber() {
    return _layerNeuralNumber;
}
//============================================================================================================
void MLP::LoadLayerNeuralNumber(const std::vector<int>& layerNeuralNumber) {
    _layerNeuralNumber = layerNeuralNumber;
}
//============================================================================================================
void MLP::LoadWeights(const std::vector<cv::Mat>& weights) {
    _weights = weights;
}
//============================================================================================================
void MLP::LoadBias(const std::vector<cv::Mat>& bias) {
    _bias = bias;
}
//============================================================================================================
void MLP::LoadLayer(const std::vector<cv::Mat>& layer) {
    _layer = layer;
}
//============================================================================================================
int MLP::GetNumberOfInputLayerNodes() {
    return _layerNeuralNumber[0];
}
//============================================================================================================
int MLP::GetNumberOfOutputLayerNodes() {
    return _layerNeuralNumber[_layerNeuralNumber.size() - 1];
}
//============================================================================================================
int MLP::GetMiniBatchSize() {
    return _miniBatchSize;
}
//============================================================================================================