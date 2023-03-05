//
// Created by howlie190 on 2023/1/2.
//

#include "MLP.h"
#include <opencv2/opencv.hpp>
#include <random>

void MLP::SetInputLayer(cv::Mat mat) { _layer[0] = mat; }

cv::Mat MLP::GetOutputLayer() { return _output; }

std::vector<cv::Mat> MLP::GetWeights() { return _weights; }

std::vector<cv::Mat> MLP::GetBias() { return _bias; }

std::vector<int> MLP::GetLayerNeuralNumber() { return _layerNeuralNumber; }

void MLP::LoadLayerNeuralNumber(std::vector<int> layerNeuralNumber) { _layerNeuralNumber = layerNeuralNumber; }

void MLP::LoadWeights(std::vector<cv::Mat> weights) { _weights = weights; }

void MLP::LoadBias(std::vector<cv::Mat> bias) { _bias = bias; }

void MLP::LoadLayer(std::vector<cv::Mat> layer) { _layer = layer; }

int MLP::GetNumberOfInputLayerNodes() { return _layerNeuralNumber[0]; }

int MLP::GetNumberOfOutputLayerNodes() { return _layerNeuralNumber[_layerNeuralNumber.size() - 1]; }

MLP::MLP() {
    _pActivationFunction = nullptr;
    _pOutputActivationFunction = nullptr;
    _pLossFunction = nullptr;
    _pDerivativeOutputFunction = nullptr;
    _loss = 0.0;
    _learningRate = 0.0;
    _lambda = 0.0;
}

void MLP::SetLearningRate(double rate) { _learningRate = rate; }

void MLP::SetLambda(double lambda) { _lambda = lambda; }

void MLP::SetTarget(cv::Mat mat) { _target = mat; }

double MLP::GetLoss() { return _loss; }

void MLP::InitNeuralNet(std::vector<int> layerNeuralNumber) {
    _layerNeuralNumber = layerNeuralNumber;
    InitNeuralNetHelper(_layer, _weights, _bias);
}

void
MLP::InitNeuralNetHelper(std::vector<cv::Mat> &layer, std::vector<cv::Mat> &weights, std::vector<cv::Mat> &bias) {
    layer.resize(_layerNeuralNumber.size());
    for (int i = 0; i < layer.size(); i++) {
        layer[i].create(_layerNeuralNumber[i], 1, CV_32FC1);
    }
    weights.resize(_layerNeuralNumber.size() - 1);
    bias.resize(_layerNeuralNumber.size() - 1);
    for (int i = 0; i < _layerNeuralNumber.size() - 1; i++) {
        weights[i].create(layer[i + 1].rows, layer[i].rows, CV_32FC1);
        bias[i] = cv::Mat::zeros(layer[i + 1].rows, 1, CV_32FC1);
    }
}

void MLP::InitWeightHelper(DISTRIBUTION distribution, cv::Mat &mat, double mean, double standardDeviation) {
    std::random_device rd;
    cv::theRNG().state = rd();
    switch (distribution) {
        case DISTRIBUTION::NORMAL:
            cv::randn(mat, mean, standardDeviation);
            break;
        case DISTRIBUTION::UNIFORM:
            cv::randu(mat, mean, standardDeviation);
            break;
    }
}

void MLP::InitWeights(DISTRIBUTION distribution, double mean, double standardDeviation) {
    for (int i = 0; i < _weights.size(); i++) {
        InitWeightHelper(distribution, _weights[i], mean, standardDeviation);
    }
}

void MLP::InitBias(cv::Scalar &&scalar) {
    for (int i = 0; i < _bias.size(); i++) {
        _bias[i] = scalar;
    }
}

void MLP::ForwardPropagation(bool training) {
    for (int i = 0; i < _layerNeuralNumber.size() - 1; i++) {
        cv::Mat product = _weights[i] * _layer[i] + _bias[i];
        if (i == _layerNeuralNumber.size() - 2) {
            _layer[i + 1] = product;
        } else {
            _layer[i + 1] = _pActivationFunction(product);
        }
    }

    _output = _pOutputActivationFunction(_layer[_layer.size() - 1]);

    if (training) {
        _loss = _pLossFunction(_output, _target);
    }
}

void MLP::BackPropagation() {
    UpdateWeights();
}

void MLP::UpdateWeights() {
    std::vector<cv::Mat> layer, weights, bias;
    InitNeuralNetHelper(layer, weights, bias);
    for (int i = layer.size() - 2; i >= 0; i--) {
        if (i == layer.size() - 2) {
            _pDerivativeOutputFunction(_output, _target, layer[i + 1]);
        } else {
            cv::Mat derivative = cv::Mat::zeros(layer[i + 1].rows, 1, CV_32FC1);
            _pDerivativeFunction(_layer[i + 1], derivative);
            layer[i + 1] = derivative.mul(layer[i + 1]);
        }
        weights[i] = layer[i + 1] * _layer[i].t() + DerivativeL2Regression(_weights[i]);
        bias[i] = layer[i + 1];
        layer[i] = _weights[i].t() * layer[i + 1];
        _weights[i] -= (_learningRate * weights[i]);
        _bias[i] -= (_learningRate * bias[i]);
    }
}

double MLP::L2Regression(std::vector<cv::Mat> weights) {
    cv::Mat squared;
    cv::Scalar sum = 0;
    for (int i = 0; i < weights.size(); i++) {
        cv::pow(weights[i], 2, squared);
        sum += cv::sum(squared);
    }
    return _lambda * sum[0];
}

cv::Mat MLP::DerivativeL2Regression(cv::Mat weights) {
    return 2 * _lambda * weights;
}

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

void MLP::SetLossFunction(LOSS_FUNCTION lossFunction) {
    switch (lossFunction) {
        case LOSS_FUNCTION::MEAN_SQUARED_ERROR:
            _pLossFunction = MeanSquaredError;
            break;
    }
}

void MLP::SetDerivativeOutputFunction(DERIVATIVE_FUNCTION derivativeOutputFunction) {
    switch (derivativeOutputFunction) {
        case DERIVATIVE_FUNCTION::SIGMOID_MSE:
            _pDerivativeOutputFunction = DerivativeSigmoidMSE;
            break;
        case DERIVATIVE_FUNCTION::SOFTMAX_MSE:
            _pDerivativeOutputFunction = DerivativeSoftmaxMSE;
    }
}

void MLP::SetDerivativeFunction(DERIVATIVE_FUNCTION derivativeFunction) {
    switch (derivativeFunction) {
        case DERIVATIVE_FUNCTION::SIGMOID:
            _pDerivativeFunction = DerivativeSigmoid;
    }
}
