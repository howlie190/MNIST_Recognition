//
// Created by howli on 2023/1/2.
//

#ifndef MLP_MLP_H
#define MLP_MLP_H

#include "Define.h"
#include <vector>
#include <tchar.h>

class MLP {
public:
    MLP() {
        _pActivationFunction        = nullptr;
        _pOutputActivationFunction  = nullptr;
        _pLossFunction              = nullptr;
        _pDerivativeOutputFunction  = nullptr;
        _loss                       = 0.0;
        _learningRate               = 0.0;
        _lambda                     = 0.0;
    }
    ~MLP() {}
    void                InitNeuralNet(std::vector<int> layerNeuralNumber);
    void                InitWeights(DISTRIBUTION distribution, double mean, double standardDeviation);
    void                InitBias(cv::Scalar &&scalar);
    void                SetActivationFunction(ACTIVATION_FUNCTION activationFunction);
    void                SetOutputActivationFunction(ACTIVATION_FUNCTION outputActivationFunction);
    void                SetLossFunction(LOSS_FUNCTION lossFunction);
    void                SetDerivativeOutputFunction(DERIVATIVE_FUNCTION derivativeOutputFunction);
    void                SetDerivativeFunction(DERIVATIVE_FUNCTION derivativeFunction);
    void                SetLearningRate(double rate) { _learningRate = rate; }
    void                SetLambda(double lambda) { _lambda = lambda; }
    void                SetTarget(cv::Mat mat) { _target = mat; };
    double              GetLoss() { return _loss; }
    virtual void        Train(char* path) = 0;
    virtual void        Test(char* path) = 0;
    virtual void        Save(char* path) = 0;
    virtual void        Load(char* path) = 0;
    virtual void        SetInput(cv::Mat mat) = 0;
protected:
    void                InitWeightHelper(DISTRIBUTION distribution, cv::Mat &mat, double mean, double standardDeviation);
    void                UpdateWeights();
    void                InitNeuralNetHelper(std::vector<cv::Mat>& layer, std::vector<cv::Mat>& weights, std::vector<cv::Mat>& bias);
    void                ForwardPropagation();
    void                SetInputLayer(cv::Mat mat) { _layer[0] = mat; }
    void                BackPropagation();
    double              L2Regression(std::vector<cv::Mat> weights);
    cv::Mat             DerivativeL2Regression(cv::Mat weights);
    cv::Mat             GetOutputLayer() { return _output; }
private:
    std::vector<int>            _layerNeuralNumber;
    std::vector<cv::Mat>        _layer;
    std::vector<cv::Mat>        _weights;
    std::vector<cv::Mat>        _bias;
    cv::Mat                     _output;
    pActivationFunction         _pActivationFunction;
    pActivationFunction         _pOutputActivationFunction;
    pDerivativeOutputFunction   _pDerivativeOutputFunction;
    pLossFunction               _pLossFunction;
    pDerivativeFunction         _pDerivativeFunction;
    cv::Mat                     _target;
    double                      _loss;
    double                      _learningRate;
    double                      _lambda;
};


#endif //MLP_MLP_H
