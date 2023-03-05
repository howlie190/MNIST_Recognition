//
// Created by howli on 2023/1/2.
//

#ifndef MLP_MLP_H
#define MLP_MLP_H

#include "Define.h"
#include <vector>
#include <tchar.h>

#ifdef MNIST_Recognition_Library_EXPORTS
#define MLP_API __declspec(dllexport)
#else
#define MLP_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

class MLP_API MLP {
public:
    MLP();

    ~MLP() {}

    void                    InitNeuralNet(std::vector<int> layerNeuralNumber);

    void                    InitWeights(DISTRIBUTION distribution, double mean, double standardDeviation);

    void                    InitBias(cv::Scalar &&scalar);

    void                    SetActivationFunction(ACTIVATION_FUNCTION activationFunction);

    void                    SetOutputActivationFunction(ACTIVATION_FUNCTION outputActivationFunction);

    void                    SetLossFunction(LOSS_FUNCTION lossFunction);

    void                    SetDerivativeOutputFunction(DERIVATIVE_FUNCTION derivativeOutputFunction);

    void                    SetDerivativeFunction(DERIVATIVE_FUNCTION derivativeFunction);

    void                    SetLearningRate(double rate);

    void                    SetLambda(double lambda);

    void                    SetTarget(cv::Mat mat);

    double                  GetLoss();

    virtual void            Train(char *path) = 0;

    virtual double          Test(char *path) = 0;

    virtual bool            Save(char *path, char *name, bool override) = 0;

    virtual bool            Load(char *path) = 0;

    virtual void            SetInput(cv::Mat mat) = 0;

protected:
    void                    InitWeightHelper(DISTRIBUTION distribution, cv::Mat &mat, double mean, double standardDeviation);

    void                    UpdateWeights();

    void                    InitNeuralNetHelper(std::vector<cv::Mat> &layer, std::vector<cv::Mat> &weights, std::vector<cv::Mat> &bias);

    void                    ForwardPropagation(bool training = false);

    void                    SetInputLayer(cv::Mat mat);

    void                    BackPropagation();

    double                  L2Regression(std::vector<cv::Mat> weights);

    cv::Mat                 DerivativeL2Regression(cv::Mat weights);

    cv::Mat                 GetOutputLayer();

    std::vector<cv::Mat>    GetWeights();

    std::vector<cv::Mat>    GetBias();

    std::vector<int>        GetLayerNeuralNumber();

    void                    LoadLayerNeuralNumber(std::vector<int> layerNeuralNumber);

    void                    LoadWeights(std::vector<cv::Mat> weights);

    void                    LoadBias(std::vector<cv::Mat> bias);

    void                    LoadLayer(std::vector<cv::Mat> layer);

    int                     GetNumberOfInputLayerNodes();

    int                     GetNumberOfOutputLayerNodes();

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

#ifdef __cplusplus
}
#endif

#endif //MLP_MLP_H
