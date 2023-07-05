//
// Created by howli on 2023/1/2.
//

#ifndef MLP_MLP_H
#define MLP_MLP_H

#include "Define.h"
#include <vector>
#include <tchar.h>
#include <boost/thread.hpp>

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
    ~MLP();

    void                        InitNeuralNet(std::vector<int>);
    void                        InitADAM();
    void                        InitWeights(DISTRIBUTION, double, double);
    void                        InitBias(cv::Scalar&&);
    void                        SetActivationFunction(ACTIVATION_FUNCTION);
    void                        SetOutputActivationFunction(ACTIVATION_FUNCTION);
    void                        SetLossFunction(LOSS_FUNCTION);
    void                        SetDerivativeOutputFunction(DERIVATIVE_FUNCTION);
    void                        SetDerivativeFunction(DERIVATIVE_FUNCTION);
    void                        SetLearningRate(double);
    void                        SetLambda(double);
    void                        SetTargetTraining(int, const cv::Mat&);
    void                        SetTarget(const cv::Mat&);
    void                        SetMiniBatchSize(size_t);

    double                      GetLoss();

    virtual std::vector<double> Train(char*) = 0;
    virtual void                SetInputTraining(const int, const cv::Mat&) = 0;
    virtual void                SetInput(const cv::Mat&) = 0;

    virtual double              Test(char*) = 0;

    virtual bool                Save(char*, char*, bool) = 0;
    virtual bool                Load(char*) = 0;

protected:
    void                    InitNeuralNetHelper(std::vector<cv::Mat>&, std::vector<cv::Mat>&);
    void                    InitNeuralLayerHelper(std::vector<std::vector<cv::Mat>>&);
    void                    InitNeuralLayerHelper(std::vector<cv::Mat>&);
    void                    InitWeightHelper(DISTRIBUTION, cv::Mat&, double, double);
    void                    ForwardPropagationMultiThread();
    void                    ForwardPropagationMultiThreadHelper(const int);
    void                    ForwardPropagation();
    void                    BackPropagation();
    void                    UpdateWeightsMultiThread();
    void                    UpdateWeightsMultiThreadWithADAM();
    void                    UpdateWeightsSingleThread();
    void                    SetInputLayerTraining(const int, const cv::Mat&);
    void                    SetInputLayer(const cv::Mat&);
    void                    LoadLayerNeuralNumber(const std::vector<int>&);
    void                    LoadWeights(const std::vector<cv::Mat>&);
    void                    LoadBias(const std::vector<cv::Mat>&);
    void                    LoadLayer(const std::vector<cv::Mat>&);
    void                    DerivativeOutputFunction(std::vector<std::vector<cv::Mat>>&);
    void                    DerivativeFunction(const int, std::vector<std::vector<cv::Mat>>&);
    void                    CalWeights(const int, const std::vector<std::vector<cv::Mat>>&, cv::Mat&);
    void                    CalBias(const int, const std::vector<std::vector<cv::Mat>>&, cv::Mat&);
    void                    CalLayer(const int, std::vector<std::vector<cv::Mat>>*);
    void                    EnableADAM();
    void                    DisableADAM();
    void                    InitIteration();
    void                    IncrementIteration();
    void                    SetBeta1(double);
    void                    SetBeta2(double);

    double                  L2Regression(const std::vector<cv::Mat>&);

    cv::Mat                 DerivativeL2Regression(const cv::Mat&);
    cv::Mat                 GetOutputLayer();

    std::vector<cv::Mat>    GetWeights();
    std::vector<cv::Mat>    GetBias();

    std::vector<int>        GetLayerNeuralNumber();

    int                     GetNumberOfInputLayerNodes();
    int                     GetNumberOfOutputLayerNodes();
    int                     GetMiniBatchSize();
private:
    std::vector<int>                    _layerNeuralNumber;
    std::vector<std::vector<cv::Mat>>   _layerThreading;
    std::vector<cv::Mat>                _layer;
    std::vector<cv::Mat>                _weights;
    std::vector<cv::Mat>                _bias;
    std::vector<boost::thread>          _vecThread;
    std::vector<double>                 _lossThreading;
    std::vector<cv::Mat>                _targetThreading;
    std::vector<cv::Mat>                _outputThreading;
    std::vector<cv::Mat>                _mtWeight;
    std::vector<cv::Mat>                _vtWeight;
    std::vector<cv::Mat>                _mtBias;
    std::vector<cv::Mat>                _vtBias;

    cv::Mat                             _target;
    cv::Mat                             _output;

    pActivationFunction                 _pActivationFunction;
    pActivationFunction                 _pOutputActivationFunction;
    pDerivativeOutputFunction           _pDerivativeOutputFunction;
    pLossFunction                       _pLossFunction;
    pDerivativeFunction                 _pDerivativeFunction;

    double                              _learningRate;
    double                              _lambda;
    double                              _beta1;
    double                              _beta2;
    const double                        _epsilon = 1e-8;

    int                                 _miniBatchSize;
    int                                 _iteration;

    bool                                _enableADAM;
};

#ifdef __cplusplus
}
#endif

#endif //MLP_MLP_H
