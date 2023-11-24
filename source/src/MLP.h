//
// Created by LG on 2023/11/3.
//

#ifndef MACHINELEARNING_MLP_H
#define MACHINELEARNING_MLP_H

#include "MachineLearning.h"
#include <vector>
#include <memory>
#include <opencv2/core/core.hpp>
#include <boost/thread.hpp>
#include <memory>
#include "Optimizer.h"
#include "Machine_Learning_Math.h"
#include <map>
#include <mutex>

#ifdef MachineLearning_EXPORTS
#define MLP_API __declspec(dllexport)
#else
#define MLP_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef cv::Mat     (*pActivationFunction)(const cv::Mat&);
typedef double      (*pLossFunction)(const cv::Mat&, const cv::Mat&);
typedef void        (*pDerivativeActivationFunction)(const cv::Mat&, cv::Mat&);
typedef void        (*pDerivativeOutputFunction)(const cv::Mat&, const cv::Mat&, cv::Mat*);

enum class DISTRIBUTION {
    NORMAL,
    UNIFORM
};

enum class ACTIVATION_FUNCTION {
    SIGMOID,
    TANH,
    RELU,
    NONE
};

enum class OUTPUT_FUNCTION {
    SIGMOID,
    TANH,
    RELU,
    SOFTMAX,
    NONE
};

enum class LOSS_FUNCTION {
    MEAN_SQUARED_ERROR,
    CROSS_ENTROPY,
    BINARY_CROSS_ENTROPY
};

enum class DERIVATIVE_ACTIVATION_FUNCTION {
    SIGMOID,
    TANH,
    RELU,
    NONE
};

enum class DERIVATIVE_OUTPUT_FUNCTION {
    SOFTMAX_CROSS_ENTROPY,
    SIGMOID_BINARY_CROSS_ENTROPY,
    TANH_MEAN_SQUARED_ERROR,
    RELU_MEAN_SQUARED_ERROR,
    NONE_MEAN_SQUARED_ERROR
};

class MLP_API MLP : public Machine_Learning {
public:
    MLP(const MLP&) = delete;
    MLP& operator=(const MLP&) = delete;

    void                                        InitNeuralNet(std::vector<int>&&);
    void                                        InitTrainNeuralNet(const size_t&);

    void                                        SetInitWeightValue(DISTRIBUTION, double, double);
    void                                        SetInitBiasValue(cv::Scalar&&);
    void                                        SetFunctions(ACTIVATION_FUNCTION,
                                                            OUTPUT_FUNCTION,
                                                            LOSS_FUNCTION);
    void                                        SetParameter(int, double, double, Optimizer::TYPE, double, double);
    void                                        SetADAM(double, double) const;
    static void                                 ShowMat(const cv::Mat&);
protected:
    MLP() { MLP::Init(); }
    virtual ~MLP();

    void                                        Init() override;
    void                                        InitNeuralNetLayer(std::vector<std::vector<cv::Mat>>&);
    void                                        InitNeuralNetLayer(std::vector<cv::Mat>&);
    void                                        InitNeuralNetWeightBias(std::vector<cv::Mat>&, std::vector<cv::Mat>&);

    void                                        SetActivationFunction(ACTIVATION_FUNCTION);
    void                                        SetOutputFunction(OUTPUT_FUNCTION);
    void                                        SetLossFunction(LOSS_FUNCTION);
    void                                        SetDerivativeActivationFunction(DERIVATIVE_ACTIVATION_FUNCTION);
    void                                        SetDerivativeOutputFunction(DERIVATIVE_OUTPUT_FUNCTION);
    void                                        SetTrainingInputLayer(const size_t&, cv::Mat&&);
    void                                        SetTrainingLabel(const size_t&, cv::Mat&&);
    void                                        SetInputLayer(cv::Mat&&);

    void                                        AssignWeight(const std::vector<cv::Mat>&);
    void                                        AssignBias(const std::vector<cv::Mat>&);
    void                                        AssignLayer(const std::vector<cv::Mat>&);
    void                                        AssignNeuralNumberOfLayer(const std::vector<int>&);

    std::vector<int>                            GetNeuralNumberOfLayer() const;
    std::vector<cv::Mat>                        GetMLPLayerWeight() const;
    std::vector<cv::Mat>                        GetMLPLayerBias() const;
    size_t                                      GetThreadSize() const;
    size_t                                      GetSectionSize() const;
    size_t                                      GetBatchSize() const;
    size_t                                      GetNumberOfInputLayer() const;
    size_t                                      GetNumberOfOutputLayer() const;
    double                                      GetLossValue() const;
    cv::Mat                                     GetPredictOutput() const;

    void                                        ForwardBatch();                                  //正向傳播
    void                                        Backward();                                 //反向傳播
    void                                        Forward();                                  //預測
private:
    std::vector<int>                            neural_number_of_layer;                     //神經網路每層神經元數

    std::vector<std::vector<cv::Mat>>           mlp_layer_values;                           //每層節點數值
    std::vector<cv::Mat>                        mlp_layer_weight_values;                    //每層節點權重值
    std::vector<cv::Mat>                        mlp_layer_bias_values;                      //每層節點偏置值
    std::vector<std::vector<cv::Mat>>           update_layer_values;                        //更新用每層節點數值
    std::vector<cv::Mat>                        update_layer_weight_values;                 //更新用每層節點權重值
    std::vector<cv::Mat>                        update_layer_bias_values;                   //更新用每層節點偏置值
    std::vector<cv::Mat>                        mlp_layer_value_predict;                    //測試預測用的節點數值

    std::vector<std::unique_ptr<boost::thread>> vec_thread;                                 //多執行續容器

    std::vector<double>                         output_loss_value;                          //損失值
    std::vector<cv::Mat>                        label_value;                                //標籤值
    std::vector<cv::Mat>                        output_value;                               //輸出值

    std::unique_ptr<Optimizer>                  optimizer;                                  //優化器
    cv::Mat                                     output_predict;

    size_t                                      thread_size{};                              //可用thread數量
    size_t                                      batch_size{};                               //設定批次大小
    size_t                                      section_size{};                             //資料區間大小

    double                                      learning_rate{};                            //學習率
    double                                      lambda{};

    bool                                        enable_optimizer{};

    Optimizer::TYPE                             optimizer_type{};

    pActivationFunction                         ActivationFunction{};                       //激活函數指標
    pActivationFunction                         OutputFunction{};                           //輸出層函數指標
    pLossFunction                               LossFunction{};                             //損失函數指標
    pDerivativeActivationFunction               DerivativeActivationFunction{};             //激活函數導數函數指標
    pDerivativeOutputFunction                   DerivativeOutputFunction{};                 //輸出層導函數指標

    std::map<std::pair<OUTPUT_FUNCTION, LOSS_FUNCTION>, DERIVATIVE_OUTPUT_FUNCTION> derivative_map = {
            {{OUTPUT_FUNCTION::SOFTMAX, LOSS_FUNCTION::CROSS_ENTROPY}, DERIVATIVE_OUTPUT_FUNCTION::SOFTMAX_CROSS_ENTROPY},
            {{OUTPUT_FUNCTION::SIGMOID, LOSS_FUNCTION::BINARY_CROSS_ENTROPY}, DERIVATIVE_OUTPUT_FUNCTION::SIGMOID_BINARY_CROSS_ENTROPY},
            {{OUTPUT_FUNCTION::RELU, LOSS_FUNCTION::MEAN_SQUARED_ERROR}, DERIVATIVE_OUTPUT_FUNCTION::RELU_MEAN_SQUARED_ERROR},
            {{OUTPUT_FUNCTION::TANH, LOSS_FUNCTION::MEAN_SQUARED_ERROR}, DERIVATIVE_OUTPUT_FUNCTION::TANH_MEAN_SQUARED_ERROR},
            {{OUTPUT_FUNCTION::NONE, LOSS_FUNCTION::MEAN_SQUARED_ERROR}, DERIVATIVE_OUTPUT_FUNCTION::NONE_MEAN_SQUARED_ERROR}
    };  //輸出函數導數映射

    void                                        ForwardBatchPropagation(const size_t&, const size_t&);       //正向傳播算法
    void                                        UpdateNeuralNet();

    void                                        DerivativeOutputLayer();
    void                                        DerivativeOutputLayerBatch(const size_t&, const size_t&);
    void                                        DerivativeHiddenLayer(const size_t&);
    void                                        DerivativeHiddenLayerBatch(const size_t&, const size_t&, const size_t&);
    void                                        DerivativeCalWeight(const size_t&);
    void                                        DerivativeCalBias(const size_t&);
    void                                        DerivativeCalLayer(const size_t&);
    cv::Mat                                     DerivativeL2Regression(const cv::Mat&) const;

    void                                        CalVecLayerProduct(std::vector<cv::Mat>&, const size_t&, const size_t&, const size_t&, const size_t&);
    void                                        CalVecMatSum(const std::vector<cv::Mat>&, cv::Mat*, const size_t&, const size_t&);
    void                                        CalMatAverage(const cv::Mat&, cv::Mat*, const size_t&);
    void                                        CalWeightLayerProduct(const size_t&, const size_t&, const size_t&, const size_t&);

    std::mutex                                  mtx_cal_vec_mat_product;
};

#ifdef __cplusplus
}
#endif

#endif //MACHINELEARNING_MLP_H
