//
// Created by LG on 2023/11/15.
//

#ifndef MACHINELEARNING_OPTIMIZER_H
#define MACHINELEARNING_OPTIMIZER_H

#include <vector>
#include <opencv2/core/core.hpp>


#ifdef MachineLearning_EXPORTS
#define OPTIMIZER_API __declspec(dllexport)
#else
#define OPTIMIZER_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

class OPTIMIZER_API Optimizer {
public:
    enum class TYPE {
        NONE,
        ADAM
    };

    Optimizer() : type(TYPE::NONE) {}
    explicit    Optimizer(TYPE t) : type(t) { Init(); }

    void        Init();
    void        SetType(TYPE);
    void        SetADAM(const std::vector<int>&, double, double);
    void        SetOptimizer();
    void        UpdateADAM(const std::vector<cv::Mat>&, const std::vector<cv::Mat>&, const size_t&);
    cv::Mat     GetWeightResult() const;
    cv::Mat     GetBiasResult() const;
    void        PlusADAMCount();

    Optimizer(const Optimizer&) = delete;
    Optimizer& operator=(const Optimizer&) = default;

private:
    TYPE                    type;

    double                  beta1{};
    double                  beta2{};

    const double            epsilon     = 1e-8;

    int                     ADAM_count  = 0;

    std::vector<cv::Mat>    mt_weight;
    std::vector<cv::Mat>    vt_weight;
    std::vector<cv::Mat>    mt_bias;
    std::vector<cv::Mat>    vt_bias;

    cv::Mat                 weight_result;
    cv::Mat                 bias_result;
    cv::Mat                 squared;
    cv::Mat                 sqrt;

    std::vector<int>        neural_number_of_layer;
};

#ifdef __cplusplus
}
#endif

#endif //MACHINELEARNING_OPTIMIZER_H
