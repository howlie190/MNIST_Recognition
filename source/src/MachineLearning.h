
// Created by LG on 2023/11/2.
//

#ifndef MACHINELEARNING_MACHINELEARNING_H
#define MACHINELEARNING_MACHINELEARNING_H

#include <string>
#include <utility>
#include "Machine_Learning_Math.h"

#ifdef MachineLearning_EXPORTS
#define MACHINE_LEARNING_API __declspec(dllexport)
#else
#define MACHINE_LEARNING_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

class MACHINE_LEARNING_API Machine_Learning {
public:
    Machine_Learning(const Machine_Learning&) = delete;
    Machine_Learning& operator=(const Machine_Learning&) = delete;

    void SetTrainDataSetPath(std::string path) { train_data_set_path = std::move(path); }
    void SetTestDataSetPath(std::string path) { test_data_set_path = std::move(path); }
    void SetPredictDataPath(std::string path) { predict_data_path = std::move(path); }
    std::string GetTrainDataSetPath() { return train_data_set_path; }
    std::string GetTestDataSetPath() { return test_data_set_path; }
    std::string GetPredictDataPath() { return predict_data_path; }
protected:
    Machine_Learning() = default;

    virtual void                    Init() = 0;
    virtual std::vector<double>     Train() = 0;
    virtual double                  Test() = 0;
    virtual int                     Predict() = 0;
    virtual bool                    SaveModel(const char*, const char*, bool) = 0;
    virtual bool                    LoadModel(const char*) = 0;

    Machine_Learning_Math                       algorithm;                                              //機器學習常用算法
private:
    std::string train_data_set_path;
    std::string test_data_set_path;
    std::string predict_data_path;
};

#ifdef __cplusplus
}
#endif



#endif //MACHINELEARNING_MACHINELEARNING_H
