
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

    void SetTrainDataSetPath(std::string path) { train_data_set_path = std::move(path); }               //設定訓練集路徑
    void SetTestDataSetPath(std::string path) { test_data_set_path = std::move(path); }                 //設定測試集路徑
    void SetPredictDataPath(std::string path) { predict_data_path = std::move(path); }                  //設定預測檔案路徑
    std::string GetTrainDataSetPath() { return train_data_set_path; }                                   //取得訓練集路徑
    std::string GetTestDataSetPath() { return test_data_set_path; }                                     //取得測試集路徑
    std::string GetPredictDataPath() { return predict_data_path; }                                      //取得預測檔案路徑
protected:
    Machine_Learning() = default;

    virtual void                    Init() = 0;
    virtual std::vector<double>     Train() = 0;                                                        //訓練
    virtual double                  Test() = 0;                                                         //測試
    virtual int                     Predict() = 0;                                                      //預測
    virtual bool                    SaveModel(const char*, const char*, bool) = 0;                      //儲存模型
    virtual bool                    LoadModel(const char*) = 0;                                         //載入模型
private:
    std::string train_data_set_path;                                                                    //訓練集路徑
    std::string test_data_set_path;                                                                     //測試集路徑
    std::string predict_data_path;                                                                      //預測檔案路徑
};

#ifdef __cplusplus
}
#endif



#endif //MACHINELEARNING_MACHINELEARNING_H
