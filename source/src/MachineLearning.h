
// Created by LG on 2023/11/2.
//

#ifndef MACHINELEARNING_MACHINELEARNING_H
#define MACHINELEARNING_MACHINELEARNING_H

#include <string>
#include <utility>
#include "Machine_Learning_Math.h"
#include <thread>
#include <queue>
#include <condition_variable>
#include <future>

#ifdef MachineLearning_EXPORTS
#define MACHINE_LEARNING_API __declspec(dllexport)
#else
#define MACHINE_LEARNING_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

//============================================================================================================
//執行緒池
class MACHINE_LEARNING_API ThreadPool {
public:
    explicit ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i) {
            workers.emplace_back(
                    [this] {
                        while(true) {
                            std::function<void()> task;

                            {
                                std::unique_lock<std::mutex> lock(this->queue_mutex);
                                this->condition.wait(lock, [this] {return this->stop || !this->tasks.empty();});

                                if(this->stop && this->tasks.empty())
                                    return;

                                task = std::move(this->tasks.front());
                                this->tasks.pop();
                            }

                            task();
                        }
                    }
            );
        }
    }

    std::future<void> enqueue(std::function<void()> task) {
        auto promise = std::make_shared<std::promise<void>>();
        auto future = promise->get_future();

        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            tasks.emplace([task, promise] () {
                try {
                    task();
                    promise->set_value();
                } catch (...) {
                    promise->set_exception(std::current_exception());
                }
            });
        }

        condition.notify_one();
        return future;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop    = true;
        }
        condition.notify_all();
        for(std::thread &worker : workers) {
            worker.join();
        }
    }
private:
    std::vector<std::thread>            workers;
    std::queue<std::function<void()>>   tasks;
    std::mutex                          queue_mutex;
    std::condition_variable             condition;
    bool                                stop;
};
//============================================================================================================
//機器學習interface
class MACHINE_LEARNING_API Machine_Learning {
public:
    Machine_Learning(const Machine_Learning&) = delete;
    Machine_Learning& operator=(const Machine_Learning&) = delete;

    virtual ~Machine_Learning() {
        delete thread_pool;
    }

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

    ThreadPool* thread_pool{};                                                                          //執行緒池
private:
    std::string train_data_set_path;                                                                    //訓練集路徑
    std::string test_data_set_path;                                                                     //測試集路徑
    std::string predict_data_path;                                                                      //預測檔案路徑
};

#ifdef __cplusplus
}
#endif



#endif //MACHINELEARNING_MACHINELEARNING_H
