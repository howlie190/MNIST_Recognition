//
// Created by LG on 2023/11/3.
//

#ifndef MACHINELEARNING_IMAGEMLP_H
#define MACHINELEARNING_IMAGEMLP_H

#include "MLP.h"
#include <windows.h>

#ifdef MachineLearning_EXPORTS
#define IMAGE_MLP __declspec(dllexport)
#else
#define MACHINE_LEARNING_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

class IMAGE_MLP Image_MLP : public MLP {
public:
    Image_MLP() = default;
    ~Image_MLP();

    void Init() override;

    bool                                                SaveModel(const char*, const char*, bool) override;
    bool                                                LoadModel(const char*) override;
    std::vector<double>                                 Train() override;
    double                                              Test() override;
    int                                                 Predict() override;
    void                                                Terminate();
    void                                                SetThreshold(double);
    void                                                SetEpoch(int);
private:
    void                                                TrainHelper();
    void                                                SetTrainingInput(const size_t&, const cv::Mat&);
    void                                                SetInput(const cv::Mat&);

    std::vector<std::pair<std::string, cv::Mat>>        train_data_set;
    std::vector<std::pair<std::string, cv::Mat>>        test_data_set;
    std::vector<std::pair<std::string, std::string>>    file_name;
    std::vector<std::unique_ptr<boost::thread>>         vec_thread;
    std::vector<double>                                 vec_loss;
    std::vector<int>                                    train_data_set_index;

    int                                                 epoch{};

    double                                              threshold{};
    double                                              loss_value{};

    bool                                                stop_training{};
    bool                                                threshold_reached{};

    std::string                                         str_log{};

    HANDLE                                              hMap_mem_data{};
    HANDLE                                              hEvent_read{};
    HANDLE                                              hEvent_write{};

    LPSTR                                               pBuf{};
};

#ifdef __cplusplus
}
#endif




#endif //MACHINELEARNING_IMAGEMLP_H
