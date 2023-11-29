//
// Created by LG on 2023/11/3.
//

#include "ImageMLP.h"
#include <fstream>
#include <regex>
#include <limits>
#include <opencv2/opencv.hpp>
#include <random>
#include <algorithm>
#include <sstream>
#include <numeric>

//============================================================================================================
void Image_MLP::Init() {
    MLP::Init();
    epoch               = 1;
    threshold           = 0.001;
    loss_value          = 0;

    stop_training       = false;
    threshold_reached   = false;

    hMap_mem_data       = nullptr;
    hEvent_read         = nullptr;
    hEvent_write        = nullptr;

    str_log.clear();

    train_data_set.clear();
    test_data_set.clear();
    file_name.clear();
    vec_loss.clear();
    train_data_set_index.clear();

    for(const auto &thd : vec_thread) {
        if(thd && thd->joinable())
            thd->join();
    }

    vec_thread.clear();
}
//============================================================================================================
bool Image_MLP::SaveModel(const char *path, const char *name, bool override) {
    std::string     file_path;

    if(strcmp(path, "") == 0) {
        file_path   = ".\\save\\" + std::string(name) + ".bin";
    } else {
        file_path   = std::string(path) + "\\" + std::string(name) + ".bin";
    }

    if(!override) {
        std::ifstream   stream_check(file_path);
        if(stream_check.is_open()) {
#ifndef MachineLearning_EXPORTS
            std::cout << "File is Already Exists!" << std::endl;
#endif
            stream_check.close();

            return false;
        }
    }

    std::ofstream   stream_save(file_path, std::ios::binary | std::ios::trunc | std::ios::out);

    std::vector<int> number_of_layer    = GetNeuralNumberOfLayer();
    stream_save << number_of_layer.size() << " ";
    for(const auto num : number_of_layer) {
        stream_save << num << " ";
    }

    std::vector<cv::Mat>    weight  = GetMLPLayerWeight();
    for(const auto &mat : weight) {
        for(int i = 0; i < mat.rows; i++) {
            for(int j = 0; j < mat.cols; j++) {
                stream_save << mat.at<float>(i, j) << " ";
            }
        }
    }

    std::vector<cv::Mat>    bias    = GetMLPLayerBias();
    for(const auto &mat : bias) {
        for(int i = 0; i < mat.rows; i++) {
            for(int j = 0; j < mat.cols; j++) {
                stream_save << mat.at<float>(i, j) << " ";
            }
        }
    }

    stream_save.close();

    return true;
}
//============================================================================================================
bool Image_MLP::LoadModel(const char *path) {
    int         layer_size;

    std::regex  pattern(".*\\.bin$");

    if(!std::regex_match(path, pattern)) {
#ifndef MachineLearning_EXPORTS
        std::cout << "File extension is not .bin!" << std::endl;
#endif
        return false;
    }

    std::ifstream   stream_load(path, std::ios::binary);
    if(!stream_load.is_open()) {
#ifndef MachineLearning_EXPORTS
        std::cout << "File Cannot be Opened!" << std::endl;
#endif
        return false;
    }

    stream_load >> layer_size;

    std::vector<int>        number_of_layer(layer_size);
    for(auto &num : number_of_layer) {
        stream_load >> num;
    }

    std::vector<cv::Mat>    weight(layer_size - 1);
    for(int i = 0; i < weight.size(); i++) {
        weight[i]       = cv::Mat::zeros(number_of_layer[i + 1], number_of_layer[i], CV_32FC1);

        for(int j = 0; j < weight[i].rows; j++) {
            for(int k = 0; k < weight[i].cols; k++) {
                stream_load >> weight[i].at<float>(j, k);
            }
        }
    }

    std::vector<cv::Mat>    bias(layer_size - 1);
    for (int i = 0; i < bias.size(); i++) {
        bias[i] = cv::Mat::zeros(number_of_layer[i + 1], 1, CV_32FC1);

        for (int j = 0; j < bias[i].rows; j++) {
            for (int k = 0; k < bias[i].cols; k++) {
                stream_load >> bias[i].at<float>(j, k);
            }
        }
    }

    std::vector<cv::Mat>    layer(layer_size);
    AssignNeuralNumberOfLayer(number_of_layer);

    InitNeuralNetLayer(layer);

    AssignWeight(weight);
    AssignBias(bias);
    AssignLayer(layer);

    stream_load.close();

    return true;
}
//============================================================================================================
std::vector<double> Image_MLP::Train() {
    WIN32_FIND_DATA tFD;

    char tPath[CHAR_MAX];
    std::string path = GetTrainDataSetPath();

    strcpy(tPath, path.c_str());
    strcat(tPath, "\\*.*");

    HANDLE hFind = ::FindFirstFile(tPath, &tFD);

    if (hFind != INVALID_HANDLE_VALUE) {
        char tFilePath[CHAR_MAX];
        char tFileName[CHAR_MAX];
        do {
            if (!(tFD.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                strcpy(tFileName, tFD.cFileName);
                strcpy(tFilePath, path.c_str());
                strcat(tFilePath, "\\");
                strcat(tFilePath, tFileName);

                file_name.emplace_back(tFilePath, tFileName);
            }
        } while (::FindNextFile(hFind, &tFD));
    }

    ::FindClose(hFind);

    size_t file_size = file_name.size();

    train_data_set.resize(file_size);
    train_data_set_index.resize(file_size);
    std::iota(train_data_set_index.begin(), train_data_set_index.end(), 0);

    vec_loss.clear();

    size_t thread_size = GetThreadSize();
    size_t section_size = file_size % thread_size == 0 ? file_size / thread_size : file_size / thread_size + 1;

    for (int i = 0; i < thread_size; i++) {
        size_t begin = i * section_size;
        size_t end = (i + 1) * section_size - 1;

        end = end > file_size ? file_size % section_size - 1 : end;

        vec_thread.emplace_back(std::make_unique<boost::thread>([this, begin, end] {
            for (size_t idx = begin; idx <= end; idx++) {
                train_data_set[idx] = std::pair<std::string, cv::Mat>(file_name[idx].second,
                                                                      cv::imread(file_name[idx].first,
                                                                                 cv::IMREAD_GRAYSCALE));
            }
        }));
    }

    for (const auto &thd: vec_thread) {
        if (thd && thd->joinable())
            thd->join();
    }

    vec_thread.clear();

    std::random_device  rd;
    std::mt19937        g(rd());
    std::shuffle(train_data_set_index.begin(), train_data_set_index.end(), g);

    hMap_mem_data       = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, TEXT("FileMapping"));
    pBuf                = (LPTSTR)MapViewOfFile(hMap_mem_data, FILE_MAP_ALL_ACCESS, 0, 0, sizeof( char[1024]));

    hEvent_read         = OpenEvent(EVENT_ALL_ACCESS, FALSE, "EventCanReadMemory");
    hEvent_write        = OpenEvent(EVENT_ALL_ACCESS, FALSE, "EventCanWriteMemory");

    stop_training       = false;
    threshold_reached   = false;

    for(int i = 0; i < epoch; i++) {
#ifndef MachineLearning_EXPORTS
        std::cout << "==================== " << i << " ====================" << std::endl;
#else
        str_log = "==================== " + std::to_string(i) + " ====================";
        WaitForSingleObject(hEvent_write, 0);
        CopyMemory(pBuf, str_log.c_str(), strlen(str_log.c_str()));
        ResetEvent(hEvent_write);
        SetEvent(hEvent_read);
#endif
        if(threshold_reached) {
            std::random_device  trd;
            std::mt19937        tg(trd());
            std::shuffle(train_data_set_index.begin(), train_data_set_index.end(), tg);

            threshold_reached   = false;
        }

        TrainHelper();

        if(stop_training) {
            break;
        }
    }

    train_data_set.clear();
    train_data_set_index.clear();
    file_name.clear();

    UnmapViewOfFile(pBuf);
    CloseHandle(hEvent_write);
    CloseHandle(hEvent_read);
    CloseHandle(hMap_mem_data);

    return vec_loss;
}
//============================================================================================================
void Image_MLP::TrainHelper() {
    size_t  batch_size      = GetBatchSize();
    size_t  data_size       = train_data_set.size();
    size_t  interval_size   = data_size % batch_size
            ? data_size / batch_size + 1
            : data_size / batch_size;

    for(size_t i = 0; i < interval_size; i++) {
#ifndef MachineLearning_EXPORTS
        std::cout << "Batch : " << i + 1 << "\t";
#endif
        int begin   = (int)i * (int)batch_size;
        int end     = (int)(i + 1) * (int)batch_size - 1;

        end         = end > data_size ? (int)data_size - 1 : end;

        for(int j = begin, k = 0; j <= end; j++, k++) {
            SetTrainingInput(k, train_data_set[train_data_set_index[j]].second);

            int     label       = std::strtol(&train_data_set[train_data_set_index[j]].first[0], nullptr, 10);
            cv::Mat label_mat   = cv::Mat::zeros((int)GetNumberOfOutputLayer(), 1, CV_32FC1);
            label_mat.at<float>(label, 0)       = 1.0;
            SetTrainingLabel(k, std::move(label_mat));
        }

        ForwardBatch();

        double loss = GetLossValue();

#ifndef MachineLearning_EXPORTS
        std::cout << "Loss Value : " << loss << "\t";
#else
        boost::thread   thd_msg([this, loss, i] {
            std::stringstream   ss;
            ss << "Batch : " << std::setw(10) << std::left << i + 1;
            str_log     = ss.str();

            char                temp[32];
            snprintf(temp, 32, "%.15f", loss);
            str_log     += "Loss Value : " + std::string(temp);

            WaitForSingleObject(hEvent_write, 0);
            CopyMemory(pBuf, str_log.c_str(), strlen(str_log.c_str()));
            ResetEvent(hEvent_write);
            SetEvent(hEvent_read);
        });
#endif
        if(stop_training) {
            break;
        } else if(loss < threshold) {
            threshold_reached = true;
            if(batch_size == 1) {
                break;
            }
        }

        vec_loss.push_back(loss);

        Backward();
    }
}
//============================================================================================================
void Image_MLP::SetTrainingInput(const size_t &idx, const cv::Mat &input) {
    cv::Mat temp((int)GetNumberOfInputLayer(), 1, CV_32FC1);

    for(int i = 0, k = 0; i < input.rows; i++) {
        for(int j = 0; j < input.cols; j++, k++) {
            temp.at<float>(k, 0)    = static_cast<float>(input.at<uchar>(i, j)) / 255;
        }
    }

    SetTrainingInputLayer(idx, std::move(temp));
}
//============================================================================================================
double Image_MLP::Test() {
    WIN32_FIND_DATA     tFD;
    char                tPath[CHAR_MAX];
    HANDLE              hFind;
    cv::Mat             output_mat;

    int                 output_result;
    int                 test_data_size_count    = 1;
    int                 valid_count             = 0;

    double              compare;

    strcpy(tPath, GetTestDataSetPath().c_str());
    strcat(tPath, "\\*.*");

    hFind       = ::FindFirstFile(tPath, &tFD);

    if(hFind != INVALID_HANDLE_VALUE) {
        do {
            if(!(tFD.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                char*   c_file_name = tFD.cFileName;
                char    file_path[CHAR_MAX];

                strcpy(file_path, GetTestDataSetPath().c_str());
                strcat(file_path, "\\");
                strcat(file_path, c_file_name);

                cv::Mat image   = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
                SetInput(image);

                Forward();

                output_mat      = GetPredictOutput();
                output_result   = INT_MIN;
                compare         = DBL_MIN;

                for(int i = 0; i < output_mat.rows; i++) {
                    if(compare < output_mat.at<float>(i, 0)) {
                        compare         = output_mat.at<float>(i, 0);
                        output_result   = i;
                    }
                }

                if(std::strtol(&c_file_name[0], nullptr, 10) == output_result) {
                    valid_count++;
                }

                test_data_size_count++;
            }
        } while(::FindNextFile(hFind, &tFD));
    }

    ::FindClose(hFind);

    return valid_count / static_cast<double>(test_data_size_count);
}
//============================================================================================================
void Image_MLP::SetInput(const cv::Mat &input) {
    cv::Mat temp((int)GetNumberOfInputLayer(), 1, CV_32FC1);

    switch (GetActivationFunctionType()) {

        case ACTIVATION_FUNCTION::SIGMOID:
            for(int i = 0, k = 0; i < input.rows; i++) {
                for(int j = 0; j < input.cols; j++, k++) {
                    temp.at<float>(k, 0)    = static_cast<float>(input.at<uchar>(i, j)) / 255;
                }
            }
            break;
        case ACTIVATION_FUNCTION::TANH:
            for(int i = 0, k = 0; i < input.rows; i++) {
                for(int j = 0; j < input.cols; j++, k++) {
                    temp.at<float>(k, 0)    = (static_cast<float>(input.at<uchar>(i, j)) - 128) / 128;
                }
            }
            break;
        case ACTIVATION_FUNCTION::RELU:
        case ACTIVATION_FUNCTION::NONE:
        default:
            for(int i = 0, k = 0; i < input.rows; i++) {
                for(int j = 0; j < input.cols; j++, k++) {
                    temp.at<float>(k, 0)    = static_cast<float>(input.at<uchar>(i, j));
                }
            }
            break;
    }

    SetInputLayer(std::move(temp));
}
//============================================================================================================
int Image_MLP::Predict() {
    cv::Mat output_mat;
    double  compare;
    int     output_result;

    cv::Mat image   = cv::imread(GetPredictDataPath(), cv::IMREAD_GRAYSCALE);
    SetInput(image);

    Forward();

    output_mat      = GetPredictOutput();
    output_result   = INT_MIN;
    compare         = DBL_MIN;

    for(int i = 0; i < output_mat.rows; i++) {
        if(compare < output_mat.at<float>(i, 0)) {
            compare         = output_mat.at<float>(i, 0);
            output_result   = i;
        }
    }

    return output_result;
}
//============================================================================================================
void Image_MLP::Terminate() {
    stop_training   = true;
}
//============================================================================================================
void Image_MLP::SetThreshold(double t) {
    threshold       = t;
}
//============================================================================================================
void Image_MLP::SetEpoch(int e) {
    epoch           = e;
}
//============================================================================================================
Image_MLP::~Image_MLP() {
    for(const auto &thd : vec_thread) {
        if(thd && thd->joinable()) {
            thd->join();
        }
    }

    CloseHandle(hEvent_write);
    CloseHandle(hEvent_read);
    CloseHandle(hMap_mem_data);
}
//============================================================================================================