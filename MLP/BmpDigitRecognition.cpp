//
// Created by howli on 2023/1/25.
//

#include "BmpDigitRecognition.h"
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <climits>
#include <fstream>
#include <regex>

void BmpDigitRecognition::SetInput(cv::Mat mat) {
    cv::Mat temp(INPUT_LAYER_SIZE, 1, CV_32FC1);
    for (int i = 0, k = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++, k++) {
            temp.at<float>(k, 0) = (float) mat.at<uchar>(i, j) / 255;
        }
    }
    SetInputLayer(std::move(temp));
}

void BmpDigitRecognition::Train(char *path) {
    for(int i = 0; i < _loopCount; i++) {
        std::cout << "==================== " << i << " ====================" << std::endl;
        TrainHelper(path);
    }
}

void BmpDigitRecognition::TrainHelper(char *path) {
    WIN32_FIND_DATA     tFD;
    char                tPath[CHAR_MAX];
    bool                tFlagThreshold  = false;
    bool                tFlagFirst      = true;
    int                 tCurrentTarget  = INT_MIN;

    strcpy(tPath, path);
    strcat(tPath, "\\*.*");

    HANDLE hFind = ::FindFirstFile(tPath, &tFD);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(tFD.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                char*   tFileName = tFD.cFileName;
                char    tFilePath[CHAR_MAX];

                strcpy(tFilePath, path);
                strcat(tFilePath, "\\");
                strcat(tFilePath, tFileName);

                cv::Mat image = cv::imread(tFilePath);

                SetInput(image);

                 if(tCurrentTarget != std::atoi(&tFileName[0])) {
                    tFlagThreshold  = false;
                    tFlagFirst      = true;
                    if(tCurrentTarget != INT_MIN)
                        std::cout << "\tEnd : " << GetLoss() << std::endl;
                } else if(tFlagThreshold) {
                    continue;
                }

                tCurrentTarget                              = std::atoi(&tFileName[0]);
                cv::Mat             tTarget                 = cv::Mat::zeros(OUTPUT_LAYER_SIZE, 1, CV_32FC1);
                tTarget.at<float>(tCurrentTarget, 0)    = 1.0;
                SetTarget(tTarget);

                if(_select && tCurrentTarget != _choose) {
                    continue;
                }

                ForwardPropagation(true);

                if(tFlagFirst) {
                    std::cout << tFileName[0] << " : " << GetLoss();
                    tFlagFirst      = false;
                }

                BackPropagation();

                if(GetLoss() < _threshold) {
                    tFlagThreshold  = true;
                }
            }
        } while (::FindNextFile(hFind, &tFD));
    }

    std::cout << "\tEnd : " << GetLoss() << std::endl;

    ::FindClose(hFind);
}

void BmpDigitRecognition::Test(char *path) {
    WIN32_FIND_DATA         tFD;
    char                    tPath[CHAR_MAX];
    HANDLE                  hFind;
    cv::Mat                 tOutput;
    int                     tOutputResult;
    int                     tTotalNumberOfData = 1;
    int                     tNumberOfValidData = 0;
    double                  tCompare;

    strcpy(tPath, path);
    strcat(tPath, "\\*.*");

    hFind = ::FindFirstFile(tPath, &tFD);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(tFD.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                char*   tFileName = tFD.cFileName;
                char    tFilePath[CHAR_MAX];

                strcpy(tFilePath, path);
                strcat(tFilePath, "\\");
                strcat(tFilePath, tFileName);

                cv::Mat image = cv::imread(tFilePath);
                SetInput(image);

                ForwardPropagation();

                tOutput         = GetOutputLayer();
                tOutputResult   = INT_MIN;
                tCompare        = DBL_MIN;

                for(int i = 0; i < tOutput.rows; i++) {
                    if(tCompare < tOutput.at<float>(i, 0)) {
                        tCompare        = tOutput.at<float>(i, 0);
                        tOutputResult   = i;
                    }
                }

                std::cout << "Target : " << tFileName[0] << " Output : " << tOutputResult << std::endl;

                if(std::atoi(&tFileName[0]) == tOutputResult) {
                    tNumberOfValidData++;
                }

                tTotalNumberOfData++;
            }
        } while (::FindNextFile(hFind, &tFD));
    }

    std::cout << "Accuracy : " << tNumberOfValidData / (float)tTotalNumberOfData << std::endl;

    ::FindClose(hFind);
}

bool BmpDigitRecognition::Save(char* path, char* name, bool override) {
    std::string filePath;
    if(strcmp(path, "") == 0) {
        filePath = ".\\save\\" + std::string(name) + ".bin";
    } else {
        filePath = std::string(path) + "\\" + std::string(name) + ".bin";
    }

    if(!override) {
        std::ifstream check(filePath);
        if(check.is_open()) {
            std::cout << "File is Already Exists!" << std::endl;
            check.close();
            return false;
        }
    }


    std::ofstream save(filePath, std::ios::binary | std::ios::trunc);

    std::vector<int> layerNumber = GetLayerNeuralNumber();
    save << layerNumber.size() << " ";
    for(int i = 0; i < layerNumber.size(); i++) {
        save << layerNumber[i] << " ";
    }

    std::vector<cv::Mat> weights = GetWeights();
    for(int i = 0; i < weights.size(); i++) {
        for(int j = 0; j < weights[i].rows; j++) {
            for(int k = 0; k < weights[i].cols; k++) {
                save << weights[i].at<float>(j, k) << " ";
            }
        }
    }

    std::vector<cv::Mat> bias = GetBias();
    for(int i = 0; i < bias.size(); i++) {
        for(int j = 0; j < bias[i].rows; j++) {
            for(int k = 0; k < bias[i].cols; k++) {
                save << bias[i].at<float>(j, k) << " ";
            }
        }
    }

    save.close();
    return true;
}

bool BmpDigitRecognition::Load(char *path) {
    int             layerSize;
    std::regex      pattern(".*\\.bin$");

    if(!std::regex_match(path, pattern)) {
        std::cout << "File extension is not .bin!" << std::endl;
        return false;
    }

    std::ifstream load(path, std::ios::binary);
    if(!load.is_open()) {
        std::cout << "File Cannot be Opened!" << std::endl;
        return false;
    }

    load >> layerSize;
    std::vector<int> layerNumber(layerSize);
    for(int i = 0; i < layerNumber.size(); i++) {
        load >> layerNumber[i];
    }

    std::vector<cv::Mat> weights(layerNumber.size() - 1);
    for(int i = 0; i < weights.size(); i++) {
        weights[i] = cv::Mat::zeros(layerNumber[i + 1], layerNumber[i], CV_32FC1);
        for(int j = 0; j < weights[i].rows; j++) {
            for(int k = 0; k < weights[i].cols; k++) {
                load >> weights[i].at<float>(j, k);
            }
        }
    }

    std::vector<cv::Mat> bias(layerNumber.size() - 1);
    for(int i = 0; i < bias.size(); i++) {
        bias[i] = cv::Mat::zeros(layerNumber[i + 1], 1, CV_32FC1);
        for(int j = 0; j < bias[i].rows; j++) {
            for(int k = 0; k < bias[i].cols; k++) {
                load >> bias[i].at<float>(j, k);
            }
        }
    }

    std::vector<cv::Mat> layer(layerNumber.size());
    for (int i = 0; i < layer.size(); i++) {
        layer[i].create(layerNumber[i], 1, CV_32FC1);
    }

    LoadLayerNeuralNumber(layerNumber);
    LoadWeights(weights);
    LoadBias(bias);
    LoadLayer(layer);

    load.close();
    return true;
}