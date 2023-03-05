//
// Created by howli on 2023/1/25.
//

#include "BmpDigitRecognition.h"
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <climits>
#include <fstream>
#include <regex>
#include <iostream>
#include <opencv2/core.hpp>
#include <shlwapi.h>
#include <sstream>
#include <minwinbase.h>

#pragma comment(lib, "Shlwapi.lib")

void BmpDigitRecognition::SetSelect(int target) {
    _choose = target;
    _select = true;
}

void BmpDigitRecognition::SetThreshold(double threshold) { _threshold = threshold; }

void BmpDigitRecognition::SetLoopCount(int loopCount) { _loopCount = loopCount; }

BmpDigitRecognition::BmpDigitRecognition() {
    _threshold = 0.001;
    _loopCount = 10;
}

void BmpDigitRecognition::SetInput(cv::Mat mat) {
    cv::Mat temp(GetNumberOfInputLayerNodes(), 1, CV_32FC1);
    for (int i = 0, k = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++, k++) {
            temp.at<float>(k, 0) = mat.at<uchar>(i, j) / (double) 255;
        }
    }
    SetInputLayer(std::move(temp));
}

void BmpDigitRecognition::Train(char *path) {
    _hMapFile   = OpenFileMapping(FILE_MAP_ALL_ACCESS, FALSE, TEXT("FileMapping"));
    _pBuf       = (LPTSTR)MapViewOfFile(_hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof( char[1024]));

    _hEventRead     = OpenEvent(EVENT_ALL_ACCESS, FALSE, "EventCanReadMemory");
    _hEventWrite    = OpenEvent(EVENT_ALL_ACCESS, FALSE, "EventCanWriteMemory");

    _stopTraining   = false;

    for (int i = 0; i < _loopCount; i++) {
#ifndef MNIST_Recognition_Library_EXPORTS
        std::cout << "==================== " << i << " ====================" << std::endl;
#else
        _logFile = "==================== " + std::to_string(i) + " ====================";
        WaitForSingleObject(_hEventWrite, INFINITE);
        CopyMemory(_pBuf, _logFile.c_str(), strlen(_logFile.c_str()));
        ResetEvent(_hEventWrite);
        SetEvent(_hEventRead);
#endif
        TrainHelper(path);
        if(_stopTraining) {
            break;
        }
    }

    UnmapViewOfFile(_pBuf);
    CloseHandle(_hEventWrite);
    CloseHandle(_hEventRead);
    CloseHandle(_hMapFile);
}

void BmpDigitRecognition::TrainHelper(char *path) {
    WIN32_FIND_DATA tFD;

    char    tPath[CHAR_MAX];
    char    tFileName[CHAR_MAX];
    char    tFilePath[CHAR_MAX];
    bool    tFlagThreshold = false;
    bool    tFlagFirst = true;
    int     tCurrentTarget = INT_MIN;

    strcpy(tPath, path);
    strcat(tPath, "\\*.*");

    HANDLE hFind = ::FindFirstFile(tPath, &tFD);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(tFD.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                strcpy(tFileName, tFD.cFileName);
                strcpy(tFilePath, path);
                strcat(tFilePath, "\\");
                strcat(tFilePath, tFileName);

                cv::Mat image = cv::imread(tFilePath, cv::IMREAD_GRAYSCALE);

                SetInput(image);

                if (tCurrentTarget != std::atoi(&tFileName[0])) {
                    tFlagThreshold = false;
                    tFlagFirst = true;
                    if (tCurrentTarget != INT_MIN) {
#ifndef MNIST_Recognition_Library_EXPORTS
                        std::cout << "\tEnd : " << GetLoss() << std::endl;
#else
                        char temp[32];
                        snprintf(temp, 32, "%.15f", GetLoss());
                        _logFile += "\tEnd : " + std::string(temp);
                        WaitForSingleObject(_hEventWrite, INFINITE);
                        CopyMemory(_pBuf, _logFile.c_str(), strlen(_logFile.c_str()));
                        ResetEvent(_hEventWrite);
                        SetEvent(_hEventRead);
#endif
                    }
                } else if (tFlagThreshold) {
                    continue;
                }

                tCurrentTarget = std::atoi(&tFileName[0]);
                cv::Mat tTarget = cv::Mat::zeros(GetNumberOfOutputLayerNodes(), 1, CV_32FC1);
                tTarget.at<float>(tCurrentTarget, 0) = 1.0;
                SetTarget(tTarget);

                if (_select && tCurrentTarget != _choose) {
                    continue;
                }

                ForwardPropagation(true);

                if (tFlagFirst) {
#ifndef MNIST_Recognition_Library_EXPORTS
                    std::cout << tFileName[0] << " : " << GetLoss();
#else
                    char temp[32];
                    snprintf(temp, 32, "%.15f", GetLoss());
                    _logFile.clear();
                    _logFile.push_back(tFileName[0]);
                    _logFile += " : Begin : " + std::string(temp);
#endif
                    tFlagFirst = false;
                }

                BackPropagation();

                if (GetLoss() < _threshold) {
                    tFlagThreshold = true;
                }
            }
        } while (::FindNextFile(hFind, &tFD));
    } else {
        throw std::runtime_error("Invalid input");
    }
#ifndef MNIST_Recognition_Library_EXPORTS
    std::cout << "\tEnd : " << GetLoss() << std::endl;
#else
    char temp[32];
    snprintf(temp, 32, "%.15f", GetLoss());
    _logFile += "\tEnd : " + std::string(temp);
    WaitForSingleObject(_hEventWrite, INFINITE);
    CopyMemory(_pBuf, _logFile.c_str(), strlen(_logFile.c_str()));
    ResetEvent(_hEventWrite);
    SetEvent(_hEventRead);
#endif

    ::FindClose(hFind);
}

double BmpDigitRecognition::Test(char *path) {
    WIN32_FIND_DATA tFD;
    char tPath[CHAR_MAX];
    HANDLE hFind;
    cv::Mat tOutput;
    int tOutputResult;
    int tTotalNumberOfData = 1;
    int tNumberOfValidData = 0;
    double tCompare;

    strcpy(tPath, path);
    strcat(tPath, "\\*.*");

    hFind = ::FindFirstFile(tPath, &tFD);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(tFD.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                char *tFileName = tFD.cFileName;
                char tFilePath[CHAR_MAX];

                strcpy(tFilePath, path);
                strcat(tFilePath, "\\");
                strcat(tFilePath, tFileName);

                cv::Mat image = cv::imread(tFilePath, cv::IMREAD_GRAYSCALE);
                SetInput(image);

                ForwardPropagation();

                tOutput = GetOutputLayer();
                tOutputResult = INT_MIN;
                tCompare = DBL_MIN;

                for (int i = 0; i < tOutput.rows; i++) {
                    if (tCompare < tOutput.at<float>(i, 0)) {
                        tCompare = tOutput.at<float>(i, 0);
                        tOutputResult = i;
                    }
                }
#ifndef MNIST_Recognition_Library_EXPORTS
                std::cout << "Target : " << tFileName[0] << " Output : " << tOutputResult << std::endl;
#endif
                if (std::atoi(&tFileName[0]) == tOutputResult) {
                    tNumberOfValidData++;
                }

                tTotalNumberOfData++;
            }
        } while (::FindNextFile(hFind, &tFD));
    }
#ifndef MNIST_Recognition_Library_EXPORTS
    std::cout << "Accuracy : " << tNumberOfValidData / (float) tTotalNumberOfData << std::endl;
#endif
    ::FindClose(hFind);

    return tNumberOfValidData / (float) tTotalNumberOfData;
}

bool BmpDigitRecognition::Save(char *path, char *name, bool override) {
    std::string filePath;
    if (strcmp(path, "") == 0) {
        filePath = ".\\save\\" + std::string(name) + ".bin";
    } else {
        filePath = std::string(path) + "\\" + std::string(name) + ".bin";
    }

    if (!override) {
        std::ifstream check(filePath);
        if (check.is_open()) {
#ifndef MNIST_Recognition_Library_EXPORTS
            std::cout << "File is Already Exists!" << std::endl;
#endif
            check.close();
            return false;
        }
    }

    std::ofstream save(filePath, std::ios::binary | std::ios::trunc | std::ios::out);

    std::vector<int> layerNumber = GetLayerNeuralNumber();
    save << layerNumber.size() << " ";
    for (int i = 0; i < layerNumber.size(); i++) {
        save << layerNumber[i] << " ";
    }

    std::vector<cv::Mat> weights = GetWeights();
    for (int i = 0; i < weights.size(); i++) {
        for (int j = 0; j < weights[i].rows; j++) {
            for (int k = 0; k < weights[i].cols; k++) {
                save << weights[i].at<float>(j, k) << " ";
            }
        }
    }

    std::vector<cv::Mat> bias = GetBias();
    for (int i = 0; i < bias.size(); i++) {
        for (int j = 0; j < bias[i].rows; j++) {
            for (int k = 0; k < bias[i].cols; k++) {
                save << bias[i].at<float>(j, k) << " ";
            }
        }
    }

    save.close();
    return true;
}

bool BmpDigitRecognition::Load(char *path) {
    int layerSize;
    std::regex pattern(".*\\.bin$");

    if (!std::regex_match(path, pattern)) {
#ifndef MNIST_Recognition_Library_EXPORTS
        std::cout << "File extension is not .bin!" << std::endl;
#endif
        return false;
    }

    std::ifstream load(path, std::ios::binary);
    if (!load.is_open()) {
#ifndef MNIST_Recognition_Library_EXPORTS
        std::cout << "File Cannot be Opened!" << std::endl;
#endif
        return false;
    }

    load >> layerSize;
    std::vector<int> layerNumber(layerSize);
    for (int i = 0; i < layerNumber.size(); i++) {
        load >> layerNumber[i];
    }

    std::vector<cv::Mat> weights(layerNumber.size() - 1);
    for (int i = 0; i < weights.size(); i++) {
        weights[i] = cv::Mat::zeros(layerNumber[i + 1], layerNumber[i], CV_32FC1);
        for (int j = 0; j < weights[i].rows; j++) {
            for (int k = 0; k < weights[i].cols; k++) {
                load >> weights[i].at<float>(j, k);
            }
        }
    }

    std::vector<cv::Mat> bias(layerNumber.size() - 1);
    for (int i = 0; i < bias.size(); i++) {
        bias[i] = cv::Mat::zeros(layerNumber[i + 1], 1, CV_32FC1);
        for (int j = 0; j < bias[i].rows; j++) {
            for (int k = 0; k < bias[i].cols; k++) {
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

void BmpDigitRecognition::SetStdInputOutput(HANDLE stdIn, HANDLE stdOut) {
    _hStdIn     = stdIn;
    _hStdOut    = stdOut;
}

int BmpDigitRecognition::SingleTest(char *path) {

    cv::Mat     tOutput;
    double      tCompare;
    int         tOutputResult;

    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    SetInput(image);

    ForwardPropagation();

    tOutput         = GetOutputLayer();

    tOutputResult   = INT_MIN;
    tCompare        = DBL_MIN;

    for (int i = 0; i < tOutput.rows; i++) {
        if (tCompare < tOutput.at<float>(i, 0)) {
            tCompare = tOutput.at<float>(i, 0);
            tOutputResult = i;
        }
    }

    return tOutputResult;
}

void BmpDigitRecognition::Terminate() {
    _stopTraining = true;
}