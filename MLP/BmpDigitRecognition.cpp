//
// Created by howli on 2023/1/25.
//

#include "BmpDigitRecognition.h"
#include <windows.h>
#include <opencv2/opencv.hpp>
#include <climits>
#include <float.h>

void BmpDigitRecognition::SetInput(cv::Mat mat) {
    cv::Mat temp(INPUT_LAYER_SIZE, 1, CV_32FC1);
    for (int i = 0, k = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++, k++) {
            temp.at<float>(k, 0) = (int) mat.at<uchar>(i, j);
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
    int                 tCurrentTarget;

    strcpy(tPath, path);
    strcat(tPath, "*.*");

    HANDLE hFind = ::FindFirstFile(tPath, &tFD);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(tFD.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                char*   tFileName = tFD.cFileName;
                char    tFilePath[CHAR_MAX];

                strcpy(tFilePath, path);
                strcat(tFilePath, tFileName);

                cv::Mat image = cv::imread(tFilePath);

                SetInput(image);

                 if(tCurrentTarget != std::atoi(&tFileName[0])) {
                    tFlagThreshold  = false;
                    tFlagFirst      = true;
                } else if(tFlagThreshold) {
                    continue;
                }

                tCurrentTarget                              = std::atoi(&tFileName[0]);
                cv::Mat             tTarget                 = cv::Mat::zeros(OUTPUT_LAYER_SIZE, 1, CV_32FC1);
                tTarget.at<float>(tCurrentTarget, 0)    = 1.0;
                SetTarget(tTarget);

                ForwardPropagation();

                if(tFlagFirst) {
                    std::cout << tFileName[0] << " : " << GetLoss() << std::endl;
                    tFlagFirst      = false;
                }

                BackPropagation();

                if(GetLoss() < _threshold) {
                    tFlagThreshold  = true;
                }
            }
        } while (::FindNextFile(hFind, &tFD));
    }

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
    strcat(tPath, "*.*");

    hFind = ::FindFirstFile(tPath, &tFD);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(tFD.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                char*   tFileName = tFD.cFileName;
                char    tFilePath[CHAR_MAX];

                strcpy(tFilePath, path);
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

void BmpDigitRecognition::Save(char *path) {

}

void BmpDigitRecognition::Load(char *path) {

}