//
// Created by howli on 2023/1/25.
//

#ifndef MLP_BMPDIGITRECOGNITION_H
#define MLP_BMPDIGITRECOGNITION_H

#include "MLP.h"
#include <windows.h>
#include <fstream>

#ifdef MNIST_Recognition_Library_EXPORTS
#define BDR_API __declspec(dllexport)
#else
#define BDR_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

class BmpDigitRecognition : public MLP {
public:
    double              Test(char*) override;

    int                 SingleTest(char*);

    bool                Save(char*, char*, bool) override;
    bool                Load(char*) override;

    std::vector<double> Train(char*) override;
    void                SetInputTraining(const int, const cv::Mat&) override;
    void                SetInput(const cv::Mat&) override;
    void                SetThreshold(const double);
    void                SetLoopCount(const int);
    void                TrainHelperMultiThread();
    void                TrainHelperSingleThread();
    void                Terminate();
    void                SetOptimizer(OPTIMIZER);
    void                SetBeta1Beta2(double, double);
private:
    std::vector<std::pair<std::string, cv::Mat>>        _trainingData;
    std::vector<std::pair<std::string, cv::Mat>>        _testingData;

    std::vector<std::pair<std::string, std::string>>    _fileName;

    std::vector<double>                                 _lossValue;

    double                                              _threshold;
    double                                              _loss;

    int                                                 _loopCount;

    bool                                                _stopTraining;
    bool                                                _thresholdReached;
    OPTIMIZER                                           _optimizer;

    std::string                                         _logFile;

    HANDLE                                              _hMapFile;
    HANDLE                                              _hEventRead;
    HANDLE                                              _hEventWrite;

    LPTSTR                                              _pBuf;
};

#ifdef __cplusplus
}
#endif

#endif //MLP_BMPDIGITRECOGNITION_H
