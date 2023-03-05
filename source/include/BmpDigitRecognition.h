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

struct DataFormat {
    std::vector<int>        layer;
    std::vector<cv::Mat>    weight;
    std::vector<cv::Mat>    bias;
};



class BmpDigitRecognition : public MLP {
public:
    BmpDigitRecognition();

    void        Train(char *path) override;

    double      Test(char *path) override;

    int         SingleTest(char *path);

    bool        Save(char *path, char *name, bool override) override;

    bool        Load(char *path) override;

    void        SetInput(cv::Mat mat) override;

    void        SetThreshold(double threshold);

    void        SetLoopCount(int loopCount);

    void        TrainHelper(char *path);

    void        SetSelect(int target);

    void        SetStdInputOutput(HANDLE, HANDLE);

    void        Terminate();

private:
    double          _threshold;

    int             _loopCount;
    int             _choose;
    int             _strDataLength;

    char            _sendData[1024];

    bool            _select = false;
    bool            _stopTraining;

    HANDLE          _hStdIn;
    HANDLE          _hStdOut;

    DWORD           _dwRead;
    DWORD           _dwWritten;

    std::string     _logFile;

    HANDLE          _hMapFile;
    HANDLE          _hMutex;
    HANDLE          _hEventRead;
    HANDLE          _hEventWrite;
    LPTSTR          _pBuf;
};

#ifdef __cplusplus
}
#endif

#endif //MLP_BMPDIGITRECOGNITION_H
