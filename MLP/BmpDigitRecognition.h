//
// Created by howli on 2023/1/25.
//

#ifndef MLP_BMPDIGITRECOGNITION_H
#define MLP_BMPDIGITRECOGNITION_H
#include "MLP.h"

struct DataFormat {
    std::vector<int> layer;
    std::vector<cv::Mat> weight;
    std::vector<cv::Mat> bias;
};


class BmpDigitRecognition : public MLP {
public:
    BmpDigitRecognition() {
        _threshold  = 0.001;
        _loopCount  = 10;
    }
    void Train(char* path) override;
    void Test(char* path) override;
    bool Save(char* path, char* name, bool override) override;
    bool Load(char* path) override;
    void SetInput(cv::Mat mat) override;
    void SetThreshold(double threshold) { _threshold = threshold; }
    void SetLoopCount(int loopCount) { _loopCount = loopCount; }
    void TrainHelper(char* path);
private:
    double  _threshold;
    int     _loopCount;
};


#endif //MLP_BMPDIGITRECOGNITION_H
