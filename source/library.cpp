#include "library.h"


void CreateBmpDigitRecognition(void) {
    pBmpDigitRecognition    = new Image_MLP();
}
//============================================================================================================
void DeleteBmpDigitRecognition(void) {
    delete pBmpDigitRecognition;
    pBmpDigitRecognition    = nullptr;
}
//============================================================================================================
bool IsPBmpDigitRecognitionEmpty(void) {
    return pBmpDigitRecognition == nullptr;
}
//============================================================================================================
void InitBmpDigitRecognitionNeuralNet(std::vector<int> &layerNeuralNumber) {
    pBmpDigitRecognition->Init();
    pBmpDigitRecognition->InitNeuralNet(std::move(layerNeuralNumber));
}
//============================================================================================================
void SetBmpDigitRecognition(int     activation_function,
                            int     output_function,
                            int     loss_function,
                            double  learning_rate,
                            double  lambda,
                            double  threshold,
                            int     epoch,
                            int     init_weight_type,
                            double  init_weight_mean,
                            double  init_weight_standard_deviation,
                            int     batch_size,
                            int     optimizer,
                            double  beta1,
                            double  beta2,
                            int     thread) {
    pBmpDigitRecognition->InitTrainNeuralNet(batch_size);
    pBmpDigitRecognition->SetFunctions(static_cast<ACTIVATION_FUNCTION>(activation_function),
                                       static_cast<OUTPUT_FUNCTION>(output_function),
                                       static_cast<LOSS_FUNCTION>(loss_function));
    pBmpDigitRecognition->SetParameter(thread,
                                       learning_rate,
                                       lambda,
                                       static_cast<Optimizer::TYPE>(optimizer),
                                       beta1,
                                       beta2);
    if(init_weight_type != 2) {
        pBmpDigitRecognition->SetInitWeightValue(static_cast<DISTRIBUTION>(init_weight_type),
                                                 init_weight_mean,
                                                 init_weight_standard_deviation);
        pBmpDigitRecognition->SetInitBiasValue(cv::Scalar(0.0));
    }

    pBmpDigitRecognition->SetEpoch(epoch);
    pBmpDigitRecognition->SetThreshold(threshold);
}
//============================================================================================================
void SetBmpDigitRecognitionModelPath(char *path) {
    pBmpDigitRecognition->LoadModel(path);
}
//============================================================================================================
std::vector<double> TrainBmpDigitRecognition(const char *path, HANDLE stdIn, HANDLE stdOut) {
    pBmpDigitRecognition->SetTrainDataSetPath(path);
    return pBmpDigitRecognition->Train();
}
//============================================================================================================
bool SaveBmpDigitRecognition(char *path, char *name, bool override) {
    return pBmpDigitRecognition->SaveModel(path, name, override);
}
//============================================================================================================
double TestBmpDigitRecognition(const char *path) {
    pBmpDigitRecognition->SetTestDataSetPath(path);
    return pBmpDigitRecognition->Test();
}
//============================================================================================================
int SingleTestBmpDigitRecognition(const char *path) {
    pBmpDigitRecognition->SetPredictDataPath(path);
    return pBmpDigitRecognition->Predict();
}
//============================================================================================================
void TerminateTrainBmpDigitRecognition() {
    pBmpDigitRecognition->Terminate();
}
//============================================================================================================