#include "library.h"

void CreateBmpDigitRecognition(void) {
    pBmpDigitRecognition = new BmpDigitRecognition();
}

void DeleteBmpDigitRecognition(void) {
    delete pBmpDigitRecognition;
    pBmpDigitRecognition = nullptr;
}

bool IsPBmpDigitRecognitionEmpty(void) {
    return pBmpDigitRecognition == nullptr ? true : false;
}

void InitBmpDigitRecognitionNeuralNet(std::vector<int> layerNeuralNumber) {
    pBmpDigitRecognition->InitNeuralNet(layerNeuralNumber);
}

void SetBmpDigitRecognition(int         activationFunction,
                            int         outputFunction,
                            int         lossFunction,
                            double      learningRate,
                            double      lambda,
                            double      threshold,
                            int         iteration,
                            int         initWeight,
                            double      mean,
                            double      standardDeviation) {
    pBmpDigitRecognition->SetActivationFunction(ACTIVATION_FUNCTION::SIGMOID);
    pBmpDigitRecognition->SetOutputActivationFunction((ACTIVATION_FUNCTION)outputFunction);
    pBmpDigitRecognition->SetLossFunction((LOSS_FUNCTION)lossFunction);
    pBmpDigitRecognition->SetLearningRate(learningRate);
    pBmpDigitRecognition->SetLambda(lambda);
    pBmpDigitRecognition->SetThreshold(threshold);
    pBmpDigitRecognition->SetLoopCount(iteration);

    if(initWeight != 2) {
        pBmpDigitRecognition->InitWeights((DISTRIBUTION)initWeight, mean, standardDeviation);
        pBmpDigitRecognition->InitBias(cv::Scalar(0.0));
    }

    if((LOSS_FUNCTION)lossFunction == LOSS_FUNCTION::MEAN_SQUARED_ERROR && (ACTIVATION_FUNCTION)outputFunction == ACTIVATION_FUNCTION::SOFTMAX) {
        pBmpDigitRecognition->SetDerivativeOutputFunction(DERIVATIVE_FUNCTION::SOFTMAX_MSE);
    }

    if((ACTIVATION_FUNCTION)activationFunction == ACTIVATION_FUNCTION::SIGMOID) {
        pBmpDigitRecognition->SetDerivativeFunction(DERIVATIVE_FUNCTION::SIGMOID);
    }
}

void SetBmpDigitRecognitionModelPath(char* path) {
    pBmpDigitRecognition->Load(path);
}

void TrainBmpDigitRecognition(char* path, HANDLE hStdIn, HANDLE hStdOut) {
    pBmpDigitRecognition->SetStdInputOutput(hStdIn, hStdOut);
    pBmpDigitRecognition->Train(path);
}

bool SaveBmpDigitRecognition(char* path, char* name, bool override) {
    return pBmpDigitRecognition->Save(path, name, override);
}

double TestBmpDigitRecognition(char* path) {
    return pBmpDigitRecognition->Test(path);
}

int SingleTestBmpDigitRecognition(char *path) {
    return pBmpDigitRecognition->SingleTest(path);
}

void TerminateTrainBmpDigitRecognition() {
    pBmpDigitRecognition->Terminate();
}