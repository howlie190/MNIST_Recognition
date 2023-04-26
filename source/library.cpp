#include "library.h"

void CreateBmpDigitRecognition(void) {
    pBmpDigitRecognition = new BmpDigitRecognition();
}
//============================================================================================================
void DeleteBmpDigitRecognition(void) {
    delete pBmpDigitRecognition;
    pBmpDigitRecognition = nullptr;
}
//============================================================================================================
bool IsPBmpDigitRecognitionEmpty(void) {
    return pBmpDigitRecognition == nullptr ? true : false;
}
//============================================================================================================
void InitBmpDigitRecognitionNeuralNet(std::vector<int> layerNeuralNumber) {
    pBmpDigitRecognition->InitNeuralNet(layerNeuralNumber);
}
//============================================================================================================
void SetBmpDigitRecognition(int         activationFunction,
                            int         outputFunction,
                            int         lossFunction,
                            double      learningRate,
                            double      lambda,
                            double      threshold,
                            int         iteration,
                            int         initWeight,
                            double      mean,
                            double      standardDeviation,
                            int         miniBatchSize) {
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

    if(miniBatchSize != 0) {
        pBmpDigitRecognition->SetMiniBatchSize(miniBatchSize);
    }

    if((LOSS_FUNCTION)lossFunction == LOSS_FUNCTION::MEAN_SQUARED_ERROR && (ACTIVATION_FUNCTION)outputFunction == ACTIVATION_FUNCTION::SOFTMAX) {
        pBmpDigitRecognition->SetDerivativeOutputFunction(DERIVATIVE_FUNCTION::SOFTMAX_MSE);
    } else if((LOSS_FUNCTION)lossFunction == LOSS_FUNCTION::CROSS_ENTROPY && (ACTIVATION_FUNCTION)outputFunction == ACTIVATION_FUNCTION::SOFTMAX) {
        pBmpDigitRecognition->SetDerivativeOutputFunction(DERIVATIVE_FUNCTION::SOFTMAX_CROSS_ENTROPY);
    } else if((LOSS_FUNCTION)lossFunction == LOSS_FUNCTION::MEAN_SQUARED_ERROR && (ACTIVATION_FUNCTION)outputFunction == ACTIVATION_FUNCTION::SIGMOID) {
        pBmpDigitRecognition->SetDerivativeOutputFunction(DERIVATIVE_FUNCTION::SIGMOID_MSE);
    } else if((LOSS_FUNCTION)lossFunction == LOSS_FUNCTION::CROSS_ENTROPY && (ACTIVATION_FUNCTION)outputFunction == ACTIVATION_FUNCTION::SIGMOID) {
        pBmpDigitRecognition->SetDerivativeOutputFunction(DERIVATIVE_FUNCTION::SIGMOID_CROSS_ENTROPY);
    }

    if((ACTIVATION_FUNCTION)activationFunction == ACTIVATION_FUNCTION::SIGMOID) {
        pBmpDigitRecognition->SetDerivativeFunction(DERIVATIVE_FUNCTION::SIGMOID);
    } else if((ACTIVATION_FUNCTION)activationFunction == ACTIVATION_FUNCTION::TANH) {
        pBmpDigitRecognition->SetDerivativeFunction(DERIVATIVE_FUNCTION::TANH);
    } else if((ACTIVATION_FUNCTION)activationFunction == ACTIVATION_FUNCTION::RELU) {
        pBmpDigitRecognition->SetDerivativeFunction(DERIVATIVE_FUNCTION::RELU);
    }
}
//============================================================================================================
void SetBmpDigitRecognitionModelPath(char* path) {
    pBmpDigitRecognition->Load(path);
}
//============================================================================================================
void TrainBmpDigitRecognition(char* path, HANDLE hStdIn, HANDLE hStdOut) {
    pBmpDigitRecognition->Train(path);
}
//============================================================================================================
bool SaveBmpDigitRecognition(char* path, char* name, bool override) {
    return pBmpDigitRecognition->Save(path, name, override);
}
//============================================================================================================
double TestBmpDigitRecognition(char* path) {
    return pBmpDigitRecognition->Test(path);
}
//============================================================================================================
int SingleTestBmpDigitRecognition(char *path) {
    return pBmpDigitRecognition->SingleTest(path);
}
//============================================================================================================
void TerminateTrainBmpDigitRecognition() {
    pBmpDigitRecognition->Terminate();
}
//============================================================================================================