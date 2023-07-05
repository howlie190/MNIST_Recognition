#ifndef MNIST_RECOGNITION_LIBRARY_LIBRARY_H
#define MNIST_RECOGNITION_LIBRARY_LIBRARY_H

#include "include/BmpDigitRecognition.h"
#include <vector>
#include <boost/thread.hpp>

#ifdef MNIST_Recognition_Library_EXPORTS
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API __declspec(dllimport)
#endif

BmpDigitRecognition *pBmpDigitRecognition = nullptr;

#ifdef __cplusplus
extern "C" {
#endif

LIBRARY_API void                CreateBmpDigitRecognition(void);
LIBRARY_API void                DeleteBmpDigitRecognition(void);
LIBRARY_API bool                IsPBmpDigitRecognitionEmpty(void);
LIBRARY_API void                InitBmpDigitRecognitionNeuralNet(std::vector<int> layerNeuralNumber);
LIBRARY_API void                SetBmpDigitRecognition( int,
                                                        int,
                                                        int,
                                                        double,
                                                        double,
                                                        double,
                                                        int,
                                                        int,
                                                        double,
                                                        double,
                                                        int,
                                                        int,
                                                        double,
                                                        double);
LIBRARY_API void                SetBmpDigitRecognitionModelPath(char *path);
LIBRARY_API std::vector<double> TrainBmpDigitRecognition(char *path, HANDLE stdIn, HANDLE stdOut);
LIBRARY_API bool                SaveBmpDigitRecognition(char* path, char* name, bool override);
LIBRARY_API double              TestBmpDigitRecognition(char*);
LIBRARY_API int                 SingleTestBmpDigitRecognition(char *path);
LIBRARY_API void                TerminateTrainBmpDigitRecognition();

#ifdef __cplusplus
}
#endif

#endif //MNIST_RECOGNITION_LIBRARY_LIBRARY_H
