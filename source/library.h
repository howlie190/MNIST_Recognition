#ifndef MACHINELEARNING_LIBRARY_H
#define MACHINELEARNING_LIBRARY_H

#include "src/ImageMLP.h"

#ifdef MachineLearning_EXPORTS
#define LIBRARY_API __declspec(dllexport)
#else
#define LIBRARY_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

Image_MLP*  pBmpDigitRecognition    = nullptr;

LIBRARY_API void                CreateBmpDigitRecognition(void);
LIBRARY_API void                DeleteBmpDigitRecognition(void);
LIBRARY_API bool                IsPBmpDigitRecognitionEmpty(void);
LIBRARY_API void                InitBmpDigitRecognitionNeuralNet(std::vector<int> &);
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
                                                        double,
                                                        int);
LIBRARY_API void                SetBmpDigitRecognitionModelPath(char *path);
LIBRARY_API std::vector<double> TrainBmpDigitRecognition(const char *path, HANDLE stdIn, HANDLE stdOut);
LIBRARY_API bool                SaveBmpDigitRecognition(char* path, char* name, bool override);
LIBRARY_API double              TestBmpDigitRecognition(const char*);
LIBRARY_API int                 SingleTestBmpDigitRecognition(const char *path);
LIBRARY_API void                TerminateTrainBmpDigitRecognition();

#ifdef __cplusplus
}
#endif

#endif //MACHINELEARNING_LIBRARY_H
