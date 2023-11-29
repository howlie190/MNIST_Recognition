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

LIBRARY_API void                CreateBmpDigitRecognition(void);                        //建立神經網路物件的實體
LIBRARY_API void                DeleteBmpDigitRecognition(void);                        //釋放神經網路
LIBRARY_API bool                IsPBmpDigitRecognitionEmpty(void);                      //確認網路是否已經建立
LIBRARY_API void                InitBmpDigitRecognitionNeuralNet(std::vector<int> &);   //初始化神經網路
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
                                                        int);                           //設定網路參數
LIBRARY_API void                SetBmpDigitRecognitionModelPath(char *path);            //讀取網路模型
LIBRARY_API std::vector<double> TrainBmpDigitRecognition(const char *path, HANDLE stdIn, HANDLE stdOut);    //訓練神經網路
LIBRARY_API bool                SaveBmpDigitRecognition(char* path, char* name, bool override);             //儲存網路模型
LIBRARY_API double              TestBmpDigitRecognition(const char*);                   //測試網路模型
LIBRARY_API int                 SingleTestBmpDigitRecognition(const char *path);        //預測
LIBRARY_API void                TerminateTrainBmpDigitRecognition();                    //終止訓練

#ifdef __cplusplus
}
#endif

#endif //MACHINELEARNING_LIBRARY_H
