//
// Created by LG on 2023/4/1.
//

#ifndef MNIST_RECOGNITION_LIBRARY_MYTHREAD_H
#define MNIST_RECOGNITION_LIBRARY_MYTHREAD_H

#include <boost/thread.hpp>
#include "MLP.h"

typedef void (MLP::*pRun)(bool, int);

class MyThread {
public:
    MyThread() : _isRunning(false), _isPaused(false) {};
    void SetRun(pRun f) { _Run = f; }
    void SetSleepTime(int t) { _sleepTime = t; }
    void SetLoop(bool l) { _isLoop = l; }
    void Start(void);
    void Pause(void);
    void Resume(void);
private:
    void Run(void);

    bool            _isRunning;
    bool            _isPaused;
    bool            _isLoop;
    int             _sleepTime;
    pRun            _Run;
    boost::thread   _thread;
};


#endif //MNIST_RECOGNITION_LIBRARY_MYTHREAD_H
