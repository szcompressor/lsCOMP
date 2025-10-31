#ifndef LSCOMP_INCLUDE_LSCOMP_TIMER_H
#define LSCOMP_INCLUDE_LSCOMP_TIMER_H

#include <cuda.h>
#include <cuda_runtime.h>

struct PrivateTimingGPU {
    cudaEvent_t start;
    cudaEvent_t stop;
};

class TimingGPU
{
    private:
        PrivateTimingGPU *privateTimingGPU;

    public:
        TimingGPU();
        ~TimingGPU();
        void StartCounter();
        void StartCounterFlags();
        float GetCounter();

};

#endif // LSCOMP_INCLUDE_LSCOMP_TIMER_H