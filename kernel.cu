
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include "fft_wrapper.h"

const char *memmap_dir =
    R"(C:\repos\CudaBigData\CudaBigData\memmaps)";

static const size_t chunk_sizes[] = {8192, 16384, 32768, 65536, 131072, 262144, 1048576, 2097152, 4194304};
std::ofstream out; // global perf logger for the lifetime of the program, all runs.

int main()
{
    // we escalate chunk size
    // open logger for all runs
    out.open("perf.txt", std::ios::app);

    for (size_t chunk_size : chunk_sizes) {
      cuda_fft(memmap_dir, chunk_size);
      //mkl_fft(memmap_dir, chunk_size);
    }
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
     cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    return 0;
}

