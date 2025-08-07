/*

*/
#define LOG_TELEMETRY 
//#define LOG_MKL

#include <cuda_runtime.h>
#include "fft_wrapper.h"
#include <filesystem> // For directory iteration
const char *mkl_report_file = "mkl_report.txt";
const char *cuda_report_file = "cuda_report.txt";

namespace fs = std::filesystem;

void FFTRun::open_all_files() 
{
  for (const auto &entry : std::filesystem::directory_iterator(mapdir_)) {
    if (!entry.is_regular_file())
      continue;

    FileMapping fmap;
    if (fmap.create(entry.path().wstring()))
      mapped_files_.emplace_back(std::move(fmap));
  }
}

void mkl_fft(const char *mapdir, const size_t chunk_size) {
#ifdef LOG_TELEMETRY
  SignalReport report;
#endif

  auto start = std::chrono::high_resolution_clock::now();
  auto voltage = std::make_unique<float[]>(chunk_size);
  auto fft_input = std::make_unique<float[]>(chunk_size * 2);

  std::printf("Beginning MKLMemMapFFT, chunk size: %zu\n", chunk_size);
  size_t row = 0;

  for (const auto &entry : std::filesystem::directory_iterator(mapdir)) {
    if (!entry.is_regular_file())
      continue;

    const std::wstring filepath = entry.path().wstring();

    HANDLE hFile =
        CreateFileW(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
      std::cerr << "Failed to open file: "
                << std::string(filepath.begin(), filepath.end()) << "\n";
      continue;
    }

    HANDLE hMapping =
        CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMapping) {
      std::cerr << "Failed to create file mapping: "
                << std::string(filepath.begin(), filepath.end()) << "\n";
      CloseHandle(hFile);
      continue;
    }

    size_t filesize = static_cast<size_t>(GetFileSize(hFile, nullptr));
    if (filesize == 0 || filesize % sizeof(float) != 0) {
      std::cerr << "Invalid file size: "
                << std::string(filepath.begin(), filepath.end()) << "\n";
      CloseHandle(hMapping);
      CloseHandle(hFile);
      continue;
    }

    void *mapped = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!mapped) {
      std::cerr << "MapViewOfFile failed: "
                << std::string(filepath.begin(), filepath.end()) << "\n";
      CloseHandle(hMapping);
      CloseHandle(hFile);
      continue;
    }

    float *data_ptr = reinterpret_cast<float *>(mapped);
    size_t usable_floats = filesize / sizeof(float);
    size_t chunks_in_file = usable_floats / chunk_size;

    for (size_t i = 0; i < chunks_in_file; ++i, ++row) {
      std::memcpy(voltage.get(), data_ptr + i * chunk_size,
                  chunk_size * sizeof(float));

      // Prepare FFT input: interleaved real/imag
      for (size_t j = 0; j < chunk_size; ++j) {
        fft_input[j * 2 + 0] = voltage[j]; // real
        fft_input[j * 2 + 1] = 0.0f;       // imag
      }

      DFTI_DESCRIPTOR_HANDLE descriptor;
      MKL_LONG status;

      status = DftiCreateDescriptor(&descriptor, DFTI_SINGLE, DFTI_COMPLEX, 1,
                                    chunk_size);
      status = DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_INPLACE);
      status = DftiCommitDescriptor(descriptor);
      status = DftiComputeForward(descriptor, fft_input.get());
      status = DftiFreeDescriptor(&descriptor);

#ifdef LOG_TELEMETRY
      report.accumulate_spectrum(fft_input.get(), chunk_size, i);
#endif

#ifdef LOG_MKL
      for (size_t k = 0; k < chunk_size; ++k) {
        float real = fft_input[k * 2 + 0];
        float imag = fft_input[k * 2 + 1];
        std::printf("Bin %4zu: % .6f + % .6fi\n", k, real, imag);
      }
#endif
    }

    UnmapViewOfFile(mapped);
    CloseHandle(hMapping);
    CloseHandle(hFile);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  log_fft("MKL Memmap", row, chunk_size, elapsed);

#ifdef LOG_TELEMETRY
  report.dump_to_text(mkl_report_file);
#endif
} 
/////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////
void cuda_fft(const char *mapdir, const size_t chunk_size) 
{
#ifdef LOG_TELEMETRY
  struct SignalReport report;
#endif

  auto start = std::chrono::high_resolution_clock::now();
  auto voltage = std::make_unique<float[]>(chunk_size); 
  auto fft_input = std::make_unique<float[]>(chunk_size * 2); 

  // Allocate device buffer for FFT
  cufftComplex *d_data = nullptr;
  cudaMalloc(&d_data, chunk_size * sizeof(cufftComplex));

  // Create FFT plan
  cufftHandle plan;
  cufftResult plan_status = cufftPlan1d(&plan, chunk_size, CUFFT_C2C, 1);
  if (plan_status != CUFFT_SUCCESS) {
    std::fprintf(stderr, "FFT plan creation failed: %d\n", plan_status);
    cudaFree(d_data);
    return;
  }

  std::printf("Beginning CudaMemMapFFT, chunk size: %zu\n", chunk_size);

  size_t row = 0;

  for (const auto &entry : std::filesystem::directory_iterator(mapdir)) {
    if (!entry.is_regular_file())
      continue;

    const std::wstring filepath = entry.path().wstring();

    HANDLE hFile =
        CreateFileW(filepath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);

    if (hFile == INVALID_HANDLE_VALUE) {
      std::cerr << "Failed to open file: "
                << std::string(filepath.begin(), filepath.end()) << "\n";
      continue;
    }

    HANDLE hMapping =
        CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMapping) {
      std::cerr << "Failed to create file mapping: "
                << std::string(filepath.begin(), filepath.end()) << "\n";
      CloseHandle(hFile);
      continue;
    }

    size_t filesize = static_cast<size_t>(GetFileSize(hFile, nullptr));
    if (filesize == 0 || filesize % sizeof(float) != 0) {
      std::cerr << "Invalid file size: "
                << std::string(filepath.begin(), filepath.end()) << "\n";
      CloseHandle(hMapping);
      CloseHandle(hFile);
      continue;
    }

    void *mapped = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!mapped) {
      std::cerr << "MapViewOfFile failed: "
                << std::string(filepath.begin(), filepath.end()) << "\n";
      CloseHandle(hMapping);
      CloseHandle(hFile);
      continue;
    }

    float *data_ptr = reinterpret_cast<float *>(mapped);
    size_t usable_floats = filesize / sizeof(float);
    size_t chunks_in_file = usable_floats / chunk_size;


    for (size_t i = 0; i < chunks_in_file; ++i, ++row) {
      std::memcpy(voltage.get(), data_ptr + i * chunk_size,// replace this with CudaMemcpy
                  chunk_size * sizeof(float)); // ditto

      // Prepare FFT input: interleaved real/imag
      for (size_t j = 0; j < chunk_size; ++j) {
        fft_input[j * 2 + 0] = voltage[j]; // real
        fft_input[j * 2 + 1] = 0.0f;       // imag
      }
      // Copy input to device
      cudaMemcpy(d_data, fft_input.get(), chunk_size * sizeof(cufftComplex),
                 cudaMemcpyHostToDevice);

      // Execute FFT
      cufftResult result = cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
      if (result != CUFFT_SUCCESS) {
        std::printf("FFT failed: %d\n", result);
        continue;
      }

      // Copy result back to host
      cudaMemcpy(fft_input.get(), d_data, chunk_size * sizeof(cufftComplex),
                 cudaMemcpyDeviceToHost);
#ifdef LOG_TELEMETRY
      report.accumulate_spectrum(fft_input.get(), chunk_size, i);
#endif
#ifdef LOG_CUDA
      for (size_t k = 0; k < chunk_size; ++k) {
        float real = fft_input[k * 2 + 0];
        float imag = fft_input[k * 2 + 1];
        std::printf("Bin %4zu: % .6f + % .6fi\n", k, real, imag);
      }
#endif
    }

    UnmapViewOfFile(mapped);
    CloseHandle(hMapping);
    CloseHandle(hFile);
  }

  cufftDestroy(plan);
  cudaFree(d_data);

  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  log_fft("Cuda fft", row, chunk_size, elapsed);
#ifdef LOG_TELEMETRY
  report.dump_to_text(cuda_report_file);
#endif
}
void log_fft(const char *label, size_t rows, size_t chunk_size,
             std::chrono::nanoseconds elapsed) 
{
  size_t total_floats = rows * chunk_size;
  double elapsed_ms = elapsed.count() / 1e6;
  double ns_per_float = static_cast<double>(elapsed.count()) / total_floats;

  std::printf("%s FFT (%zu rows), %zu floats took %.2f ms (%.2f ns/float)\n",
              label, rows, total_floats, elapsed_ms, ns_per_float);

}
