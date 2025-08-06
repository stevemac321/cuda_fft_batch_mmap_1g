/*

*/

#include <cuda_runtime.h>
#include "fft_wrapper.h"
#include <filesystem> // For directory iteration

namespace fs = std::filesystem;

extern float outliers[10][128];

// USER CODE BEGIN
#define LOG_FFT
void CUDAMemMapFFT(const char *mapdir, const size_t chunk_size) {
  auto voltage = std::make_unique<float[]>(chunk_size);
  auto fft_input =
      std::make_unique<float[]>(chunk_size * 2); // interleaved real/imag

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

  std::printf("Beginning FFT...\n");
  auto start = std::chrono::high_resolution_clock::now();

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

      // SVM prediction on raw voltage data
      auto pred = 0;
      arm_svm_polynomial_predict_f32(&svm, voltage.get(), &pred);
      if (pred != 1) {
        std::printf("Anomaly detected at row %zu (pred = %d)\n", row, pred);
      }

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

#ifdef LOG_FFT
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
  std::chrono::duration<double, std::micro> elapsed = end - start;

  std::printf("CUDA FFT (%zu rows), %zu floats took %.2f ms\n", row,
              row * chunk_size, elapsed.count() / 1000.0);
}

// #define LOG_FFT
void CUDAFFT(const char *logfile, const int header_offset,
             const size_t total_rows, const size_t chunk_size) {

  auto voltage = std::make_unique<float[]>(chunk_size);
  auto fft_input =
      std::make_unique<float[]>(chunk_size * 2); // interleaved real/imag

  std::ifstream ifs(logfile, std::ios::binary);
  if (!ifs) {
    std::cerr << "Failed to open file: " << logfile << "\n";
    return;
  }

  // Skip header if needed
  ifs.seekg(header_offset, std::ios::beg);

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

  printf("Beginning FFT...\n");
  auto start = std::chrono::high_resolution_clock::now();

  for (size_t row = 0; row < total_rows; ++row) {
    ifs.read(reinterpret_cast<char *>(voltage.get()),
             chunk_size * sizeof(float));
    if (ifs.gcount() != chunk_size * sizeof(float)) {
      std::cerr << "Incomplete row read at row " << row << "\n";
      break;
    }

    // Populate complex buffer: real = voltage[i], imag = 0.0f
    for (size_t i = 0; i < chunk_size; ++i) {
      fft_input[i * 2 + 0] = voltage[i]; // Real
      fft_input[i * 2 + 1] = 0.0f;       // Imag
    }

    // Copy input to device
    cudaMemcpy(d_data, fft_input.get(), chunk_size * sizeof(cufftComplex),
               cudaMemcpyHostToDevice);

    // Perform FFT
    cufftResult exec_status = cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    if (exec_status != CUFFT_SUCCESS) {
      std::printf("FFT execution failed at row %zu: %d\n", row, exec_status);
      continue;
    }

    // Copy result back to host
    cudaMemcpy(fft_input.get(), d_data, chunk_size * sizeof(cufftComplex),
               cudaMemcpyDeviceToHost);

#ifdef LOG_FFT
    for (size_t k = 0; k < chunk_size; ++k) {
      float real = fft_input[k * 2 + 0];
      float imag = fft_input[k * 2 + 1];
      std::printf("Bin %4zu: % .6f + % .6fi\n", k, real, imag);
    }
#endif
  }

  cufftDestroy(plan);
  cudaFree(d_data);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> elapsed = end - start;

  std::printf("CUDA FFT (%zu rows), %zu Floats took %.2f ms\n", total_rows,
              total_rows * chunk_size, elapsed.count() / 1000.0);
}
void SVMOutliers() {

  printf("outlier test \n");
  int i = 0;
  int correct = 0;

  for (; i < 10; i++) {

    int32_t pred = 0;
    arm_svm_polynomial_predict_f32(&svm, outliers[i], &pred);
    if (pred == -1) {
      correct++;
    }
    std::printf("Row: %d, pred: %d\n", i, pred);
  }
  std::printf("\n---- Final Report ----\n");
  std::printf("Total Rows Processed: %d\n", i);
  std::printf("Correct Predictions:    %d\n", correct);
  std::printf("----------------------\n");
}
