#include "fft_wrapper.h"
#include <filesystem> // For directory iteration

namespace fs = std::filesystem;

// USER CODE BEGIN

void CUDAMemMapFFT(const char *mapdir, const size_t chunk_size) {
  auto voltage = std::make_unique<float[]>(chunk_size);
  auto fft_input = std::make_unique<float[]>(chunk_size * 2);

  cufftHandle plan;
  cufftComplex *data = reinterpret_cast<cufftComplex *>(fft_input.get());
  cufftPlan1d(&plan, chunk_size / 2, CUFFT_C2C, 1);

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

      for (size_t j = 0; j < chunk_size; ++j) {
        fft_input[j * 2 + 0] = voltage[j];
        fft_input[j * 2 + 1] = 0.0f;
      }

      cufftExecC2C(plan, data, data, CUFFT_FORWARD);

#ifdef LOG_FFT
      for (size_t k = 0; k < chunk_size / 2; ++k) {
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
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> elapsed = end - start;

  std::printf("CUDA FFT (%zu rows), %zu floats took %.2f ms\n", row,
              row * chunk_size, elapsed.count() / 1000.0);
}
// #define LOG_FFT
void CUDAFFT(const char *logfile, const int header_offset,
             const size_t total_rows, const size_t chunk_size) {

  auto voltage = std::make_unique<float[]>(chunk_size);
  auto fft_input = std::make_unique<float[]>(chunk_size * 2);

  std::ifstream ifs(logfile, std::ios::binary);
  if (!ifs) {
    std::cerr << "Failed to open file: " << logfile << "\n";
    return;
  }

  // FFT setup once, reused each pass
  cufftHandle plan;
  cufftComplex *data = reinterpret_cast<cufftComplex *>(fft_input.get());
  cufftPlan1d(&plan, chunk_size / 2, CUFFT_C2C, 1);

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

    // Perform FFT
    cufftExecC2C(plan, data, data, CUFFT_FORWARD);

    // Optionally post-process FFT result
#ifdef LOG_FFT
    for (size_t i = 0; i < chunk_size / 2; ++i) {
      float real = fft_input[i * 2 + 0];
      float imag = fft_input[i * 2 + 1];
      std::printf("Bin %4zu: % .6f + % .6fi\n", i, real, imag);
    }

#endif
  }
  cufftDestroy(plan);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> elapsed = end - start;

  std::printf("CUDA FFT (%zu rows), %zu Floats took %.2f ms\n", total_rows,
              total_rows * chunk_size, elapsed.count() / 1000.0);
}
/*
void CUDAFFT(const char *logfile, const int header_offset,
             const size_t total_samples, const size_t chunk_size)
{

  auto voltage = std::make_unique<float[]>(chunk_size);

  std::ifstream ifs(logfile, std::ios::binary);
  if (!ifs) {
    std::cerr << "Failed to open file: " << logfile << "\n";
    return;
  }

  // Reads all floats
  ifs.read(reinterpret_cast<char *>(voltage.get()), rowsize * sizeof(float));

  if (!ifs) {
    std::cerr << "File read incomplete or failed\n";
    return;
  }
  // Prepare FFT input (interleaved complex)
  auto fft_input = std::make_unique<float[]>(fft_size * 2);
  for (size_t i = 0; i < chunk_size; ++i) {
    fft_input[i * 2 + 0] = voltage[i]; // Real
    fft_input[i * 2 + 1] = 0.0f;       // Imag
  }

  cufftComplex *data = reinterpret_cast<cufftComplex *>(voltage.get());
  cufftHandle plan;
  const int fft_buffer = (chunk_size / 2);

  printf("beginning FFT\n");
  auto start = std::chrono::high_resolution_clock::now();
  // put FFT call here


  cufftPlan1d(&plan, fft_buffer, CUFFT_C2C, 1);
  cufftExecC2C(plan, data, data, CUFFT_FORWARD);
  cufftDestroy(plan);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> elapsed = end - start;


#ifdef LOG_FFT
  for (size_t i = 0; i < fft_size; ++i) {
    float real = fft_input[i * 2 + 0];
    float imag = fft_input[i * 2 + 1];
    std::printf("Bin %4zu: % .6f + % .6fi\n", i, real, imag);
  }
#endif

  std::printf("CUDA FFT (%zu samples) took %.2f ms\n", fft_size,
              elapsed.count());

}
*/