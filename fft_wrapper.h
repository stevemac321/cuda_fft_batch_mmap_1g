#pragma once
// Standard Library
#include <algorithm>
#include <array>
#include <cctype>           // std::isspace
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>           // std::unique_ptr
#include <sstream>
#include <string>
#include <vector>

// Windows API
#include <windows.h>
#include <cufft.h>
#include "mkl.h"

void cuda_fft(const char *mapdir, const size_t chunk_size);

void mkl_fft(const char *mapdir, const size_t chunk_size);
void log_fft(const char *label, size_t rows, size_t chunk_size,
             std::chrono::nanoseconds elapsed);

