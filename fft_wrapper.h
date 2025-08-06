#pragma once
#include <algorithm>
#include <cctype> // For std::isspace
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cufft.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory> // For std::unique_ptr
#include <sstream>
#include <string> // For std::string
#include <vector>
#include <windows.h>


void CUDAFFT(const char *logfile, const int header_offset,
             const size_t total_rows, const size_t chunk_size);

void CUDAMemMapFFT(const char *mapdir, const size_t chunk_size);

