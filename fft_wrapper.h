#pragma once
#include <algorithm>
#include <array>
#include <cctype>       // For std::isspace
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>       // For std::unique_ptr
#include <sstream>
#include <string>
#include <vector>
#include <windows.h>

#include <cufft.h>
#include "svm_model.h"
struct FrequencyResponseMatrix {
  // Placeholder: define dimensions and structure as needed
  std::vector<std::vector<float>> matrix;
};

struct SignalReport {
  // Metadata
  size_t chunk_index = 0;
  double timestamp_offset = 0.0; // seconds
  std::string signal_name;

  // Voltage Statistics
  float max_voltage = 0.0f;
  float min_voltage = 0.0f;
  float voltage_range = 0.0f;
  float mean_voltage = 0.0f;
  float std_dev_voltage = 0.0f;
  size_t outlier_count = 0;
  float outlier_density = 0.0f;

  // Frequency & Power Metrics
  float dominant_frequency = 0.0f;
  float spectral_centroid = 0.0f;
  float spectral_spread = 0.0f;
  float power_per_batch = 0.0f;
  FrequencyResponseMatrix freq_response;

  // Distribution & Trend Metrics
  std::vector<size_t> histogram_bins; // e.g., 10–20 bins
  float rolling_mean = 0.0f;          // optional
  float rolling_stddev = 0.0f;        // optional

  // Anomaly Flags
  bool saturation_detected = false;
  bool dropout_detected = false;
  int svm_prediction = -1;    // -1 = normal, other = anomaly class
  float anomaly_score = 0.0f; // optional confidence

  // Methods (optional)
  void reset() { *this = SignalReport{}; }

  // Extend with serialization, comparison, etc. as needed
};




void CUDAFFT(const char *logfile, const int header_offset,
             const size_t total_rows, const size_t chunk_size);

void CUDAMemMapFFT(const char *mapdir, const size_t chunk_size);
void SVMOutliers();

