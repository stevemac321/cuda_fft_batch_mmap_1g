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
#include <complex>
#include <cstddef>
#include <algorithm>
#include <cmath>

// Windows API
#include <windows.h>
#include <cufft.h>
#include "mkl.h"

struct SignalReport {
  int chunk_index = 0;

  struct FrequencyResponse {
    std::vector<std::vector<float>> matrix;
    std::vector<float> dominant_frequency;
    std::vector<float> spectral_centroid;
    std::vector<float> spectral_spread;
    std::vector<float> power;
  } freq_response;

  // Set chunk index
  void set_chunk_index(int idx) { chunk_index = idx; }

  // Compute and store dominant frequency
  void compute_dominant_frequency(const std::vector<float> &magnitudes) {
    auto max_it = std::max_element(magnitudes.begin(), magnitudes.end());
    float value = std::distance(magnitudes.begin(), max_it);
    freq_response.dominant_frequency.push_back(value);
  }

  // Compute and store spectral centroid
  void compute_spectral_centroid(const std::vector<float> &magnitudes) {
    float sum = 0.0f, weighted_sum = 0.0f;
    for (size_t i = 0; i < magnitudes.size(); ++i) {
      weighted_sum += i * magnitudes[i];
      sum += magnitudes[i];
    }
    float value = (sum > 0.0f) ? weighted_sum / sum : 0.0f;
    freq_response.spectral_centroid.push_back(value);
  }

  // Compute and store spectral spread
  void compute_spectral_spread(const std::vector<float> &magnitudes) {
    float centroid =
        freq_response.spectral_centroid.back(); // use last computed centroid
    float sum = 0.0f, variance = 0.0f;
    for (size_t i = 0; i < magnitudes.size(); ++i) {
      float diff = i - centroid;
      variance += diff * diff * magnitudes[i];
      sum += magnitudes[i];
    }
    float value = (sum > 0.0f) ? std::sqrt(variance / sum) : 0.0f;
    freq_response.spectral_spread.push_back(value);
  }

  // Compute and store power
  void compute_power(const std::vector<float> &magnitudes) {
    float total = 0.0f;
    for (float m : magnitudes)
      total += m * m;
    freq_response.power.push_back(total);
  }

  // Append magnitudes to matrix
  void append_spectrum(const std::vector<float> &magnitudes) {
    freq_response.matrix.push_back(magnitudes);
  }

  // Dump all data to text file
  void dump_to_text(const std::string &filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
      std::cerr << "Failed to open file: " << filename << "\n";
      return;
    }

    out << "Spectrum Report (" << freq_response.matrix.size()
        << " chunks):\n\n";

    for (size_t i = 0; i < freq_response.matrix.size(); ++i) {
      out << "Chunk Index: " << i << "\n";
      out << "Dominant Frequency: " << freq_response.dominant_frequency[i]
          << "\n";
      out << "Spectral Centroid: " << freq_response.spectral_centroid[i]
          << "\n";
      out << "Spectral Spread: " << freq_response.spectral_spread[i] << "\n";
      out << "Power per Batch: " << freq_response.power[i] << "\n";
      out << "Spectrum:";
      for (float val : freq_response.matrix[i]) {
        out << " " << val;
      }
      out << "\n\n";
    }

    out.close();
  }
};
void cuda_fft(const char *mapdir, const size_t chunk_size = 8192);

void mkl_fft(const char *mapdir, const size_t chunk_size = 8192);

void log_fft(const char *label, size_t rows, size_t chunk_size,
             std::chrono::nanoseconds elapsed);



