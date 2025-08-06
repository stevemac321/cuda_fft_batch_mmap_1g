## 🚀 `cuda_fft_batch_mmap_1gb`
## TODO - grab SVM stuff from DSPServer, copy in CMSIS-DSP, demo the big logfile compared to memory mapped files, add the .json code that captures all the global FFT data to pass to Matplotlib##
### Overview

This project processes high-throughput voltage telemetry collected from STM32F4xx microcontrollers. It performs batched FFT analysis on 1GB of data using CUDA, (TODO applies anomaly detection via CMSIS-DSP and scikit-learn SVM), and generates detailed signal reports with Matplotlib. Performance metrics are captured using CUDA Insight and Visual Studio debugging tools.

---

### 🔧 Pipeline Summary

1. **Data Acquisition**
   - Voltage packets streamed from STM32F4xx MCUs
   - Stored as binary float32 arrays via memory-mapped files
   - 12 files × 16MB = 192MB per batch (scaled to 1GB in full runs)

2. **FFT Processing**
   - (TODO run the raw voltage chunk thru outlier detection)
   - CUDA FFT on 128-float chunks (393216 rows per 192MB batch)
   - cuFFT throughput: ~106 ms for 50M+ floats
   - Frequency domain features extracted per chunk

3. **Signal Statistics (per chunk)**
   - Max Voltage  
   - Min Voltage  
   - Voltage Range  
   - Mean Voltage  
   - Standard Deviation (Noise Level)  
   - Outlier Count  
   - Outlier Density  

4. **Frequency & Power Metrics**
   - Dominant Frequency  
   - Spectral Centroid  
   - Spectral Spread  
   - Power per Batch  
   - Frequency Response Matrix (for heatmap)

5. **Distribution & Trend Metrics**
   - Histogram Binning  
   - Rolling Statistics (optional)  
   - Batch Timestamp / Time Offset  
   - Chunk Index  
   - Anomaly Flags (e.g., saturation, dropout)

6. **Anomaly Detection**
   - CMSIS-DSP SVM classifier (trained externally via scikit-learn)  
   - Inference performed on embedded targets or host  
   - Flags injected into JSON report

7. **Visualization**
   - Matplotlib report generation  
   - Signal stats, FFT heatmaps, anomaly overlays

8. **Performance Profiling (CUDA Insight / VS)**
   - Kernel Execution Time per Batch  
   - Memory Bandwidth (Host ↔ Device)  
   - SM Utilization  
   - Thread Block & Warp Efficiency  
   - Shared vs Global Memory Access  
   - Launch Config Visualizer  
   - Stream Overlap Analysis  
   - Instruction Throughput  
   - FFT Plan Efficiency
---

### 🧪 Dependencies

- Python 3.10+
- NumPy, Matplotlib
- scikit-learn (for training SVM)
- cuFFT (CUDA FFT)
- CUDA Toolkit + Visual Studio (for profiling)

---

### 📺 Demo

A YouTube demo will showcase:

- Real-time FFT and anomaly detection
- Matplotlib report generation
- CUDA Insight profiling walkthrough

---

Licence GPL v.2
