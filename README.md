## 🚀 `cuda_fft_batch_mmap_1gb`
## demo the big logfile compared to memory mapped files, add the .json code that captures all the global FFT data to pass to Matplotlib##
### Overview

This project processes high-throughput voltage telemetry collected from STM32F4xx microcontrollers. It performs batched FFT analysis on 1GB of data using CUDA, applies anomaly detection via CMSIS-DSP and scikit-learn SVM, and generates detailed signal reports with Matplotlib. Performance metrics are captured using CUDA Insight and Visual Studio debugging tools.

##NOTE## The file CudaFFT-Demo.pdf is the spec.  TODO: conditional compilation for telemetry data, separate from timing data, write to mmSignalReport, signal_report.py and benchmark.py.  
-also benchmark CMSIS-DSP FFT outside of the Cuda Engine.

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
- scikit-learn (for training SVM if you want to train, it works for STM32F4xx voltage data).
- cuFFT (CUDA FFT)
- CUDA Toolkit + Visual Studio (for profiling)

---Project--
-I use VS, install CUDA SDK, CMSIS/DSP it is a zipfile in this project, just extract it and make it a peer to lib and the .cpp and .h file

---

### 🎯 CUDA 12.1 + Visual Studio Setup

#### 🔗 **Download Installer**
- [CUDA Toolkit 12.1 (Windows)](https://developer.nvidia.com/cuda-12-1-0-download-archive)
- Choose **Local Installer** for full offline installation
- Confirm **Visual Studio Integration** is selected during setup

#### ✅ **Supported Visual Studio Versions**
- **Visual Studio 2019** (MSVC v142)
- **Visual Studio 2022** (MSVC v143)

> ⚠️ If you're using VS 2022 17.10 or newer, CUDA 12.1 may require a newer toolkit (e.g. 12.4+) or manual workaround. CUDA 12.1 officially supports up to VS 2022 17.4.

---

### 🛠️ Visual Studio Project Configuration

#### 📁 **Include Directories**
Add to **Project → Properties → C/C++ → General → Additional Include Directories**:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include
```

#### 📁 **Library Directories**
Add to **Project → Properties → Linker → General → Additional Library Directories**:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64
```

#### 📄 **Linker Input**
Add to **Project → Properties → Linker → Input → Additional Dependencies**:
```
cudart.lib
cufft.lib
```

---

### 🧩 Optional: CUDA Build Customization

If you're compiling `.cu` files:
- Set **Item Type** to `CUDA C/C++`
- Use **CUDA C/C++ → Device → Code Generation** to match your GPU architecture (e.g. `compute_86,sm_86`)
- Use **CUDA C/C++ → Common → Additional Options** for flags like `--use_fast_math` or `--ptxas-options=-v`

---

### 📺 Demo Coming soon...

A YouTube demo will showcase:

- Real-time FFT and anomaly detection
- Matplotlib report generation
- CUDA Insight profiling walkthrough

---

Licence GPL v.2
