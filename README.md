## 🚀 `cuda_fft_batch_mmap_1gb`

### Overview  (todo, open all the memmap files at once, get an array of handles, then read one at a time).

This project processes high-throughput voltage telemetry collected from STM32F4xx microcontrollers. It performs batched FFT analysis on 1GB of data using CUDA and Intel MKL libraries.  It generates detailed signal reports with Matplotlib. Performance metrics are captured using CUDA Insight and Visual Studio debugging tools.

##NOTE## The file CudaFFT-Demo.pdf is the spec.  TODO: Telemetry and matplotlib graphs are working without impacting perf.  I separate telemetry from perf.  I need to pipe the perf to a file.  I need to add steps on running the analyze_spectrum_report.py on the mkl_report.txt and cuda_report.txt.  Need to have a script that builds comparisons between two runs.  


---

### 🔧 Pipeline Summary

1. **Data Acquisition**
   - Voltage packets streamed from STM32F4xx MCUs
   - Stored as binary float32 arrays via memory-mapped files
   - 12 files × 16MB = 192MB per batch (scaled to 1GB in full runs)

2. **FFT Processing**
   - scripts to capture voltage or any signal from your microcontroller included.
   - memmaps1.zip and memmaps2.zip are memory mapped files of volatage samples off my STM32F4xx controllers, they are just floats.  I could not give you the 1G, the zips contain four 16M memory mapped files.  Extract them into on directory and have const char *memmap_dir = specify that directory.  
   - CUDA FFT on 8192 float chunks on memory mapped files of 16M each.
   - Same with Intel MKL, compare results
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
- cuFFT (CUDA FFT)
- CUDA Toolkit + Visual Studio (for profiling)
- Intel MKL - one API for intel machines on windows AFAIKT

---

### 🎯 CUDA 12.1 + Visual Studio Setup

#### 🔗 **Download Installer**
- [CUDA Toolkit 12.1 (Windows)](https://developer.nvidia.com/cuda-12-1-0-download-archive)
- Choose **Local Installer** for full offline installation
- Confirm **Visual Studio Integration** is selected during setup

#### ✅ **Supported Visual Studio Versions**
- **Visual Studio 2019** (MSVC v142)
- **Visual Studio 2022** (MSVC v143)

> ⚠️ If you're using VS 2022 17.10 or newer, CUDA 12.1 may require a newer toolkit (e.g. 12.4+) or manual workaround. CUDA 12.1 officially supports up to VS 2022 17.4.  I used 17.14.11 with Cuda 12.9

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

### 🧩 Option CUDA Build Customization 

If you're compiling `.cu` files:
- Set **Item Type** to `CUDA C/C++`
- Use **CUDA C/C++ → Device → Code Generation** to match your GPU architecture (e.g. `compute_86,sm_86`)
- Use **CUDA C/C++ → Common → Additional Options** for flags like `--use_fast_math` or `--ptxas-options=-v`
---

## 🧠 MKL Setup for FFT Acceleration

This project supports Intel MKL for high-performance CPU-based FFT processing. To enable MKL support:

### ✅ Step 1: Install Intel oneAPI Base Toolkit

Download and install from [Intel's official site](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html).

Make sure to include:

- **Intel MKL**
- **Intel oneAPI command-line environment**

### ✅ Step 2: Configure Visual Studio

1. Open your project in Visual Studio
2. Right-click your project → **Properties**
3. Under **VC++ Directories**:
   - **Include Directories**:  
     `C:\Program Files (x86)\Intel\oneAPI\mkl\latest\include`
   - **Library Directories**:  
     `C:\Program Files (x86)\Intel\oneAPI\mkl\latest\lib\intel64`

4. Under **Linker → Input → Additional Dependencies**:
   Add:
   ```
   mkl_intel_lp64.lib
   mkl_sequential.lib
   mkl_core.lib
   ```

5. (Optional) If using dynamic linking, ensure `mkl_rt.dll` is in your runtime path

### ✅ Step 3: Build and Run

Once configured, the MKL backend will be available via `mkl_fft()` in the benchmark suite. You can toggle between CUDA and MKL using the runtime selector or build flags.

### 📺 Demo Coming soon...

A YouTube demo will showcase:

- Real-time FFT and anomaly detection
- Matplotlib report generation
- CUDA Insight profiling walkthrough

---
### ⚔️ CUDA vs. MKL FFT Performance Comparison (this is prior implementing opening all the files handles first)

| Chunk Size | Rows per FFT | CUDA Time (ms) | CUDA ns/float | MKL Time (ms) | MKL ns/float | Faster |
|------------|---------------|----------------|----------------|----------------|----------------|--------|
| 8192       | 32768         | 2532.28        | 9.43           | 1358.43        | 5.06           | MKL    |
| 32768      | 8192          | 1772.56        | 6.60           | 1331.14        | 4.96           | MKL    |
| 65536      | 4096          | 1619.60        | 6.03           | 1372.75        | 5.11           | MKL    |
| 131072     | 2048          | 1482.64        | 5.52           | 1824.29        | 6.80           | CUDA   |
| 262144     | 1024          | 1437.17        | 5.35           | 1915.56        | 7.14           | CUDA   |

> **Note**: MKL outperforms CUDA at smaller chunk sizes due to lower overhead and CPU cache locality. CUDA overtakes at larger sizes where GPU parallelism and memory bandwidth dominate.

Licence GPL v.2
