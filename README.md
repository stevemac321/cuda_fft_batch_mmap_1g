
```markdown
# 🚀 `cuda_fft_batch_mmap_1gb`

## Overview

This project processes high-throughput voltage telemetry collected from STM32F4xx microcontrollers. It performs batched FFT analysis on 1GB of data using CUDA and Intel MKL libraries. It generates detailed signal reports with Matplotlib. Performance metrics are captured using CUDA Insight and Visual Studio debugging tools.
##NOTE Cuda 13.0 SDK came out, but the Windows 11 driver for my GPU is not out yet, so this is still 12.9
-Also, I added pinned Cuda pinning host memory for easy fetch for chunks >= 64K, two versions of cuda_fft (pinned and unpinned) wrapper calls the correct one.

> 📄 **Spec**: See `CudaFFT-Demo.pdf`  
> ✅ **TODOs**:
> - Telemetry and matplotlib graphs are working without impacting performance
> - Separate telemetry from performance
> - Pipe performance to file
> - Document `analyze_spectrum_report.py` usage on `mkl_report.txt` and `cuda_report.txt`
> - Script needed to compare two runs automatically

---

## 📚 Table of Contents

1. [Pipeline Summary](#pipeline-summary)  
2. [Dependencies](#dependencies)  
3. [CUDA + Visual Studio Setup](#cuda--visual-studio-setup)  
4. [Visual Studio Project Configuration](#visual-studio-project-configuration)  
5. [CUDA Build Customization](#cuda-build-customization)  
6. [MKL Setup](#mkl-setup)  
7. [Performance Comparison](#performance-comparison)  
8. [Benchmark Workflow](#benchmark-workflow)  
9. [License](#license)

---

## 🔧 Pipeline Summary

### 1. Data Acquisition

- Voltage packets streamed from STM32F4xx MCUs
- Stored as binary `float32` arrays via memory-mapped files
- 12 files × 16MB = 192MB per batch (scaled to 1GB in full runs)

### 2. FFT Processing

- Scripts included to capture signal data from your microcontroller
- `memmaps1.zip` and `memmaps2.zip` contain float-based memory-mapped files from STM32F4xx
- Extract to a directory and set `const char *memmap_dir` accordingly
- CUDA FFT on 8192-float chunks across each 16MB file
- Same processing with Intel MKL for performance comparison
- Frequency domain features extracted per chunk

### 3. Signal Statistics (per chunk)

- Max / Min Voltage  
- Voltage Range  
- Mean Voltage  
- Standard Deviation (Noise Level)  
- Outlier Count / Density  

### 4. Frequency & Power Metrics

- Dominant Frequency  
- Spectral Centroid  
- Spectral Spread  
- Power per Batch  
- Frequency Response Matrix (for heatmap)

### 5. Distribution & Trend Metrics

- Histogram Binning  
- Rolling Statistics (optional)  
- Batch Timestamp / Time Offset  
- Chunk Index  

### 6. Visualization

- Matplotlib report generation  
- Signal statistics, FFT heatmaps  

### 7. Performance Profiling (CUDA Insight / VS)

- Kernel execution time per batch  
- Host ↔ Device memory bandwidth  
- SM utilization  
- Thread block & warp efficiency  
- Shared vs. global memory access  
- Stream overlap  
- Instruction throughput  
- FFT plan efficiency  

---

## 🧪 Dependencies

- Python 3.10+
- NumPy, Matplotlib
- cuFFT (CUDA FFT)
- CUDA Toolkit + Visual Studio
- Intel MKL (via oneAPI toolkit)

---

## 🎯 CUDA + Visual Studio Setup

### Download Installer

- [CUDA Toolkit 12.1 (Windows)](https://developer.nvidia.com/cuda-12-1-0-download-archive)
- Choose **Local Installer**
- Ensure **Visual Studio Integration** is selected

### Supported Visual Studio Versions

- Visual Studio 2019 (MSVC v142)
- Visual Studio 2022 (MSVC v143)

> ⚠️ CUDA 12.1 officially supports up to VS 2022 v17.4. I used 17.14.11 with CUDA 12.9 successfully.

---

## 🛠️ Visual Studio Project Configuration

### Include Directories

```

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include

```

### Library Directories

```

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64

```

### Linker Input

```

cudart.lib
cufft.lib

```

---

## 🧩 CUDA Build Customization

If compiling `.cu` files:

- Set **Item Type** to `CUDA C/C++`
- Set **Code Generation** under `CUDA C/C++ → Device` (e.g., `compute_86,sm_86`)
- Add flags like `--use_fast_math` or `--ptxas-options=-v` under `CUDA C/C++ → Common → Additional Options`

---

## 🧠 MKL Setup

### Step 1: Install Intel oneAPI Base Toolkit

- [Download Intel oneAPI Base Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html)
- Ensure **Intel MKL** and **oneAPI CLI Environment** are selected

### Step 2: Configure Visual Studio

**VC++ Directories**:
```

Include:  C:\Program Files (x86)\Intel\oneAPI\mkl\latest\include
Library:  C:\Program Files (x86)\Intel\oneAPI\mkl\latest\lib\intel64

```

**Linker → Input → Additional Dependencies**:
```

mkl\_intel\_lp64.lib
mkl\_sequential.lib
mkl\_core.lib

````

**Note**: For dynamic linking, ensure `mkl_rt.dll` is in your system path

### Step 3: Build and Run

Once configured, use `mkl_fft()` in the benchmark suite. You can toggle between MKL and CUDA via build flags or runtime switch.

---

## 📊 Performance Comparison

### CUDA vs. MKL FFT

| Chunk Size | Rows | CUDA (ms) | CUDA ns/float | MKL (ms) | MKL ns/float | Faster |
|------------|------|-----------|----------------|----------|---------------|--------|
| 8192       | 32768| 2532.28   | 9.43           | 1358.43  | 5.06          | MKL    |
| 32768      | 8192 | 1772.56   | 6.60           | 1331.14  | 4.96          | MKL    |
| 65536      | 4096 | 1619.60   | 6.03           | 1372.75  | 5.11          | MKL    |
| 131072     | 2048 | 1482.64   | 5.52           | 1824.29  | 6.80          | CUDA   |
| 262144     | 1024 | 1437.17   | 5.35           | 1915.56  | 7.14          | CUDA   |

> MKL outperforms at smaller sizes due to CPU cache locality. CUDA dominates at larger scales.

---

### CUDA MemMap FFT (Before Pre-Mapping)

| Chunk Size | Rows | Time (ms) | ns/float |
|------------|------|-----------|----------|
| 8192       | 32768| 2532.28   | 9.43     |
| 32768      | 8192 | 1772.56   | 6.60     |
| 65536      | 4096 | 1619.60   | 6.03     |
| 131072     | 2048 | 1482.64   | 5.52     |
| 262144     | 1024 | 1437.17   | 5.35     |

### ✅ CUDA (After Pre-Mapping)

| Chunk Size | Rows | Time (ms) | ns/float |
|------------|------|-----------|----------|
| 8192       | 32768| 2465.30   | 9.18     |
| 32768      | 8192 | 1673.58   | 6.23     |
| 65536      | 4096 | 1461.92   | 5.45     |
| 131072     | 2048 | 1493.22   | 5.56     |
| 262144     | 1024 | 1253.41   | 4.67     |

### MKL MemMap FFT (Before Pre-Mapping)

| Chunk Size | Rows | Time (ms) | ns/float |
|------------|------|-----------|----------|
| 8192       | 32768| 1363.60   | 5.08     |
| 32768      | 8192 | 1335.30   | 4.97     |
| 65536      | 4096 | 1408.26   | 5.25     |
| 131072     | 2048 | 1822.70   | 6.79     |
| 262144     | 1024 | 1876.66   | 6.99     |

### ✅ MKL (After Pre-Mapping)

| Chunk Size | Rows | Time (ms) | ns/float |
|------------|------|-----------|----------|
| 8192       | 32768| 1283.55   | 4.78     |
| 32768      | 8192 | 1257.65   | 4.69     |
| 65536      | 4096 | 1282.02   | 4.78     |
| 131072     | 2048 | 1717.00   | 6.40     |
| 262144     | 1024 | 1780.56   | 6.63     |

---

## 📘 Benchmark Workflow

```powershell
# Open 2 terminals and activate Python env:
.\.venv\Scripts\Activate.ps1

# Terminal 1: Monitor
python monitor.py

# Terminal 2: Benchmark without telemetry
cd C:\repos\CudaBigData
.\x64\Release\CudaBigData.exe
python benchmark.py
````

### With Telemetry Logging

```powershell
# Terminal 1: Monitor
python monitor.py

# Terminal 2: Benchmark with telemetry
cd C:\repos\CudaBigData
$env:LOG_TELEMETRY=1
.\x64\Release\CudaBigData.exe
python benchmark.py

# Post-run: Analyze
python analyze_spectrum_report.py
```

**Notes**:

* Scripts (`monitor.py`, `benchmark.py`, `analyze_spectrum_report.py`) must be in project root
* `utilization_log.txt` is generated by `monitor.py`
* `LOG_TELEMETRY=1` enables detailed logging

---

## 📄 License

GPL v2

---

```

Let me know if you'd like this saved as a downloadable file or auto-generated with collapsible details or badges.
```
