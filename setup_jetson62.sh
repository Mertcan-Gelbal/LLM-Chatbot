#!/bin/bash
# Jetson Orin Nano Setup Script - JetPack 6.2 L4T 36.4.3
# ======================================================
# Agricultural RAG System için optimize edilmiş kurulum

set -e

echo "🚀 Jetson Orin Nano Setup - JetPack 6.2 L4T 36.4.3"
echo "=================================================="

# System info
echo "📱 Sistem Bilgileri:"
uname -a
cat /etc/nv_tegra_release || echo "Tegra release bilgisi bulunamadı"
nvidia-smi || echo "⚠️ nvidia-smi bulunamadı"

# JetPack version check
if [ -f /etc/nv_tegra_release ]; then
    JETPACK_VERSION=$(cat /etc/nv_tegra_release | grep "R36" | head -1)
    echo "🎯 JetPack Version: $JETPACK_VERSION"
fi

# Update system for JetPack 6.2
echo "🔄 Sistem güncelleniyor (JetPack 6.2)..."
sudo apt update
sudo apt upgrade -y

# Essential packages for JetPack 6.2
echo "📦 JetPack 6.2 temel paketler..."
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3-pip \
    python3-dev \
    python3-venv \
    python3.10-dev \
    libssl-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    libblas-dev \
    liblapack-dev \
    libhdf5-dev

# CUDA 12.2 development (JetPack 6.2)
echo "🔧 CUDA 12.2 development tools..."
sudo apt install -y \
    nvidia-cuda-toolkit \
    nvidia-cuda-dev \
    nvidia-cudnn9-dev \
    nvidia-tensorrt-dev \
    libnvonnxparsers-dev \
    libnvinfer-dev \
    libnvparsers-dev

# Python virtual environment
echo "🐍 Python virtual environment oluşturuluyor..."
python3 -m venv jetson_rag_env
source jetson_rag_env/bin/activate

# Upgrade pip and tools
pip install --upgrade pip setuptools wheel

echo "📚 PyTorch 2.3.0 ecosystem yükleniyor (JetPack 6.2 uyumlu)..."

# PyTorch 2.3.0 for JetPack 6.2
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# TensorRT 10.x for JetPack 6.2
pip install torch-tensorrt==2.3.0

echo "🤗 Transformers ecosystem (JetPack 6.2 optimized)..."

# Core ML libraries (latest compatible versions)
pip install \
    transformers==4.42.0 \
    tokenizers==0.19.0 \
    datasets==2.20.0 \
    accelerate==0.32.0 \
    optimum[onnxruntime-gpu]==1.21.0

# RAG specific components
pip install \
    sentence-transformers==3.0.0 \
    faiss-cpu==1.8.0

echo "🔧 JetPack 6.2 specific optimizations..."

# ONNX Runtime GPU 1.18.0 (JetPack 6.2 uyumlu)
pip install onnxruntime-gpu==1.18.0

# System monitoring (updated versions)
pip install \
    psutil==5.9.8 \
    gpustat==1.1.1 \
    nvidia-ml-py==12.555.43

# Scientific computing (optimized for ARM64)
pip install \
    numpy==1.26.4 \
    scipy==1.13.0 \
    pandas==2.2.0 \
    scikit-learn==1.5.0

# Visualization
pip install \
    matplotlib==3.9.0 \
    seaborn==0.13.0 \
    tqdm==4.66.0

# Development tools
pip install \
    jupyter==1.0.0 \
    ipykernel==6.29.0 \
    black==24.0.0 \
    flake8==7.0.0 \
    pytest==8.2.0

# Additional JetPack 6.2 optimizations
pip install \
    cupy-cuda12x==13.2.0 \
    numba==0.60.0 \
    pynvml==11.5.0

echo "⚙️ JetPack 6.2 System optimizations..."

# Jetson Orin performance mode (JetPack 6.2)
sudo nvpmodel -m 0  # Max performance mode
sudo jetson_clocks  # Max clocks

# Memory settings for JetPack 6.2
echo "vm.swappiness=1" | sudo tee -a /etc/sysctl.conf
echo "vm.vfs_cache_pressure=50" | sudo tee -a /etc/sysctl.conf
echo "vm.dirty_ratio=80" | sudo tee -a /etc/sysctl.conf
echo "vm.dirty_background_ratio=5" | sudo tee -a /etc/sysctl.conf

# GPU memory settings for CUDA 12.2
echo "options nvidia NVreg_PreserveVideoMemoryAllocations=1" | sudo tee -a /etc/modprobe.d/nvidia.conf

# CPU governor for maximum performance
echo 'GOVERNOR="performance"' | sudo tee -a /etc/default/cpufrequtils

echo "📁 Enhanced directory structure..."

# Create comprehensive directory structure
mkdir -p {logs,models,checkpoints,data,results,benchmarks,exports}

echo "🔍 JetPack 6.2 Environment test..."

# Test CUDA 12.2
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Device capability: {torch.cuda.get_device_capability(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB')
"

# Test Transformers
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test FAISS
python3 -c "import faiss; print(f'FAISS: {faiss.__version__}')"

# Test TensorRT
python3 -c "
try:
    import tensorrt as trt
    print(f'TensorRT: {trt.__version__}')
except ImportError:
    print('TensorRT Python bindings not found')
"

echo "📋 JetPack 6.2 Environment info..."

# Enhanced environment info
cat > jetson62_env_info.txt << EOF
Jetson Agricultural RAG Environment - JetPack 6.2
================================================
Date: $(date)
Hostname: $(hostname)
Architecture: $(uname -m)
Kernel: $(uname -r)
JetPack: $(cat /etc/nv_tegra_release 2>/dev/null || echo "Unknown")
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "Not detected")
CUDA: $(nvcc --version 2>/dev/null | grep release || echo "Not detected")
Python: $(python3 --version)
PyTorch: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "Not installed")
CUDA Available: $(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "Unknown")
Transformers: $(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "Not installed")
Available Memory: $(free -h | grep Mem | awk '{print $2}')
Available GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "Unknown") MB
CPU Cores: $(nproc)
Storage: $(df -h / | tail -1 | awk '{print $4}') available
EOF

echo "📊 JetPack 6.2 Performance benchmark..."

# Enhanced performance test
python3 -c "
import time
import torch
import numpy as np

print('🔥 JetPack 6.2 Performance Benchmark')
print('=' * 40)

if torch.cuda.is_available():
    device = torch.device('cuda')
    
    # GPU Info
    props = torch.cuda.get_device_properties(0)
    print(f'GPU: {props.name}')
    print(f'Compute Capability: {props.major}.{props.minor}')
    print(f'Memory: {props.total_memory/1e9:.1f}GB')
    print()
    
    # Matrix multiplication benchmark
    sizes = [1000, 2000, 4000]
    for size in sizes:
        x = torch.randn(size, size, device=device, dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            _ = torch.mm(x, x)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            y = torch.mm(x, x)
        torch.cuda.synchronize()
        end = time.time()
        
        gflops = (2 * size**3 * 100) / ((end - start) * 1e9)
        print(f'{size}x{size} MatMul: {(end-start)*1000:.1f}ms, {gflops:.1f} GFLOPS')
    
    print()
    print(f'Memory allocated: {torch.cuda.memory_allocated()/1e6:.1f}MB')
    print(f'Memory cached: {torch.cuda.memory_reserved()/1e6:.1f}MB')
else:
    print('CUDA not available for benchmark')
"

echo "✅ JetPack 6.2 setup tamamlandı!"
echo ""
echo "🚀 Tam performans eğitimi için:"
echo "   source jetson_rag_env/bin/activate"
echo "   cd jetson_training"
echo "   python full_performance_trainer.py --gpu --mixed_precision --epochs 5 --batch_size 8"
echo ""
echo "📊 Environment bilgileri: jetson62_env_info.txt"
echo ""
echo "💡 JetPack 6.2 Performance tips:"
echo "   - sudo nvpmodel -m 0    # Max performance mode"
echo "   - sudo jetson_clocks    # Max clocks"
echo "   - tegrastats            # System monitoring"
echo "   - nvidia-smi -l 1       # GPU monitoring"

# Create enhanced activation script for JetPack 6.2
cat > activate_rag62.sh << 'EOF'
#!/bin/bash
# JetPack 6.2 Quick activation script
source jetson_rag_env/bin/activate
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# JetPack 6.2 optimizations
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

echo "🌾 Agricultural RAG environment activated (JetPack 6.2)!"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo "CUDA: $(python3 -c 'import torch; print(torch.version.cuda)')"
EOF

chmod +x activate_rag62.sh

echo "🎯 JetPack 6.2 hızlı aktivasyon: ./activate_rag62.sh" 