# 🚀 Jetson Orin Nano - JetPack 6.2 L4T 36.4.3 Deployment

[![JetPack 6.2](https://img.shields.io/badge/JetPack-6.2-green.svg)](https://developer.nvidia.com/embedded/jetpack)
[![CUDA 12.2](https://img.shields.io/badge/CUDA-12.2-blue.svg)](https://developer.nvidia.com/cuda-downloads)
[![PyTorch 2.3](https://img.shields.io/badge/PyTorch-2.3.0-orange.svg)](https://pytorch.org/)

Bu rehber, Agricultural RAG sistemini **JetPack 6.2 L4T 36.4.3** ile çalıştırmak için optimize edilmiştir.

## 📋 JetPack 6.2 Ön Gereksinimler

### Sistem Doğrulaması
```bash
# JetPack version kontrolü
cat /etc/nv_tegra_release
# Beklenen: # R36 (release), REVISION: 4.3

# CUDA version kontrolü
nvcc --version
# Beklenen: Cuda compilation tools, release 12.2

# GPU kontrolü
nvidia-smi
```

### Sistem Bilgileri
- **JetPack:** 6.2 L4T 36.4.3
- **CUDA:** 12.2
- **TensorRT:** 10.x
- **PyTorch:** 2.3.0
- **RAM:** 8GB minimum (16GB önerilen)

## 🔗 1. Repository Clone ve Setup

```bash
# SSH bağlantısı sonrası
cd ~
git clone https://github.com/[USERNAME]/agricultural-rag-jetson.git
cd agricultural-rag-jetson

# JetPack 6.2 setup scriptini çalıştır
chmod +x setup_jetson62.sh
./setup_jetson62.sh
```

**⏰ Kurulum süresi:** 20-30 dakika (internet hızına bağlı)

## ⚙️ 2. JetPack 6.2 Optimizasyonları

### Performance Mode
```bash
# Maksimum performans modu
sudo nvpmodel -m 0
sudo jetson_clocks

# Fan control (gerekirse)
sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
```

### Memory Optimizasyonları
```bash
# Swap ayarları
sudo sysctl vm.swappiness=1
sudo sysctl vm.vfs_cache_pressure=50

# GPU memory
echo 'options nvidia NVreg_PreserveVideoMemoryAllocations=1' | sudo tee -a /etc/modprobe.d/nvidia.conf
```

## 🎯 3. Tam Performans Eğitimi

### Environment Aktivasyonu
```bash
# Hızlı aktivasyon
./activate_rag62.sh

# Veya manuel
source jetson_rag_env/bin/activate
cd jetson_training
```

### Full Performance Training
```bash
# Maksimum performans eğitimi
python full_performance_trainer.py \
    --gpu \
    --mixed_precision \
    --epochs 5 \
    --batch_size 8 \
    --auto_batch_size \
    --custom_optimizer

# Konservatif eğitim (düşük memory)
python full_performance_trainer.py \
    --gpu \
    --mixed_precision \
    --epochs 3 \
    --batch_size 4
```

### Training Parametreleri
| Parametre | Açıklama | JetPack 6.2 Önerilen |
|-----------|----------|----------------------|
| `--epochs` | Eğitim epoch sayısı | 5 |
| `--batch_size` | Hedef batch size | 8 |
| `--learning_rate` | Learning rate | 3e-5 |
| `--auto_batch_size` | Otomatik batch size | ✅ |
| `--mixed_precision` | FP16 kullan | ✅ |
| `--tensorrt` | TensorRT export | ✅ |

## 📊 4. JetPack 6.2 Monitoring

### Real-time Monitoring
```bash
# GPU monitoring
nvidia-smi -l 1

# System stats (JetPack 6.2)
tegrastats

# Temperature monitoring
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'

# Comprehensive monitoring
python -c "
from jetson_training.gpu_optimizer_jp62 import JetsonOptimizerJP62
opt = JetsonOptimizerJP62()
opt.monitor_jetson_gpu()
"
```

### Performance Profiling
```bash
# Benchmark test
python -c "
from jetson_training.gpu_optimizer_jp62 import JetsonOptimizerJP62
from transformers import AutoModel
opt = JetsonOptimizerJP62()
model = AutoModel.from_pretrained('distilbert-base-uncased')
opt.benchmark_performance_jp62(model, (128,))
"
```

## 🔧 5. Troubleshooting JetPack 6.2

### Memory Issues
```bash
# OOM hatası alırsanız
python full_performance_trainer.py --batch_size 2

# Memory temizleme
python -c "
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
"
```

### CUDA 12.2 Issues
```bash
# CUDA environment check
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'Available: {torch.cuda.is_available()}')
"

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### TensorRT Issues
```bash
# TensorRT test
python -c "
try:
    import tensorrt as trt
    print(f'TensorRT: {trt.__version__}')
except ImportError as e:
    print(f'TensorRT error: {e}')
"
```

## 📈 6. Expected Performance (JetPack 6.2)

### Training Performance
- **Training Time:** 1.5-2.5 saat (5 epochs)
- **Memory Usage:** ~4-5GB GPU, ~6GB RAM
- **Temperature:** <80°C (fan aktif)
- **Power:** ~20W (performance mode)

### Inference Performance
- **Response Time:** <80ms per query
- **Throughput:** 12+ queries/second
- **Memory:** <2GB GPU
- **Accuracy:** 0.87+ F1 score

### Memory Efficiency (JetPack 6.2)
| Model Size | GPU Memory | Batch Size | Performance |
|------------|------------|------------|-------------|
| Small (110M) | 2GB | 8 | ⚡⚡⚡ |
| Medium (355M) | 4GB | 6 | ⚡⚡ |
| Large (770M) | 6GB | 4 | ⚡ |

## 🎯 7. Advanced JetPack 6.2 Features

### CUDA 12.2 Optimizations
```python
# Training script'inde otomatik aktif
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

### PyTorch 2.3 Compilation
```python
# Model compilation (otomatik)
model = torch.compile(model, mode="reduce-overhead")
```

### TensorRT 10.x Integration
```bash
# TensorRT export (training sonrası)
python export_onnx.py \
    --model_dir models/full_performance_YYYYMMDD_HHMM \
    --tensorrt \
    --verify
```

## 🔥 8. Production Deployment

### Model Serving
```bash
# ONNX model test
python -c "
import onnxruntime as ort
providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
session = ort.InferenceSession('models/onnx_optimized/model.onnx', providers=providers)
print('✅ ONNX model ready')
"
```

### API Server (Coming Soon)
```bash
# FastAPI server
python api_server_jp62.py --port 8000 --workers 4 --tensorrt
```

## 📞 Support & Troubleshooting

### Log Files
```bash
# Training logs
tail -f logs/full_training.log

# System logs
sudo journalctl -u nvidia-docker

# CUDA logs
sudo dmesg | grep -i cuda
```

### Performance Issues
1. **Thermal throttling:** Fan kontrolü, thermal paste
2. **Memory pressure:** Batch size küçültme
3. **CUDA errors:** Driver güncelleme
4. **Slow training:** nvpmodel -m 0 kontrolü

### Contact
- **GitHub Issues:** Repository'de issue açın
- **Environment Info:** `cat jetson62_env_info.txt`
- **Performance Report:** Benchmark sonuçlarını paylaşın

---

**🌾 Agricultural RAG System - JetPack 6.2 L4T 36.4.3 Ready!** 🚀

### Quick Start Commands
```bash
# Full setup
./setup_jetson62.sh && ./activate_rag62.sh

# Max performance training
python full_performance_trainer.py --gpu --mixed_precision --epochs 5

# Monitor performance
tegrastats & nvidia-smi -l 1
``` 