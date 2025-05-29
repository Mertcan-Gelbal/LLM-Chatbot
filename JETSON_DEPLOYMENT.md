# 🚀 Jetson Orin Nano Deployment Guide

Bu rehber, Agricultural RAG sistemini Jetson Orin Nano Super'da çalıştırmak için adım adım talimatları içerir.

## 📋 Ön Gereksinimler

### Jetson Sistem Gereksinimleri
- **Jetson Orin Nano Super** (8GB+ RAM)
- **JetPack 5.1+** yüklü
- **CUDA 11.4+** aktif
- **Internet bağlantısı**
- **SSH erişimi** aktif

### SSH Bağlantısı Kontrolü
```bash
# Jetson IP adresini öğren
hostname -I

# SSH servisini kontrol et
sudo systemctl status ssh

# SSH'ı etkinleştir (gerekirse)
sudo systemctl enable ssh
sudo systemctl start ssh
```

## 🔗 1. GitHub Repository Clone

SSH ile Jetson'a bağlandıktan sonra:

```bash
# Home dizinine git
cd ~

# Repository'yi klonla
git clone https://github.com/[USERNAME]/agricultural-rag-jetson.git

# Proje dizinine gir
cd agricultural-rag-jetson

# Dosyaları kontrol et
ls -la
```

## ⚙️ 2. Environment Setup

```bash
# Setup script'ini çalıştırılabilir yap
chmod +x setup_jetson.sh

# Jetson environment kurulumunu başlat (15-20 dakika sürer)
./setup_jetson.sh

# Kurulum tamamlandıktan sonra environment'ı aktive et
source jetson_rag_env/bin/activate

# Hızlı aktivasyon için
./activate_rag.sh
```

## 📊 3. System Performance Check

```bash
# GPU durumunu kontrol et
nvidia-smi

# Sistem kaynaklarını kontrol et
htop

# Jetson performance mode'u kontrol et
sudo nvpmodel -q

# Maximum performance için
sudo nvpmodel -m 0
sudo jetson_clocks
```

## 🎯 4. Model Training

### Basit Training (Test)
```bash
cd jetson_training

# CPU ile test
python jetson_train.py --epochs 1 --batch_size 1

# GPU ile test  
python jetson_train.py --gpu --epochs 1 --batch_size 2

# Full training with optimizations
python jetson_train.py --gpu --mixed_precision --epochs 3 --batch_size 4
```

### Memory Optimized Training
```bash
# Düşük memory için
python jetson_train.py --gpu --mixed_precision --batch_size 1 --epochs 2

# Optimal batch size testi
python -c "
from gpu_optimizer import JetsonOptimizer
opt = JetsonOptimizer()
opt.monitor_gpu()
"
```

## 📦 5. ONNX Export

Training tamamlandıktan sonra:

```bash
# Model export
python export_onnx.py --model_dir ../models/rag_jetson_YYYYMMDD_HHMM --export_dir ../models/onnx_optimized --verify
```

## 🔍 6. Monitoring ve Debug

### Real-time Monitoring
```bash
# GPU monitoring
watch -n 1 nvidia-smi

# System resources
watch -n 1 'free -h && df -h'

# Temperature monitoring
watch -n 1 'cat /sys/class/thermal/thermal_zone*/temp'
```

### Performance Profiling
```bash
# Memory usage profiling
python -c "
from gpu_optimizer import JetsonProfiler
profiler = JetsonProfiler()
profiler.start()
# Your model operations here
profiler.end()
"
```

## ⚡ 7. Performance Optimization

### Memory Management
```bash
# Clear cache
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# GPU memory cleanup
python -c "import torch; torch.cuda.empty_cache()"
```

### System Optimization
```bash
# Disable GUI (SSH kullanıyorsan)
sudo systemctl set-default multi-user.target

# CPU governor performance mode
sudo cpufreq-set -g performance

# Swap optimization
sudo sysctl vm.swappiness=10
```

## 🚨 8. Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```bash
# Batch size'ı küçült
python jetson_train.py --batch_size 1

# Mixed precision kullan
python jetson_train.py --mixed_precision
```

**CUDA Errors**
```bash
# CUDA installation check
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
```

**Model Loading Issues**
```bash
# Fallback to smaller model
python jetson_train.py --fallback_model
```

### Error Logs
```bash
# Training logs
tail -f logs/training.log

# System logs
sudo dmesg | grep -i cuda
sudo journalctl -u nvidia-docker
```

## 📈 9. Expected Performance

### Training Performance
- **Training Time:** 2-3 saat (3 epochs)
- **Memory Usage:** ~3GB GPU, ~4GB RAM
- **Temperature:** <75°C (normal)
- **Power:** ~15W (performance mode)

### Inference Performance
- **Response Time:** <100ms per query
- **Throughput:** 10+ queries/second
- **Memory:** <2GB GPU
- **Accuracy:** 0.85+ F1 score

## 🎯 10. Production Deployment

### Model Serving
```bash
# ONNX model ile hızlı inference
python -c "
import onnxruntime as ort
session = ort.InferenceSession('models/onnx_optimized/model.onnx')
print('Model ready for inference')
"
```

### API Server (Gelecek)
```bash
# FastAPI server başlat
python api_server.py --port 8000 --workers 2
```

## 📞 Destek

Sorun yaşarsanız:

1. **GitHub Issues:** Repository'de issue açın
2. **Logs:** Hata loglarını paylaşın
3. **System Info:** `cat jetson_env_info.txt` çıktısını ekleyin

---

**🌾 Agricultural RAG System - Jetson Orin Nano'da Çalışıyor!** 🚀 