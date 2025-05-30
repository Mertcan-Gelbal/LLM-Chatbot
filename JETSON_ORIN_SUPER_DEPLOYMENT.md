# 🌟 Jetson Orin Nano Super Deployment

## Özel Optimizasyonlar

### Hardware Specifications
- **GPU:** 1024-core NVIDIA Ampere architecture GPU
- **CPU:** 6-core Arm Cortex-A78AE v8.2 64-bit CPU
- **Memory:** 8GB 128-bit LPDDR5
- **Storage:** microSD (64GB+ önerilir)

### Agricultural BERT Optimizations

#### Memory Management
```python
# Optimal batch sizes for Orin Nano Super
BERT_BASE_BATCH_SIZE = 8
BERT_LARGE_BATCH_SIZE = 4
MAX_SEQUENCE_LENGTH = 512
```

#### Performance Tuning
```bash
# GPU frequency scaling
sudo jetson_clocks

# Memory optimization
echo 1 | sudo tee /proc/sys/vm/drop_caches

# Swap configuration
sudo systemctl disable nvzramconfig
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Agricultural Dataset Processing

#### 13,200 Chunk Processing
- **Plant Disease:** 2,640 chunks
- **Crop Management:** 2,640 chunks  
- **Plant Genetics:** 2,640 chunks
- **Environmental Factors:** 2,640 chunks
- **Food Security:** 2,640 chunks
- **Technology:** 2,640 chunks

#### Training Pipeline
```bash
# Dataset generation
python3 agricultural_test_generator.py

# BERT training
cd jetson_training
python3 bert_classification_trainer.py
```

### Monitoring & Debugging

#### Real-time Monitoring
```bash
# GPU utilization
watch -n 1 nvidia-smi

# Temperature monitoring
watch -n 1 'cat /sys/devices/virtual/thermal/thermal_zone*/temp'

# Memory usage
watch -n 1 free -h
```

#### Log Analysis
```bash
# Training logs
tail -f results/training.log

# System logs
journalctl -f -u agricultural-bert
``` 