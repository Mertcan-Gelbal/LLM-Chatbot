# 📜 Scripts

Bu klasör, proje için yardımcı scriptleri içerir.

## Mevcut Scriptler

Şu anda bu klasör boş. Gerektiğinde aşağıdaki türde scriptler eklenebilir:

### 🔧 Yardımcı Scriptler
- `setup.sh` - Otomatik kurulum
- `test_model.sh` - Model test scripti  
- `deploy.sh` - Deployment scripti

### 📊 Veri İşleme
- `data_preprocessor.py` - Veri ön işleme
- `dataset_generator.py` - Dataset oluşturma
- `data_validator.py` - Veri doğrulama

### 🚀 Deployment
- `docker_build.sh` - Docker image oluşturma
- `k8s_deploy.yaml` - Kubernetes deployment
- `api_server.py` - Flask API server

### 📈 Monitoring
- `performance_monitor.py` - Performans izleme
- `log_analyzer.py` - Log analizi
- `health_check.py` - Sistem sağlık kontrolü

## Kullanım

Scripts klasöründeki dosyalar şu şekilde çalıştırılabilir:

```bash
# Shell scripti
cd Scripts
chmod +x script_name.sh
./script_name.sh

# Python scripti  
python script_name.py

# Proje kök dizininden
python Scripts/script_name.py
```

## Script Ekleme Kuralları

1. **Naming Convention:** `snake_case` kullanın
2. **Documentation:** Her script için docstring ekleyin
3. **Error Handling:** Hata yakalama ekleyin
4. **Logging:** İşlem logları ekleyin
5. **Configuration:** Configurable parametreler kullanın

## Örnek Script Template

```python
#!/usr/bin/env python3
"""
Script Açıklaması
Scriptin ne yaptığını açıklayan detaylı bilgi
"""

import os
import sys
import logging

def main():
    """Ana fonksiyon"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Script logic here
        logger.info("Script başlatıldı")
        
        # İşlemler...
        
        logger.info("Script başarıyla tamamlandı")
        
    except Exception as e:
        logger.error(f"Script hatası: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 