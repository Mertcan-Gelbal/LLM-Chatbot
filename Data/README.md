# 📊 Dataset Dosyaları

Bu klasör, Botanical BERT modeli için kullanılan tarımsal dataset'leri içerir.

## Dataset Yapısı

### Ana Eğitim Dosyaları
- `train.csv` - Model eğitimi için veri (1,260 örnekler)
- `val.csv` - Validation verisi (270 örnekler)  
- `test.csv` - Test verisi (270 örnekler)

### Dataset Kategorileri (6 Sınıf)

| Kategori | Açıklama | Örnek |
|----------|----------|-------|
| `plant_disease` | Bitki hastalıkları | "Domates yaprak yanıklığı" |
| `crop_management` | Mahsul yönetimi | "Buğday ekim zamanı" |
| `plant_genetics` | Bitki genetiği | "GMO mısır çeşitleri" |
| `environmental_factors` | Çevre faktörleri | "İklim değişikliği" |
| `food_security` | Gıda güvenliği | "Depolama kayıpları" |
| `technology` | Tarım teknolojisi | "Drone ile hastalık tespiti" |

## CSV Format

```csv
text,label
"Domates bitkilerinde yaprak yanıklığı hastalığı",plant_disease
"Mısır için en uygun ekim zamanı",crop_management
```

## Toplam Dataset İstatistikleri

- **Toplam Örnekler:** 1,800
- **Eğitim:** %70 (1,260)
- **Validation:** %15 (270) 
- **Test:** %15 (270)
- **Dil:** Türkçe/İngilizce karışık
- **Kaynak:** Tarımsal makaleler + uzman bilgisi

## Kullanım

```python
import pandas as pd

# Dataset yükle
train_df = pd.read_csv('Data/train.csv')
val_df = pd.read_csv('Data/val.csv')
test_df = pd.read_csv('Data/test.csv')

print(f"Eğitim verisi: {len(train_df)} örnekler")
print(f"Kategori dağılımı:\n{train_df['label'].value_counts()}")
``` 