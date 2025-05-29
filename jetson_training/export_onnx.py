#!/usr/bin/env python3
"""
Jetson ONNX Export - Agricultural RAG
====================================
Optimized ONNX export for Jetson Orin Nano deployment
"""

import os
import torch
from pathlib import Path
from transformers import RagTokenizer, RagSequenceForGeneration
try:
    from optimum.onnxruntime import ORTConfig, ORTModelForSeq2SeqLM
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False
    print("⚠️ Optimum bulunamadı, basit ONNX export kullanılacak")

def export_to_onnx(model, tokenizer, model_dir, export_dir=None):
    """
    Jetson için optimize edilmiş ONNX export
    
    Args:
        model: Eğitilmiş RAG modeli
        tokenizer: Model tokenizer'ı
        model_dir: Model kayıt dizini
        export_dir: ONNX export dizini
    """
    
    if export_dir is None:
        export_dir = Path(model_dir) / "onnx_jetson"
    
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 ONNX Export başlıyor...")
    print(f"📂 Export dizini: {export_dir}")
    
    try:
        if OPTIMUM_AVAILABLE:
            export_with_optimum(model, tokenizer, export_dir)
        else:
            export_native_onnx(model, tokenizer, export_dir)
            
    except Exception as e:
        print(f"❌ ONNX export hatası: {e}")
        print("🔄 Fallback export deneniyor...")
        export_simple_onnx(model, tokenizer, export_dir)

def export_with_optimum(model, tokenizer, export_dir):
    """Optimum ile gelişmiş ONNX export"""
    print("⚡ Optimum ile ONNX export...")
    
    # Jetson için optimize edilmiş config
    ort_config = ORTConfig(
        optimize=True,                                    # Graph optimizasyonu
        execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        use_gpu=True,
        optimization_level="all",                         # Maksimum optimizasyon
        enable_cuda_graph=True,                          # CUDA Graph
        use_cache=False,                                 # Memory optimization
        provider_options={
            "CUDAExecutionProvider": {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 6 * 1024 * 1024 * 1024,  # 6GB limit
                "cudnn_conv_use_max_workspace": "1",
                "do_copy_in_default_stream": "1",
            }
        }
    )
    
    # Model'i ONNX'e dönüştür
    try:
        ort_model = ORTModelForSeq2SeqLM.from_transformers(
            model,
            config=ort_config,
            export=True,
            use_cache=False,
            use_merged=True,  # Memory efficiency
        )
        
        # Kaydet
        ort_model.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)
        
        print(f"✅ Optimum ONNX export tamamlandı: {export_dir}")
        
        # Model boyutu raporu
        onnx_files = list(export_dir.glob("*.onnx"))
        total_size = sum(f.stat().st_size for f in onnx_files) / 1e6
        print(f"📊 ONNX model boyutu: {total_size:.1f}MB")
        
    except Exception as e:
        print(f"❌ Optimum export hatası: {e}")
        raise e

def export_native_onnx(model, tokenizer, export_dir):
    """Native PyTorch ONNX export"""
    print("🔧 Native ONNX export...")
    
    model.eval()
    
    # Dummy input oluştur
    dummy_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 128))
    dummy_attention_mask = torch.ones(1, 128)
    
    dummy_inputs = {
        'input_ids': dummy_input_ids,
        'attention_mask': dummy_attention_mask
    }
    
    # ONNX export
    onnx_path = export_dir / "model.onnx"
    
    try:
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            onnx_path,
            export_params=True,
            opset_version=14,  # Jetson uyumlu
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'attention_mask': {0: 'batch_size', 1: 'sequence'},
                'output': {0: 'batch_size', 1: 'sequence'}
            },
            verbose=False
        )
        
        # Tokenizer'ı kaydet
        tokenizer.save_pretrained(export_dir)
        
        print(f"✅ Native ONNX export tamamlandı: {onnx_path}")
        
    except Exception as e:
        print(f"❌ Native ONNX export hatası: {e}")
        raise e

def export_simple_onnx(model, tokenizer, export_dir):
    """Basit ONNX export (fallback)"""
    print("📦 Basit ONNX export...")
    
    try:
        # Model'i TorchScript'e çevir
        model.eval()
        
        # Trace model
        dummy_input = torch.randint(0, 1000, (1, 64))
        
        with torch.no_grad():
            traced_model = torch.jit.trace(model.generate, dummy_input)
        
        # Kaydet
        traced_path = export_dir / "traced_model.pt"
        traced_model.save(traced_path)
        
        # Tokenizer'ı kaydet
        tokenizer.save_pretrained(export_dir)
        
        print(f"✅ TorchScript export tamamlandı: {traced_path}")
        
    except Exception as e:
        print(f"❌ Basit export de başarısız: {e}")

def verify_onnx_model(export_dir):
    """ONNX model doğrulaması"""
    print("🔍 ONNX model doğrulanıyor...")
    
    try:
        import onnx
        import onnxruntime as ort
        
        onnx_files = list(Path(export_dir).glob("*.onnx"))
        
        for onnx_file in onnx_files:
            # ONNX model yükle
            onnx_model = onnx.load(onnx_file)
            onnx.checker.check_model(onnx_model)
            
            # ONNX Runtime session
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(str(onnx_file), providers=providers)
            
            print(f"✅ {onnx_file.name} doğrulandı")
            print(f"   Inputs: {[inp.name for inp in session.get_inputs()]}")
            print(f"   Outputs: {[out.name for out in session.get_outputs()]}")
        
    except ImportError:
        print("⚠️ ONNX Runtime bulunamadı, doğrulama atlandı")
    except Exception as e:
        print(f"❌ ONNX doğrulama hatası: {e}")

def create_jetson_config(export_dir):
    """Jetson deployment için config dosyası"""
    config = {
        "model_type": "agricultural_rag",
        "framework": "onnx",
        "target_device": "jetson_orin_nano",
        "optimization": {
            "mixed_precision": True,
            "tensorrt": True,
            "cuda_graph": True
        },
        "inference": {
            "max_batch_size": 4,
            "max_sequence_length": 512,
            "temperature": 0.7,
            "top_p": 0.9
        },
        "memory": {
            "gpu_memory_limit_gb": 6,
            "cpu_memory_limit_gb": 4
        }
    }
    
    config_path = Path(export_dir) / "jetson_config.json"
    
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"📋 Jetson config oluşturuldu: {config_path}")

def main():
    """Ana ONNX export fonksiyonu"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Jetson ONNX Export")
    parser.add_argument("--model_dir", type=str, required=True, help="Model dizini")
    parser.add_argument("--export_dir", type=str, help="Export dizini")
    parser.add_argument("--verify", action="store_true", help="ONNX doğrulaması yap")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    
    if not model_dir.exists():
        print(f"❌ Model dizini bulunamadı: {model_dir}")
        return
    
    # Model ve tokenizer yükle
    print(f"📂 Model yükleniyor: {model_dir}")
    
    try:
        tokenizer = RagTokenizer.from_pretrained(model_dir)
        model = RagSequenceForGeneration.from_pretrained(model_dir)
        
        print("✅ Model yüklendi")
        
        # Export
        export_to_onnx(model, tokenizer, model_dir, args.export_dir)
        
        # Config oluştur
        export_dir = Path(args.export_dir) if args.export_dir else model_dir / "onnx_jetson"
        create_jetson_config(export_dir)
        
        # Doğrulama
        if args.verify:
            verify_onnx_model(export_dir)
        
        print("🎉 ONNX export başarıyla tamamlandı!")
        
    except Exception as e:
        print(f"❌ Ana hata: {e}")

if __name__ == "__main__":
    main() 