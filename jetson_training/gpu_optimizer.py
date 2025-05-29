#!/usr/bin/env python3
"""
Jetson GPU Optimizer
===================
GPU performance optimizations for Jetson Orin Nano
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import gc

class JetsonOptimizer:
    """Jetson Orin Nano için GPU optimizasyonları"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_cuda_optimizations()
    
    def setup_cuda_optimizations(self):
        """CUDA optimizasyonları"""
        if torch.cuda.is_available():
            # CUDA ayarları
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Memory pool ayarları
            torch.cuda.empty_cache()
            
            print("🔧 CUDA optimizasyonları aktif")
    
    def optimize_model(self, model):
        """Model optimizasyonları"""
        print("⚡ Model optimizasyonları uygulanıyor...")
        
        # Gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Model parallelism (if multiple GPUs)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"📱 {torch.cuda.device_count()} GPU kullanılıyor")
        
        # Compile for better performance (PyTorch 2.0+)
        try:
            import torch._dynamo as dynamo
            model = torch.compile(model)
            print("🚀 Model compiled")
        except:
            print("ℹ️ Torch compile kullanılamıyor")
        
        return model
    
    def optimize_memory(self):
        """Memory optimizasyonları"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
            # Memory stats
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            
            print(f"💾 GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    
    def get_optimal_batch_size(self, model, input_shape, max_memory_gb=6):
        """Optimal batch size hesaplama"""
        print("🧮 Optimal batch size hesaplanıyor...")
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        optimal_batch = 1
        
        for batch_size in batch_sizes:
            try:
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Test forward pass
                model.eval()
                with torch.no_grad():
                    with autocast():
                        _ = model(dummy_input)
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated() / 1e9
                
                if memory_used < max_memory_gb:
                    optimal_batch = batch_size
                    print(f"✅ Batch {batch_size}: {memory_used:.2f}GB")
                else:
                    print(f"❌ Batch {batch_size}: {memory_used:.2f}GB (çok yüksek)")
                    break
                
                # Clear memory
                del dummy_input
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ Batch {batch_size}: OOM")
                    break
                else:
                    raise e
        
        print(f"🎯 Optimal batch size: {optimal_batch}")
        return optimal_batch
    
    def setup_mixed_precision(self):
        """Mixed precision setup"""
        scaler = torch.cuda.amp.GradScaler()
        print("🔢 Mixed precision (FP16) aktif")
        return scaler
    
    def monitor_gpu(self):
        """GPU monitoring"""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            
            print(f"📊 GPU Info:")
            print(f"   Name: {props.name}")
            print(f"   Memory: {props.total_memory / 1e9:.1f}GB")
            print(f"   SM Count: {props.multi_processor_count}")
            print(f"   CUDA Capability: {props.major}.{props.minor}")
            
            # Current usage
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            free = (props.total_memory - torch.cuda.memory_reserved()) / 1e9
            
            print(f"   Allocated: {allocated:.2f}GB")
            print(f"   Cached: {cached:.2f}GB") 
            print(f"   Free: {free:.2f}GB")
    
    def optimize_inference(self, model):
        """Inference optimizasyonları"""
        print("🚀 Inference optimizasyonları...")
        
        # Eval mode
        model.eval()
        
        # No grad
        for param in model.parameters():
            param.requires_grad = False
        
        # TorchScript (optional)
        try:
            model = torch.jit.optimize_for_inference(model)
            print("✅ TorchScript inference optimizasyonu")
        except:
            print("ℹ️ TorchScript kullanılamıyor")
        
        return model
    
    def get_tensorrt_engine(self, model, input_shape):
        """TensorRT engine oluştur"""
        try:
            import torch_tensorrt
            
            print("🔥 TensorRT engine oluşturuluyor...")
            
            # Compile with TensorRT
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[torch_tensorrt.Input(input_shape, dtype=torch.float16)],
                enabled_precisions=[torch.float16],
                workspace_size=2 << 30,  # 2GB
            )
            
            print("✅ TensorRT engine hazır")
            return trt_model
            
        except ImportError:
            print("❌ TensorRT bulunamadı")
            return model
        except Exception as e:
            print(f"❌ TensorRT hatası: {e}")
            return model

class JetsonProfiler:
    """Jetson performance profiler"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
    
    def start(self):
        """Profiling başlat"""
        self.start_time = torch.cuda.Event(enable_timing=True)
        self.end_time = torch.cuda.Event(enable_timing=True)
        
        if torch.cuda.is_available():
            self.start_memory = torch.cuda.memory_allocated()
            torch.cuda.synchronize()
            self.start_time.record()
    
    def end(self):
        """Profiling sonlandır"""
        if torch.cuda.is_available():
            self.end_time.record()
            torch.cuda.synchronize()
            
            # Time calculation
            elapsed_time = self.start_time.elapsed_time(self.end_time)  # ms
            
            # Memory calculation
            end_memory = torch.cuda.memory_allocated()
            memory_used = (end_memory - self.start_memory) / 1e6  # MB
            
            print(f"⏱️  Süre: {elapsed_time:.2f}ms")
            print(f"💾 Memory: {memory_used:.2f}MB")
            
            return elapsed_time, memory_used
        
        return 0, 0 