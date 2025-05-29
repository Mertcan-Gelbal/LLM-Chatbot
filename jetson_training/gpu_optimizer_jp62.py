#!/usr/bin/env python3
"""
Jetson GPU Optimizer - JetPack 6.2 CUDA 12.2
===========================================
JetPack 6.2 L4T 36.4.3 için optimize edilmiş GPU performans sistemi
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import gc
import os
import psutil
import time

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

class JetsonOptimizerJP62:
    """JetPack 6.2 için gelişmiş GPU optimizasyonları"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_jp62_optimizations()
        
        if NVML_AVAILABLE:
            pynvml.nvmlInit()
    
    def setup_jp62_optimizations(self):
        """JetPack 6.2 CUDA 12.2 optimizasyonları"""
        if torch.cuda.is_available():
            # CUDA 12.2 specific optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # JetPack 6.2 specific settings
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Memory management for CUDA 12.2
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
            
            # Clear cache
            torch.cuda.empty_cache()
            
            print("🔧 JetPack 6.2 CUDA 12.2 optimizasyonları aktif")
    
    def optimize_model_jp62(self, model):
        """JetPack 6.2 için model optimizasyonları"""
        print("⚡ JetPack 6.2 model optimizasyonları...")
        
        # Gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
        
        # Model parallelism
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print(f"📱 {torch.cuda.device_count()} GPU kullanılıyor")
        
        # PyTorch 2.3 compilation (JetPack 6.2)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(
                    model,
                    mode="reduce-overhead",  # JetPack 6.2 için optimize
                    fullgraph=False,
                    dynamic=True
                )
                print("🚀 PyTorch 2.3 model compiled (JetPack 6.2)")
        except Exception as e:
            print(f"ℹ️ Torch compile kullanılamıyor: {e}")
        
        return model
    
    def get_optimal_batch_size_jp62(self, model, input_shape, max_memory_gb=7):
        """JetPack 6.2 için optimal batch size"""
        print("🧮 JetPack 6.2 optimal batch size hesaplanıyor...")
        
        # Test batch sizes for Jetson Orin Nano
        batch_sizes = [1, 2, 4, 6, 8, 12, 16]
        optimal_batch = 1
        
        model.eval()
        
        for batch_size in batch_sizes:
            try:
                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()
                
                # Create dummy input
                dummy_input = torch.randn(batch_size, *input_shape).to(self.device)
                
                # Test forward pass with autocast
                with torch.no_grad():
                    with autocast():
                        try:
                            _ = model(dummy_input)
                        except:
                            # Fallback for complex models
                            _ = model.generate(dummy_input, max_length=50)
                
                # Check memory usage
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                
                if memory_used < max_memory_gb and memory_reserved < max_memory_gb * 1.2:
                    optimal_batch = batch_size
                    print(f"✅ Batch {batch_size}: {memory_used:.2f}GB used, {memory_reserved:.2f}GB reserved")
                else:
                    print(f"❌ Batch {batch_size}: {memory_used:.2f}GB used, {memory_reserved:.2f}GB reserved (çok yüksek)")
                    break
                
                # Clear memory
                del dummy_input
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"❌ Batch {batch_size}: OOM")
                    break
                else:
                    print(f"❌ Batch {batch_size}: Error - {e}")
                    break
        
        print(f"🎯 JetPack 6.2 Optimal batch size: {optimal_batch}")
        return optimal_batch
    
    def monitor_jetson_gpu(self):
        """JetPack 6.2 GPU monitoring"""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            
            print(f"📊 JetPack 6.2 GPU Info:")
            print(f"   Name: {props.name}")
            print(f"   Memory: {props.total_memory / 1e9:.1f}GB")
            print(f"   SM Count: {props.multi_processor_count}")
            print(f"   CUDA Capability: {props.major}.{props.minor}")
            print(f"   CUDA Version: {torch.version.cuda}")
            
            # Current usage
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            free = (props.total_memory - torch.cuda.memory_reserved()) / 1e9
            
            print(f"   Allocated: {allocated:.2f}GB")
            print(f"   Reserved: {reserved:.2f}GB") 
            print(f"   Free: {free:.2f}GB")
            
            # Additional NVML info if available
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # W
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    print(f"   Temperature: {temp}°C")
                    print(f"   Power: {power:.1f}W")
                    print(f"   GPU Util: {utilization.gpu}%")
                    print(f"   Memory Util: {utilization.memory}%")
                    
                except Exception as e:
                    print(f"   NVML error: {e}")
    
    def optimize_memory_jp62(self):
        """JetPack 6.2 memory optimizasyonları"""
        if torch.cuda.is_available():
            # CUDA memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # System memory cleanup
            gc.collect()
            
            # Memory stats
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            
            print(f"💾 Memory optimized - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            # System memory
            mem = psutil.virtual_memory()
            print(f"💾 System Memory - Used: {mem.used/1e9:.2f}GB, Available: {mem.available/1e9:.2f}GB")
    
    def setup_tensorrt_jp62(self, model, input_shape):
        """JetPack 6.2 TensorRT optimizasyonu"""
        try:
            import torch_tensorrt
            
            print("🔥 JetPack 6.2 TensorRT engine oluşturuluyor...")
            
            # JetPack 6.2 TensorRT settings
            trt_model = torch_tensorrt.compile(
                model,
                inputs=[torch_tensorrt.Input(
                    min_shape=input_shape,
                    opt_shape=input_shape,
                    max_shape=input_shape,
                    dtype=torch.float16
                )],
                enabled_precisions=[torch.float16],
                workspace_size=3 << 30,  # 3GB workspace
                max_batch_size=8,
                use_python_runtime=True,
                truncate_long_and_double=True,
                refit=False,
                debug=False,
                device_type=torch_tensorrt.DeviceType.GPU,
                capability=torch_tensorrt.EngineCapability.STANDARD
            )
            
            print("✅ TensorRT engine hazır")
            return trt_model
            
        except ImportError:
            print("❌ TensorRT bulunamadı")
            return model
        except Exception as e:
            print(f"❌ TensorRT hatası: {e}")
            return model
    
    def benchmark_performance_jp62(self, model, input_shape, num_runs=100):
        """JetPack 6.2 performance benchmark"""
        print("📊 JetPack 6.2 Performance Benchmark...")
        
        model.eval()
        
        # Warmup
        dummy_input = torch.randn(1, *input_shape).to(self.device)
        with torch.no_grad():
            for _ in range(10):
                with autocast():
                    try:
                        _ = model(dummy_input)
                    except:
                        _ = model.generate(dummy_input, max_length=20)
        
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            
            with torch.no_grad():
                with autocast():
                    try:
                        _ = model(dummy_input)
                    except:
                        _ = model.generate(dummy_input, max_length=20)
            
            torch.cuda.synchronize()
            end = time.time()
            
            times.append((end - start) * 1000)  # ms
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"⚡ Inference Performance (JetPack 6.2):")
        print(f"   Average: {avg_time:.2f}ms")
        print(f"   Min: {min_time:.2f}ms")
        print(f"   Max: {max_time:.2f}ms")
        print(f"   Throughput: {1000/avg_time:.1f} samples/sec")
        
        return avg_time

class JetsonProfilerJP62:
    """JetPack 6.2 performance profiler"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.start_temp = None
    
    def start(self):
        """Profiling başlat"""
        if torch.cuda.is_available():
            self.start_time = torch.cuda.Event(enable_timing=True)
            self.end_time = torch.cuda.Event(enable_timing=True)
            self.start_memory = torch.cuda.memory_allocated()
            
            # Temperature monitoring
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    self.start_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    self.start_temp = None
            
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
            
            # Temperature check
            temp_info = ""
            if NVML_AVAILABLE and self.start_temp:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    end_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    temp_diff = end_temp - self.start_temp
                    temp_info = f", Temp: {end_temp}°C ({temp_diff:+d}°C)"
                except:
                    pass
            
            print(f"⏱️  JetPack 6.2 Profile - Süre: {elapsed_time:.2f}ms, Memory: {memory_used:.2f}MB{temp_info}")
            
            return elapsed_time, memory_used
        
        return 0, 0 