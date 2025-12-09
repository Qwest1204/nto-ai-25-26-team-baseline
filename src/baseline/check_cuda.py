import torch
import sys


def final_check():
    print("=== FINAL CHECK ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print("🎉 SUCCESS! GPU acceleration enabled!")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

        # Тест производительности
        print("\n🚀 Performance test:")
        device = torch.device("cuda")

        # Создаем тестовые данные
        x = torch.randn(5000, 5000).to(device)
        y = torch.randn(5000, 5000).to(device)

        # GPU вычисления
        result = x @ y
        print(f"GPU matrix multiplication: {result.shape}")
        print("✅ GPU is working correctly!")

        return True
    else:
        print("❌ Still using CPU version")
        print("Troubleshooting steps:")
        print("1. Run: poetry run pip uninstall torch -y")
        print("2. Run: poetry run pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("3. Restart your terminal/IDE")
        return False


def setup_nomic_config():
    """Финальная настройка NOMIC параметров"""
    config = {
        'NOMIC_MODEL_NAME': 'nomic-ai/nomic-embed-text-v1.5',
        'NOMIC_MAX_LENGTH': 8192,
        'NOMIC_EMBEDDING_DIM': 768,
    }

    if torch.cuda.is_available():
        config['NOMIC_DEVICE'] = "cuda"
        config['NOMIC_BATCH_SIZE'] = 8  # Можно увеличить для GPU
        config['NOMIC_GPU_MEMORY_FRACTION'] = 0.95
        print(f"\n✅ NOMIC will use GPU with batch_size={config['NOMIC_BATCH_SIZE']}")
    else:
        config['NOMIC_DEVICE'] = "cpu"
        config['NOMIC_BATCH_SIZE'] = 2  # Меньше для CPU
        config['NOMIC_GPU_MEMORY_FRACTION'] = None
        print(f"\n⚠️ NOMIC will use CPU with batch_size={config['NOMIC_BATCH_SIZE']}")

    return config


if __name__ == "__main__":
    success = final_check()
    config = setup_nomic_config()

    print("\n=== FINAL CONFIG ===")
    for key, value in config.items():
        print(f"{key}: {value}")
