import sys
import subprocess
import platform

def has_nvidia_gpu():
    """Sprawdza, czy system ma kartę NVIDIA."""
    try:
        output = subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return b"NVIDIA" in output
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def install_torch_with_cuda():
    """Instaluje wersję PyTorch z obsługą CUDA, jeśli to możliwe."""
    torch_cuda_url = "https://download.pytorch.org/whl/cu126"
    packages = ["torch", "torchvision", "torchaudio"]
    install_command = [sys.executable, "-m", "uv", "pip", "install", f"--index-url={torch_cuda_url}"] + packages

    print("🔍 Sprawdzanie karty graficznej...")
    if has_nvidia_gpu():
        print("✅ Wykryto kartę NVIDIA! Instalowanie wersji z CUDA...")
        subprocess.run(install_command, check=True)
        print("🚀 CUDA PyTorch zainstalowany!")
    else:
        print("⚠️ Brak karty NVIDIA lub `nvidia-smi` niedostępne. Pozostaję przy wersji CPU.")

if __name__ == "__main__":
    if platform.system() == "Linux" or platform.system() == "Windows":
        install_torch_with_cuda()
    else:
        print("⚠️ macOS wykryty – pozostaję przy wersji CPU.")

