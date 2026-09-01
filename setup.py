from setuptools import setup, find_packages
import subprocess

REQUIRED_DRIVER_VERSION_CUDA12 = "525.60.13"
REQUIRED_DRIVER_VERSION_CUDA13 = "580.65.06"

def get_driver_version():
    try:
        output = subprocess.check_output("nvidia-smi --query-gpu=driver_version --format=csv,noheader", shell=True)
        driver_version = output.decode("utf-8").strip()
        return driver_version
    except subprocess.CalledProcessError:
        return None

def is_compatible_gpu_available():
    driver_version = get_driver_version()
    if driver_version is None:
        print("No GPU detected or unable to get driver version.")
        return 0
    elif driver_version >= REQUIRED_DRIVER_VERSION_CUDA13:
        print(f"CUDA13-compatible GPU detected with driver version {driver_version}.")
        return 1
    elif driver_version >= REQUIRED_DRIVER_VERSION_CUDA12:
        print(f"CUDA12-compatible GPU detected with driver version {driver_version}.")
        return 2
    else:
        print(f"GPU driver version neither compatible with CUDA12 nor CUDA13: {driver_version}.")
        return 0

# Conditional JAX requirement
compatible = is_compatible_gpu_available()
if compatible == 1:
    install_requires = ['jax[cuda13]']
elif compatible == 2:
    install_requires = ['jax[cuda12]']
else:
    install_requires = ['jax']
    print("Falling back to CPU installation.")

# Add additional hardcoded requirements
install_requires.extend([
    'pytest',
    'pytest-cov'
])

setup(
    name="jsolver",
    version="0.3.0",
    packages=find_packages(),
    install_requires=install_requires, # set to [] for manual dependency management
)
