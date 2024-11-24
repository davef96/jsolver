from setuptools import setup, find_packages
import subprocess

REQUIRED_DRIVER_VERSION = "525.60.13"

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
        return False
    elif driver_version >= REQUIRED_DRIVER_VERSION:
        print(f"Compatible GPU detected with driver version {driver_version}.")
        return True
    else:
        print(f"Incompatible GPU driver version {driver_version}.")
        return False

# Conditional `jax` requirement
if is_compatible_gpu_available():
    install_requires = ['jax[cuda12]']
else:
    install_requires = ['jax']
    print("Falling back to CPU installation.")

# Add additional hardcoded requirements
install_requires.extend([
    'simple-pytree',
    'pytest',
    'pytest-cov'
])

setup(
    name="jsolver",
    version="0.1",
    packages=find_packages(),
    install_requires=install_requires,
)
