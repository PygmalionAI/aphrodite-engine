import io
import os
import re
import subprocess
from typing import List, Set

from packaging.version import parse, Version
import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)

CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
NVCC_FLAGS = ["-O3", "-std=c++17"]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if CUDA_HOME is None:
    raise RuntimeError(
        f"Cannot find CUDA_HOME. CUDA must be available in order to build the package.")

def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version

device_count = torch.cuda.device_count()
compute_capabilities: Set[int] = set()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 7:
        raise RuntimeError(
            "GPUs with compute capability less than 7.0 are not supported.")
    compute_capabilities(major * 10 + minor)
if not compute_capabilities:
    compute_capabilities = {70, 75, 80, 86, 90}
for capability in compute_capabilities:
    NVCC_FLAGS += ["-gencode", f"arch=compute_{capability},code=sm_{capability}"]

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if nvcc_cuda_version < Version("11.0"):
    raise RuntimeError("CUDA 11.0 or higher is required to build the package.")
if 86 in compute_capabilities and nvcc_cuda_version < Version("11.1"):
    raise RuntimeError(
        "CUDA 11.1 or higher is required for GPUs with compute capability 8.6.")
if 90 in compute_capabilities and nvcc_cuda_version < Version("11.8"):
    raise RuntimeError(
        "CUDA 11.8 or higher is required for GPUs with compute capability 9.0.")

if nvcc_cuda_version >= Version("11.2"):
    num_threads = min(os.cpu_count(), 8)
    NVCC_FLAGS += ["--threads", str(num_threads)]

ext_modules = []

# Cache operations
cache_extension = CUDAExtension(
    name="aphrodite.cache_ops",
    sources=["kernels/cache.cpp", "kernels/cache_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(cache_extension)

# Attention operations
attention_extension = CUDAExtension(
    name="aphrodite.attention_ops",
    sources=["kernels/attention.cpp", "kernels/attention/attention_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(attention_extension)

# Positional encoding operations
positional_encoding_extension = CUDAExtension(
    name="aphrodite.pos_encoding_ops",
    sources=["kernels/pos_encoding.cpp", "kernels/pos_encoding_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(positional_encoding_extension)

# Layer normalization operations
layernorm_extension = CUDAExtension(
    name="aphrodite.layernorm_ops",
    sources=["kernels/layernorm.cpp", "kernels/layernorm_kernels.cu"],
    extra_compile_args={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(layernorm_extension)

# Activation operations
activation_extension = CUDAExtension(
    name="aphrodite.activation_ops",
    sources=["kernels/activation.cpp", "kernels/activation_kernels.cu"],
    extra_compile_flags={"cxx": CXX_FLAGS, "nvcc": NVCC_FLAGS},
)
ext_modules.append(activation_extension)


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str):
    with open(filepath) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")
    

def read_readme() -> str:
    return io.open(get_path("README.md"), "r", encoding="utf-8").read()


def get_requirements() -> List[str]:
    with open(get_path("requirements.txt")) as f:
        requirements = f.read().strip().split("\n")
    return requirements


setuptools.setup(
    name="aphrodite",
    version=find_version(get_path("aphrodite", "__init__.py")),
    author="PygmalionAI Team",
    license="AGPL-3.0",
    description="The official inference engine for PygmalionAI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/PygmalionAI/aphrodite-engine",
    project_urls={
        "Homepage": "https://pygmalion.chat/",
        "Documentation": "https://docs.pygmalion.chat/",
        "HuggingFace": "https://huggingface.co/PygmalionAI",
        "GitHub": "https://github.com/PygmalionAI",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU AFFERO GENERAL PUBLIC LICENSE",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=setuptools.find_packages(
        exclude=("assets", "kernels", "examples")),
    python_requirements=">=3.8",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
    