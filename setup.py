import io
import os
import re
import subprocess
import sys
from typing import List, Set
import warnings
from pathlib import Path

from packaging.version import parse, Version
import setuptools
import torch
import torch.utils.cpp_extension as torch_cpp_ext
from setuptools.command.build_ext import build_ext
from setuptools import setup, find_packages, Extension
from torch.utils.cpp_extension import (BuildExtension, CUDAExtension,
                                       CUDA_HOME, ROCM_HOME)
from shutil import which

ROOT_DIR = os.path.dirname(__file__)

MAIN_CUDA_VERSION = "12.1"

assert sys.platform.startswith(
    "linux"), "Aphrodite only supports Linux at the moment (including WSL)."


def _is_cuda() -> bool:
    return torch.version.cuda is not None and not _is_neuron()


def _is_hip() -> bool:
    return torch.version.hip is not None


def _is_neuron() -> bool:
    torch_neuronx_installed = True
    try:
        subprocess.run(["neuron-ls"], capture_output=True, check=True)
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        torch_neuronx_installed = False
    return torch_neuronx_installed


# Compiler flags.
CXX_FLAGS = ["-g", "-O2", "-std=c++17"]
# TODO: Should we use -O3?
NVCC_FLAGS = ["-O2", "-std=c++17"]

if _is_hip():
    if ROCM_HOME is None:
        raise RuntimeError(
            "Cannot find ROCM_HOME. ROCm must be available to build the "
            "package.")
    NVCC_FLAGS += ["-DUSE_ROCM"]

if _is_cuda() and CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]


def get_hipcc_rocm_version():
    # Run the hipcc --version command
    result = subprocess.run(['hipcc', '--version'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True)

    # Check if the command was executed successfully
    if result.returncode != 0:
        print("Error running 'hipcc --version'")
        return None

    # Extract the version using a regular expression
    match = re.search(r'HIP version: (\S+)', result.stdout)
    if match:
        # Return the version string
        return match.group(1)
    else:
        print("Could not find HIP version in the output")
        return None


def get_neuronxcc_version():
    import sysconfig
    site_dir = sysconfig.get_paths()["purelib"]
    version_file = os.path.join(site_dir, "neuronxcc", "version",
                                "__init__.py")

    # Check if the command was executed successfully
    with open(version_file, "rt") as fp:
        content = fp.read()

    # Extract the version using a regular expression
    match = re.search(r"__version__ = '(\S+)'", content)
    if match:
        # Return the version string
        return match.group(1)
    else:
        raise RuntimeError("Could not find HIP version in the output")


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc.

    Adapted from https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/setup.py
    """
    nvcc_output = subprocess.check_output([CUDA_HOME + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def find_version(filepath: str) -> str:
    """Extract version information from the given filepath.

    Adapted from https://github.com/ray-project/ray/blob/0b190ee1160eeca9796bc091e07eaebf4c85b511/python/setup.py
    """
    with open(filepath) as fp:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                                  fp.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


def get_aphrodite_version() -> str:
    version = find_version(get_path("aphrodite", "__init__.py"))

    if _is_cuda():
        cuda_version = str(get_nvcc_cuda_version())
        if cuda_version != MAIN_CUDA_VERSION:
            cuda_version_str = cuda_version.replace(".", "")[:3]
            version += f"+cu{cuda_version_str}"
    elif _is_hip():
        # Get the HIP version
        hipcc_version = get_hipcc_rocm_version()
        if hipcc_version != MAIN_CUDA_VERSION:
            rocm_version_str = hipcc_version.replace(".", "")[:3]
            version += f"+rocm{rocm_version_str}"
    elif _is_neuron():
        # Get the Neuron version
        neuron_version = str(get_neuronxcc_version())
        if neuron_version != MAIN_CUDA_VERSION:
            neuron_version_str = neuron_version.replace(".", "")[:3]
            version += f"+neuron{neuron_version_str}"
    else:
        raise RuntimeError("Unknown environment. Only "
                           "CUDA, HIP, and Neuron are supported.")

    return version


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        return io.open(get_path("README.md"), "r", encoding="utf-8").read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    if _is_cuda():
        with open(get_path("requirements.txt")) as f:
            requirements = f.read().strip().split("\n")
    elif _is_hip():
        with open(get_path("requirements-rocm.txt")) as f:
            requirements = f.read().strip().split("\n")
    elif _is_neuron():
        with open(get_path("requirements-neuron.txt")) as f:
            requirements = f.read().strip().split("\n")
    else:
        raise ValueError(
            "Unsupported platform, please use CUDA, ROCm or Neuron.")
    return requirements


ext_modules = []

if _is_cuda():
    ext_modules.append(CMakeExtension(name="aphrodite._moe_C"))

    if _install_punica():
        ext_modules.append(CMakeExtension(name="aphrodite._punica_C"))

    if _install_hadamard():
        ext_modules.append(CMakeExtension(name="aphrodite._hadamard_C"))

if not _is_neuron():
    ext_modules.append(CMakeExtension(name="aphrodite._C"))

package_data = {
    "aphrodite": [
        "endpoints/kobold/klite.embd",
        "modeling/layers/quantization/hadamard.safetensors", "py.typed",
        "modeling/layers/fused_moe/configs/*.json"
    ]
}
if os.environ.get("APHRODITE_USE_PRECOMPILED"):
    package_data["aphrodite"].append("*.so")

setup(
    name="aphrodite-engine",
    version=get_aphrodite_version(),
    author="PygmalionAI",
    license="AGPL 3.0",
    description="The inference engine for PygmalionAI models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/PygmalionAI/aphrodite-engine",
    project_urls={
        "Homepage": "https://pygmalion.chat",
        "Documentation": "https://docs.pygmalion.chat",
        "GitHub": "https://github.com/PygmalionAI",
        "Huggingface": "https://huggingface.co/PygmalionAI",
    },
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",  # noqa: E501
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=("kernels", "examples",
                                               "tests")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={"flash-attn": [
        "flash-attn==2.5.6",
    ]},
    ext_modules=ext_modules,
    cmdclass={"build_ext": cmake_build_ext} if not _is_neuron() else {},
    include_package_data=True,
)
