import io
import logging
import os
import re
import subprocess
import sys
import warnings
from shutil import which
from typing import List

import torch
from packaging.version import Version, parse
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import CUDA_HOME

ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)
# Target device of Aphrodite, supporting [cuda (by default), rocm, neuron, cpu]
APHRODITE_TARGET_DEVICE = os.getenv("APHRODITE_TARGET_DEVICE", "cuda")


def embed_commit_hash():
    try:
        commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                            encoding="utf-8").strip()
        short_commit_id = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], encoding="utf-8").strip()

        commit_contents = f'__commit__ = "{commit_id}"\n'
        short_commit_contents = f'__short_commit__ = "{short_commit_id}"\n'

        version_file = os.path.join(ROOT_DIR, "aphrodite", "commit_id.py")
        with open(version_file, "w", encoding="utf-8") as f:
            f.write(commit_contents)
            f.write(short_commit_contents)

    except subprocess.CalledProcessError as e:
        warnings.warn(f"Failed to get commit hash:\n{e}",
                      RuntimeWarning,
                      stacklevel=2)
    except Exception as e:
        warnings.warn(f"Failed to embed commit hash:\n{e}",
                      RuntimeWarning,
                      stacklevel=2)


embed_commit_hash()

if not sys.platform.startswith("linux"):
    logger.warning(
        "Aphrodite only supports Linux platform (including WSL). "
        f"Building on {sys.platform}, "
        "so APhrodite may not be able to run correctly")
    APHRODITE_TARGET_DEVICE = "empty"

MAIN_CUDA_VERSION = "12.4"


def is_sccache_available() -> bool:
    return which("sccache") is not None


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


class CMakeExtension(Extension):

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = os.environ.get("MAX_JOBS", None)
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info(f"Using MAX_JOBS={num_jobs} as the number of jobs.")
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
                logger.info(f"Using {num_jobs} CPUs as the number of jobs.")
            except AttributeError:
                num_jobs = os.cpu_count()
                logger.info(f"Using os.cpu_count()={num_jobs} as the number of"
                            " jobs.")

        nvcc_threads = None
        if _is_cuda() and get_nvcc_cuda_version() >= Version("11.2"):
            # `nvcc_threads` is either the value of the NVCC_THREADS
            # environment variable (if defined) or 1.
            # when it is set, we reduce `num_jobs` to avoid
            # overloading the system.
            nvcc_threads = os.getenv("NVCC_THREADS", None)
            if nvcc_threads is not None:
                nvcc_threads = int(nvcc_threads)
                logger.info(f"Using NVCC_THREADS={nvcc_threads} as the number"
                            " of nvcc threads.")
            else:
                nvcc_threads = 1
            num_jobs = max(1, num_jobs // nvcc_threads)

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = os.getenv("CMAKE_BUILD_TYPE", default_cfg)

        # where .so files will be written, should be the same for all extensions
        # that use the same CMakeLists.txt.
        outdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name)))

        cmake_args = [
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(outdir),
            '-DCMAKE_ARCHIVE_OUTPUT_DIRECTORY={}'.format(self.build_temp),
            '-DAPHRODITE_TARGET_DEVICE={}'.format(APHRODITE_TARGET_DEVICE),
        ]

        verbose = bool(int(os.getenv('VERBOSE', '0')))
        if verbose:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=ON']

        if is_sccache_available():
            cmake_args += [
                '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache',
            ]
            logger.info("Using sccache as the compiler launcher.")
        elif is_ccache_available():
            cmake_args += [
                '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache',
            ]
            logger.info("Using ccache as the compiler launcher.")

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += [
            '-DAPHRODITE_PYTHON_EXECUTABLE={}'.format(sys.executable)
        ]

        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ['-DNVCC_THREADS={}'.format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ['-G', 'Ninja']
            cmake_args += [
                '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []

        subprocess.check_call(
            ['cmake', ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp)

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e
        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []
        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(remove_prefix(ext.name, "aphrodite."))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)


def _no_device() -> bool:
    return APHRODITE_TARGET_DEVICE == "empty"

def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return (APHRODITE_TARGET_DEVICE == "cuda" and has_cuda
            and not (_is_neuron() or _is_tpu()))


def _is_hip() -> bool:
    return (APHRODITE_TARGET_DEVICE == "cuda"
            or APHRODITE_TARGET_DEVICE == "rocm") \
            and torch.version.hip is not None


def _is_neuron() -> bool:
    torch_neuronx_installed = True
    try:
        subprocess.run(["neuron-ls"], capture_output=True, check=True)
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        torch_neuronx_installed = False
    return torch_neuronx_installed


def _is_tpu() -> bool:
    return APHRODITE_TARGET_DEVICE == "tpu"


def _is_cpu() -> bool:
    return APHRODITE_TARGET_DEVICE == "cpu"


def _is_openvino() -> bool:
    return APHRODITE_TARGET_DEVICE == "openvino"


def _is_xpu() -> bool:
    return APHRODITE_TARGET_DEVICE == "xpu"


def _build_custom_ops() -> bool:
    return _is_cuda() or _is_hip() or _is_cpu()


def _build_core_ext() -> bool:
    return not _is_neuron() and not _is_tpu()


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
    version = find_version(get_path("aphrodite", "version.py"))

    if _no_device():
        version += "+empty"
    elif _is_cuda():
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
    elif _is_openvino():
        version += "+openvino"
    elif _is_tpu():
        version += "+tpu"
    elif _is_cpu():
        version += "+cpu"
    elif _is_xpu():
        version += "+xpu"
    else:
        raise RuntimeError("Unknown runtime environment, "
                           "must be either CUDA, ROCm, CPU, or Neuron.")

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

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    if _no_device():
        requirements = _read_requirements("requirements-cuda.txt")
    elif _is_cuda():
        requirements = _read_requirements("requirements-cuda.txt")
        cuda_major, cuda_minor = torch.version.cuda.split(".")
        modified_requirements = []
        for req in requirements:
            if ("aphrodite-flash-attn" in req
                    and not (cuda_major == "12" and cuda_minor == "4")):
                # aphrodite-flash-attn is built only for CUDA 12.4.
                # Skip for other versions.
                continue
            modified_requirements.append(req)
    elif _is_hip():
        requirements = _read_requirements("requirements-rocm.txt")
    elif _is_neuron():
        requirements = _read_requirements("requirements-neuron.txt")
    elif _is_openvino():
        requirements = _read_requirements("requirements-openvino.txt")
    elif _is_tpu():
        requirements = _read_requirements("requirements-tpu.txt")
    elif _is_cpu():
        requirements = _read_requirements("requirements-cpu.txt")
    elif _is_xpu():
        requirements = _read_requirements("requirements-xpu.txt")
    else:
        raise ValueError(
            "Unsupported platform, please use CUDA, ROCm, Neuron, CPU or "
            "OpenVINO.")
    return requirements


ext_modules = []

if _build_core_ext():
    ext_modules.append(CMakeExtension(name="aphrodite._core_C"))

if _is_cuda() or _is_hip():
    ext_modules.append(CMakeExtension(name="aphrodite._moe_C"))

if _build_custom_ops():
    ext_modules.append(CMakeExtension(name="aphrodite._C"))

package_data = {
    "aphrodite": [
        "endpoints/kobold/klite.embd", "quantization/hadamard.safetensors",
        "py.typed", "modeling/layers/fused_moe/configs/*.json"
    ]
}
if os.environ.get("APHRODITE_USE_PRECOMPILED"):
    ext_modules = []
    package_data["aphrodite"].append("*.so")

if _no_device():
    ext_modules = []

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
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",  # noqa: E501
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=("kernels", "examples", "tests*")),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    extras_require={
        "flash-attn": ["flash-attn==2.5.8"],
        "tensorizer": ["tensorizer>=2.9.0"],
        "ray": ["ray>=2.9"],
    },
    ext_modules=ext_modules,
    cmdclass={"build_ext": cmake_build_ext} if len(ext_modules) > 0 else {},
    package_data=package_data,
    entry_points={
        "console_scripts": [
            "aphrodite=aphrodite.endpoints.cli:main",
        ],
    },
)
