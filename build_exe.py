import os
import sys
import shutil
import PyInstaller.__main__
import torch
import site
from pathlib import Path
import outlines
import pyairports

def collect_cuda_dlls():
    """Collect all required CUDA DLLs"""
    cuda_dlls = []
    
    # Get CUDA DLLs from torch
    torch_path = os.path.dirname(torch.__file__)
    cuda_path = os.path.join(torch_path, "lib")
    
    # Standard CUDA DLLs from torch
    for file in os.listdir(cuda_path):
        if file.endswith('.dll'):
            cuda_dlls.append((os.path.join(cuda_path, file), '.'))

    # Additional CUDA locations to check
    potential_cuda_paths = [
        os.environ.get('CUDA_PATH', ''),
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2',
    ]

    required_cuda_dlls = [
        'cublas64*.dll',
        'cublasLt64*.dll',
        'cudart64*.dll',
        'cufft64*.dll',
        'curand64*.dll',
        'cusolver64*.dll',
        'cusparse64*.dll',
        'nvrtc64*.dll',
        'nvrtc-builtins64*.dll',
    ]

    # Search for CUDA DLLs in potential locations
    for cuda_path in potential_cuda_paths:
        if cuda_path and os.path.exists(cuda_path):
            bin_path = os.path.join(cuda_path, 'bin')
            if os.path.exists(bin_path):
                for dll in required_cuda_dlls:
                    for file in Path(bin_path).glob(dll):
                        cuda_dlls.append((str(file), '.'))

    return cuda_dlls

def collect_torch_dlls():
    """Collect PyTorch and related DLLs"""
    torch_dlls = []
    torch_path = os.path.dirname(torch.__file__)
    
    # Add PyTorch DLLs
    for file in os.listdir(torch_path):
        if file.endswith('.dll'):
            torch_dlls.append((os.path.join(torch_path, file), '.'))
            
    return torch_dlls

def collect_outlines_data():
    """Collect Outlines grammar files"""
    outlines_path = os.path.dirname(outlines.__file__)
    grammar_files = []
    
    # Add grammar files
    grammar_path = os.path.join(outlines_path, 'grammars')
    if os.path.exists(grammar_path):
        for file in os.listdir(grammar_path):
            if file.endswith('.lark'):
                src = os.path.join(grammar_path, file)
                # Note: The destination needs to match the expected path
                dst = os.path.join('outlines', 'grammars')
                grammar_files.append((src, dst))

    return grammar_files

def collect_pyairports_data():
    """Collect PyAirports data files"""
    pyairports_path = os.path.dirname(pyairports.__file__)
    data_files = []
    
    # Add data files
    data_path = os.path.join(pyairports_path, 'data')
    if os.path.exists(data_path):
        for file in os.listdir(data_path):
            if file.endswith('.json'):
                src = os.path.join(data_path, file)
                dst = os.path.join('pyairports', 'data')
                data_files.append((src, dst))
    
    return data_files

def main():
    # Collect all necessary DLLs
    binaries = []
    binaries.extend(collect_cuda_dlls())
    binaries.extend(collect_torch_dlls())

    outlines_data = collect_outlines_data()
    pyairports_data = collect_pyairports_data()

    # Define hidden imports
    hidden_imports = [
        'torch',
        'xformers',
        'ninja',
        'packaging',
        'torch._C',
        'torch.cuda',
        'torch.cuda.amp',
        'torch.cuda.memory',
        'torch._utils',
        'torch.version',
        'torch.jit',
        'torch.nn',
        'torch.nn.functional',
        'torch.nn.parallel',
        'torch.utils.data',
        'winloop',
        'winloop.loop',
        'winloop._noop',
        'winloop._windows',
        'yaml',
        'openai',
        'argparse',
        'asyncio',
        'signal',
        'subprocess',
        'multiprocessing',
        'multiprocessing.pool',
        'multiprocessing.managers',
        'multiprocessing.popen_spawn_win32',
        'multiprocessing.spawn',
        'multiprocessing.reduction',
        'multiprocessing.synchronize',
        'multiprocessing.util',
        'multiprocessing.context',
        'multiprocessing.connection',
        'zmq',
        'zmq.backend',
        'zmq.backend.cython',
        'zmq.utils',
        'zmq.utils.jsonapi',
        'zmq.utils.strtypes',
    ]

    entry_point = os.path.join('aphrodite', 'endpoints', 'cli.py')
    if not os.path.exists(entry_point):
        raise FileNotFoundError(f"Entry point not found: {entry_point}")

    # Create runtime hook with better multiprocessing support
    with open('runtime_hook.py', 'w') as f:
        f.write("""
import os
import sys
import logging
import asyncio

# Fix for Windows event loop
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('aphrodite_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('aphrodite')

def runtime_hook():
    if getattr(sys, 'frozen', False):
        logger.debug('Running in frozen environment')
        logger.debug(f'Executable path: {sys.executable}')
        logger.debug(f'Working directory: {os.getcwd()}')
        logger.debug(f'Command line args: {sys.argv}')
        
        # Fix multiprocessing args
        if len(sys.argv) > 1:
            # Check if this is a child process
            parent_pid_args = [arg for arg in sys.argv if arg.startswith('--multiprocessing-fork')]
            if parent_pid_args:
                # This is a child process, reconstruct the original command
                # Remove the multiprocessing arguments
                sys.argv = [arg for arg in sys.argv if not arg.startswith('--multiprocessing-fork')]
                if not sys.argv[1:]:  # If no args left, add default 'run'
                    sys.argv.append('run')
                logger.debug(f'Modified args for child process: {sys.argv}')
        
        os.environ['CUDA_PATH'] = os.path.dirname(sys.executable)
        os.environ['CUDA_HOME'] = os.path.dirname(sys.executable)
        os.environ['PATH'] = os.path.dirname(sys.executable) + os.pathsep + os.environ['PATH']
""")

    # PyInstaller command line arguments
    pyinstaller_args = [
        entry_point,
        '--name=aphrodite',
        '--onedir',
        '--clean',
        # Data files
        '--add-data=aphrodite/endpoints/kobold/klite.embd;aphrodite/endpoints/kobold',
        '--add-data=aphrodite/quantization/hadamard.safetensors;aphrodite/quantization',
        '--add-data=aphrodite/modeling/layers/fused_moe/configs/*;aphrodite/modeling/layers/fused_moe/configs',
        '--add-data=aphrodite/endpoints/openai/api_server.py;aphrodite/endpoints/openai',
        '--add-data=aphrodite/endpoints/openai/rpc/server.py;aphrodite/endpoints/openai/rpc',
        # Add Outlines grammar files
        *[f'--add-data={src};{dst}' for src, dst in outlines_data],
        # Add PyAirports data files
        *[f'--add-data={src};{dst}' for src, dst in pyairports_data],
        # Hidden imports
        *[f'--hidden-import={imp}' for imp in hidden_imports],
        # Binaries
        *[f'--add-binary={src};{dst}' for src, dst in binaries],
        # Additional PyInstaller options
        '--collect-all=torch',
        '--collect-all=xformers',
        '--collect-all=aphrodite',
        '--collect-all=outlines',
        '--collect-all=pyairports',
        '--collect-all=winloop',
        # Exclude unnecessary torch components to reduce size
        '--exclude-module=torch.distributions',
        '--exclude-module=torch.testing',
        '--exclude-module=torch.utils.tensorboard',
        # Additional options for better compatibility
        '--runtime-hook=runtime_hook.py',
        '--collect-submodules=multiprocessing',
    ]

    # Run PyInstaller
    PyInstaller.__main__.run(pyinstaller_args)

if __name__ == '__main__':
    main()