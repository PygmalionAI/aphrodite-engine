
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
