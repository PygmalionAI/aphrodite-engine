def bootstrap():
    import os
    import sys

    has_pinnocchio = False
    pwd = os.path.dirname(__file__)
    if os.path.exists(os.path.join(pwd, 'lib')):
        has_pinnocchio = True
    if os.name == 'nt' and has_pinnocchio:
        if sys.version_info[:2] >= (3, 8):
            CUDA_PATH = os.getenv('CUDA_PATH')
            os.add_dll_directory(os.path.join(CUDA_PATH, 'bin'))


bootstrap()
