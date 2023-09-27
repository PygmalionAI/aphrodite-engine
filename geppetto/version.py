from typing import Tuple

__version__ = "0.0"
short_version = __version__

def parse_version_info(short_version: str) -> Tuple:
    _version_info = []
    for x in version_str.split('.'):
        if x.isdigit():
            _version_info.append(int(x))
        elif x.find('rc') != -1:
            patch_version = x.split('rc')
            _version_info.append(int(patch_version[0]))
            _version_info.append(f'rc{patch_version[1]}')
    return tuple(_version_info)

version_info = parse_version_info(__version__)

__all__ = ['__version__', '_version_info', 'parse_version_info']
