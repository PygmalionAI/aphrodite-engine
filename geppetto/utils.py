import logging
from typing import Optional

logger_initialized = {}

def get_logger(name: str,
               log_file: Optional[str] = None,
               log_level: int = logging.INFO,
               file_mode: str = 'w'):
    try:
        from mmengine.logging import MMLogger
        if MMLogger.check_instance_created(name):
            logger = MMLogger.get_instance(name)
        else:
            logger = MMLogger.get_instance(name,
                                           logger_name=name,
                                           log_file=log_file,
                                           log_level=log_level,
                                           file_mode=file_mode)
        return logger

    except Exception:
        pass

    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names. e.g., logger `a` is initialized 
    # then logger `a.b` will skip the initialization since it is 
    # a child of `a`
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
        
    logger.setLevel(log_level)
    logger_initialized[name] = True

    return logger
