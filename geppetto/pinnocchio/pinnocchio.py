import asyncio
import os.path as osp
import sys
from configparser import ConfigParser
from contextlib import contextmanager
from queue import Queue
from threading import Thread
from typing import Iterable, List

import numpy as np 
import torch
from torch.nn.utils.rnn import pad_sequence

import geppetto
from geppetto.model import MODELS
from geppetto.pinnocchio import Tokenizer
from geppetto.utils import get_logger

geppetto_dir = osp.split(geppetto.__file__)[0]
sys.path.append(osp.join(geppetto_dir, 'lib'))
import _pinnocchio as _pn 

def _stop_words(stop_words: List[str], tokenizer: Tokenizer):
    if stop_words is None:
        return None
    assert isinstance(stop_words, List) and 
