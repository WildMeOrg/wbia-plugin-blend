# -*- coding: utf-8 -*-
from wbia_blend import _plugin  # NOQA
from wbia_blend import train_blend  # NOQA


try:
    from wbia_blend._version import __version__  # NOQA
except ImportError:
    __version__ = '0.0.0'
