#!/usr/bin/env python3
"""
Compilation engine for parsing C/C++ source code and extracting semantic metadata.
"""

import os
import logging
import clang
from .manager import CompilationManager
from .types import SourceSpan, MacroSpan, TypeAliasSpan, IncludeRelation
from utils import FileExtensions

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def _configure_libclang_from_env() -> None:
    """Configure libclang from LIBCLANG_PATH, as either a DLL file or a directory."""
    libclang_path = os.getenv("LIBCLANG_PATH")
    if not libclang_path:
        return

    libclang_path = os.path.abspath(libclang_path)

    if os.path.isfile(libclang_path):
        try:
            clang.cindex.Config.set_library_file(libclang_path)
            logger.debug(f"Configured libclang from file: {libclang_path}")
            return
        except Exception as e:
            logger.debug(f"Could not configure libclang from file '{libclang_path}': {e}")

    if os.path.isdir(libclang_path):
        try:
            clang.cindex.Config.set_library_path(libclang_path)
            logger.debug(f"Configured libclang search path: {libclang_path}")
        except Exception as e:
            logger.debug(f"Could not configure libclang from path '{libclang_path}': {e}")


_configure_libclang_from_env()

__all__ = [
    'CompilationManager',
    'SourceSpan',
    'MacroSpan',
    'TypeAliasSpan',
    'IncludeRelation',
    'FileExtensions'
]
