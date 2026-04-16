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


def _patch_clang_utf8_decoding() -> None:
    """Patch clang bindings to tolerate non-UTF-8 bytes, which can occur in
    places such as included vendor headers.

    libclang returns raw bytes for token spellings/cursors. The Python bindings
    decode these as strict UTF-8, which crashes on vendor headers (ffmpeg, Microsoft
    GDK, 3ds Max SDK, etc.) that contain accented characters in author names and
    comments encoded as Windows-1252 or with corrupted multi-byte sequences.
    Using errors='replace' substitutes bad bytes with U+FFFD instead of raising
    UnicodeDecodeError, which is harmless since C++ identifiers are ASCII.
    """
    from clang.cindex import c_interop_string
    import ctypes

    @property
    def _value(self) -> str | None:
        val = super(ctypes.c_char_p, self).value
        if val is None:
            return None
        return val.decode("utf-8", errors="replace")

    c_interop_string.value = _value


_patch_clang_utf8_decoding()

__all__ = [
    'CompilationManager',
    'SourceSpan',
    'MacroSpan',
    'TypeAliasSpan',
    'IncludeRelation',
    'FileExtensions'
]
