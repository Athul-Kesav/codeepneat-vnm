"""Lets the CoDeepNEAT model code be imported in the VENOM env (torch 1.13) for benchmarking.

Packages that the code only imports for logging, plotting or data augmentation are replaced with inert stubs; model
construction never calls into them.
"""
import importlib.abc
import importlib.machinery
import sys
import types

STUBS = set()


class _Stub(types.ModuleType):
    __path__ = []

    def __getattr__(self, k):
        if k.startswith("__"):
            raise AttributeError(k)
        return _Stub(f"{self.__name__}.{k}")

    def __call__(self, *a, **k):
        return _Stub("call")

    def __mro_entries__(self, bases):
        return (object,)


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in STUBS:
            return importlib.machinery.ModuleSpec(name, self)
        return None

    def create_module(self, spec):
        return _Stub(spec.name)

    def exec_module(self, module):
        pass


def install(names):
    STUBS.update(names)
    sys.meta_path.insert(0, _Finder())
