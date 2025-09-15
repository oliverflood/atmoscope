from typing import Callable, Dict

_DATA: Dict[str, Callable] = {}
_MODEL: Dict[str, Callable] = {}

def register_data(name: str):
    def deco(fn): 
        _DATA[name] = fn
        return fn
    return deco

def register_model(name: str):
    def deco(fn): 
        _MODEL[name] = fn
        return fn
    return deco

def get_data(name: str):
    return _DATA[name]
def get_model(name: str):
    return _MODEL[name]
