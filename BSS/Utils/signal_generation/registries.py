"""Registres des types de signaux disponibles."""
from .TypedSignal import TypedSignal

_SOURCE_SIGNAL_TYPE: dict[str, type[TypedSignal]] = {}
_LOCAL_NOISE_SIGNAL_TYPE: dict[str, type[TypedSignal]] = {}
_CONTINUOUS_NOISE_SIGNAL_TYPE: dict[str, type[TypedSignal]] = {}


def register_SourceSignal(signal_cls: type[TypedSignal]) -> None:
    if not issubclass(signal_cls, TypedSignal):
        raise TypeError("La classe doit heriter de TypedSignal.")
    signal_cls.validate_generation_contract()
    _SOURCE_SIGNAL_TYPE[signal_cls.signal_type] = signal_cls


def register_LocalNoiseSignal(signal_cls: type[TypedSignal]) -> None:
    if not issubclass(signal_cls, TypedSignal):
        raise TypeError("La classe doit heriter de TypedSignal.")
    signal_cls.validate_generation_contract()
    _LOCAL_NOISE_SIGNAL_TYPE[signal_cls.signal_type] = signal_cls


def register_ContinuousNoiseSignal(signal_cls: type[TypedSignal]) -> None:
    if not issubclass(signal_cls, TypedSignal):
        raise TypeError("La classe doit heriter de TypedSignal.")
    signal_cls.validate_generation_contract()
    _CONTINUOUS_NOISE_SIGNAL_TYPE[signal_cls.signal_type] = signal_cls
