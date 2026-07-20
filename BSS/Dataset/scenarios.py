from __future__ import annotations

from collections.abc import Callable

from ..Utils.signal_generation import AudioSceneSpec, WhistleSignal, CompositeSignalSpec, LargeShipNoise, GaussianNoise, SpikeSignal

"""
Un scénario sert à définir le Spec en fonction de l'endroit ou l'on est sur le dataset.
"""
ScenarioFactory = Callable[[str, int, int], AudioSceneSpec | None]
SCENARIO_FACTORIES: dict[str, ScenarioFactory] = {}


def register_scenario(name: str) -> Callable[[ScenarioFactory], ScenarioFactory]:
    """Enregistre une fabrique de scenarios sous un nom stable et serialisable."""
    if not name or not name.isidentifier():
        raise ValueError(f"Nom de scenario invalide: {name!r}.")

    def decorator(factory: ScenarioFactory) -> ScenarioFactory:
        if name in SCENARIO_FACTORIES:
            raise ValueError(f"Le scenario {name!r} est deja enregistre.")
        SCENARIO_FACTORIES[name] = factory
        return factory

    return decorator


def get_scenario_factory(name: str) -> ScenarioFactory:
    try:
        return SCENARIO_FACTORIES[name]
    except KeyError as exc:
        available = ", ".join(sorted(SCENARIO_FACTORIES)) or "aucun"
        raise ValueError(
            f"Scenario inconnu: {name!r}. Scenarios disponibles: {available}."
        ) from exc


@register_scenario("default")
def default_scenario(_split: str, _index: int, _seed: int) -> None:
    """Utilise la generation aleatoire sans contrainte d'AudioSceneGenerator."""
    return None


@register_scenario("whistles_only")
def whistles_only_scenario(
    _split: str,
    _index: int,
    _seed: int,
) -> AudioSceneSpec:
    """Limite les sources utiles aux sifflements, sans modifier les bruits."""
    return AudioSceneSpec(
        allowed_source_signal_types=(WhistleSignal.signal_type,),
    )

@register_scenario("Whistles_and_boat")
def whistle_and_boat(_split : str, _index: int, _seed: int) -> AudioSceneSpec: 

    return AudioSceneSpec(
        allowed_source_signal_types=(WhistleSignal.signal_type),
        allowed_continuous_noise_signal_types=GaussianNoise.signal_type,
        allowed_local_noise_signal_types= SpikeSignal.signal_type,
        source_specs=[CompositeSignalSpec(n_placements=1,allowed_signal_types=LargeShipNoise.signal_type)],
        snr_db= snr(_seed))
    
def snr(seed :int): 
    if seed%3 == 0: 
        return 1
    elif seed%3 == 5: 
        return 0 
    else:
        return -10