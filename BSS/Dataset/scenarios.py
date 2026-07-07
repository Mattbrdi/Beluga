from __future__ import annotations

from collections.abc import Callable

from ..Utils.signal_generation import AudioSceneSpec, WhistleSignal


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
        random_source_signal_types=(WhistleSignal.signal_type,),
    )
