"""Bruit sous-marin synthétique d'un grand navire à moteur."""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ..common import _random_value
from .base import TypedSignal


class LargeShipNoise(TypedSignal):
    """Bruit sous-marin continu d'un grand cargo en navigation.

    Hypotheses physiques
    --------------------
    Le modele représente un cargo propulse par un moteur diesel deux-temps
    lent directement couple a l'arbre de l'helice. Pour ce type de navire, le
    moteur, l'arbre et l'helice partagent donc la meme frequence de rotation.
    Le bruit rayonne dans l'eau est approxime par la somme de trois familles :

    1. un bruit large bande colore produit principalement par la cavitation ;
    2. des raies aux frequences de passage des pales et a leurs harmoniques ;
    3. des raies de combustion du moteur transmises a l'eau par la coque.

    Ce modele produit une signature acoustique plausible et parametrable. Il
    ne constitue pas un modele vibroacoustique calibre pour un navire precis :
    il ne simule ni la propagation dans la coque, ni la directivite, ni les
    pertes de propagation dans l'ocean.

    Relations entre le moteur et l'helice
    --------------------------------------
    Si ``f_shaft`` est ``shaft_rotation_frequency``, alors la frequence de
    passage des pales est :

    ``f_blade = propeller_blade_count * f_shaft``.

    Le moteur est suppose deux-temps : chaque cylindre produit une combustion
    par tour. La cadence globale de combustion est donc :

    ``f_engine = engine_cylinder_count * f_shaft``.

    Ces deux frequences sont disponibles apres generation avec les proprietes
    ``blade_rate`` et ``engine_firing_frequency``.

    Formule du signal
    -----------------
    Chaque composante est d'abord normalisee en valeur efficace. Le signal
    avant normalisation finale est :

    ``x = w_cavitation * cavitation``
    ``    + tonal_mix * propeller_tones``
    ``    + engine_mix * engine_tones``

    avec :

    ``w_cavitation = 1 - tonal_mix - engine_mix``.

    Le pic du resultat est finalement ramene a 1. Le niveau acoustique relatif
    dans une scene est ensuite fixe par le gain du bruit continu.

    Cavitation large bande
    ----------------------
    Un spectre complexe aleatoire est pondere par une forme possedant un
    maximum autour de ``cavitation_peak_frequency``. Sa densite de puissance
    est proportionnelle a :

    ``r**low_frequency_slope``
    ``/ (1 + r**(low_frequency_slope + high_frequency_slope))``

    ou ``r = frequency / cavitation_peak_frequency``. La premiere pente
    controle la montee sous le pic et la seconde la decroissance au-dessus.
    Une transformee de Fourier inverse produit ensuite le signal temporel.

    L'amplitude de cette cavitation est modulee lentement suivant :

    ``1 + modulation_depth * sin(2*pi*modulation_frequency*t + phase)``.

    Cela represente les variations cycliques du chargement de l'helice dans un
    ecoulement non uniforme. La phase de modulation est aleatoire.

    Raies de l'helice et du moteur
    --------------------------------
    Les raies de l'helice sont placees a ``k * f_blade`` et leur amplitude
    suit ``k**(-tonal_decay)``. Les raies moteur sont placees a
    ``k * f_engine`` et suivent ``k**(-engine_harmonic_decay)``. Une phase
    aleatoire independante est affectee a chaque harmonique.

    Parametres
    ----------
    freq:
        Frequence d'echantillonnage en hertz. Toutes les raies generees doivent
        rester strictement sous la frequence de Nyquist ``freq / 2``.
    time_duration:
        Duree du bruit en secondes. La generation aleatoire utilise
        ``DURATION_RANGE`` ; une AudioScene impose la duree complete de scene.
    shaft_rotation_frequency:
        Vitesse commune du moteur et de l'arbre en tours par seconde. La plage
        aleatoire 0.8--2 Hz correspond a 48--120 tours par minute.
    propeller_blade_count:
        Nombre de pales. Il determine directement ``blade_rate``. La
        generation aleatoire choisit une valeur dans ``PROPELLER_BLADE_COUNTS``.
    tonal_harmonic_count:
        Nombre de raies conservees pour l'helice, fondamentale comprise.
    tonal_decay:
        Exposant de decroissance des raies de l'helice. Une petite valeur rend
        les harmoniques elevees plus presentes ; une grande valeur concentre
        davantage l'energie sur la fondamentale.
    engine_cylinder_count:
        Nombre de cylindres du diesel principal. Il determine directement la
        cadence globale de combustion ``engine_firing_frequency``.
    engine_harmonic_count:
        Nombre de raies de combustion conservees, fondamentale comprise.
    engine_harmonic_decay:
        Exposant de decroissance des harmoniques moteur. Une petite valeur
        produit une signature moteur spectralement plus riche.
    cavitation_peak_frequency:
        Frequence en hertz autour de laquelle la composante large bande atteint
        son maximum spectral. La plage aleatoire est 35--90 Hz.
    low_frequency_slope:
        Pente positive de la montee du spectre sous le pic de cavitation.
    high_frequency_slope:
        Pente positive de la decroissance au-dessus du pic de cavitation.
    modulation_frequency:
        Frequence en hertz de la pulsation lente appliquee a la cavitation.
        Elle decrit une periodicite de chargement et non la vitesse du navire.
    modulation_depth:
        Profondeur de cette modulation. Zero la desactive ; une valeur proche
        de 1 produit une pulsation tres marquee. Elle doit rester dans [0, 1[.
    tonal_mix:
        Poids relatif des raies de passage des pales dans le melange.
    engine_mix:
        Poids relatif des raies moteur. La somme de ``tonal_mix`` et
        ``engine_mix`` ne peut pas depasser 1 ; le reste revient a la cavitation.
    seed:
        Graine controlant le bruit de cavitation, la phase de modulation et les
        phases des raies. Les memes parametres et la meme graine reproduisent
        exactement le meme signal.

    Attributs de classe et regles de tirage
    ---------------------------------------
    Les ``ClassVar`` en majuscules definissent l'identite statistique de ce type
    de signal. Elles ne sont pas stockees dans chaque instance : elles servent
    a tirer les parametres d'une nouvelle realisation par ``generate_random``.

    DURATION_RANGE:
        Bornes minimale et maximale, en secondes, du tirage uniforme de
        ``time_duration``. Cette regle est ignoree lorsqu'une AudioScene force
        le bruit continu a couvrir toute la scene.
    SHAFT_ROTATION_FREQUENCY_RANGE:
        Bornes du tirage uniforme de ``shaft_rotation_frequency``, en tours par
        seconde. ``(0.8, 2.0)`` correspond a 48--120 tours par minute.
    PROPELLER_BLADE_COUNTS:
        Ensemble discret des nombres de pales possibles. Une valeur est choisie
        uniformement pour produire ``propeller_blade_count``.
    TONAL_HARMONIC_COUNT_RANGE:
        Bornes inclusives du nombre entier d'harmoniques de passage des pales.
        Le resultat devient ``tonal_harmonic_count``.
    TONAL_DECAY_RANGE:
        Bornes du tirage uniforme de l'exposant ``tonal_decay`` qui controle la
        decroissance ``k**(-tonal_decay)`` des raies de l'helice.
    ENGINE_CYLINDER_COUNTS:
        Ensemble discret des nombres de cylindres autorises pour le diesel
        principal. Une valeur produit ``engine_cylinder_count``.
    ENGINE_HARMONIC_COUNT_RANGE:
        Bornes inclusives du nombre entier de raies moteur conservees. Le
        resultat devient ``engine_harmonic_count``.
    ENGINE_HARMONIC_DECAY_RANGE:
        Bornes du tirage uniforme de ``engine_harmonic_decay``, l'exposant de
        decroissance des harmoniques de combustion.
    ENGINE_MIX_RANGE:
        Bornes du tirage uniforme de ``engine_mix``, poids relatif de la
        composante moteur dans le melange avant normalisation finale.
    CAVITATION_PEAK_FREQUENCY_RANGE:
        Bornes en hertz du tirage uniforme de
        ``cavitation_peak_frequency``, position du maximum spectral du bruit de
        cavitation.
    LOW_FREQUENCY_SLOPE_RANGE:
        Bornes du tirage uniforme de ``low_frequency_slope``, pente de montee
        du spectre sous le pic de cavitation.
    HIGH_FREQUENCY_SLOPE_RANGE:
        Bornes du tirage uniforme de ``high_frequency_slope``, pente de
        decroissance du spectre au-dessus du pic.
    MODULATION_FREQUENCY_RANGE:
        Bornes en hertz du tirage uniforme de ``modulation_frequency``, vitesse
        de pulsation de l'enveloppe de cavitation.
    MODULATION_DEPTH_RANGE:
        Bornes du tirage uniforme de ``modulation_depth``. Cette grandeur sans
        unite controle l'intensite de la pulsation et doit rester inferieure a 1.
    TONAL_MIX_RANGE:
        Bornes du tirage uniforme de ``tonal_mix``, poids relatif des raies de
        l'helice. Sa somme avec ``engine_mix`` doit rester inferieure ou egale a
        1 afin de conserver un poids de cavitation positif ou nul.
    RANDOM_SEED_MAX:
        Borne superieure exclusive utilisee pour tirer une graine entiere quand
        ``seed`` n'est pas fixe. Elle correspond a la plage d'un entier uint32.

    Un parametre passe explicitement a ``generate_random`` reste toujours
    prioritaire sur la ``ClassVar`` correspondante. Une sous-classe peut changer
    ces attributs pour definir un autre type statistique de bruit de navire sans
    reimplementer la logique de synthese.
    """

    signal_type = "large_ship_noise"
    allowed_windows = (None,)
    default_window = None

    DURATION_RANGE: ClassVar[tuple[float, float]] = (5, 10.0)
    SHAFT_ROTATION_FREQUENCY_RANGE: ClassVar[tuple[float, float]] = (0.8, 2.0)
    PROPELLER_BLADE_COUNTS: ClassVar[tuple[int, ...]] = (4, 5, 6)
    TONAL_HARMONIC_COUNT_RANGE: ClassVar[tuple[int, int]] = (3, 7)
    TONAL_DECAY_RANGE: ClassVar[tuple[float, float]] = (0.8, 1.5)
    ENGINE_CYLINDER_COUNTS: ClassVar[tuple[int, ...]] = (6, 7, 8, 9, 10, 11, 12)
    ENGINE_HARMONIC_COUNT_RANGE: ClassVar[tuple[int, int]] = (5, 12)
    ENGINE_HARMONIC_DECAY_RANGE: ClassVar[tuple[float, float]] = (0.6, 1.3)
    ENGINE_MIX_RANGE: ClassVar[tuple[float, float]] = (0.3,0.5)
    CAVITATION_PEAK_FREQUENCY_RANGE: ClassVar[tuple[float, float]] = (35.0, 90.0)
    LOW_FREQUENCY_SLOPE_RANGE: ClassVar[tuple[float, float]] = (1.0, 2.0)
    HIGH_FREQUENCY_SLOPE_RANGE: ClassVar[tuple[float, float]] = (0.8, 1.4)
    MODULATION_FREQUENCY_RANGE: ClassVar[tuple[float, float]] = (1.0, 4.0)
    MODULATION_DEPTH_RANGE: ClassVar[tuple[float, float]] = (0.1, 0.4)
    TONAL_MIX_RANGE: ClassVar[tuple[float, float]] = (0.15, 0.4)
    RANDOM_SEED_MAX: ClassVar[int] = int(np.iinfo(np.uint32).max)

    def __init__(
        self,
        freq: float,
        data: np.ndarray,
        time_duration: float,
        shaft_rotation_frequency: float,
        propeller_blade_count: int,
        tonal_harmonic_count: int,
        tonal_decay: float,
        engine_cylinder_count: int,
        engine_harmonic_count: int,
        engine_harmonic_decay: float,
        cavitation_peak_frequency: float,
        low_frequency_slope: float,
        high_frequency_slope: float,
        modulation_frequency: float,
        modulation_depth: float,
        tonal_mix: float,
        engine_mix: float,
        seed: int,
    ):
        super().__init__(data=data, freq=freq)
        self.time_duration = time_duration
        self.shaft_rotation_frequency = shaft_rotation_frequency
        self.propeller_blade_count = propeller_blade_count
        self.tonal_harmonic_count = tonal_harmonic_count
        self.tonal_decay = tonal_decay
        self.engine_cylinder_count = engine_cylinder_count
        self.engine_harmonic_count = engine_harmonic_count
        self.engine_harmonic_decay = engine_harmonic_decay
        self.cavitation_peak_frequency = cavitation_peak_frequency
        self.low_frequency_slope = low_frequency_slope
        self.high_frequency_slope = high_frequency_slope
        self.modulation_frequency = modulation_frequency
        self.modulation_depth = modulation_depth
        self.tonal_mix = tonal_mix
        self.engine_mix = engine_mix
        self.seed = seed

    @classmethod
    def generate(
        cls,
        freq: float,
        time_duration: float,
        shaft_rotation_frequency: float,
        propeller_blade_count: int,
        tonal_harmonic_count: int,
        tonal_decay: float,
        engine_cylinder_count: int,
        engine_harmonic_count: int,
        engine_harmonic_decay: float,
        cavitation_peak_frequency: float,
        low_frequency_slope: float,
        high_frequency_slope: float,
        modulation_frequency: float,
        modulation_depth: float,
        tonal_mix: float,
        engine_mix: float,
        seed: int,
    ) -> "LargeShipNoise":
        cls._validate_parameters(
            freq=freq,
            time_duration=time_duration,
            shaft_rotation_frequency=shaft_rotation_frequency,
            propeller_blade_count=propeller_blade_count,
            tonal_harmonic_count=tonal_harmonic_count,
            tonal_decay=tonal_decay,
            engine_cylinder_count=engine_cylinder_count,
            engine_harmonic_count=engine_harmonic_count,
            engine_harmonic_decay=engine_harmonic_decay,
            cavitation_peak_frequency=cavitation_peak_frequency,
            low_frequency_slope=low_frequency_slope,
            high_frequency_slope=high_frequency_slope,
            modulation_frequency=modulation_frequency,
            modulation_depth=modulation_depth,
            tonal_mix=tonal_mix,
            engine_mix=engine_mix,
        )
        rng = np.random.default_rng(seed)
        n_samples = int(round(freq * time_duration))
        time = np.arange(n_samples) / freq

        broadband = cls._colored_cavitation_noise(
            rng=rng,
            n_samples=n_samples,
            freq=freq,
            peak_frequency=cavitation_peak_frequency,
            low_slope=low_frequency_slope,
            high_slope=high_frequency_slope,
        )
        modulation_phase = rng.uniform(0.0, 2.0 * np.pi)
        modulation = 1.0 + modulation_depth * np.sin(
            2.0 * np.pi * modulation_frequency * time + modulation_phase
        )
        broadband *= modulation
        broadband = cls._normalize_rms(broadband)

        blade_rate = shaft_rotation_frequency * propeller_blade_count
        propeller_tones = cls._harmonic_tones(
            rng, time, blade_rate, tonal_harmonic_count, tonal_decay
        )
        engine_firing_frequency = shaft_rotation_frequency * engine_cylinder_count
        engine_tones = cls._harmonic_tones(
            rng,
            time,
            engine_firing_frequency,
            engine_harmonic_count,
            engine_harmonic_decay,
        )

        broadband_mix = 1.0 - tonal_mix - engine_mix
        data = (
            broadband_mix * broadband
            + tonal_mix * propeller_tones
            + engine_mix * engine_tones
        )
        peak = float(np.max(np.abs(data)))
        if peak > 0:
            data /= peak

        return cls(
            freq=freq,
            data=data,
            time_duration=time_duration,
            shaft_rotation_frequency=shaft_rotation_frequency,
            propeller_blade_count=int(propeller_blade_count),
            tonal_harmonic_count=int(tonal_harmonic_count),
            tonal_decay=tonal_decay,
            engine_cylinder_count=int(engine_cylinder_count),
            engine_harmonic_count=int(engine_harmonic_count),
            engine_harmonic_decay=engine_harmonic_decay,
            cavitation_peak_frequency=cavitation_peak_frequency,
            low_frequency_slope=low_frequency_slope,
            high_frequency_slope=high_frequency_slope,
            modulation_frequency=modulation_frequency,
            modulation_depth=modulation_depth,
            tonal_mix=tonal_mix,
            engine_mix=engine_mix,
            seed=int(seed),
        )

    @classmethod
    def generate_random_params(
        cls,
        rng: np.random.Generator,
        freq: float,
        fixed_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        fixed_params = fixed_params or {}
        count_min, count_max = cls.TONAL_HARMONIC_COUNT_RANGE
        tonal_harmonic_count = (
            int(fixed_params["tonal_harmonic_count"])
            if "tonal_harmonic_count" in fixed_params
            else int(rng.integers(count_min, count_max + 1))
        )
        propeller_blade_count = (
            int(fixed_params["propeller_blade_count"])
            if "propeller_blade_count" in fixed_params
            else int(rng.choice(cls.PROPELLER_BLADE_COUNTS))
        )
        engine_cylinder_count = (
            int(fixed_params["engine_cylinder_count"])
            if "engine_cylinder_count" in fixed_params
            else int(rng.choice(cls.ENGINE_CYLINDER_COUNTS))
        )
        engine_count_min, engine_count_max = cls.ENGINE_HARMONIC_COUNT_RANGE
        engine_harmonic_count = (
            int(fixed_params["engine_harmonic_count"])
            if "engine_harmonic_count" in fixed_params
            else int(rng.integers(engine_count_min, engine_count_max + 1))
        )
        seed = (
            int(fixed_params["seed"])
            if "seed" in fixed_params
            else int(rng.integers(0, cls.RANDOM_SEED_MAX, dtype=np.uint32))
        )
        return {
            "freq": freq,
            "time_duration": _random_value(
                rng, fixed_params.get("time_duration"), *cls.DURATION_RANGE
            ),
            "shaft_rotation_frequency": _random_value(
                rng,
                fixed_params.get("shaft_rotation_frequency"),
                *cls.SHAFT_ROTATION_FREQUENCY_RANGE,
            ),
            "propeller_blade_count": propeller_blade_count,
            "tonal_harmonic_count": tonal_harmonic_count,
            "tonal_decay": _random_value(
                rng, fixed_params.get("tonal_decay"), *cls.TONAL_DECAY_RANGE
            ),
            "engine_cylinder_count": engine_cylinder_count,
            "engine_harmonic_count": engine_harmonic_count,
            "engine_harmonic_decay": _random_value(
                rng,
                fixed_params.get("engine_harmonic_decay"),
                *cls.ENGINE_HARMONIC_DECAY_RANGE,
            ),
            "cavitation_peak_frequency": _random_value(
                rng,
                fixed_params.get("cavitation_peak_frequency"),
                *cls.CAVITATION_PEAK_FREQUENCY_RANGE,
            ),
            "low_frequency_slope": _random_value(
                rng,
                fixed_params.get("low_frequency_slope"),
                *cls.LOW_FREQUENCY_SLOPE_RANGE,
            ),
            "high_frequency_slope": _random_value(
                rng,
                fixed_params.get("high_frequency_slope"),
                *cls.HIGH_FREQUENCY_SLOPE_RANGE,
            ),
            "modulation_frequency": _random_value(
                rng,
                fixed_params.get("modulation_frequency"),
                *cls.MODULATION_FREQUENCY_RANGE,
            ),
            "modulation_depth": _random_value(
                rng,
                fixed_params.get("modulation_depth"),
                *cls.MODULATION_DEPTH_RANGE,
            ),
            "tonal_mix": _random_value(
                rng, fixed_params.get("tonal_mix"), *cls.TONAL_MIX_RANGE
            ),
            "engine_mix": _random_value(
                rng, fixed_params.get("engine_mix"), *cls.ENGINE_MIX_RANGE
            ),
            "seed": seed,
        }

    @classmethod
    def _harmonic_tones(
        cls,
        rng: np.random.Generator,
        time: np.ndarray,
        fundamental_frequency: float,
        harmonic_count: int,
        harmonic_decay: float,
    ) -> np.ndarray:
        tones = np.zeros_like(time)
        for harmonic in range(1, harmonic_count + 1):
            amplitude = harmonic ** (-harmonic_decay)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            tones += amplitude * np.sin(
                2.0 * np.pi * harmonic * fundamental_frequency * time + phase
            )
        return cls._normalize_rms(tones)

    @staticmethod
    def _colored_cavitation_noise(
        rng: np.random.Generator,
        n_samples: int,
        freq: float,
        peak_frequency: float,
        low_slope: float,
        high_slope: float,
    ) -> np.ndarray:
        frequencies = np.fft.rfftfreq(n_samples, d=1.0 / freq)
        ratio = frequencies / peak_frequency
        power_shape = np.zeros_like(frequencies)
        positive = frequencies > 0
        positive_ratio = ratio[positive]
        power_shape[positive] = (
            positive_ratio**low_slope
            / (1.0 + positive_ratio ** (low_slope + high_slope))
        )
        random_spectrum = (
            rng.normal(size=len(frequencies))
            + 1j * rng.normal(size=len(frequencies))
        )
        random_spectrum *= np.sqrt(power_shape)
        random_spectrum[0] = 0.0
        return np.fft.irfft(random_spectrum, n=n_samples)

    @staticmethod
    def _normalize_rms(data: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(data))))
        return data if rms == 0 else data / rms

    @staticmethod
    def _validate_parameters(
        freq: float,
        time_duration: float,
        shaft_rotation_frequency: float,
        propeller_blade_count: int,
        tonal_harmonic_count: int,
        tonal_decay: float,
        engine_cylinder_count: int,
        engine_harmonic_count: int,
        engine_harmonic_decay: float,
        cavitation_peak_frequency: float,
        low_frequency_slope: float,
        high_frequency_slope: float,
        modulation_frequency: float,
        modulation_depth: float,
        tonal_mix: float,
        engine_mix: float,
    ) -> None:
        values = (
            freq,
            time_duration,
            shaft_rotation_frequency,
            tonal_decay,
            engine_harmonic_decay,
            cavitation_peak_frequency,
            low_frequency_slope,
            high_frequency_slope,
            modulation_frequency,
            modulation_depth,
            tonal_mix,
            engine_mix,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("Tous les parametres doivent etre finis.")
        if freq <= 0 or time_duration <= 0:
            raise ValueError("freq et time_duration doivent etre strictement positifs.")
        if shaft_rotation_frequency <= 0 or cavitation_peak_frequency <= 0:
            raise ValueError("Les frequences caracteristiques doivent etre positives.")
        if (
            isinstance(tonal_harmonic_count, bool)
            or not isinstance(tonal_harmonic_count, (int, np.integer))
            or tonal_harmonic_count < 1
            or tonal_decay <= 0
        ):
            raise ValueError("La structure tonale doit contenir des harmoniques decroissantes.")
        for name, value in (
            ("propeller_blade_count", propeller_blade_count),
            ("engine_cylinder_count", engine_cylinder_count),
            ("engine_harmonic_count", engine_harmonic_count),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or value < 1
            ):
                raise ValueError(f"{name} doit etre un entier strictement positif.")
        if engine_harmonic_decay <= 0:
            raise ValueError("Les parametres du moteur doivent etre strictement positifs.")
        if low_frequency_slope <= 0 or high_frequency_slope <= 0:
            raise ValueError("Les pentes spectrales doivent etre strictement positives.")
        if modulation_frequency <= 0 or not 0 <= modulation_depth < 1:
            raise ValueError("Parametres de modulation invalides.")
        if not 0 <= tonal_mix <= 1 or not 0 <= engine_mix <= 1:
            raise ValueError("Les proportions du melange doivent etre comprises entre 0 et 1.")
        if tonal_mix + engine_mix > 1:
            raise ValueError("tonal_mix + engine_mix ne doit pas depasser 1.")
        blade_rate = shaft_rotation_frequency * propeller_blade_count
        if max(blade_rate, cavitation_peak_frequency) >= freq / 2.0:
            raise ValueError("Les frequences caracteristiques doivent rester sous Nyquist.")
        if tonal_harmonic_count * blade_rate >= freq / 2.0:
            raise ValueError("La plus haute raie de propulsion depasse Nyquist.")
        engine_firing_frequency = shaft_rotation_frequency * engine_cylinder_count
        if engine_harmonic_count * engine_firing_frequency >= freq / 2.0:
            raise ValueError("La plus haute raie moteur depasse Nyquist.")

    @property
    def blade_rate(self) -> float:
        return self.shaft_rotation_frequency * self.propeller_blade_count

    @property
    def engine_firing_frequency(self) -> float:
        return self.shaft_rotation_frequency * self.engine_cylinder_count
