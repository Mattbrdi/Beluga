"""Chemins centralises vers les enregistrements de test de mai 2026."""

from pathlib import Path


AUDIO_DATA_DIR = Path(__file__).resolve().parent / "test_data2026_all" / "data"
GROUND_TRUTH_PATH = (
    r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\ground_truth\trace_gps_calibration.csv"
)
MODEL_PATH = "jsons/models/mobile_net_8_layers_qat.pt"
PARAM_PATH = "jsons/parameters/default_parameters.json"
ENV_PATH = "jsons/environments/env_cacouna_may2026.json"

POINT_TIMESTAMPS: dict[int, str] = {
    7: "260511123530",
    8: "260511124248",
    9: "260511125520",
    10: "260511130305",
    11: "260511131406",
    12: "260511132244",
    13: "260511133026",
    14: "260511134030",
    15: "260511134901",
    16: "260511135634",
    17: "260511140435",
    18: "260511141244",
    19: "260511141906",
    20: "260511142534",
}

POINT_NUMBERS: list[int] = sorted(POINT_TIMESTAMPS)

TEST_DATA2026_ALL_AUDIO_PATHS: dict[int, list[str]] = {
    point_number: [
        str(AUDIO_DATA_DIR / f"8295.{timestamp}.wav"),
        str(AUDIO_DATA_DIR / f"8296.{timestamp}.wav"),
    ]
    for point_number, timestamp in POINT_TIMESTAMPS.items()
}
