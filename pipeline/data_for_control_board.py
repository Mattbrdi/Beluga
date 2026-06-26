from main_module import positions_from_audio
from src.utils.dashboard import set_dashboard
from src.utils.sub_classes import Environment, Parameters
import numpy as np
from pyproj import Transformer
import csv
from datetime import datetime, timedelta

from math import radians, sin, cos, sqrt, atan2


# audio_path =  [r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data\8295.240729065600.wav", 
#                r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data\8296.240729065600.wav",
#                ]

# audio_path =  [r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026\8295.260511123039.wav", 
#                r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026\8296.260511123039.wav",
#                ]    
# audio_path = [r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026\8295.260511132236.wav", 
#               r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026\8296.260511132236.wav"]

# audio_path = [r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026\8295.260511141242.wav", 
#               r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026\8296.260511141242.wav"]
# audio_path = [r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all_all\8295.260511134902.wav", 
#               r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all_all\8296.260511134902.wav"]

# audio_path = [r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026\8295.260511133808.wav", 
#               r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026\8296.260511133808.wav"]

TEST_DATA2026_ALL_AUDIO_PATHS = {
    7: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511123530.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511123530.wav",
    ],
    8: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511124248.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511124248.wav",
    ],
    9: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511125520.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511125520.wav",
    ],
    10: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511130305.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511130305.wav",
    ],
    11: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511131406.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511131406.wav",
    ],
    12: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511132244.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511132244.wav",
    ],
    13: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511133026.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511133026.wav",
    ],
    14: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511134030.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511134030.wav",
    ],
    15: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511134901.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511134901.wav",
    ],
    16: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511135634.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511135634.wav",
    ],
    17: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511140435.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511140435.wav",
    ],
    18: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511141244.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511141244.wav",
    ],
    19: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511141906.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511141906.wav",
    ],
    20: [
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8295.260511142534.wav",
        r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data2026_all\8296.260511142534.wav",
    ],
}

point_number = 9


audio_path = TEST_DATA2026_ALL_AUDIO_PATHS[point_number]
ground_truth_path = r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\ground_truth\trace_gps_calibration.csv"
#ces fichier n'ont pas exactrement la bonne date 
model_path = 'jsons/models/mobile_net_8_layers_qat.pt'
param_path = 'jsons/parameters/default_parameters.json'
# env_path = 'jsons/environments/env_cacouna.json'

env_path = 'jsons/environments/env_cacouna_may2026.json'

parameters = Parameters(param_path)
environment = Environment(env_path, parameters.location_parameters.use_h4)

(
    positions, errors, timestamps, durations, call_types,
    event_times, event_durations, event_call_types, event_status,
    detections_dfs
) = positions_from_audio(model_path, env_path, param_path, audio_path)

def lat_long(csv_path, datetime_str):
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)

        if reader.fieldnames is None:
            raise ValueError(f"CSV vide ou invalide : {csv_path}")

        normalized_fieldnames = [field.strip() for field in reader.fieldnames]
        reader.fieldnames = normalized_fieldnames

        required_columns = {"datetime_correct", "lat", "long"}
        missing_columns = required_columns.difference(normalized_fieldnames)
        if missing_columns:
            raise ValueError(
                f"Colonnes manquantes dans {csv_path}: {', '.join(sorted(missing_columns))}"
            )

        for row in reader:
            normalized_row = {key.strip(): value.strip() for key, value in row.items()}
            if normalized_row["datetime_correct"] == datetime_str:
                return (
                    float(normalized_row["lat"]),
                    float(normalized_row["long"]),
                )

    raise ValueError(
        f"Aucune ligne trouvée pour datetime_correct = '{datetime_str}' dans {csv_path}"
    )
def sum_i(date :str , i: int ):
    try:
        resultat = (datetime.strptime(date, "%Y/%m/%d %H:%M:%S") + timedelta(seconds=i)).strftime("%Y/%m/%d %H:%M:%S")
    except: 
        print("format date valide requis")
    return resultat

def date_from_audio_path(audio_path):
    first_audio_path = audio_path[0]
    filename = first_audio_path.split("\\")[-1]
    date_str = filename.split(".")[1]
    parsed_date = datetime.strptime(date_str, "%y%m%d%H%M%S")
    return parsed_date.strftime("%Y/%m/%d %H:%M:%S")

#WARNIGN, fichier audio doivent avoir le meme nom de début et de fin 
if __name__ == "__main__":
    app = set_dashboard(
        audio_path,
        positions,
        errors,
        timestamps,
        durations,
        call_types,
        #2024-07-27 [10h30-11h30]
        #[(47.939782, -69.526838), (47.935598, -69.539854), (47.934512, -69.537531),
        # (47.932280, -69.538654), (47.934000, -69.539888), (47.934626, -69.536155),
        # (47.933660, -69.538341)],

        #2024-07-27 [13h30-14h30]
        #[(47.936382, -69.529243), (47.936290, -69.529107), (47.936922, -69.530982),
        # (47.937467, -69.527804), (47.940074, -69.528866), (47.938619, -69.522967),
        # (47.939892, -69.530096), (47.937919, -69.528822), (47.939719, -69.528054),
        # (47.939303, -69.524555), (47.938312, -69.523180), (47.939280, -69.523034),
        # (47.938790, -69.523244), (47.941022, -69.520652)],

        #2024-07-28 ~11h15
        #[(47.943012, -69.518671)],

        #2024-07-29 [07h00-07h10]
        # [(47.940056, -69.530229), (47.939948, -69.530897), (47.939373, -69.531312),
        # (47.938277, -69.533283), (47.937416, -69.535017), (47.937167, -69.535564),
        # (47.939913, -69.534175)],

        #2024-07-29 [11h40-12h40]
        #[(47.939973, -69.528317), (47.941761, -69.522871), (47.941849, -69.522566),
        # (47.941926, -69.522651), (47.939867, -69.524520), (47.939924, -69.525104),
        # (47.940690, -69.530254), (47.938799, -69.535928)],

        #2024-08-04 [08h30-09h50]
        # [(47.936815, -69.524449), (47.933553, -69.540572), (47.932641, -69.539965),
        #  (47.932844, -69.540278), (47.932842, -69.540286), (47.945641, -69.523906),
        #  (47.945017, -69.524993), (47.944608, -69.526129), (47.941017, -69.533308),
        #  (47.935525, -69.535717), (47.935611, -69.535486), (47.936317, -69.530894),
        #  (47.937244, -69.528370)],
        #
        #2026 test_data 12h30
        # [(47.9440129827708,-69.5248331129551), (47.9440762661397,-69.524761447683),(47.9441651981324,-69.5246633794159)],
        
        # point 12 
        # [(47.9440129827708,-69.5248331129551)] + [lat_long(ground_truth_path, f"2026/05/11 13:22:{50+i}") for i in range(9)] + [ lat_long(ground_truth_path, f"2026/05/11 13:23:{i}") for i in range(10,60)] ,
        # point 17
        # [(47.9440129827708,-69.5248331129551)] + [lat_long(ground_truth_path, f"2026/05/11 14:04:{37+i}") for i in range(20)],  

        # [(47.9440129827708,-69.5248331129551)] + [lat_long(ground_truth_path, f"2026/05/11 14:12:{42+i}") for i in range(15)],  
       
        # [(47.9440129827708,-69.5248331129551)] + [lat_long(ground_truth_path, f"2026/05/11 13:49:{10+i}") for i in range(15)],  
        [lat_long(ground_truth_path, sum_i(date_from_audio_path(audio_path), i)) for i in range(30)] ,
        # dummy (modifier le code pour pouvoir utiliser sans ground truth?)
        #[(47.942531, -69.528000)],
        environment=environment,
        event_times=event_times,
        event_durations=event_durations,
        event_call_types=event_call_types,
        event_status=event_status,
        detections_dfs=detections_dfs,
        detection_threshold=0.5,
    )
    app.run(debug=False)
    
    
