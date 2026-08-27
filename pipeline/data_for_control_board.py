from main_module import positions_from_audio
from src.utils.dashboard import set_dashboard
from src.utils.sub_classes import Environment, Parameters
import numpy as np
from pyproj import Transformer

from math import radians, sin, cos, sqrt, atan2


# audio_path =  [r"C:\Users\Admin\Desktop\belugaWatch\beluga-watch\beluga-watch-main\test_data\full audios\8296\8296.240804084225.wav", 
#                r"C:\Users\Admin\Desktop\belugaWatch\beluga-watch\beluga-watch-main\test_data\full audios\8295\8295.240804084225.wav",
#                ]


audio_path = [
    r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data\8295.240729065600.wav",
    r"C:\Users\BORDERIES\Desktop\Cours\Stage canada\Beluga\pipeline\test_data\8296.240729065600.wav",
]

model_path = 'jsons/models/mobile_net_8_layers_qat.pt'
param_path = 'jsons/parameters/default_parameters.json'
env_path = 'jsons/environments/env_cacouna.json'

parameters = Parameters(param_path)
environment = Environment(env_path, parameters.location_parameters.use_h4)

(
    positions, errors, timestamps, durations, call_types,
    event_times, event_durations, event_call_types, event_status,
    detections_dfs
) = positions_from_audio(model_path, env_path, param_path, audio_path)

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
        [(47.940056, -69.530229), (47.939948, -69.530897), (47.939373, -69.531312),
        (47.938277, -69.533283), (47.937416, -69.535017), (47.937167, -69.535564),
        (47.939913, -69.534175)],

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
