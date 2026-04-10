"""Defines all the classes that will be used in 'final' functions"""

from typing import Optional
from dataclasses import dataclass
from scipy.spatial.distance import pdist
import numpy as np
import json
from src.utils.rotation_bricks import enu2lla, lla2enu

#########################################
########## Environment classes ##########
#########################################

class Tetrahedra:
    """Generate a tetrahedra from hydrophones' position. Environment (ie pregenerated)."""
    def __init__(self, tetra_name, tetra_data, use_h4, sound_speed, enu_ref):
        """hydro_pos[i] is the position in ENU ref of the hydrophone with id H[i]"""
        self.origin_lla = tetra_data["coordinates_lla"]
        self.origin_enu = lla2enu(enu_ref, self.origin_lla)
        self.relative_hydro_coords = np.array(tetra_data["coordinates_hydrophones"]) #WARNING : In the referential of the origin of the tetrahedra, not ENU nor lla
        self.rotation_matrix = np.array(tetra_data["rotation_matrix"])
        
        self.rotated_hydro_pos = (self.rotation_matrix @ (self.relative_hydro_coords.T)).T
        self.rotated_hydro_pos_enu = self.origin_enu + self.rotated_hydro_pos
        
        self.id = tetra_name
        self.is_active = True # False if no beluga found 
        self.use_h4 = use_h4 
        self.v_matrix = self.compute_vmatrix()
        self.max_delay_seconds = 1.1 * self.compute_max_distance() / sound_speed
        
    def compute_max_distance(self):
        """Compute the maximum distance between all hydrophone pairs.
        Returns:
            float: La distance maximale entre deux hydrophones."""
        return np.max(pdist(self.relative_hydro_coords))
    
    def compute_vmatrix(self):
        hydros = self.rotated_hydro_pos
        if self.use_h4:
            v_mat = np.array([hydros[i] - hydros[j]
                        for i in range(len(hydros))
                        for j in range(i + 1, len(hydros))])
        else:
            v_mat = np.array([hydros[i,0:2] - hydros[j,0:2]
                for i in range(len(hydros)-1)
                for j in range(i + 1, len(hydros)-1)])
        return v_mat
    
class Environment:
    """Parse an environment with N tetrahedras from a json environment."""
    def __init__(self, env_json, use_h4):
        self.tetrahedras = {}
        with open(env_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.sensitivity = data["system_parameters"]["system_sensitivity"] 
        self.sound_speed = data["system_parameters"]["sound_speed_in_water"]
        self.enu_ref = data["system_parameters"]["enu_ref"]
        for tetra_dict in data["tetrahedras"]:
            tetra_name, tetra_data = next(iter(tetra_dict.items()))
            self.tetrahedras[f'{tetra_name}'] = Tetrahedra(tetra_name, tetra_data, use_h4, self.sound_speed, self.enu_ref)
    
####################################################    
########## Parameters processing classes ###########
####################################################

class VMDOptions:
    def __init__(self, max_iter=1000, nb_modes=5, tolerance=1E-7, alpha=3000, tau=0.):
        self.max_iter = max_iter
        self.nb_modes = nb_modes
        self.tolerance = tolerance  # TODO try bigger tolerance
        self.alpha = alpha
        self.tau = tau # Tau addresses the use of the lagrangian update : having tau != 0 means we want strict equality

        
    def set_k_alpha(self,K,alpha):
        self.nb_modes = int(K)
        self.alpha = int(alpha)

@dataclass
class WOAOptions:
    pop_size : int = 15
    max_iter : int = 50
    min_k_bound : int = 3
    max_k_bound : int = 9
    min_alpha_bound :int = 1000
    max_alpha_bound : int = 9000

@dataclass
class PreFilterParameters:
    order : int # Filter order
    filtering_method : Optional[str] = None

@dataclass
class VMDDenoiseParameters:
    use_vmd : bool
    use_woa : bool
    print_level : int
    modes_choice_method : str
    vmd_options : VMDOptions
    woa_options : WOAOptions
    imf_threshold : float
    output_vmd_log : bool
    remove_dc : bool
    compute_noisy_imfs : bool
    cc_threshold : float
    
@dataclass
class LocationParameters:
    use_h4 : bool
    fusion_type : str
    projection_plan : Optional[float] # Projection plan in ENU_ref for 2D processing

class Parameters:
    def __init__(self, path_to_json):
        with open(path_to_json, 'r') as file:
            data = json.load(file)
        self.debug_mode = data["debug_mode"]
        self.print_level = data["print_level"]
        self.max_position_frames = data["max_position_frames"]
        self.location_parameters = LocationParameters(**data["location_parameters"])
        self.use_gcc = data["use_gcc"]
        vmd_options_dict = data["vmd_denoise_parameters"]["vmd_options"]
        vmd_options = VMDOptions(
            vmd_options_dict["max_iter"],
            vmd_options_dict["nb_modes"],
            vmd_options_dict["tolerance"],
            vmd_options_dict["alpha"],
            vmd_options_dict["tau"]
        )
        woa_dict = data["vmd_denoise_parameters"]["woa_options"]
        woa_options = WOAOptions(
            woa_dict["pop_size"],
            woa_dict["max_iter"],
            woa_dict["min_k_bound"],
            woa_dict["max_k_bound"],
            woa_dict["min_alpha_bound"],
            woa_dict["max_alpha_bound"]
        )
        self.vmd_denoise_parameters = VMDDenoiseParameters(
            data["vmd_denoise_parameters"]["use_vmd"],
            data["vmd_denoise_parameters"]["use_woa"],
            data["print_level"],
            data["vmd_denoise_parameters"]["modes_choice_method"],
            vmd_options,
            woa_options,
            data["vmd_denoise_parameters"]["imf_threshold"],
            data["vmd_denoise_parameters"]["output_vmd_log"],
            data["vmd_denoise_parameters"]["remove_dc"],
            data["vmd_denoise_parameters"]["compute_noisy_imfs"],
            data["vmd_denoise_parameters"]["cc_threshold"]
        )
        self.pre_filter_parameters = PreFilterParameters(data["pre_filter_parameters"]["order"], data["pre_filter_parameters"]["filtering_method"])

########## Audio processing classes ##########
@dataclass
class AudioMetadata:
    tetra_id : str
    beluga_call_type : str # In hfpc, cc, ec, whistle
    call_duration : float # [s]
    start_time : float # [s]
    snr_power : Optional[list[float]] # Pressure not dB
    sample_rate : int # [ech/s]
    frequency_range : Optional[tuple[float,float]] # [Hz]
    central_frequency : Optional[float] #[Hz]
    
class AudioArray:
    """A class that parses wav data into a usable array. It is a 4 canals audio corresponding to one tetra.
    The audio is approx 1.5s.
    The output of Emmanuel is an AudioArray.
    The audio array also contains info about the SNR and the call type (one call type, four SNR)"""
    def __init__(self, audio_metadata : AudioMetadata, tetra_data : Tetrahedra, use_h4 : bool, data_array : np.ndarray):
        self.data_array = data_array
        self.metadata = audio_metadata
        self.use_h4 = use_h4
        self.pairs_dict = self.generate_pairs_dict(tetra_data, self.use_h4) # A dict with HydrophonePair id (8295_H1H2 for instance) matching a reference (array[0], array[h1])
    
    def generate_pairs_dict(self, tetra_data:Tetrahedra, use_h4 : bool):
        pairs_dict = {}
        max_delay_idx = int(tetra_data.max_delay_seconds*self.metadata.sample_rate)
        if self.metadata.snr_power is not None:
            hydrophones = [Hydrophone(f'{self.metadata.tetra_id}_H{i+1}', self.data_array[i], tetra_data.rotated_hydro_pos[i], self.metadata.snr_power[i]) for i in range(self.data_array.shape[0])]
        else:
            hydrophones = [Hydrophone(f'{self.metadata.tetra_id}_H{i+1}', self.data_array[i], tetra_data.rotated_hydro_pos[i], None) for i in range(self.data_array.shape[0])]
        h4_remover = int(1-use_h4) # 0 if we use h4, 1 else
        for i in range(self.data_array.shape[0]-h4_remover):
            for j in range(i+1,self.data_array.shape[0]-h4_remover):
                pairs_dict[f'{self.metadata.tetra_id}_H{i+1}H{j+1}'] = HydrophonePair(hydrophones[i],hydrophones[j], max_delay_idx)
        return pairs_dict
    
    def update_snr(self, snr_power):
        h4_remover = int(1-self.use_h4) # 0 if we use h4, 1 else
        for i in range(self.data_array.shape[0]-h4_remover):
            for j in range(i+1,self.data_array.shape[0]-h4_remover):
                self.pairs_dict[f'{self.metadata.tetra_id}_H{i+1}H{j+1}'].hydrophone_ref.snr_power = snr_power[i]
                self.pairs_dict[f'{self.metadata.tetra_id}_H{i+1}H{j+1}'].hydrophone_delta.snr_power = snr_power[j]

        

class Hydrophone:
    """Sub-class of AudioArray with one canal. It only stores REFERENCES to the mother class with the full array."""
    def __init__(self, id_hydro, audio_r, enu_pos, snr_power):
        self.id = id_hydro
        self.audio_r = audio_r
        self.enu_pos = enu_pos
        self.snr_power = snr_power

class HydrophonePair:
    """Sub-class of AudioArray with only two canals. It only stores REFERENCES to the mother class with the full array."""
    def __init__(self, hydrophone_ref : Hydrophone, hydrophone_delta : Hydrophone, max_delay_idx : int):
        self.hydrophone_ref = hydrophone_ref
        self.hydrophone_delta = hydrophone_delta
        self.max_delay_idx = max_delay_idx


    
