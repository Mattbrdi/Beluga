"""Module for beluga vocalizes matching over the tetrahedra."""

##########################
####### Imports ##########
##########################

########## Sound modules ##########
from datetime import datetime
import soundfile as sf

########## Computation modules ##########
from scipy.signal import correlate
import numpy as np

##################################################
########## Auxiliary matching functions ##########
##################################################

def path_to_datetime(path):
    """Transforms the path's name into a datetime format."""
    x = path.split('.')[-2]
    year, month, day, hour, minute, second = int(x[0:2]) + 2000, int(x[2:4]), int(x[4:6]), int(x[6:8]), int(x[8:10]), int(x[10:12])
    return datetime(year, month, day, hour, minute, second)

def datetimes_to_timestamps(datetimes, seconds_since_file_start_T1, print_level = 0):
    diffs = [abs(new_date-datetimes[0]) for new_date in datetimes] # Contains diff T1 = 0 !
    seconds_since_file_starts= [seconds_since_file_start_T1 + diff for diff in diffs]
    if print_level > 1:
        print(f"seconds_since_file_starts {seconds_since_file_starts} ")
    total_seconds = [seconds_since_start.total_seconds() for seconds_since_start in seconds_since_file_starts]
    return total_seconds

def slice_audio(file_path, time0_idx, duration_idx, sr=384000, print_level = 0):
    
    sliced_audio, sample_rate = sf.read(file_path, start=time0_idx, stop=time0_idx + duration_idx)
    
    #sf.write(output_path, sliced_audio, sr)
    return(sliced_audio, sample_rate)

def find_pattern_in_T2(sliced_T1, sr1, sliced_T2, sr2, start_time, duration):
    start_sample = int(start_time * sr1)
    end_sample = int((start_time + duration) * sr1)
    pattern = sliced_T1[start_sample:end_sample]

    correlation = correlate(sliced_T2, pattern, mode='valid')
    match_index = np.argmax(correlation)
    match_time = match_index / sr2
    # TODO !! Ajouter un critère béluga détecté pour virer les faux positifs (ie cross corr crado)

    return match_time

############################################
########## Main matching function ##########
############################################

def spotting_to_location_preparation(audio_files, timestamps, pad, print_level = 0):
    """Spot patterns to return 

    Args:
        audio_files (_type_): _description_
        seconds_since_file_start_T1 (_type_): _description_
        pad (_type_): _description_
        duration (_type_): _description_
    """
    ##### Finding lags #####
    datetimes = [path_to_datetime(audio_path) for audio_path in audio_files]
    if print_level >2:
        print(f"datetimes : {datetimes}")
        print(f"timestamps {timestamps}")
    sliced_audios = []
    sample_rates = []
    #WARNING : SAMPLE RATE IS NOT FROM SOURCE
    ###########################################
    ###########################################
    sample_rate_guess = 384000 ######################
    ###########################################
    ###########################################
    duration_idx = int(1.4 * pad * sample_rate_guess)
    for j, (audio_path, timestamp) in enumerate(zip(audio_files, timestamps)):
        time0 = max(0, int((timestamp.total_seconds() - 0.2 * pad) * sample_rate_guess))
        new_sliced_audio, sr_real = slice_audio(audio_path, time0, duration_idx)

        # Ajuste une fois avec le vrai SR
        if j == 0 and sr_real != sample_rate_guess:
            sample_rate_guess = sr_real
            duration_idx = int(1.4 * pad * sample_rate_guess)
            time0 = max(0, int((timestamp.total_seconds() - 0.2 * pad) * sample_rate_guess))
            new_sliced_audio, sr_real = slice_audio(audio_path, time0, duration_idx)

        sliced_audios.append(new_sliced_audio)
        sample_rates.append(sr_real)
    return(sliced_audios, sample_rates)