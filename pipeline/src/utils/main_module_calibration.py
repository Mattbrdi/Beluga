"""_summary_
"""

import pandas as pd
import numpy as np
import pyproj
from pyproj import Transformer
import scipy.io.wavfile as wavfile
from scipy.signal import firwin, butter, lfilter, filtfilt
import scipy


# Initialiser les parametres des projections
geod = pyproj.Geod(ellps='WGS84')
wgs84 = pyproj.Proj(proj="latlong", datum="WGS84")
ecef = pyproj.Proj(proj="geocent", datum="WGS84")

def estimate_wavector(V_M, t_M):
    """Calcul du vecteur d'ondes avec le LMS algorithm.

    Args:
        V_M (array): matrices des distances entre les hydrophones d'un tetraedre
        t_M (array): TDOA differences entre chaque paires

    Returns:
        array: wavectors
    """
    return - np.linalg.pinv(V_M.T @ V_M) @ V_M.T @ t_M

def select_frequency_range_fft(sample_rate, data, frequency_range, order=None):
    """Réalise un filre passe bande en utilisant la fft. (lent mais très efficace). 
    On pourrait penser a faire un filtre passse bande plus "classique" si on souhaite aller plus vite.

    Args:
        sample_rate (int): sample rate
        data (list): audio provenant d'un micro
        frequency_range (list): fréquence basse et haute pour le filtre
        order (int): argument inutile, ne sert qu'à la symetrie avec les autres fonctions

    Raises:
        ValueError: On n'accepte qu'un seul audio, donc une liste et pas un tableau de plusieur audios

    Returns:
        liste: audio filtré entre les fréquences choisies
    """

    if frequency_range[0] == -1:
        return data # pas de filtrage
    else:
        if len(data.shape) > 1:
            raise ValueError("Les fichiers audio doivent être en mono - pas de stéréo - pour la sélection de fréquence")

        fft_data = np.fft.fft(data)
        n = len(data)
        freqs = np.fft.fftfreq(n, d=1.0/sample_rate)
        indices = np.where(np.logical_and(freqs >= frequency_range[0], freqs <= frequency_range[1]))[0]
        fft_filtered = np.zeros_like(fft_data)
        fft_filtered[indices] = fft_data[indices]
        data_filtered = np.fft.ifft(fft_filtered).real

        return(data_filtered)

def select_frequency_range_linear_phase(sample_rate, data, frequency_range, order=101):
    b = firwin(order, [frequency_range[0], frequency_range[1]], pass_zero=False, window=('kaiser', 8), fs=sample_rate) # ON PEUT CHANGER LA WINDOW
    filtered_signal = lfilter(b, 1.0, data)

    return(filtered_signal)

def select_frequency_range_butter(sample_rate, data, frequency_range, order=3):
    b, a = butter(order, [frequency_range[0], frequency_range[1]], fs=sample_rate, btype='band')
    filtered_signal = lfilter(b, a, data)
    return(filtered_signal)

def select_frequency_range_two_way(sample_rate, data, frequency_range, order=2):
    b, a = butter(order, frequency_range[0], btype="high", fs=sample_rate)
    filtered_signal_inter = filtfilt(b, a, data) # Le high pass ne fonctionne pas !
    c, d = butter(order, frequency_range[1], btype="low", fs=sample_rate)
    filtered_signal = filtfilt(c, d, filtered_signal_inter)

    return(filtered_signal)

def high_pass_filtering(sample_rate, data, frequency_cut=900, order=3):
    b, a = butter(order, frequency_cut, btype="high", fs=sample_rate)
    filtered_signal = filtfilt(b, a, data) 
    return filtered_signal

def lla_to_ecef(lat, lon, alt=0):
    """Fonction pour convertir LLA en ECEF

    Args:
        lat (float): coordonnee de latitude (Nord, posiif au Canada) (degres decimaux)
        lon (float): longitude (Est, negatif au Canada) (degres decimaux)
        alt (float, optional): altitude. Defaults to 0.

    Returns:
        tuple: coordonnees en ECEF
    """
    transformer = Transformer.from_crs('epsg:4326', 'epsg:4978')
    x, y, z = transformer.transform(lat, lon, alt)
    return x, y, z

def ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref=0):
    """Fonction pour convertir ECEF en ENU centré sur un point de reference

    Args:
        x (float): 1ere coordonnee ECEF
        y (float): 2eme coordonnee ECEF
        z (float): 3eme coordonnee ECEF
        lat_ref (float): latitude (degres decimaux)
        lon_ref (float): longitude (degres decimaux)
        alt_ref (float, optional): altitude. Defaults to 0.

    Returns:
        tuple: coordonnees ENU
    """
    # Convertir le point de référence LLA en ECEF
    x_ref, y_ref, z_ref = lla_to_ecef(lat_ref, lon_ref, alt_ref)
    
    # Convertir les angles en radians
    lat_ref_rad = np.radians(lat_ref)
    lon_ref_rad = np.radians(lon_ref)
    
    # Matrice de rotation
    t = np.array([
        [-np.sin(lon_ref_rad),  np.cos(lon_ref_rad), 0],
        [-np.sin(lat_ref_rad) * np.cos(lon_ref_rad), -np.sin(lat_ref_rad) * np.sin(lon_ref_rad), np.cos(lat_ref_rad)],
        [ np.cos(lat_ref_rad) * np.cos(lon_ref_rad),  np.cos(lat_ref_rad) * np.sin(lon_ref_rad), np.sin(lat_ref_rad)]
    ])
    
    # Différence entre le point de référence et le point en question
    ecef_diff = np.array([x - x_ref, y - y_ref, z - z_ref])
    
    # Convertir ECEF en ENU
    enu = np.dot(t, ecef_diff)
    
    return enu[0], enu[1], enu[2]

def lla_to_enu(lat, lon, alt, lat_ref, lon_ref, alt_ref=0):
    """Fonction pour convertir LLA en ENU centré sur un point de reference 

    Args:
        lat (float): coordonnee de latitude (Nord, posiif au Canada) (degres decimaux)
        lon (float): longitude (Est, negatif au Canada) (degres decimaux)
        alt (float, optional): altitude. Defaults to 0.
        lat_ref (float): latitude (degres decimaux)
        lon_ref (float): longitude (degres decimaux)
        alt_ref (float, optional): altitude. Defaults to 0.

    Returns:
        tuple: coordonnees ENU
    """
    x, y, z = lla_to_ecef(lat, lon, alt)
    e, n, u = ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref)
    return e, n, u

def select_plage_audio(data, sample_rate, time_start, duration):
    """récupère les audios sous la forme d'un tableau (de nombre-hydrophones colonnes), 
    et renvoie les audio d'une duréee duration commencant au time_start

    Args:
        data (array): data des 4 canaux
        sample_rate (int): sample rate
        time_start (array): [heures, minutes, secondes, ms]
        duration (array): [heures, minutes, secondes, ms]

    Returns:
        array: data des 4 canaux
    """
    start_index = sample_rate*(time_start[0]*3600 + time_start[1] * 60 + time_start[2] + time_start[3] * 0.001)
    if duration[0] == -1:
        return data[int(start_index):,:]
    else:
        end_index = start_index + sample_rate*(duration[0]*3600 + duration[1] * 60 + duration[2] + duration[3] * 0.001)
        return data[int(start_index):int(end_index),:]

def calculate_max_distance_between_hydrophones(coordinates_hydrophones):
    """
    Calcule la distance maximale entre deux hydrophones de notre structure.
    
    Parameters:
    coords_hydrophones (np.ndarray): Un tableau 2D où chaque ligne représente les coordonnées d'un hydrophone.
    
    Returns:
    float: La distance maximale entre deux hydrophones.
    """
    max_distance = 0
    num_hydrophones = coordinates_hydrophones.shape[0]
    
    for i in range(num_hydrophones):
        for j in range(i + 1, num_hydrophones):
            distance = np.linalg.norm(coordinates_hydrophones[i] - coordinates_hydrophones[j])
            if distance > max_distance:
                max_distance = distance
    
    return max_distance

def prepare_audio_data(input_files, coordinates_hydrophones, time_window_duration, frequency_range, start_time, duration, filtering_method, order):
    """_summary_

    Args:
        input_files (str): path du fichier audio des 4 cannaux 
        coordinates_hydrophones (list): coordonnées des hydrophones dans le repere ENU
        time_window_duration (float): duree de la fenetre de calcul de la cross correlation
        frequency_range (list): fréquence basse et haute pour le filtre
        time_start (array): [heures, minutes, secondes, ms]
        duration (array): [heures, minutes, secondes, ms]

    Raises:
        ValueError: le temps de calcul de la fenetre de cross correlation doit etre superieur au temps maximal parcouru par le son d'un hydrophone à l'autre

    Returns:
        list: fichiers audios du tétraedre filtrés en temps et fréquence
    """

    number_of_hydrophones = len(coordinates_hydrophones)
    audio_data_list = np.empty((0, number_of_hydrophones))
    #audio_data_list = []

    
    #def process_audio_file(input_file, number_of_hydrophones):
    #    sample_rate, data_x = wavfile.read(input_file)
    #    return sample_rate, data_x[:, :number_of_hydrophones]
    
    #results = Parallel(n_jobs=-1)(delayed(process_audio_file)(input_file, number_of_hydrophones) for input_file in input_files)

    #sample_rate = results[0][0]
    #audio_data = [result[1] for result in results]


    #for result in audio_data:
    #    audio_data_list = np.vstack((audio_data_list, result))

    for input_file in input_files:
        sample_rate, data_x = wavfile.read(input_file)
        audio_data_list = np.vstack((audio_data_list, data_x[:, :number_of_hydrophones]))

    
    audio_data_list = select_plage_audio(audio_data_list, sample_rate, start_time, duration)

    UNDERWATER_SOUND_SPEED = 1450
    ARRAY_MAX_LENGTH = calculate_max_distance_between_hydrophones(coordinates_hydrophones)
    max_decalage = 1.1 * ARRAY_MAX_LENGTH / UNDERWATER_SOUND_SPEED * sample_rate
    
    if max_decalage > time_window_duration * sample_rate:
        raise ValueError("La fenetre de calcul de la cross correlation doit être plus grande que le delais maximal")

    if filtering_method == "fft":
        select_frequency_range = select_frequency_range_fft
    elif filtering_method == "butter":
        select_frequency_range = select_frequency_range_butter
    elif filtering_method == "two_way":
        select_frequency_range = select_frequency_range_two_way
    elif filtering_method == "high_pass_linear_phase":
        select_frequency_range = high_pass_filtering
    else :
        select_frequency_range = select_frequency_range_linear_phase

    for i in range(number_of_hydrophones):
        if filtering_method == "high_pass_linear_phase":
            audio_data_list[:, i] = select_frequency_range(sample_rate, audio_data_list[:, i], frequency_range[0], order)
        else:
            audio_data_list[:, i] = select_frequency_range(sample_rate, audio_data_list[:, i], frequency_range, order)

    return audio_data_list, sample_rate, max_decalage

def normalize_vectors_matrix(matrix):
    """Fonction qui normalise toutes le lignes d'une matrice

    Args:
        matrix (array):

    Returns:
        array: matrice
    """
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

def intersection_point(k, h, point_on_vector=[0, 0, 0]):
    """Calcul de la position de la source a la surface dans le repére du tétraèdre

    Args:
        k (list): vecteur d'onde représenté sous forme d'une liste [x, y, z]
        h (float): hauteur du plan horizontal par rapport a l'origine du repere
        point_on_vector (list, optional): Point d'origine du vecteur d'onde. Defaults to [0, 0, 0]

    Returns:
        list: coordonnee du point d'intersection du vecteur d'onde et de la surface 
    """

    # Calculer du coefficient t de l'équation paramétrique
    t = (h - point_on_vector[2]) / k[2]

    # Calculer les coordonnées du point d'intersection
    intersection_point = [point_on_vector[0] + t * k[0],
                          point_on_vector[1] + t * k[1],
                          h]

    return intersection_point

def xcorr_freq_norm(signal1, signal2):
    """ Calcul la cross-correlation et normalise

    Args:
        signal1 (list): 1er audio
        signal2 (list): 2nd audio

    Returns:
        list: cross correlation normalisee
    """
    corr = scipy.signal.correlate(signal1, signal2, mode='full')
    return corr / np.max(np.abs(corr))

def process_audio(audio1, audio2, window_size, max_delay, cross_corr_func = xcorr_freq_norm):
    """ Fonction qui calcule la corrélation de chaque sample du signal et plot le graphe en cascade

    Args:
        audio1 (list): 1er fichier audio
        audio2 (list): 2nd fichier audio
        window_size (int): nombre de sample d'une fenetre de calcul de cross correlation
        max_delay (int): delay maximal entre 2 hydrophones
        cross_corr_func (func, optional): fonction de calcul de cross correlation. Defaults to xcorr_freq_norm.

    Returns:
        _type_: _description_
    """

    # Découper les fichiers audio en fenêtres
    num_windows = len(audio1) // window_size

    cross_corr_matrix = np.zeros((2 * window_size - 1, num_windows))
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        segment1 = audio1[start:end]
        segment2 = audio2[start:end]

        cross_corr_matrix[:, i] = cross_corr_func(segment1, segment2)

    lower_bound = int((2 * window_size - 1)/2 - max_delay)
    upper_bound = int((2 * window_size - 1)/2 + max_delay)

    return cross_corr_matrix[lower_bound:upper_bound , :]

def interpolate_linear(start, end):
    """Fonction d'interpolation linéaire

    Args:
        start (list): 1er point
        end (list): 2eme point 

    Returns:
        list: points d'interpolation
    """

    x1, y1 = start
    x2, y2 = end
    interpolated_points = []
    for x in range(x1 + 1, x2):
        y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        interpolated_points.append(y)

    return interpolated_points

def filter_list(values, ecart_tolere, fenetre):
    """Fonction qui filtre la liste : objectif supprimer le bruit
1ere methode de retrait des valerus absurdes :
1. On va parcourir la liste de gauche a droite
2. On calcule les ecarts entre chque point et son suivant dans une nouvelle liste
3. on commence lorsque l'on a 10 points de suite avec un ecart inferieur a un seuil en moyenne absolue avec leur voisin
4. a partir de la, on parcourt le tableau : tant que l ecart est inferieur au seuil, on avance sans rien faire
5. lorsque l on rencontre un ecart superieur au seuil en valeur absolue, on garde en memoire les indices puis on s arrette lorsque l on rencontre un point avec un ecart superieur au seuil du signe opposé ! (si au bout de x valeurs on est pas tombé dessus, on ne change rien et on s ereplace a l'indice suivnt l ecart qu on avait rouvé) et on remplace le ou les points points de la fenetre par une interpolation lineaire entre le dernier point qui ne presentait pas de decalage>20, jusqau dernier point absurde. (par exmple, si on avait a la suite 110, 111, 112, 212, 114, 116, cela sera remplace par 110, 111, 112, 113, 114, 116; si on avait a la suite 110, 111, 112, 212, 220, 114, 116, cela sera remplace par 110, 111, 112, 113, 113, 114, 116;). On garde uniquement des valeurs entiere lors de l'interpolation (les nombre pairs sont arrondis au superieur, les nombres impair a l inferieur)
6. On applique plusieurs fois de suite l'algo avec le suil et x qui diminuent

    Args:
        values (list): TDOAS
        ecart_tolere (int): ecart tolorés en unités minimale de mesure des TDOA (relatifs au sample_rate)
        fenetre (int): nombres de sample considérés avant de revenir au 2eme pour detecter une incoherence

    Returns:
        _type_: _description_
    """
    n = len(values)
    if n < 4:
        return values  # Pas assez de points pour l'analyse
    
    # Calcul des écarts
    ecarts = [values[i+1] - values[i] for i in range(n-1)]
    
    # Recherche de la première séquence de 10 écarts consécutifs < ecart_tolere
    start_idx = -1
    for i in range(n - 4):
        if all(abs(e) < ecart_tolere for e in ecarts[i:i+2]):
            start_idx = i + 3
            break
    
    if start_idx == -1:
        return values  # Aucune séquence trouvée
    
    # Filtrage des valeurs aberrantes
    i = start_idx
    while i < n - 1:
        if abs(ecarts[i]) < ecart_tolere:
            i += 1
        else:
            # On a trouvé une valeur aberrante
            j = i + 1
            while j < n - 1 and (abs(ecarts[j]) < ecart_tolere or (ecarts[j] * ecarts[i] > 0)) and j-i < fenetre: # Si on a pas trouvé de redescente dans les fenetre valeurs suivantes, on abandonne
                j += 1
            # Si la séquence aberrante est terminée par un écart de signe opposé
            if j < n - 1 and j-i < fenetre and ecarts[j] * ecarts[i] < 0:
                interpolated = interpolate_linear((i, values[i]), (j+1, values[j+1]))
                values = values[:i+1] + interpolated + values[j+1:]
                ecarts = [values[k+1] - values[k] for k in range(n-1)]  # Recalcul des écarts
                i += len(interpolated)  # Continuer après la séquence interpolée
            else:
                i = j
    
    return values

def debruitage_crosscorr(delays):
    """Débruite une liste de TDOA en utilisatio nla fonction filter_list

    Args:
        delays (array): TDOAs

    Returns:
        array: TDOA débruités
    """

    filtres = filter_list(list(delays), 20, 20)
    filtres = filter_list(filtres, 10, 10)
    filtres = filter_list(filtres, 5, 5)
    filtres = filter_list(filtres, 2, 2)    # Potentiellement fenetres a adapter

    return np.array(filtres)

def calculate_wavectors(audio_data_list, coordinates_hydrophones, time_window_duration, sample_rate, max_delay, plot_tdoa=False, threshold=-2):
    """Cette fonction récupere les audio filtrés, calcul les TDOA entre les paires d'hydrophones d'un tétraedre, résoud un systeme
    et calcule le vecteur pointant vers la source sonore

    Args:
        audio_data_list (list): fichiers audio a analyser
        coordinates_hydrophones (list): coordonnees des hydrophones
        time_window_duration (float): duree en seconde d'une fenetre de calcul de cross correlation
        sample_rate (int): fréquence d'échantillonnage
        max_delay (int): delais maximum pour la propagation du son entre 2 hydrophones
        plot_tdoa (bool, optional): Plot des correlogrammes. Defaults to False.
        threshold (int, optional): entre 0 et 1, valeur minimal pour qu'un point de la cross corr soit plot. Defaults to -2 = tout plot.

    Returns:
        array: wavectors pointant vers la source
    """

    number_of_hydrophones = len(coordinates_hydrophones)
    window_size = int(sample_rate * time_window_duration)
    num_windows = len(audio_data_list[:, 0]) // window_size
    cross_corr_matrix_list = []
    list_hydrophones_numbers = []
    lower_bound = int((2 * window_size - 1) / 2 - max_delay)
    delay_tdoa_in_seconds_list = []
    
    for i in range(number_of_hydrophones):
        for j in range(i + 1, number_of_hydrophones):
            cross_corr_matrix = process_audio(audio_data_list[:, i], audio_data_list[:, j], window_size, max_delay)
            cross_corr_matrix_list.append(cross_corr_matrix)
            delay_tdoa_in_seconds_list.append(debruitage_crosscorr(np.argmax(cross_corr_matrix, axis=0) - (window_size - 1) + lower_bound) / sample_rate)
            list_hydrophones_numbers.append(f"H{i+1} et H{j+1}")

    delay_tdoa_in_seconds_array = np.array(delay_tdoa_in_seconds_list)

    v_m = np.array([coordinates_hydrophones[i] - coordinates_hydrophones[j]
                    for i in range(number_of_hydrophones)
                    for j in range(i + 1, number_of_hydrophones)])

    wavevectors = []
    for i in range(delay_tdoa_in_seconds_array.shape[1]):
        wavevectors.append(estimate_wavector(v_m, delay_tdoa_in_seconds_array[:, i]))

    return np.array(wavevectors)

def calculate_surface_position_from_wavector(wavectors, time_window_duration, base_depth, plot_position=False):
    """fonction qui determine la position d'une source a plusieurs pas de temps a la surface grace a l elevation et l azimut

    Args:
        wavectors (array): vecteurs d'onde
        time_window_duration (float): duree d une fenetre de calcul en secondes
        base_depth (float): profondeur de la base du tetraedre
        plot_position (bool, optional): Plot la position si True. Defaults to False.

    Returns:
        list: positions [x, y] de la source a la surface
    """

    NUMBER_OF_WINDOWS = len(wavectors)
    positions = []
    for i in range(NUMBER_OF_WINDOWS):   
        positions.append(intersection_point(wavectors[i], base_depth))

    timestamps = np.linspace(0, time_window_duration*(NUMBER_OF_WINDOWS-1), NUMBER_OF_WINDOWS)

    return positions

def compute_rotation_matrix(A, B):
    """Resolution du probleme de Wahba via SVD pour estimer la matrice de rotation entre le repere ENU et le repere du tetraedre

    Args:
        A (array): Vecteurs d'onde (normalisés) estimés par TDOA dans le repere du tetraedre
        B (array): Vecteurs d'onde (normalisés) obtenus grace a la trace GPS dans le repere ENU

    Returns:
        array: matrice de rotation de ENU vers le repere du tetraedre
    """

    # Compute the covariance matrix
    H = np.dot(B.T , A)
    
    # Compute the Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = U @ np.diag([1, 1, np.linalg.det(U)*np.linalg.det(Vt)]) @ Vt

    return R

def find_rotation_matrix_3D(input_datagps, input_files, start_time, end_time, time_window_duration, coordinates_tetrahedron, coordinates_hydrophones, frequency_range, FILTERING_METHOD, FILTER_ORDER, plot_repere=False):
    """Calcule la matrice de rotation 3x3 entre le repere ENU et du tetraedre

    Args:
        input_datagps (str): path du data GPS de reference
        input_files (str): path des audios du tetraedre
        start_time (datetime): timing du debut de l'audio
        end_time (datetim): timing de la fin de l'audio
        time_window_duration (float): _description_
        coordinates_tetrahedron (list): coordonnees du tetraedre en LLA
        coordinates_hydrophones (array): coordonnees des hydrophones dans le repere de l'hydrophone
        frequency_range (list): bornes des fréquences a selectionner
        plot_repere (bool, optional): Si True plot le repere de l'hydrophone, default = False

    Returns:
        array: matrice de rotation 3x3 entre le repere ENU et du tetraedre
    """

    START_DELAY = [0, 0, (start_time-pd.to_datetime(input_files[0].split('\\')[-1][5:17], format='%y%m%d%H%M%S')).total_seconds(), 0] # h, m, s, ms
    AUDIO_DURATION = [0, 0, (end_time-start_time).total_seconds(), 0]


    # Calculer les vecteurs d'onde dans le repere du tetraedre et les interpoller a la seconde
    audio_data_list, sample_rate, max_decalage = prepare_audio_data(input_files, coordinates_hydrophones, time_window_duration, frequency_range, START_DELAY, AUDIO_DURATION, FILTERING_METHOD, FILTER_ORDER)
    wavectors = calculate_wavectors(audio_data_list, coordinates_hydrophones, time_window_duration, sample_rate, max_decalage)
    wavectors_seconds = wavectors[::int(1/time_window_duration)]

    #Lire les vraies coordonnees GPS
    data_gps = pd.read_csv(input_datagps)
    data_gps = data_gps[['datetime_correct', 'lat', 'long']].rename(columns={'datetime_correct': 'time', 'lat':'latitude','long':'longitude'}) # selectionner et renommer les colonnes
    data_gps['time'] = pd.to_datetime(data_gps['time']) # transformation du type object vers datetime
    data_gps = data_gps[(data_gps['time'] >= start_time) & (data_gps['time'] < end_time)]
    data_gps['altitude'] = 0 # le spoint de bateau sont a altitude nulle dans le repereem LLA
    
    # Passer en ENU
    data_ref_gps = np.array(data_gps[['latitude', 'longitude', 'altitude']])
    for i in range(len(data_ref_gps)):
        lat, lon, alt = data_ref_gps[i][:3]
        e, n, u = lla_to_enu(lat, lon, alt, coordinates_tetrahedron[0], coordinates_tetrahedron[1], coordinates_tetrahedron[2]) #Le point de reference est le centre du tetraedre dans le repere LLA
        #u = -coordonnees_base[2] # la conversion cree une incertitude d environ 0.001 m sur le u, on peut la supprimer 
        data_ref_gps[i][:3] = [e, n, u]

    An = normalize_vectors_matrix(wavectors_seconds)
    Bn = normalize_vectors_matrix(data_ref_gps)

    R = compute_rotation_matrix(An, Bn)

    return R


###################################################################################################
###################################################################################################
###################################################################################################


TETRAHEDRON_COUNT = 2 # 1 ou 2 tétraedre
ROTATION_DIMENSION = 3 # 2 ou 3

FILTERING_METHODS = ["linear_phase", "fft", "butter", "two_way"] 
FILTERING_METHOD = FILTERING_METHODS[2] # linear_phase par defaut
FILTER_ORDER = 3
FREQUENCY_RANGE = [500, 25000] #En hz, la plage de fréquence de l'audio que l'on souhaite conserver, si [-1, _] tout l'audio est conservé
TIME_WINDOW_DURATION = .1

DEPTH_T1 = 19.8-0.74
COORDINATES_T1 = [47.93961, -69.52256, -DEPTH_T1] # coordonnées du centre du tétraedre en lat, long, alt
DEPTH_T2 = 19.8-0.74
COORDINATES_T2 = [47.94323, -69.51839, -DEPTH_T2] # coordonnées du centre du tétraedre en lat, long, alt

ARRAY_LENGTH_1 = .3 # Longueur des arretes en mètres
# Coordonnées des sommets du tétraèdre dans le référentiel défini DANS LE MEME ORDRE QUE LES CANAUX D'ENREGISTREMENT
COORDINATES_HYDROPHONES_1 = np.array([
    [np.sqrt(3)*ARRAY_LENGTH_1/3, 0, 0], # Canal 0
    [-np.sqrt(3)*ARRAY_LENGTH_1/6, -ARRAY_LENGTH_1/2, 0], # Canal 1
    [-np.sqrt(3)*ARRAY_LENGTH_1/6, ARRAY_LENGTH_1/2, 0], # Canal 2
    [0, 0, np.sqrt(6)*ARRAY_LENGTH_1/3] # Canal 3
])

ARRAY_LENGTH_2 = .3 
COORDINATES_HYDROPHONES_2 = np.array([ # DANS LE MEME ORDRE QUE LES CANAUX D'ENREGISTREMENT
    [np.sqrt(3)*ARRAY_LENGTH_2/3, 0, 0], # Canal 0
    [-np.sqrt(3)*ARRAY_LENGTH_2/6, -ARRAY_LENGTH_2/2, 0], # Canal 1
    [-np.sqrt(3)*ARRAY_LENGTH_2/6, ARRAY_LENGTH_2/2, 0], # Canal 2
    [0, 0, np.sqrt(6)*ARRAY_LENGTH_2/3] # Canal 3
])

# Audio au 1er tetra
AUDIO_FILES_T1 = [r"C:\Users\plgal\qilaluga\app\__meta\_inputs\audios_beluga\8296.240726161122.wav"] # Audio file
GROUNDTRUTH_T1 = r'C:\Users\plgal\qilaluga\___dominic\data\Exp3_CAC\groundtruth\trace_gps_calibration.csv' # Track GPS de la source

# Définition des dates de début et de fin
START_TIME_T1 = pd.to_datetime('2024-07-26 16:19:11.00', format='%Y-%m-%d %H:%M:%S.%f')
END_TIME_T1 = pd.to_datetime('2024-07-26 16:19:31.00', format='%Y-%m-%d %H:%M:%S.%f')

# Auio au 2eme tetra
AUDIO_FILES_T2 = [r"C:\Users\plgal\qilaluga\app\__meta\_inputs\audios_beluga\8295.240726162234.wav"] # Audio file

GROUNDTRUTH_T2 = r'C:\Users\plgal\qilaluga\___dominic\data\Exp3_CAC\groundtruth\trace_gps_calibration.csv' # Track GPS de la source

# Définition des dates de début et de fin
START_TIME_T2 = pd.to_datetime('2024-07-26 16:29:13.00', format='%Y-%m-%d %H:%M:%S.%f')
END_TIME_T2 = pd.to_datetime('2024-07-26 16:29:33.00', format='%Y-%m-%d %H:%M:%S.%f')

ROTATION_MATRIX_TETRAHEDRON_1 = find_rotation_matrix_3D(GROUNDTRUTH_T1, AUDIO_FILES_T1, START_TIME_T1, END_TIME_T1, TIME_WINDOW_DURATION, COORDINATES_T1, COORDINATES_HYDROPHONES_1, FREQUENCY_RANGE, FILTERING_METHOD, FILTER_ORDER)
ROTATION_MATRIX_TETRAHEDRON_2 = find_rotation_matrix_3D(GROUNDTRUTH_T2, AUDIO_FILES_T2, START_TIME_T2, END_TIME_T2, TIME_WINDOW_DURATION, COORDINATES_T2, COORDINATES_HYDROPHONES_2, FREQUENCY_RANGE, FILTERING_METHOD, FILTER_ORDER)

print(f"Tétraèdre 1: {ROTATION_MATRIX_TETRAHEDRON_1}")
print(f"Tétraèdre 2: {ROTATION_MATRIX_TETRAHEDRON_2}")