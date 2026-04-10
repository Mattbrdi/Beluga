"""Visualization tools for algorithm evaluation."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram
from pathlib import Path

from sklearn.cluster import KMeans

from src.utils.rotation_bricks import enu2lla, lla2enu
from src.utils.sub_classes import Environment, AudioArray, Parameters
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import correlate
from typing import Optional
import librosa
import librosa.display
from src.location_bricks.tdoa_brick import tdoas, gcc_phat_from_pair, compute_correlation_score
from src.location_bricks.low_level_fusion import low_fusion
from src.location_bricks.high_level_fusion import high_fusion

#################################################
########## Old functions for VMD plots ##########
#################################################

def centroids_from_pos(positions, nb_clusters = 1):
    poisitions_reshaped = np.array([position.reshape(3,) for position in positions if not np.any(np.isnan(position))])
    kmeans = KMeans(n_clusters=nb_clusters, random_state=0)
    kmeans.fit(poisitions_reshaped)
    centroids = kmeans.cluster_centers_
    return centroids


def plot_from_data_array(data, sample_rate, output_path, print_logs = False, igff = None, share_y = True):
    """Visualize an array of modes in both time and spectral domain.

    Args:
        data (np.ndarray): A (N, M) array of different modes 
        sample_rate (int): Sampling frequency
    """
    N, M = data.shape
    if N == 0:
        if print_logs:
            print(f"Warning : {output_path} is an empty array.")
        return False
    fig, axes = plt.subplots(N, 3, figsize=(18, 3 * N))
    if N == 1:
        axes = [axes] 
    for i in range(N):
        # Domaine temporel
        axes[i][0].plot(np.arange(M)/sample_rate,data[i])
        if igff is None:
            axes[i][0].set_title(f'Signal temporel {i+1}')
            if i >0 and share_y:
                axes[i][0].sharey(axes[0][0])
        else:
            if i >0:
                axes[i][0].set_title(f'Mode {i+1}, IGFF {igff[i-1]:.2f}')
                if i >1 :
                    axes[i][0].sharey(axes[1][0])
            elif i ==0:
                axes[0][0].set_title(f'DC mode')
        axes[i][0].set_xlabel('Temps')
        axes[i][0].set_ylabel('Amplitude')

        # Domaine spectral
        yf = fft(data[i] - np.mean(data[i]))  # Soustraire la moyenne pour éliminer la composante continue
        xf = fftfreq(M, 1 / sample_rate)[:M//2]  # Ne garder que la moitié positive du spectre
        axes[i][1].plot(xf, 2.0/M * np.abs(yf[:M//2]))  # Normaliser l'amplitude
        axes[i][1].set_title(f'Spectre {i+1}')
        axes[i][1].set_xlabel('Fréquence [Hz]')
        axes[i][1].set_ylabel('Amplitude')

        # Spectrogramme
        f, t, Sxx = spectrogram(data[i], fs=sample_rate)
        axes[i][2].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
        axes[i][2].set_title(f'Spectrogramme {i+1}')
        axes[i][2].set_xlabel('Temps [s]')
        axes[i][2].set_ylabel('Fréquence [Hz]')

    fig.tight_layout()
    plt.savefig(output_path)
    plt.close()
    if print_logs:
        print(f"Successfully exported file at {output_path}.")
    return True

def plot_from_output(denoising_outputs, output_folder : Path, print_logs : bool=False):
    tetra_folder = output_folder / Path(denoising_outputs.tetra_id)
    tetra_folder.mkdir(parents=False, exist_ok=True)
    sample_rate = denoising_outputs[0].fech
    plot_from_data_array(denoising_outputs.tetra_array, sample_rate, tetra_folder / Path(denoising_outputs.tetra_id + "_original_signal.png"),print_logs, share_y = False)
    plot_from_data_array(denoising_outputs.filtered_tetra_array, sample_rate, tetra_folder / Path(denoising_outputs.tetra_id + "_filtered_signal.png"),print_logs)
    plot_from_data_array(denoising_outputs.noisy_tetra_array, sample_rate, tetra_folder / Path(denoising_outputs.tetra_id + "_noisy_signal.png"),print_logs)
    if print_logs:
        print(f"Finished plotting original and filtered signals of tetrahydra {denoising_outputs.tetra_id}")
    for output in denoising_outputs:
        this_output_path = str(tetra_folder / Path(output.hydrophone_id + "_"))
        # VMD echo brick
        plot_from_data_array(output.pure_imfs, sample_rate, Path(this_output_path + "pure_imfs.png"),print_logs)
        plot_from_data_array(output.mixed_imfs, sample_rate, Path(this_output_path + "mixed_imfs.png"),print_logs)
        plot_from_data_array(output.noisy_imfs, sample_rate, Path(this_output_path + "noisy_imfs.png"),print_logs)
        #Plotting all modes
        plot_from_data_array(output.all_imfs, sample_rate, Path(this_output_path + "all_modes.png"),print_logs, output.imfs_igff)
        print(f"Finished plotting {output.hydrophone_id} of tetrahydra {denoising_outputs.tetra_id}")
    return True

#######################################################
########## Plotting the positions on the map ##########
#######################################################

def two_dim_plot_enu(positions_enu,
                    positions_error_variance,
                    associated_times,
                    environment,
                    showfigs = False,
                    output_path = None,
                    groundtruths : Optional[np.ndarray] = None,
                    groundtruths_labels = None,
                    call_types : Optional[list[str]] = None):
    positions = np.array(positions_enu)
    positions_error_variance = np.array(positions_error_variance)
    fig, ax = plt.subplots(figsize=(10, 10))

    centroids = centroids_from_pos(positions, nb_clusters = 2)
    ax.scatter(centroids[:,0], centroids[:,1], color='tab:orange', marker='o', s=30, label= 'Positions centroids')

    ##### Trace positions #####
    whistle_mask = np.where(np.array(call_types)=='Whistle')[0]
    hfpc_mask = np.where(np.array(call_types)=='HFPC')[0]

    ax.scatter(positions[whistle_mask, 0], positions[whistle_mask, 1], color='tab:red', marker='x', s=10, label= 'Whistle')
    ax.scatter(positions[hfpc_mask, 0], positions[hfpc_mask, 1], color='tab:purple', marker='x', s=10, label= 'HFPC')
    
    ##### Groundtruths #####
    if groundtruths is not None:
        ax.scatter(groundtruths[:,0], groundtruths[:,1],color='yellow', marker='x', s=50, label= 'Groundtruth')
        if groundtruths_labels is not None:
            for pos,label in zip(groundtruths, groundtruths_labels):
                ax.text(pos[0], pos[1], label, fontsize=9, color='black')
                
    ##### Add incertitudes #####
    for pos, inc in zip(positions, positions_error_variance):
        ellipse = Ellipse(pos, inc[0]*2, inc[1]*2, fill=True, edgecolor = None, color='blue', alpha=0.1)
        ax.add_patch(ellipse)

    ##### Labels #####
    all_distances = []
    for i, (pos, time) in enumerate(zip(positions, associated_times)):
        label = str(i)
        #label =  f'T : {time}'
        if groundtruths is not None:
            #label += ' dist : '+str(np.linalg.norm(groundtruths[i][:2]- pos[:2]))
            distance = np.linalg.norm(groundtruths[0][:2].ravel()- pos[:2].ravel())
            if not np.isnan(distance):
                label += f' - {distance:.1f}m'
                all_distances.append(distance)
        ax.text(pos[0], pos[1], label, fontsize=6, color='black')
    print(all_distances)
    distance_mediane = np.nanmedian(all_distances)
    print(distance_mediane)
        
    ##### Tetrahedras #####
    origins0 = [tetrahedra.origin_enu[0] for tetrahedra in environment.tetrahedras.values()]
    origins1 = [tetrahedra.origin_enu[1] for tetrahedra in environment.tetrahedras.values()]
    ax.scatter(origins0,origins1, color = 'tab:green', marker = '^', label = 'Tetrahedras')
    ax.set_aspect('equal')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'2D pos of belugas\nMedian 2D dist : {distance_mediane}')
    ax.set_xlim(left=-1000, right = 1000)
    ax.set_ylim(bottom=-1000, top = 1000)

    plt.legend()
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path)
    if showfigs:
        plt.show()
    plt.close()
    return ax

def three_dim_plot_enu(positions_enu, positions_error_variance, associated_times, environment : Environment, showfigs=False, output_path=None, groundtruths: Optional[np.ndarray] = None, groundtruths_labels=None):
    positions = np.array(positions_enu)
    positions_error_variance = np.array(positions_error_variance)

    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')

    # Trace positions
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color='red', marker='x')

    # Groundtruths
    if groundtruths is not None:
        ax.scatter(groundtruths[:, 0], groundtruths[:, 1], groundtruths[:, 2], color='yellow', marker='x')
        if groundtruths_labels is not None:
            for pos, label in zip(groundtruths, groundtruths_labels):
                ax.text(pos[0], pos[1], pos[2], s=label, fontsize=9, color='black')

    # Add incertitudes
    for pos, inc in zip(positions, positions_error_variance):
        # Approximation of ellipsoid using circles in each plane
        for angle in np.linspace(0, 2*np.pi, 100):
            x = pos[0] + inc[0] * np.cos(angle)
            y = pos[1] + inc[1] * np.sin(angle)
            ax.plot([pos[0], x], [pos[1], y], [pos[2], pos[2]], color='blue', alpha=0.3)

        for angle in np.linspace(0, 2*np.pi, 100):
            x = pos[0] + inc[0] * np.cos(angle)
            z = pos[2] + inc[2] * np.sin(angle)
            ax.plot([pos[0], x], [pos[1], pos[1]], [pos[2], z], color='blue', alpha=0.3)

        for angle in np.linspace(0, 2*np.pi, 100):
            y = pos[1] + inc[1] * np.cos(angle)
            z = pos[2] + inc[2] * np.sin(angle)
            ax.plot([pos[0], pos[0]], [pos[1], y], [pos[2], z], color='blue', alpha=0.3)

    # Labels
    for i, (pos, time) in enumerate(zip(positions, associated_times)):
        label =  f'Point {i+1} - time : {time}'
        if groundtruths is not None:
            label += f' dist : {np.linalg.norm(groundtruths[i]- pos):.1f}'
        ax.text(pos[0], pos[1], pos[2], s = label, fontsize=9, color='black')
    # Tetrahedras
    for tetrahedra in environment.tetrahedras.values():
        ax.plot([tetrahedra.origin_enu[0]], [tetrahedra.origin_enu[1]], [tetrahedra.origin_enu[2]], color='green', marker='o', label=tetrahedra.id)
        ax.text(tetrahedra.origin_enu[0], tetrahedra.origin_enu[1], tetrahedra.origin_enu[2], s = tetrahedra.id, fontsize=9, color='black')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D ENU pos of belugas')
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path)
    if showfigs:
        plt.show()
    plt.close()
    return ax

# TODO lla version

def plot_cross_pairs(audio_array : AudioArray, output_folder, showfigs):
    """Trace a figure with all hydrophone pairs."""
    fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharey=True)
    axs = axs.ravel()

    # Parcourir toutes les paires de canaux
    """for i, hydrophone_pair in enumerate(audio_array.pairs_dict.values()):
        canal_one = hydrophone_pair.hydrophone_ref.audio_r
        id_one = hydrophone_pair.hydrophone_ref.id
        
        canal_two = hydrophone_pair.hydrophone_delta.audio_r
        id_two = hydrophone_pair.hydrophone_delta.id
        
        corr = correlate(canal_one, canal_two, mode='full')
        #mid_index = int((2 * canal_one.shape[0] - 1)/2)
        taus = np.linspace(-canal_one.shape[0]/float(audio_array.metadata.sample_rate), canal_one.shape[0]/float(audio_array.metadata.sample_rate), 2* canal_one.shape[0]-1)

        max_index = np.argmax(corr)

        axs[i].plot(taus, corr, label=f'Canaux {id_one} & {id_two}')
        axs[i].axvline(x=taus[max_index], color='red', linestyle='--', label=f'Max {taus[max_index]}')
        axs[i].set_title(f'Cross-corrélation Canaux {id_one} & {id_two}')
        axs[i].legend()

    plt.tight_layout()
    if output_folder is not None:
        plt.savefig(output_folder + f'{audio_array.metadata.tetra_id} cross corr full.png')
    if showfigs:
        plt.show()
    plt.close()"""
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharey=True)
    axs = axs.ravel()

    # Parcourir toutes les paires de canaux
    for i, hydrophone_pair in enumerate(audio_array.pairs_dict.values()):
        canal_one = hydrophone_pair.hydrophone_ref.audio_r
        id_one = hydrophone_pair.hydrophone_ref.id
        
        canal_two = hydrophone_pair.hydrophone_delta.audio_r
        id_two = hydrophone_pair.hydrophone_delta.id
        
        corr = correlate(canal_one, canal_two, mode='full')
        mid_index = canal_one.shape[0]
        max_delay_idx = hydrophone_pair.max_delay_idx
        taus = np.linspace((-max_delay_idx-100)/float(audio_array.metadata.sample_rate), (max_delay_idx+100) /float(audio_array.metadata.sample_rate), 2* max_delay_idx + 200)

        max_index = np.argmax(corr[mid_index-max_delay_idx-100:mid_index+max_delay_idx+100])
        score, score_text = compute_correlation_score(corr[mid_index-max_delay_idx-100:mid_index+max_delay_idx+100],audio_array.metadata.sample_rate)
        
        axs[i].plot(taus, corr[mid_index-max_delay_idx-100:mid_index+max_delay_idx+100], label=f'Canaux {id_one} & {id_two}')
        axs[i].axvline(x=taus[max_index], color='red', linestyle='--', label=f'Max {taus[max_index]}')
        axs[i].set_title(f'Cross-corrélation Canaux {id_one} & {id_two} restreinte sur max index')
        axs[i].text(0.05, 0.95, score_text,
            transform=axs[i].transAxes,  # coordonnées relatives à l'axe (0 à 1)
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
        axs[i].legend(loc = 'upper right')
    plt.tight_layout()
    if output_folder is not None:
        plt.savefig(output_folder + f'{audio_array.metadata.tetra_id} cross corr shrinked.png')
    if showfigs:
        plt.show()
    plt.close()
    
    
        
def plot_gcc_phat(audio_array : AudioArray, output_folder, showfigs):
    """Trace a figure with all hydrophone pairs."""
    fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharey=True)
    axs = axs.ravel()

    # Parcourir toutes les paires de canaux
    for i, hydrophone_pair in enumerate(audio_array.pairs_dict.values()):
        canal_one = hydrophone_pair.hydrophone_ref.audio_r
        id_one = hydrophone_pair.hydrophone_ref.id
        
        canal_two = hydrophone_pair.hydrophone_delta.audio_r
        id_two = hydrophone_pair.hydrophone_delta.id
        
        corr = gcc_phat_from_pair(canal_one, canal_two)
        #mid_index = int((2 * canal_one.shape[0] - 1)/2)
        taus = np.linspace(-canal_one.shape[0]/float(audio_array.metadata.sample_rate), canal_one.shape[0]/float(audio_array.metadata.sample_rate), 2* canal_one.shape[0]-1)

        max_index = np.argmax(corr)
        
        axs[i].plot(taus, corr, label=f'Canaux {id_one} & {id_two}')
        axs[i].axvline(x=taus[max_index], color='red', linestyle='--', label=f'Max {taus[max_index]}')
        axs[i].set_title(f'GCC PHAT Canaux {id_one} & {id_two}')
        axs[i].legend()

    plt.tight_layout()
    if output_folder is not None:
        plt.savefig(output_folder + f'{audio_array.metadata.tetra_id} gcc phat full.png')
    if showfigs:
        plt.show()
    plt.close()
    
    fig, axs = plt.subplots(3, 2, figsize=(12, 12), sharey=True)
    axs = axs.ravel()

    # Parcourir toutes les paires de canaux
    for i, hydrophone_pair in enumerate(audio_array.pairs_dict.values()):
        canal_one = hydrophone_pair.hydrophone_ref.audio_r
        id_one = hydrophone_pair.hydrophone_ref.id
        
        canal_two = hydrophone_pair.hydrophone_delta.audio_r
        id_two = hydrophone_pair.hydrophone_delta.id
        
        corr = gcc_phat_from_pair(canal_one, canal_two)
        mid_index = canal_one.shape[0]
        max_delay_idx = hydrophone_pair.max_delay_idx
        taus = np.linspace((-max_delay_idx-100)/float(audio_array.metadata.sample_rate), (max_delay_idx+100) /float(audio_array.metadata.sample_rate), 2* max_delay_idx + 200)
    
        max_index = np.argmax(corr[mid_index-max_delay_idx-100:mid_index+max_delay_idx+100])
        
        score, score_text = compute_correlation_score(corr[mid_index-max_delay_idx-100:mid_index+max_delay_idx+100],audio_array.metadata.sample_rate)
        
        axs[i].plot(taus, corr[mid_index-max_delay_idx-100:mid_index+max_delay_idx+100], label=f'Canaux {id_one} & {id_two}')
        axs[i].axvline(x=taus[max_index], color='red', linestyle='--', label=f'Max {taus[max_index]}')
        axs[i].set_title(f'GCC PHAT Canaux {id_one} & {id_two} restreinte sur max index')
        axs[i].text(0.05, 0.95, score_text,
            transform=axs[i].transAxes,  # coordonnées relatives à l'axe (0 à 1)
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
        axs[i].legend(loc = 'upper right')

    plt.tight_layout()
    if output_folder is not None:
        plt.savefig(output_folder + f'{audio_array.metadata.tetra_id} GCC PHAT shrinked.png')
    if showfigs:
        plt.show()
    plt.close()

    
def plot_spectro(audio_array : AudioArray, output_folder, showfigs, vmin = None, vmax = None):
    """Trace le spectrogramme de chaque canal d'un tableau audio (4 canaux), se coupe à 30kHz"""
    
    fig, axs = plt.subplots(4, 1, figsize=(30, 20), sharex=True)
    mappable = None
    
    central_freqs = [1516.1,3884.7,2863.7,4003.6]
    min_bands = [557.9, 550.8,523.8,504.0]
    max_bands = [1627.0,4170.6,4746.0,10881.7]
    
    for i, canal in enumerate(audio_array.data_array):
        # Calcul du spectrogramme
        D = librosa.amplitude_to_db(np.abs(librosa.stft(canal, n_fft = 4096, hop_length = 1024)), ref=np.max)
        if vmin is None or vmax is None:
            vmin = np.min(D)
            vmax = np.max(D)
        # Affichage du spectrogramme
        mappable = librosa.display.specshow(
            D,
            y_axis='log',
            x_axis='time',
            ax=axs[i],
            sr=audio_array.metadata.sample_rate,
            cmap='magma',
            hop_length=1024,
            vmin = vmin,
            vmax = vmax
        )
        axs[i].set_title(f'Spectrogramme du Canal {i+1}')
        axs[i].set_ylim(audio_array.metadata.frequency_range)
        #axs[i].axhline(y = central_freqs[i], linestyle = '-', linewidth = 5.0, color = 'tab:green')
        #axs[i].axhline(y = min_bands[i], linestyle = '--', linewidth = 4.0, color = 'black')
        #axs[i].axhline(y = max_bands[i], linestyle = '--',  linewidth = 4.0, color = 'black')


    
        plt.colorbar(mappable, ax=axs[i], format='%+2.0f dB')
    plt.tight_layout()
    
    if output_folder is not None:
        plt.savefig(output_folder + f'{audio_array.metadata.tetra_id} spectrogram.png', dpi = 300)
    if showfigs:
        plt.show()
    plt.close()
    
def plot_wave(audio_array : AudioArray, output_folder, showfigs):
    """Trace le wav de chaque canal d'un tableau audio (4 canaux)."""
    
    fig, axs = plt.subplots(4, 1, figsize=(15, 10), sharex=True, sharey = True)
    
    for i, canal in enumerate(audio_array.data_array):
        padding_idx = 200
        times = np.linspace(padding_idx/float(audio_array.metadata.sample_rate),(len(canal)-padding_idx)/float(audio_array.metadata.sample_rate),len(canal)-2*padding_idx)
        axs[i].plot(times, canal[padding_idx:-padding_idx], color = "#0077b6")
        #axs[i].set_title(f'Wave of channel {i+1}')
        axs[i].patch.set_alpha(0.0)
        axs[i].axis('off')
    fig.patch.set_alpha(0.0)
    plt.tight_layout()
    if output_folder is not None:
        plt.savefig(output_folder + f'{audio_array.metadata.tetra_id} wave.png', dpi = 300)
    if showfigs:
        plt.show()
    plt.close()
    
def plot_3d_pos(audio_arrays : list[AudioArray], environment : Environment, parameters : Parameters, output_folder, showfigs, groundtruth_enu):
    ##### TDOAs and CRBs computation #####
    tdoas_measured = []
    tdoas_error_variance = []
    tdoas_mask = []
    for audio_array in audio_arrays:
        new_tdoa, new_crb, tdoa_mask, _ = tdoas(audio_array, parameters.use_gcc, False)
        tdoas_measured.append(new_tdoa)
        tdoas_error_variance.append(new_crb)
        tdoas_mask.append(tdoa_mask)
    print(f'\n TDOAs : {tdoas_measured}')
    print(f'\n Variances : {tdoas_error_variance}')
    print(f'\n TDOA mask : {tdoas_mask}\n')
    ##### Position computation #####
    if parameters.location_parameters.fusion_type == 'low':
        position_enu, position_error_variance = low_fusion(tdoas_measured, tdoas_error_variance, tdoas_mask, environment, parameters.location_parameters.projection_plan)
        print(f'Low pos : {position_enu}')
    else : #fusion_type == 'high'
        position_enu, position_error_variance = high_fusion(tdoas_measured, tdoas_error_variance, environment, projection_plan = parameters.location_parameters.projection_plan)
        print(f'High pos : {position_enu}')
    output_path_3d = output_folder + '3D position '+parameters.location_parameters.fusion_type+ ' fusion.png'
    output_path_2d = output_folder + '2D position '+parameters.location_parameters.fusion_type+ ' fusion.png'
    error_vector = np.array(position_error_variance).reshape(1,3)
    error_vector = np.zeros((1,3))
    three_dim_plot_enu(np.array(position_enu).reshape(1,3),error_vector, [0], environment, showfigs=showfigs, output_path=output_path_3d, groundtruths = np.array(groundtruth_enu).reshape(1,-1), groundtruths_labels=['Groundtruth'])
    two_dim_plot_enu(np.array(position_enu).reshape(1,3), error_vector, [0], environment, showfigs = showfigs, output_path = output_path_2d, groundtruths = np.array(groundtruth_enu).reshape(1,-1), groundtruths_labels = ['Groundtruth'])