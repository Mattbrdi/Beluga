from src.location_bricks.low_level_fusion import pos_crb_matrix, weight_matrices, tdoas_covariances, low_fusion
from src.utils.sub_classes import Environment
from src.utils.four_can_generator import tdoa_from_travel_time, travel_times

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from typing import Optional, Callable

##################################
##### Intermediary functions #####
##################################

def generate_true_masks(environment : Environment):
    true_mask = []
    for tetra in environment.tetrahedras.values():
        true_mask.append(np.array(len(tetra.v_matrix)*[True]))
    return true_mask

def generate_constant_variance(estimated_std : float, environment : Environment):
    variances = []
    for tetra in environment.tetrahedras.values():
        variances.append(np.array(len(tetra.v_matrix)*[estimated_std**2]))
    return variances

def compute_heatmap_value(pos : np.ndarray, estimated_tdoa_std : float, environment : Environment, draw_number = None):
    truth_mask = generate_true_masks(environment)
    variances = generate_constant_variance(estimated_tdoa_std, environment)
    w = weight_matrices(variances, truth_mask)
    crb_matrix = pos_crb_matrix(pos, w, truth_mask, environment)
    return np.sqrt(np.sum(np.diag(crb_matrix)))

def monte_carlo_values(pos : np.ndarray, estimated_tdoa_std : float, environment : Environment, draw_number : Optional[int]):
    if draw_number is None:
        draw_number = 10
    arrival_times = [travel_times(tetra, environment.sound_speed, pos) for tetra in environment.tetrahedras.values()]
    true_tdoa_dicts = [tdoa_from_travel_time(arrival_time, name) for arrival_time, name in zip(arrival_times, environment.tetrahedras.keys())]
    tdoa_values = [[tdoa_value for tdoa_value in true_tdoa_dict.values()] for true_tdoa_dict in true_tdoa_dicts]
    # Pay attention to modelling. We should not add a Gaussian noise to all independtly, but a 
    # constant sigma at each hydrophone, resulting in a 2* variance at each pair and cov terms.
    truth_mask = generate_true_masks(environment)
    variances = generate_constant_variance(estimated_tdoa_std, environment)
    a_matrix = np.array([
        [1,1,0,0],
        [1,0,1,0],
        [1,0,0,1],
        [0,1,1,0],
        [0,1,0,1],
        [0,0,1,1]
    ])
    individual_variances = [np.linalg.pinv(a_matrix) @ tdoa_error.reshape(-1,1) for tdoa_error in variances]
    covs = [tdoas_covariances(individual_variance, tdoa_error, tdoa_mask) for individual_variance, tdoa_error, tdoa_mask in zip(individual_variances, variances, truth_mask)]
    random_tdoas = [np.random.multivariate_normal(tdoa, cov, draw_number) for tdoa, cov in zip(tdoa_values, covs)]
    
    all_estimated_pos = []
    
    for i in range(draw_number):
        tdoas = [random_tdoa[i] for random_tdoa in random_tdoas]
        new_pos , _ = low_fusion(tdoas, variances, truth_mask, environment, None)
        all_estimated_pos.append(new_pos.ravel())
    
    mean_distance = np.mean(np.linalg.norm(np.array(all_estimated_pos) - pos.reshape(1,-1), axis = 1))
    
    return float(mean_distance)
    
    
#############################
##### Plotting function #####
#############################
    
def heatmap(
    estimated_tdoa_std : float,
    environment : Environment,
    value_function : Callable[[np.ndarray, float, Environment, Optional[int]], float],
    xmin = -3E3, xmax = 3E3, ymin = -3E3, ymax = 3E3, z = 0.0,
    meshpoints = 300,
    vmax = 1000,
    nb_draws = None,
    save_path = None,
    ):
    
    x = np.linspace(xmin, xmax, meshpoints)
    y = np.linspace(ymin, ymax, meshpoints)
    meshx, meshy = np.meshgrid(x,y)
    mesh_values = np.zeros(meshx.shape)
    
    total_points = meshx.shape[0] * meshx.shape[1]
    
    for i in range(meshx.shape[0]):
        for j in range(meshx.shape[1]):
            pos = np.array([meshx[i,j], meshy[i,j], z])
            mesh_values[i,j] = value_function(pos, estimated_tdoa_std, environment, nb_draws)
        print(f'Computed {(i+1) * meshx.shape[1]}/{total_points} values ({float((i+1) * meshx.shape[1])/total_points * 100 :.2f}%)')

    if vmax is not None:
        mesh_values = np.clip(mesh_values, None, vmax)
    
    plt.figure(figsize=(12, 12))
    plt.contourf(meshx, meshy, mesh_values, levels=20, cmap='jet')
    plt.colorbar(label='Estimated error distance')
    
    ax = plt.gca()
    
    for name, tetra in environment.tetrahedras.items():
        pos = tetra.origin_enu[0:2]
        ax.plot(pos[0], pos[1], marker='^', color='white', markersize=15, markeredgecolor='white')
        # label à côté
        ax.text(pos[0] + 50, pos[1] + 50, name, color='white', fontsize=12)
    
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Distance error heatmap with tetrahedras')
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()

    
if __name__== '__main__':
    environment = Environment('jsons/environments/env_cacouna.json',True)
    estimated_tdoa_std = (1. / 384000) / np.sqrt(12.) # 2 fois fech
    meshpoints = 500 # Actually meshpoints ^2 meshpoints
    n_draws = 500
    # Plot CRB
    save_path_crb = f"c:/Users/pauld/OneDrive/Bureau/Césure/Stage UQO/crb_meshpoints{meshpoints}_errorstd{int(estimated_tdoa_std * 384000)}fech.png"
    heatmap(estimated_tdoa_std,
            environment,
            xmin=-3000,
            xmax=3000,
            ymin = -3000,
            ymax=3000,
            value_function = compute_heatmap_value,
            save_path=save_path_crb, 
            meshpoints = meshpoints)
    # Plot Monte Carlo
    #save_path_mc = f"c:/Users/pauld/OneDrive/Bureau/Césure/Stage UQO/montecarlo_meshpoints{meshpoints}_draws{n_draws}_errorstd{int(estimated_tdoa_std * 384000)}fech.png"
    #heatmap(estimated_tdoa_std, environment, value_function = monte_carlo_values, save_path=save_path_mc, meshpoints = meshpoints, nb_draws=n_draws)

    
    