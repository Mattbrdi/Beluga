from typing import Optional
import numpy as np
from src.utils.sub_classes import Environment

############################################
########## Intermediary functions ##########
############################################

def tdoas_covariances(individual_variance, mixed_error_variance, tdoa_mask):
    covariance_diag = np.diag(mixed_error_variance)
    s1, s2, s3, s4 = individual_variance.ravel()
    covariance_terms = np.array([
        [0, s1, s1, s2, s2, 0],
        [s1, 0, s1, s3, 0, s3],
        [s1, s1, 0, 0, s4, s4],
        [s2, s3, 0, 0, s2, s3],
        [s2, 0, s4, s2, 0, s4],
        [0, s3, s4, s3, s4, 0]
    ])
    full_matrix = covariance_diag + covariance_terms
    valid_idx = np.where(tdoa_mask)[0]
    filtered_matrix = full_matrix[np.ix_(valid_idx, valid_idx)]
    return filtered_matrix


def weight_matrices(tdoas_error_variance : list[np.ndarray], tdoas_mask):
    a_matrix = np.array([
        [1,1,0,0],
        [1,0,1,0],
        [1,0,0,1],
        [0,1,1,0],
        [0,1,0,1],
        [0,0,1,1]
    ])
    individual_variances = [np.linalg.pinv(a_matrix) @ tdoa_error.reshape(-1,1) for tdoa_error in tdoas_error_variance]
    
    covs = [tdoas_covariances(individual_variance, tdoa_error, tdoa_mask) for individual_variance, tdoa_error, tdoa_mask in zip(individual_variances, tdoas_error_variance, tdoas_mask)]
    if np.any(np.abs(np.ravel(tdoas_error_variance)) < 1E-35):
        print('Warning : An error variance is below 1E-35, not supposed to happen')
        weight_matrices = [np.eye(len(tdoa_error[tdoa_mask])) for tdoa_error, tdoa_mask in zip(tdoas_error_variance, tdoas_mask)] # No weighting
    else:
        try :
            # REMARQUE IMPORTANTE : On met pinv car dans le cas où les covariances sont toutes
            #  égales (typiquement le cas avant d'avoir le bruit estimé), la matrice n'est pas
            # inversible. Autrement dit, le np.linalg.inv explose.
            weight_matrices = [np.linalg.pinv(cov) for cov in covs]
        except :
            # A matrix was non inversible
            print("Warning : Weight matrix pinv failed")
            weight_matrices = [np.eye(len(tdoa_error[tdoa_mask])) for tdoa_error, tdoa_mask in zip(tdoas_error_variance, tdoas_mask)]
    return weight_matrices

def bold_v_matrix(v_matrices, weights):
    return np.sum([(v_matrix.T) @ weight @ v_matrix for weight, v_matrix in zip(weights, v_matrices)], axis = 0)
        

def line_matrix(v_matrices, tdoas_filtered, weights):
    return np.hstack([-v_matrix.T @ weight @ tdoa_filtered.reshape(-1,1) for v_matrix, tdoa_filtered, weight in zip(v_matrices, tdoas_filtered, weights)])

def weighted_norm(matrix, weight):
    return (matrix.T)@ weight @ matrix

def schur_inverse(bold_v_mat, line_mat, tdoas_filtered, weights):
    d_inv =  np.diag([1./(weighted_norm(tdoa, weight)+1E-34) for tdoa, weight in zip(tdoas_filtered, weights)]) # D
    # S^{-1} = D^{-1} - D^{-1} L^T (- V + L D^{-1} L^T)^{-1} L D^{-1} 
    middle_mat = (-bold_v_mat + line_mat @ d_inv @ (line_mat.T))
    try:
        schur_inv = d_inv - (d_inv @ (line_mat.T) @ np.linalg.inv(middle_mat) @ line_mat @ d_inv)
    except np.linalg.LinAlgError:
        print("WARNING : Schur inverse failed")
        schur_inv = None
    return schur_inv

def right_terms(v_matrices, weights, tdoas_filtered, environment : Environment):
    """Compute Cx and Cmu"""
    if v_matrices[0].shape[1] == 2: #2D
        origins = [np.array(tetra.origin_enu)[:2].reshape(2,1) for tetra in environment.tetrahedras.values()]
    else: #3D
        origins = [np.array(tetra.origin_enu).reshape(3,1) for tetra in environment.tetrahedras.values()]
    
    ##### Cx #####
    constant_pos = np.sum([(v_matrix.T) @ weight @ v_matrix @ origin for v_matrix, origin, weight in zip(v_matrices, origins, weights)],axis=0)
    
    ##### Cmu #####
    constant_variables = -np.array([tdoa.T @ weight @ v_matrix @ origin for tdoa, v_matrix, origin, weight in zip(tdoas_filtered, v_matrices, origins, weights) ]).reshape(-1,1)
    
    return constant_pos, constant_variables

def tilde_s(inv_bold_v, l_matrix, l_tilde, filtered_tdoas, tdoas_errors, tdoas_mask, weights):
    s1 = (l_matrix.T) @ inv_bold_v @ l_tilde
    s2 = (l_tilde.T) @ inv_bold_v @ l_tilde
    d2 = np.diag([weighted_norm(tdoa_error[mask], weight) for tdoa_error, weight, mask in zip(tdoas_errors, weights, tdoas_mask)])
    d1 = np.diag([(tdoa_error[tdoa_mask].T) @ weight @ tdoa for tdoa_error, tdoa, weight, tdoa_mask in zip(tdoas_errors, filtered_tdoas, weights, tdoas_mask)])
    return d2 + 2*d1 - s1 - s1.T - s2

def tilde_s_inv(s_inv : np.ndarray, s_tilde : np.ndarray):
    s_inv_s_tilde = -s_inv @ s_tilde
    return s_inv @ s_inv_s_tilde, s_inv @ s_inv_s_tilde @ s_inv_s_tilde

def tilde_lambda(l_matrix, l_tilde, schur_inverse, s_itilde_1, s_itilde_2):
    lambda_tilde = (
        (l_matrix + l_tilde) @ (schur_inverse + s_itilde_1) @ ((l_matrix + l_tilde).T)
        - l_matrix @ schur_inverse @ (l_matrix.T)
        - l_tilde @ s_itilde_1 @ (l_tilde.T)
        + l_matrix @ s_itilde_2 @ (l_matrix.T)
    )
    return lambda_tilde

def tilde_gamma(l_matrix, l_tilde, schur_inverse, s_itilde_1, s_itilde_2, cmu, cmu_tilde):
    gamma_tilde = (
        (l_matrix + l_tilde) @ (schur_inverse + s_itilde_1) @ (cmu + cmu_tilde)
        - l_matrix @ schur_inverse @ cmu
        - l_tilde @ s_itilde_1 @ cmu_tilde
        + l_matrix @ s_itilde_2 @ cmu
    )
    return gamma_tilde

def pos_crb_matrix(position : np.ndarray, weight_matrices : list[np.ndarray], tdoas_mask : list[np.ndarray], environment : Environment):
    
    cwater = environment.sound_speed
    all_hydrophones_coordinates = [tetra.rotated_hydro_pos_enu for tetra in environment.tetrahedras.values()]
    differential_taus = []
    
    for tdoa_mask, hydrophones_coordinates in zip(tdoas_mask, all_hydrophones_coordinates):
        
        nb_hydrophones = len(hydrophones_coordinates)

        relative_distances = np.linalg.norm(position.ravel()-hydrophones_coordinates, axis = 1)

        x_diffs = np.array([((position[0] - hydrophones_coordinates[j,0]) / relative_distances[j]) - ((position[0] - hydrophones_coordinates[i,0]) / relative_distances[i]) for i in range(nb_hydrophones) for j in range(i + 1, nb_hydrophones)])[tdoa_mask]
        y_diffs = np.array([((position[1] - hydrophones_coordinates[j,1]) / relative_distances[j]) - ((position[1] - hydrophones_coordinates[i,1]) / relative_distances[i]) for i in range(nb_hydrophones) for j in range(i + 1, nb_hydrophones)])[tdoa_mask]
        z_diffs = np.array([((position[2] - hydrophones_coordinates[j,2]) / relative_distances[j]) - ((position[2] - hydrophones_coordinates[i,2]) / relative_distances[i]) for i in range(nb_hydrophones) for j in range(i + 1, nb_hydrophones)])[tdoa_mask]

        dtk_dX= (1./cwater) * np.column_stack((x_diffs, y_diffs, z_diffs))

        differential_taus.append(dtk_dX)

    fischer_matrix = np.sum([dtau_dx.T @ w @ dtau_dx for dtau_dx, w in zip(differential_taus, weight_matrices)], axis = 0)
    try:
        crb_matrix = np.linalg.inv(fischer_matrix)
    except:
        crb_matrix = np.full((3,3),np.nan) 
    #print(f'CRB Matrix : {crb_matrix}')
    #print(f'Estimated distance error : {np.sqrt(np.sum(np.diag(np.abs(crb_matrix))))}')
    return crb_matrix

###################################
########## Main function ##########
###################################

def low_fusion(tdoas_measured : list[np.ndarray], tdoas_error_variance : list[np.ndarray], tdoas_mask : list[np.ndarray], environment : Environment, projection_plan : Optional[float]):
    ##### Preliminary lists #####
    v_matrices = [tetrahedra.v_matrix[tdoa_mask] for tetrahedra, tdoa_mask in zip(environment.tetrahedras.values(), tdoas_mask)] # Already 2D or 3D depending on use_h4
    tdoas_filtered = [tdoa_measured[tdoa_mask] for tdoa_measured, tdoa_mask in zip(tdoas_measured, tdoas_mask)]
    
    ##### Main matrices #####
    w = weight_matrices(tdoas_error_variance, tdoas_mask)
    bold_v = bold_v_matrix(v_matrices, w)
    v_inv = np.linalg.inv(bold_v)
    l = line_matrix(v_matrices, tdoas_filtered, w)
    s_inv = schur_inverse(bold_v, l, tdoas_filtered, w)
    if s_inv is None:
        return np.full((3,1), np.nan), np.full(3 , np.nan)
    cx, cmu = right_terms(v_matrices, w, tdoas_filtered, environment)

    ##### Position computation #####
    pos = (
        (v_inv + v_inv @ l @ s_inv @ (l.T) @ v_inv) @ cx
        - v_inv @ l @ s_inv @ cmu
    )
    
    error_matrix = pos_crb_matrix(np.array(pos), w, tdoas_mask, environment)
    diag_terms = np.abs(np.diag(error_matrix)) # Sometimes the error is < 0
    
    if np.sqrt(np.sum(diag_terms)) > 1000 :
        print(f'Position unreliable, expecting > 1000m error')
        return np.full((3,1), np.nan), np.full(3 , np.nan)
    
    pos_tilde = np.sqrt(diag_terms)

    return pos, pos_tilde