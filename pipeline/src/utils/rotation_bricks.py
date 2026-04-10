"""Two intermediaries functions for alternating lla - enu."""
    
from pyproj import Transformer
import numpy as np
    
def enu2lla(enu_ref, enu_coords):
    # Remark : enu_coords can be an array of coordinates. Each line should be of shape 3
    """Fonction pour convertir ENU centre sur un point de reference a LLA

    Args:
        enu_ref (np.ndarray) : Reference coordinates in lla
        enu_coords (np.ndarray) : Array with enu coords to turn into lla

    Returns:
        lla_coords (np.ndarray) : Array with lla coords from former ENU
    """
    transformer = Transformer.from_crs('epsg:4326', 'epsg:4978')
    X_ref, Y_ref, Z_ref  = transformer.transform(enu_ref[0],enu_ref[1], enu_ref[2])

    lat_ref = np.radians(enu_ref[0])
    lon_ref = np.radians(enu_ref[1])

    R = np.array([[-np.sin(lon_ref), np.cos(lon_ref), 0],
                  [-np.sin(lat_ref)*np.cos(lon_ref), -np.sin(lat_ref)*np.sin(lon_ref), np.cos(lat_ref)],
                  [np.cos(lat_ref)*np.cos(lon_ref), np.cos(lat_ref)*np.sin(lon_ref), np.sin(lat_ref)]])
    
    d = R.T @ enu_coords

    X = d[0] + X_ref
    Y = d[1] + Y_ref
    Z = d[2] + Z_ref

    transformer = Transformer.from_crs("epsg:4978", "epsg:4326", always_xy=True)
    lon, lat, alt = transformer.transform(X, Y, Z)

    return np.array([lat, lon, alt])
    
def lla2enu(enu_ref, lla_coords):
    """Fonction pour convertir LLA en ENU centré sur un point de reference 
    
    Args:
        enu_ref (np.ndarray) : Reference coordinates in lla
        lla_coords (np.ndarray) : Array with lla coords to turn into enu

    Returns:
        tuple: coordonnees enu
    """
    transformer = Transformer.from_crs('epsg:4326', 'epsg:4978')
    x, y, z = transformer.transform(lla_coords[0], lla_coords[1], lla_coords[2])

    transformer = Transformer.from_crs('epsg:4326', 'epsg:4978')
    x_ref, y_ref, z_ref = transformer.transform(enu_ref[0], enu_ref[1], enu_ref[2])

    # Convertir les angles en radians
    lat_ref_rad = np.radians(enu_ref[0])
    lon_ref_rad = np.radians(enu_ref[1])
    
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
    
    return np.array([enu[0], enu[1], enu[2]])

# TODO Ajouter une brique calibration