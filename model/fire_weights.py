"""Compute fire model weight tensors in numpy."""

import math
import numpy as np


def radius_tensor(kernelShape):
    """Compute r tensor (constant everywhere)"""

    kernelRadius = int((kernelShape[0] - 1) / 2)

    r = np.empty((kernelShape[0], kernelShape[1], 2))
    for i in range(kernelShape[0]):
        for j in range(kernelShape[1]):
            r[i, j, 0] = kernelRadius - j
            r[i, j, 1] = kernelRadius - i
    
    return r


def normalize(tensor):
    """Normalize the given tensor along its last axis."""

    mag = np.linalg.norm(tensor, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        tensor_norm = tensor / np.expand_dims(mag, -1)
    tensor_norm[np.isnan(tensor_norm)] = 0

    return tensor_norm


def pad_tensor_np(tensor_np, kernelShape, boundaryCondition):
    """Pad numpy array according to boundary condition."""

    paddingLayers = int((kernelShape[0] - 1) / 2)
    
    lateralPadding = np.zeros((tensor_np.ndim, 2), dtype=np.int8)
    lateralPadding[0] = [paddingLayers, paddingLayers]

    verticalPadding = np.zeros((tensor_np.ndim, 2), dtype=np.int8)
    verticalPadding[1] = [paddingLayers, paddingLayers]

    bothPadding = np.zeros((tensor_np.ndim, 2), dtype=np.int8)
    bothPadding[0] = [paddingLayers, paddingLayers]
    bothPadding[1] = [paddingLayers, paddingLayers]

    if boundaryCondition == 'infinite':
        tensor_np_padded = np.pad(tensor_np,
                                  bothPadding,
                                  mode='constant',
                                  constant_values=(0))
    
    elif boundaryCondition == 'periodic':
        tensor_np_padded = np.pad(tensor_np,
                                  bothPadding,
                                  mode='wrap')
    
    elif boundaryCondition == 'lateralPeriodic':
        tensor_np_padded = np.pad(tensor_np,
                                  verticalPadding,
                                  mode='constant',
                                  constant_values=(0))
        tensor_np_padded = np.pad(tensor_np_padded,
                                  lateralPadding,
                                  mode='wrap')

    return tensor_np_padded


def get_weight(field, alpha, weight, kernelShape, boundaryCondition):
    """Compute the 4D slope factor array based on the given weight function."""

    # Shape of the field
    fieldShape = field.shape

    # Half-width of the kernel
    kernelRadius = int((kernelShape[0] - 1) / 2)

    # Pad based on boundary condition
    field_padded = pad_tensor_np(field, kernelShape, boundaryCondition)

    # Initialize weight factor tensor
    phi = np.empty((fieldShape[0], fieldShape[1], kernelShape[0], kernelShape[1]))

    # Iterate over the field
    for i in range(fieldShape[0]):
        for j in range(fieldShape[1]):

            # Adjust indices for padding
            i_padded = i + kernelRadius
            j_padded = j + kernelRadius

            # Get the data within the neighborhood
            field_local = field_padded[i_padded - kernelRadius: i_padded + kernelRadius + 1,
                                       j_padded - kernelRadius: j_padded + kernelRadius + 1]
            
            phi[i, j] = weight(field_local, alpha)

    return phi


def slope(slope_local, alpha):
    """Compute slope weight factor."""

    slope_local = np.flip(slope_local, axis=2)

    # Normalized r vectors
    r = radius_tensor(slope_local.shape)
    r_norm = normalize(r)

    # Compute the slope factor
    phi_local = np.exp(alpha * np.einsum('ijk,ijk->ij', r_norm, slope_local))

    return phi_local


def wind(wind_local, alpha):
    """Compute wind weight factor."""

    wind_local = np.flip(wind_local, axis=2)

    # Normalized r vectors
    r = radius_tensor(wind_local.shape)
    r_norm = normalize(r)

    # Normalized U vectors
    mag_u = np.linalg.norm(wind_local, axis=2)
    u_norm = normalize(wind_local)

    # Angle between U and r
    theta = np.arccos(np.einsum('ijk,ijk->ij', u_norm, r_norm))
    
    # Wind factor
    Z = 1 + 0.25 * mag_u
    e = np.sqrt(1 - Z**-2)
    a = mag_u / (1 + e)
    b = a / Z
    gamma = np.sqrt(
        (a**2)*np.sin(theta)**2
        - (a**2)*(e**2)*np.sin(theta)**2
        + (b**2)*np.cos(theta)**2)
    num = a*(b**2)*e*np.cos(theta) + a*b*gamma
    den = (a**2)*np.sin(theta)**2 + (b**2)*np.cos(theta)**2

    with np.errstate(divide='ignore', invalid='ignore'):
        frac = num / den
    frac[np.isnan(frac)] = 0

    psi_local = 1 + alpha * frac

    return psi_local
