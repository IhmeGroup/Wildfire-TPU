"""Utilities for generating convolution kernels."""

import numpy as np
import tensorflow as tf

FLOAT_TYPE = np.float32


def termial(n):
    """Return the sum of i from i=1 to n."""
    if n == 1:
        return 1
    else:
        return n + termial(n-1)


def get_stencil(neighborhoodSize):
    """Generate a list of indices of neighbors for a given neighborhood size."""

    distArray = np.empty([termial(neighborhoodSize+1)-1, 3])

    i = 0

    for x in range(1, neighborhoodSize+1):
        for y in range(0, x+1):

            # Compute distance to origin and fill distArray
            distArray[i] = [x, y, np.linalg.norm([x, y])]

            i = i+1

    # Sort by distance to origin
    distArray_sorted = distArray[distArray[:, 2].argsort()]

    # Select desired closest number of neighbors
    stencil = distArray_sorted[:neighborhoodSize, :2]

    # Convert to integers
    stencil = stencil.astype('int')

    # Add reflection points
    stencil_refxy = np.column_stack((stencil[:, 1], stencil[:, 0]))
    stencil = np.concatenate((stencil, stencil_refxy))

    stencil_refx = np.column_stack((stencil[:, 0], -stencil[:, 1]))
    stencil = np.concatenate((stencil, stencil_refx))

    stencil_refy = np.column_stack((-stencil[:, 0], stencil[:, 1]))
    stencil = np.concatenate((stencil, stencil_refy))

    # Remove duplicates created by reflection
    stencil = np.unique(stencil, axis=0)

    return stencil


def apply_boundaryCondition(pts, fieldShape, boundaryCondition):
    """Apply the given boundary condition to the given points."""

    out_w = pts[:, 0] < 0
    out_e = pts[:, 0] >= fieldShape[0]
    out_s = pts[:, 1] < 0
    out_n = pts[:, 1] >= fieldShape[1]

    if boundaryCondition == 'infinite':

        # Implement wall boundary conditions on all sides
        out = np.logical_or.reduce((out_w, out_e, out_s, out_n))
        pts = pts[np.logical_not(out), :]

    elif boundaryCondition == 'periodic':

        # Implement periodic boundary conditions on all sides
        pts[out_w, 0] = pts[out_w, 0] + fieldShape[0]
        pts[out_e, 0] = pts[out_e, 0] - fieldShape[0]
        pts[out_s, 1] = pts[out_s, 1] + fieldShape[1]
        pts[out_n, 1] = pts[out_n, 1] - fieldShape[1]

    elif boundaryCondition == 'lateralPeriodic':

        # Implement lateral periodic boundary condition
        pts[out_w, 0] = pts[out_w, 0] + fieldShape[0]
        pts[out_e, 0] = pts[out_e, 0] - fieldShape[0]

        # Implement south and north wall boundary conditions
        out_nb = np.logical_or(out_n, out_s)
        pts = pts[np.logical_not(out_nb), :]

    return pts


def pad_tensor(tensor, tensorShape, kernelShape, boundaryCondition):
    """Pad convolution image tensor according to boundary condition."""

    # Number of layers to pad is dictated by kernel size
    paddingLayers = int((kernelShape[0] - 1) / 2) # pylint: disable=E1136  # pylint/issues/3139

    if boundaryCondition == 'infinite':

        # Vertical zero padding
        tensor_padded = tf.concat([tf.zeros([tensorShape[0],
                                             tensorShape[1],
                                             paddingLayers,
                                             1],
                                            dtype=FLOAT_TYPE),
                                   tensor,
                                   tf.zeros([tensorShape[0],
                                             tensorShape[1],
                                             paddingLayers,
                                             1],
                                            dtype=FLOAT_TYPE)],
                                  2)

        # Lateral zero padding
        tensor_padded = tf.concat([tf.zeros([tensorShape[0],
                                             paddingLayers,
                                             tensorShape[2] + 2 * paddingLayers,
                                             1],
                                            dtype=FLOAT_TYPE),
                                   tensor_padded,
                                   tf.zeros([tensorShape[0],
                                             paddingLayers,
                                             tensorShape[2] + 2 * paddingLayers,
                                             1],
                                            dtype=FLOAT_TYPE)],
                                  1)

        return tensor_padded

    elif boundaryCondition == 'periodic':

        # Vertical periodic padding
        tensor_padded = tf.concat([tensor[:, :, -paddingLayers:, :],
                                   tensor,
                                   tensor[:, :, 0:paddingLayers, :]], 2)

        # Lateral periodic padding
        tensor_padded = tf.concat([tensor_padded[:, -paddingLayers:, :, :],
                                   tensor_padded,
                                   tensor_padded[:, 0:paddingLayers, :, :]], 1)

        return tensor_padded

    elif boundaryCondition == 'lateralPeriodic':

        # Lateral periodic padding
        tensor_padded = tf.concat([tensor[:, -paddingLayers:, :, :],
                                   tensor,
                                   tensor[:, 0:paddingLayers, :, :]], 1)

        # Vertical zero padding
        tensor_padded = tf.concat([tf.zeros([tensorShape[0],
                                             tensorShape[1] + 2 * paddingLayers,
                                             paddingLayers,
                                             1],
                                            dtype=FLOAT_TYPE),
                                   tensor_padded,
                                   tf.zeros([tensorShape[0],
                                             tensorShape[1] + 2 * paddingLayers,
                                             paddingLayers,
                                             1],
                                            dtype=FLOAT_TYPE)],
                                  2)

        return tensor_padded


def get_kernel(neighborhoodSize):
    """Generate the kernel matrix for a given neighborhood size."""

    # Compute stencil to get list of indices
    stencil = get_stencil(neighborhoodSize)

    # Determine necessary kernel size
    maxIndex = np.amax(stencil)
    kernelSize = 2 * maxIndex + 1

    # Create kernel array
    kernel = np.zeros((kernelSize, kernelSize))

    # Set kernel elements corresponding to stencil to 1
    kernel[stencil[:, 0] + maxIndex, stencil[:, 1] + maxIndex] = 1

    return kernel


def get_neighbors(location, stencil, fieldShape, boundaryCondition):
    """Get the indices of neighbors of a given point."""

    # Center stencil at location
    neighbors = np.column_stack((stencil[:, 0] + location[0],
                                 stencil[:, 1] + location[1]))

    return apply_boundaryCondition(neighbors, fieldShape, boundaryCondition)


def gen_dynKernel_np(kernel, weights):
    """Compute dynamic kernel given kernel and 4D weight array."""

    # Ensure all weight tensors are correctly shaped
    firstKey = list(weights)[0]
    phi_shape = weights[firstKey].shape

    for key in weights:
        assert weights[key].shape == phi_shape, "Weight '" + key + "' not set properly"

    # Initialize dynamic kernel
    dynKernel = np.zeros((phi_shape[0],
                          phi_shape[1],
                          phi_shape[2] * phi_shape[3]))

    # Iterate over all points in field
    for i in range(phi_shape[0]):
        for j in range(phi_shape[1]):

            # Compute product of all weights with neighborhood kernel
            product = kernel
            for key in weights:
                product = np.multiply(product, weights[key][i, j])
            
            # Set dynamic kernel entry
            dynKernel[i, j] = product.flatten()

    return dynKernel


def conv2d_dynamic(image, dynKernel, kernelShape, padding):
    """Compute convolution using dynamic kernel."""
    
    sizes = [1, kernelShape[0], kernelShape[1], 1]

    # Gather image patches from the input matrix
    patches = tf.image.extract_patches(images=image,
                                       sizes=sizes,
                                       strides=[1, 1, 1, 1],
                                       rates=[1, 1, 1, 1],
                                       padding=padding)

    # Elementwise multiplication of the patches tensor and dynamic kernel
    # Summing the products yields the result of the convolution
    # Note fusion of multiply and reduce for XLA compiler
    conv = tf.reduce_sum(tf.multiply(patches, dynKernel), axis=3)

    # Add dimension to maintain consistency with TensorFlow convention
    return tf.expand_dims(conv, -1)

def set_border(field, val, layers=1):
    """Set borders of field to given value."""

    field_out = field
    field_out[:layers, :] = val
    field_out[-layers:, :] = val
    field_out[:, :layers] = val
    field_out[:, -layers:] = val
    return field_out

def gradient_o1(field, h):
    """Compute first-order accurate 2D gradient."""
    # First order central difference in interior
    # Forward / backward difference at edges

    grad = np.zeros((field.shape[0], field.shape[1], 2))

    grad[1:-2, :, 0] = (field[2:-1, :] - field[0:-3, :]) / (2*h)
    grad[0, :, 0] = (field[1, :] - field[0, :]) / h
    grad[-1, :, 0] = (field[-1, :] - field[-2, :]) / h

    grad[:, 1:-2, 1] = (field[:, 2:-1] - field[:, 0:-3]) / (2*h)
    grad[:, 0, 1] = (field[:, 1] - field[:, 0]) / h
    grad[:, -1, 1] = (field[:, -1] - field[:, -2]) / h

    return grad