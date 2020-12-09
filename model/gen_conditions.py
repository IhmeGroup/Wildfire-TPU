"""Generate random conditions for fire models."""

import math
import numpy as np

from model import tensor_utils


def density(fieldShape, mean=1, stdev=0.25):
    """Generate random field of vegetation."""

    normDist = np.random.normal(loc=mean,
                                scale=stdev,
                                size=fieldShape)

    return np.abs(normDist)


def density_bool(fieldShape, burnable_p):
    """Generate boolean density."""

    raw = np.random.rand(*fieldShape)
    density_np = np.less_equal(raw, burnable_p)

    return density_np.astype(np.float32)


def density_patchy(fieldShape, d, p_large, p_small, mean=1, stdev=0.25):
    """Generate patchy vegetation"""

    A_field = fieldShape[0] * fieldShape[1]
    A_patch = np.pi * d**2

    density_np = np.zeros(fieldShape)
    field_withPatches = np.zeros(fieldShape, dtype=bool)
    p_actual = 0

    x = np.arange(fieldShape[0])
    y = np.arange(fieldShape[1])
    X,Y = np.meshgrid(x,y)
    
    while p_actual <= p_large:
        loc = np.round(np.random.rand(2) * np.array(fieldShape))

        X_dist = X - loc[0]
        Y_dist = Y - loc[1]
        R = np.sqrt(X_dist**2 + Y_dist**2)
        
        patch_bool = np.less_equal(R, d/2)
        field_withPatches = np.logical_or(field_withPatches, patch_bool)

        p_actual = np.sum(field_withPatches) / (fieldShape[0] * fieldShape[1])

    density_np = field_withPatches * np.random.normal(loc=mean,
                                                      scale=stdev,
                                                      size=fieldShape)
    density_np *= density_bool(fieldShape, p_small)
    return density_np


def moisture(fieldShape, mean=0, stdev=0.25):
    """Generate random field of vegetation."""

    normDist = np.random.normal(loc=mean,
                                scale=stdev,
                                size=fieldShape)

    return np.abs(normDist)


def terrain_slope(fieldShape, azimuth, elevation):
    """Generate terrain with constant given slope in given direction.
    All angles in radians."""

    planeNorm = np.array([-np.sin(elevation) * np.cos(azimuth),
                          -np.sin(elevation) * np.sin(azimuth),
                          np.cos(elevation)])
    
    x = np.arange(fieldShape[0])
    y = np.arange(fieldShape[1])
    [X, Y] = np.meshgrid(x, y)

    terrain_np = -(planeNorm[0]*X + planeNorm[1]*Y) / planeNorm[2]
    
    return terrain_np.T - np.amin(terrain_np)


def terrain_diamond_step(terrain_np, tempFieldSize, size, half, n, roughness, height):
    """Diamond-square algorithm - diamond step."""

    for i in range(half, tempFieldSize-1, size):
        for j in range(half, tempFieldSize-1, size):

            offset = ((np.random.rand() - 0.5) * roughness * height) / 2**n

            terrain_np[i, j] = np.mean([terrain_np[i + half, j + half],
                                        terrain_np[i + half, j - half],
                                        terrain_np[i - half, j + half],
                                        terrain_np[i - half, j - half]]) + offset

    return terrain_np


def terrain_square_step(terrain_np, tempFieldSize, size, half, n, roughness, height):
    """Diamond-square algorithm - square step."""

    for i in range(0, tempFieldSize, half):
        for j in range((i+half) % size, tempFieldSize, size):

            offset = ((np.random.rand() - 0.5) * roughness * height) / 2**n

            if i == 0:
                terrain_np[i, j] = np.mean([terrain_np[i, j + half],
                                            terrain_np[i, j - half],
                                            terrain_np[i + half, j]]) + offset

            elif i == tempFieldSize-1:
                terrain_np[i, j] = np.mean([terrain_np[i, j + half],
                                            terrain_np[i, j - half],
                                            terrain_np[i - half, j]]) + offset

            elif j == 0:
                terrain_np[i, j] = np.mean([terrain_np[i, j + half],
                                            terrain_np[i + half, j],
                                            terrain_np[i - half, j]]) + offset

            elif j == tempFieldSize-1:
                terrain_np[i, j] = np.mean([terrain_np[i, j - half],
                                            terrain_np[i + half, j],
                                            terrain_np[i - half, j]]) + offset

            else:
                terrain_np[i, j] = np.mean([terrain_np[i, j + half],
                                            terrain_np[i, j - half],
                                            terrain_np[i + half, j],
                                            terrain_np[i - half, j]]) + offset
    
    return terrain_np


def terrain_ds(fieldShape, height, roughness):
    """Generate random terrain using diamond-square algorithm."""

    assert fieldShape[0] == fieldShape[1], """Random terrain only compatible with square fields."""

    nextPow2 = lambda x: math.ceil(math.log2(abs(x)))

    # Padded field size determined by next 2^n + 1
    tempFieldSize = 2**nextPow2(fieldShape[0] - 1) + 1

    # Maximum possible iterations
    iterationCount = nextPow2(tempFieldSize - 1)

    # Initialize field
    terrain_np = np.zeros((tempFieldSize, tempFieldSize))

    # Seed corner values
    terrain_np[0, 0] = np.random.rand() * roughness * height
    terrain_np[-1, 0] = np.random.rand() * roughness * height
    terrain_np[0, -1] = np.random.rand() * roughness * height
    terrain_np[-1, -1] = np.random.rand() * roughness * height

    # Perform diamond-square algorithm

    size = tempFieldSize - 1

    for n in range(iterationCount):

        half = int(size / 2)

        terrain_np = terrain_diamond_step(terrain_np,
                                          tempFieldSize,
                                          size,
                                          half,
                                          n,
                                          roughness,
                                          height)

        terrain_np = terrain_square_step(terrain_np,
                                         tempFieldSize,
                                         size,
                                         half,
                                         n,
                                         roughness,
                                         height)

        size = half

    # Remove padding
    padding = tempFieldSize - fieldShape[0] - 1

    terrain_np = terrain_np[padding : tempFieldSize-1,
                            padding : tempFieldSize-1]

    # Shift terrain vertically by setting lowest point to 0
    return terrain_np - np.amin(terrain_np)


def wind_uniform(fieldShape, components):
    """Generate wind field from constant components."""

    try:
        assert isinstance(components, (list, tuple, np.ndarray))
        assert len(components) == 2

        u_matrix = float(components[0]) * np.ones(fieldShape)
        v_matrix = float(components[1]) * np.ones(fieldShape)

        wind = np.stack((u_matrix, v_matrix), axis=2)

    except (AttributeError, AssertionError):
        raise ValueError("Wind must be a tuple of constants with length 2.")

    return wind


def location_rand(bounds):
    """Generate random coordinates within field."""

    return (np.random.randint(bounds[0][0], bounds[0][1]),
            np.random.randint(bounds[1][0], bounds[1][1]))


def in_domain(fieldShape, pts):
    """Determine which points are in the domain."""

    if isinstance(pts, (list, tuple)):
        pts = np.array(pts)
    if len(pts.shape) == 1:
        pts = pts[np.newaxis,:]

    in_w = pts[:, 0] >= 0
    in_e = pts[:, 0] < fieldShape[0]
    in_s = pts[:, 1] >= 0
    in_n = pts[:, 1] < fieldShape[1]
    return np.all(np.stack([in_w, in_e, in_s, in_n]), axis=0)


def dist(pt1, pt2):
    """Computes the distance between the given points."""

    return np.linalg.norm(pt2 - pt1, axis=0)


def elevation(pt1, pt2):
    """Computes elevation angle of line connecting the given points."""

    return np.arctan2((pt2[1] - pt1[1]), (pt2[0] - pt1[0]))


def rot2d(theta):
    """Generates 2D rotation matrix for given angle."""

    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]])


def lit_fromPts(fieldShape, firePts):
    """Generates lit array from fire coordinates."""

    try:
        assert firePts.shape[1] == 2
    except TypeError:
        raise ValueError("'firePts' must be a numpy array")

    lit_np = np.zeros(fieldShape, dtype=bool)
    for n in range(firePts.shape[0]):
        lit_np[firePts[n, 0], firePts[n, 1]] = True
    
    return lit_np


def lit_lowern(fieldShape, rows):
    """Ignite bottom rows of field."""

    lit_np = np.zeros(fieldShape, dtype=bool)
    lit_np[:,0:rows+1] = True
    return lit_np


def lit_circle(fieldShape, location, size, boundaryCondition):
    """Ignite circular region in field."""

    try:
        assert len(location) == 2, "'location' must have length 2"
        assert in_domain(fieldShape, location), "location outside domain"
    except TypeError:
        raise ValueError("'location' must be a tuple or list")
    
    if size == 0:
        firePts = np.array([location])
    else:
        initStencil = tensor_utils.get_stencil(size)
        firePts = tensor_utils.get_neighbors(location,
                                             initStencil,
                                             fieldShape,
                                             boundaryCondition)
        firePts = np.concatenate(([location], firePts))
    
    return lit_fromPts(fieldShape, firePts)


def lit_line(fieldShape, endPts, thickness, boundaryCondition, endcaps=False):
    """Start line fire between given endpoints."""

    try:
        assert endPts.shape == (2, 2), "'endPts' must have shape (2, 2)"
        assert np.all(in_domain(fieldShape, endPts)),  "endPts outside domain"
    except TypeError:
        raise ValueError("'endPts' must be a numpy array with shape (2, 2)")

    l = dist(endPts[0], endPts[1])
    w = thickness / 2
    t = elevation(endPts[0], endPts[1])

    # Compute transformation matrices
    to_prime = rot2d(-t)
    # to_orig = rot2d(t)

    # Compute bounding box
    box_xlims = np.array([np.min(endPts[:,0]) - w,
                          np.max(endPts[:,0]) + w])
    box_ylims = np.array([np.min(endPts[:,1]) - w,
                          np.max(endPts[:,1]) + w])

    box_xlims = np.round(box_xlims)
    box_ylims = np.round(box_ylims)

    # Get all points within box
    x = np.arange(box_xlims[0], box_xlims[1]+1)
    y = np.arange(box_ylims[0], box_ylims[1]+1)
    [X, Y] = np.meshgrid(x, y)
    pts_orig = np.stack([X.flatten(), Y.flatten()])
    pts_shift = pts_orig - endPts[0,:][np.newaxis, :].T

    # Get coords in prime csys
    pts_prime = np.matmul(to_prime, pts_shift)

    # Get coords within fire region
    masks = np.stack([np.greater_equal(pts_prime[0], 0),
                      np.less_equal(pts_prime[0], l),
                      np.greater_equal(pts_prime[1], -w),
                      np.less_equal(pts_prime[1], w)])
    fireMask = np.all(masks, axis=0)

    # Add endcaps if necessary
    if endcaps:
        dist1 = dist(endPts[0,:][np.newaxis, :].T, pts_orig)
        dist2 = dist(endPts[1,:][np.newaxis, :].T, pts_orig)
        capMask = np.logical_or(np.less_equal(dist1, w),
                                np.less_equal(dist2, w))
        fireMask = np.logical_or(fireMask, capMask)
    
    firePts = pts_orig[:, fireMask].astype(int).T
    firePts = tensor_utils.apply_boundaryCondition(firePts, fieldShape, boundaryCondition)

    return lit_fromPts(fieldShape, firePts)


def lit_curve(fieldShape, curvePts, thickness, boundaryCondition):
    """Start curve fire following points, using recusion and line fires."""

    try:
        assert curvePts.shape[1] == 2, "'curvePts' must have shape (n, 2)"
        assert curvePts.shape[0] >= 2, "curve must have at least 2 points"
        assert np.all(in_domain(fieldShape, curvePts)), "curvePts outside domain"
    except TypeError:
        raise ValueError("'curvePts' must be a numpy array with shape (n, 2)")

    if curvePts.shape[0] == 2:
        return lit_line(fieldShape, curvePts, thickness, boundaryCondition, endcaps=True)
    else:
        return np.logical_or(lit_line(fieldShape,
                                      curvePts[0:2],
                                      thickness,
                                      boundaryCondition,
                                      endcaps=True),
                             lit_curve(fieldShape,
                                       curvePts[1:],
                                       thickness,
                                       boundaryCondition))


def lit_walking(fieldShape, curvePts, thickness, speed, boundaryCondition):

    theta1 = elevation(curvePts[1], curvePts[0])
    thetaN = elevation(curvePts[-2], curvePts[-1])

    curvePts_new = curvePts.copy()

    i = 0
    inbounds = in_domain(fieldShape, curvePts[[0,-1],:])
    while np.any(inbounds):

        lit_np_step = lit_curve(fieldShape, curvePts, thickness, boundaryCondition)
        if i == 0:
            lit_np = lit_np_step[np.newaxis,:]
        else:
            lit_np = np.concatenate([lit_np, lit_np_step[np.newaxis,:]], axis=0)
        
        curvePts_new[0,0] = curvePts[0,0] + speed * np.cos(theta1)
        curvePts_new[0,1] = curvePts[0,1] + speed * np.sin(theta1)
        curvePts_new[-1,0] = curvePts[-1,0] + speed * np.cos(thetaN)
        curvePts_new[-1,1] = curvePts[-1,1] + speed * np.sin(thetaN)
        
        inbounds = in_domain(fieldShape, curvePts_new[[0,-1],:])
        if inbounds[0]:
            curvePts[0] = curvePts_new[0]
        if inbounds[1]:
            curvePts[-1] = curvePts_new[-1]

        i += 1
    
    return lit_np
