"""Run basic percolation case with slope and/or wind."""

import os
import numpy as np
import json

from model import wildfire as wf
from model import tensor_utils
from model import gen_conditions

numCases = 1000
casesPerBatch = 100
fieldShape = (110, 110)
neighborhoodSize = 3
nomIgnitionHeat = 3
nomBurnDuration = 3
burnable_p = 0.5

outRoot = r'./out_4'
outDir = outRoot + '_1/'
filePath = lambda run, name, case : outRoot + '_{0}/{1}_{2}.npy'.format(run, name, case)

# Iterate over batches
for initCaseID in range(0, numCases, casesPerBatch):
    batchSize = numCases - initCaseID
    if batchSize > casesPerBatch:
        batchSize = casesPerBatch
    
    # Generate cases and fill batch
    batch = wf.FireBatch()
    for i in range(batchSize):
        loc = gen_conditions.location_rand(((25, 75), (25, 75)))
        lit_init = gen_conditions.lit_circle(fieldShape, loc, 5, 'infinite')
        lit_source_np = np.expand_dims(lit_init, axis=0)

        # Field has some empty spaces dictated by burnable_p. Where there are trees, random density
        # density_np_bool = gen_conditions.density_bool(fieldShape, burnable_p)
        # density_np_raw = gen_conditions.density(fieldShape, 1, 0.25)
        # density_np = density_np_bool * density_np_raw
        # density_np = tensor_utils.set_border(density_np, 0, 5)
        
        density_np = gen_conditions.density_patchy(fieldShape, 8, 0.7, 0.6, 1, 0.25)
        density_np = tensor_utils.set_border(density_np, 0, 5)

        # moisture_np = np.zeros(fieldShape)
        moisture_np = gen_conditions.moisture(fieldShape, 0, 0.25)

        # terrain_np = np.zeros(fieldShape)
        # terrain_np = gen_conditions.terrain_slope(fieldShape,
        #                                           np.random.uniform(0, 2*np.pi),
        #                                           np.random.uniform(0, np.pi/4))
        terrain_np = gen_conditions.terrain_ds(fieldShape, 50, 0.5)

        # wind_np = gen_conditions.wind_uniform(fieldShape, (0, 0))
        wind_np = gen_conditions.wind_uniform(fieldShape, np.random.uniform(-12, 12, 2))

        case = wf.FireCase(fieldShape=fieldShape,
                           neighborhoodSize=neighborhoodSize,
                           nomIgnitionHeat=nomIgnitionHeat,
                           nomBurnDuration=nomBurnDuration,
                           lit_source_np=lit_source_np,
                           density_np=density_np,
                           moisture_np=moisture_np,
                           terrain_np=terrain_np,
                           wind_np=wind_np,
                           boundaryCondition='infinite',
                           termConditions=['exhausted', 'step'],
                           termStep=30,
                           initCaseID=initCaseID,
                           outDir=outDir,
                           outInterval=1)
        batch.append(case)

    # Run model
    batch.initialize_tensors()
    (buildTime, runTime, fireDuration) = wf.simulate(batch)
    batch.export_params(append=True)
    wf.export_perfStats(buildTime, runTime, fireDuration, outDir=outDir, append=True)

    print("{0} cases completed...".format(initCaseID + batchSize))
    print("")

print("Changing wind direction and restarting...")
print("")

with open(outDir + "perfStats.json") as f:
    perfStats = json.load(f)
outDir = outRoot + '_2/'

for initCaseID in range(0, numCases, casesPerBatch):
    batchSize = numCases - initCaseID
    if batchSize > casesPerBatch:
        batchSize = casesPerBatch
    
    batch = wf.FireBatch()
    for i in range(batchSize):
        ID = i + initCaseID

        if perfStats['fireDuration'][ID] == 0:
            lit_init = np.zeros(fieldShape)
            lit_source_np = np.expand_dims(lit_init, axis=0)
            fire_np = np.zeros(fieldShape)
            density_np = np.load(filePath(1, "density", ID))

        else:
            # Use final state from previous run as initial condition
            lit_fromFile = np.load(filePath(1, "lit", ID))
            lit_init = lit_fromFile[-1,:,:]
            lit_source_np = np.expand_dims(lit_init, axis=0)
    
            # Rebuild density, removing burnt trees
            fire_np = np.load(filePath(1, "fire", ID))
            burnDuration_np = np.load(filePath(1, "burnDuration", ID))
            density_np = np.load(filePath(1, "density", ID))
            density_np = np.where(fire_np[-1] == burnDuration_np, 0, density_np)

        terrain_np = np.load(filePath(1, "terrain", ID))
        wind_np = gen_conditions.wind_uniform(fieldShape, np.random.uniform(-12, 12, 2))
        case = wf.FireCase(fieldShape=fieldShape,
                           neighborhoodSize=neighborhoodSize,
                           nomIgnitionHeat=nomIgnitionHeat,
                           nomBurnDuration=nomBurnDuration,
                           lit_source_np=lit_source_np,
                           density_np=density_np,
                           moisture_np=np.zeros(fieldShape),
                           terrain_np=terrain_np,
                           wind_np=wind_np,
                           boundaryCondition='infinite',
                           termConditions=['exhausted'],
                           initCaseID=initCaseID,
                           outDir=outDir,
                           outInterval=1)
        batch.append(case)
    
    # Run model
    batch.initialize_tensors()
    (buildTime, runTime, fireDuration) = wf.simulate(batch)
    batch.export_params(append=True)
    wf.export_perfStats(buildTime, runTime, fireDuration, outDir=outDir, append=True)

    print("{0} cases completed...".format(initCaseID + batchSize))
    print("")

print("Merging runs...")
print("")

outDir = outRoot + '_3/'
if not os.path.exists(outDir):
    os.makedirs(outDir)

for initCaseID in range(0, numCases, casesPerBatch):
    batchSize = numCases - initCaseID
    if batchSize > casesPerBatch:
        batchSize = casesPerBatch
    
    batch = wf.FireBatch()
    for i in range(batchSize):
        ID = i + initCaseID

        for var in ["density", "moisture", "terrain", "ignitionHeat", "burnDuration"]:
            shutil.copyfile(filePath(1, var, ID), filePath(3, var, ID))

        if perfStats['fireDuration'][ID] < 30:
            shutil.copyfile(filePath(1, "lit", ID), filePath(3, "lit", ID))
            shutil.copyfile(filePath(1, "fire", ID), filePath(3, "fire", ID))
            wind_single = np.load(filePath(1, "wind", ID))
            wind = np.repeat(wind_single[np.newaxis, :], perfStats['fireDuration'][ID], axis=0)
            np.save(filePath(3, "wind", ID), wind)

        else:
            lit_1 = np.load(filePath(1, "lit", ID))
            lit_2 = np.load(filePath(2, "lit", ID))
            lit = np.concatenate((lit_1, lit_2[1:]), axis=0)
            np.save(filePath(3, "lit", ID), lit)

            fire_1 = np.load(filePath(1, "fire", ID))
            fire_2 = np.load(filePath(2, "fire", ID))
            fire = np.concatenate((fire_1, fire_2[1:]), axis=0)
            np.save(filePath(3, "fire", ID), fire)

            shape_1 = lit_1.shape
            shape_2 = lit_2.shape
            fireDuration_1 = shape_1[0]
            fireDuration_2 = shape_2[0]

            wind_1_single = np.load(filePath(1, "wind", ID))
            wind_2_single = np.load(filePath(2, "wind", ID))
            wind_1 = np.repeat(wind_1_single[np.newaxis, :], fireDuration_1, axis=0)
            wind_2 = np.repeat(wind_2_single[np.newaxis, :], fireDuration_2, axis=0)
            wind = np.concatenate((wind_1, wind_2[1:]), axis=0)
            np.save(filePath(3, "wind", ID), wind)

