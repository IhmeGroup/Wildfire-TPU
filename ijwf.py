"""Run basic percolation case with slope and/or wind."""

import numpy as np

from model import wildfire as wf
from model import tensor_utils
from model import gen_conditions

numCases = 1000
casesPerBatch = 100
outDir = r'./out_2/'
fieldShape = (110, 110)
burnable_p = 0.5

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
        density_np_bool = gen_conditions.density_bool(fieldShape, burnable_p)
        density_np_raw = gen_conditions.density(fieldShape, 1, 0.25)
        density_np = density_np_bool * density_np_raw
        density_np = tensor_utils.set_border(density_np, 0, 5)

        # terrain_np = np.zeros(fieldShape)
        terrain_np = gen_conditions.terrain_slope(fieldShape,
                                                  np.random.uniform(0, 2*np.pi),
                                                  np.random.uniform(0, np.pi/4))

        # wind_np = gen_conditions.wind_uniform(fieldShape, (0, 0))
        wind_np = gen_conditions.wind_uniform(fieldShape, np.random.uniform(-7, 7, 2))

        case = wf.FireCase(fieldShape=fieldShape,
                           neighborhoodSize=3,
                           nomIgnitionHeat=3,
                           nomBurnDuration=3,
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
