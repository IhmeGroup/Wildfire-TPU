"""Run single fire case."""

import numpy as np

from model import wildfire as wf
from model import gen_conditions

outDir = r'./out/'
fieldShape = (100, 50)

lit_1 = gen_conditions.lit_circle(fieldShape, (25, 10), 7, 'infinite')

lit_source_np = np.array([lit_1])

case1 = wf.FireCase(
    fieldShape=fieldShape,
    neighborhoodSize=5,
    ignitionHeat=5,
    burnDuration=5,
    lit_source_np=lit_source_np,
    density_np=gen_conditions.density_bool(fieldShape, 0.5),
    moisture_np=gen_conditions.moisture(fieldShape),
    terrain_np=gen_conditions.terrain_slope(fieldShape, 0, np.pi/4),
    wind_np=gen_conditions.wind_uniform(fieldShape, (10, 5)),
    boundaryCondition='infinite',
    termConditions=['walls'],
    termWalls=['E', 'N'],
    outInterval=1)

# Build batch
batch = wf.FireBatch([case1])
batch.initialize_tensors()

# Run model
wf.simulate(batch)
batch.export_params()
