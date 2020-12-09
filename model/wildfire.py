"""Dynamic percolation fire spread model, with added weights.
Implemented using TensorFlow to enable execution on TPUs.
"""

import os
import time
import json
import numpy as np
import tensorflow as tf
from numpy.lib.format import open_memmap

from model import tensor_utils
from model import fire_weights

DEFAULTS_MODELCONFIGS = {
    'neighborhoodSize': 3,
    'boundaryCondition': 'infinite',
    'termConditions': ['exhausted'],
    'termWalls': [],
    'termStep': np.inf,
    'lengthScale': 1,
    'timeScale': 1}

DEFAULTS_WEIGHTPARAMETERS = {
    'nomIgnitionHeat': 3,
    'nomBurnDuration': 3,
    'moistureAlpha': 1,
    'slopeAlpha': 0.7,
    'windAlpha': 2}

DEFAULTS_RUNCONFIGS = {
    'useTPU': False,
    'runIters': 1,
    'outInterval': None,
    'initCaseID': 0,
    'outDir': r'./out/'}

DEFAULTS = {**DEFAULTS_MODELCONFIGS,
            **DEFAULTS_WEIGHTPARAMETERS,
            **DEFAULTS_RUNCONFIGS}

PARAMS_NODEFAULT = ['fieldShape']

FIELDCONDS = ['lit_source_np',
              'density_np',
              'moisture_np',
              'terrain_np',
              'wind_np']

CHOICES = {'boundaryCondition': ['infinite', 'periodic', 'lateralPeriodic'],
           'termConditions': ['exhausted', 'walls', 'step'],
           'termWalls': ['W', 'E', 'S', 'N']}

PRE_TENSORS = ['dynKernel_np',
               'lit_np',
               'heat_np',
               'fire_np',
               'ignitionHeat_np',
               'burnDuration_np']

BATCH_SHARED_PARAMS = [*{**DEFAULTS_MODELCONFIGS, **DEFAULTS_RUNCONFIGS}] + ['fieldShape', 'kernelShape']

FLOAT_TYPE = np.float32
MEMORY_LIMIT = 8 * 10**9

EXPORT_PARAMS = [key for key in [*DEFAULTS] + PARAMS_NODEFAULT if key not in ['initCaseID']]

EXPORT_FIELDS = ['density',
                 'moisture',
                 'burnDuration',
                 'ignitionHeat',
                 'terrain',
                 'wind']

def fire_active_np(lit, termConditions, termWalls):
    """Determine whether any batch in the case is active."""

    # Fire has exhausted if lit is entirely False for a given field
    if 'exhausted' in termConditions:
        exhausted = np.logical_not(np.any(lit, axis=(1,2)))
    else:
        exhausted = False
    
    # Fire has exhausted if lit is True at any of the selected walls
    if 'walls' in termConditions:
        wall_w = np.any(lit[:, 0, :], axis=1)
        wall_e = np.any(lit[:, -1, :], axis=1)
        wall_s = np.any(lit[:, :, 0], axis=1)
        wall_n = np.any(lit[:, :, -1], axis=1)
        walls = np.zeros_like(wall_w[np.newaxis,:], dtype=bool)
        for i in termWalls:
            if i == 'W':
                walls = np.concatenate([walls, wall_w[np.newaxis,:]], axis=0)
            elif i == 'E':
                walls = np.concatenate([walls, wall_e[np.newaxis,:]], axis=0)
            elif i == 'S':
                walls = np.concatenate([walls, wall_s[np.newaxis,:]], axis=0)
            elif i == 'N':
                walls = np.concatenate([walls, wall_n[np.newaxis,:]], axis=0)
        penetrated = np.any(walls, axis=0)
    else:
        penetrated = False

    return np.logical_not(np.logical_or(exhausted, penetrated))


def fire_active(lit, termConditions, termWalls):
    """Determine whether each fire in the batch is active."""

    # Fire has exhausted if lit is entirely False for a given field
    if 'exhausted' in termConditions:
        exhausted = tf.logical_not(tf.reduce_any(lit, [1, 2, 3]))
    else:
        exhausted = False

    if 'walls' in termConditions:
        wall_w = tf.reduce_any(lit[:, 0, :, :], axis=[1, 2])
        wall_e = tf.reduce_any(lit[:, -1, :, :], axis=[1, 2])
        wall_s = tf.reduce_any(lit[:, :, 0, :], axis=[1, 2])
        wall_n = tf.reduce_any(lit[:, :, -1, :], axis=[1, 2])
        walls = tf.zeros_like(tf.expand_dims(wall_w, 0), dtype=bool)
        for i in termWalls:
            if i == 'W':
                walls = tf.concat([walls, tf.expand_dims(wall_w, 0)], axis=0)
            elif i == 'E':
                walls = tf.concat([walls, tf.expand_dims(wall_e, 0)], axis=0)
            elif i == 'S':
                walls = tf.concat([walls, tf.expand_dims(wall_s, 0)], axis=0)
            elif i == 'N':
                walls = tf.concat([walls, tf.expand_dims(wall_n, 0)], axis=0)
        penetrated = tf.reduce_any(walls, axis=0)
    else:
        penetrated = False

    return tf.logical_not(tf.logical_or(exhausted, penetrated))


class FireCase(object):
    """Class representing a fire case."""


    def __init__(self, **kwargs):
        """Initialize the fire case.

        Keyword arguments:

        Model configuration:
        fieldShape          -- (y, x) tuple size of field
        neighborhoodSize    -- interaction radius, nth nearest neighbor
        boundaryCondition   -- boundary conditions, either 'infinite', 'periodic', or 'lateralPeriodic'
        termConditions      -- list of termination conditions, either 'exhausted', 'wall', or 'step'
        termWalls           -- list of walls to evaluate penetration condition
        termStep            -- step at which to stop simulation

        Initial conditions:
        lit_source_np       -- boolean numpy array of fire source term

        Field conditions:
        density_np          -- numpy array of density
        moisture_np         -- numpy array of moisture content
        terrain_np          -- numpy array of terrain
        wind_np             -- numpy array of wind

        Weight parameters:
        nomIgnitionHeat     -- heat to cause ignition of dry fuel when density = 1
        nomBurnDuration     -- burn duration of fuel when density = 1
        moistureAlpha       -- ignition heat sensitivity to moisture
        slopeAlpha          -- heat transfer sensitivity to slope
        windAlpha        -- heat transfer sensitivity to wind magnitude

        Run configuration - These must agree for all FireCases in a FireBatch:
        useTPU              -- bool indicating whether to use TPU
        runIters            -- Number of repeated cases to run (for benchmarking)
        outInterval         -- interval of steps to run between output, or None
        initCaseID          -- ID of the first case in the FireBatch
        outDir              -- path to output directory
        """

        print("Building case...")
        
        # Parse inputs
        for key in kwargs:
            setattr(self, key, kwargs[key])
        for key in DEFAULTS:
            if key not in kwargs:
                setattr(self, key, DEFAULTS[key])

        # Check mode selection strings
        for key in CHOICES:
            if isinstance(getattr(self, key), (list, tuple)):
                for item in getattr(self, key):
                    assert item in CHOICES[key], "Invalid definition for '" + key + "'"
            else:
                assert getattr(self, key) in CHOICES[key], "Invalid definition for '" + key + "'"

        self.weights = {'slope': None,
                        'wind': None}

        # Generate fire spread kernel
        kernel_np = tensor_utils.get_kernel(self.neighborhoodSize)
        self.kernelShape = kernel_np.shape

        # Set scalar weight factors
        self.set_ignitionHeat(self.density_np, self.moisture_np)
        self.set_burnDuration(self.density_np)
        
        # Set vector weight factors
        self.set_weight_slope(self.terrain_np)
        self.set_weight_wind(self.wind_np)

        # Generate dynamic kernel
        self.dynKernel_np = tensor_utils.gen_dynKernel_np(kernel_np, self.weights)

        # Set initial fire
        self.start_fire(self.lit_source_np)
        
        # Create output directory if necessary
        if not os.path.exists(self.outDir):
            os.makedirs(self.outDir)
    

    def parse_field_np(self, field_np, name, arrShape=None, ignore=[]):
        """Parse input field array."""

        if arrShape == None:
            arrShape = self.fieldShape

        try:
            assert field_np.ndim == len(arrShape), "'" + name + "' has invalid dimensionality"
            for i in range(len(field_np.shape)):
                if i not in ignore:
                    assert field_np.shape[i] == arrShape[i], "'" + name + "' must have shape " + str(arrShape)
        except AttributeError:
            raise ValueError(name + " must be a numpy array")
    

    def set_ignitionHeat(self, density_np, moisture_np):
        """Define the ignition heat for the field."""

        self.parse_field_np(density_np, 'density')
        self.density_np = density_np

        self.parse_field_np(moisture_np, 'moisture')
        self.moisture_np = moisture_np

        dryIgHeat = self.nomIgnitionHeat * self.density_np
        self.ignitionHeat_np = dryIgHeat + (self.moistureAlpha * self.moisture_np)
        self.ignitionHeat_np[np.equal(self.density_np, 0)] = float('inf')
    

    def set_burnDuration(self, density_np):
        """Define the burn duration for the field."""

        self.parse_field_np(density_np, 'density')
        self.density_np = density_np

        self.burnDuration_np = self.nomBurnDuration * self.density_np


    def set_weight_slope(self, terrain_np):
        """Define the terrain for the field."""

        self.parse_field_np(terrain_np, 'terrain')
        self.terrain_np = terrain_np
        slope_np = tensor_utils.gradient_o1(self.terrain_np, 1)

        # Compute slope factor tensor
        self.weights['slope'] = fire_weights.get_weight(slope_np,
                                                        self.slopeAlpha,
                                                        fire_weights.slope,
                                                        self.kernelShape,
                                                        self.boundaryCondition)


    def set_weight_wind(self, wind_np):
        """Define the wind for the field."""

        windShape = (self.fieldShape[0],
                     self.fieldShape[1],
                     2)

        self.parse_field_np(wind_np, 'wind', arrShape=windShape)
        self.wind_np = wind_np

        # Compute wind weight tensor
        self.weights['wind'] = fire_weights.get_weight(self.wind_np,
                                                       self.windAlpha,
                                                       fire_weights.wind,
                                                       self.kernelShape,
                                                       self.boundaryCondition)


    def start_fire(self, lit_source_np):
        """Set the initial fire."""

        sourceShape = (1,) + self.fieldShape
        self.parse_field_np(lit_source_np,
                            'lit_source',
                            arrShape=sourceShape,
                            ignore=(0,))
        
        self.lit_np = lit_source_np[0]
        self.fire_np = np.zeros(self.fieldShape)

        self.heat_np = np.zeros(self.fieldShape)
        self.heat_np[np.where(self.lit_np)] = self.ignitionHeat_np[np.where(self.lit_np)]


    def export_params(self):
        """Export model parameters to json file in output directory."""

        params = dict()
        for key in EXPORT_PARAMS:
            params[key] = getattr(self, key)

        with open(self.outDir + 'params.json', 'w') as outFile:
            json.dump(params, outFile)
    

    def export_fields(self):
        """Export field parameters to files in output directory."""

        for field in EXPORT_FIELDS:
            np.save(self.outDir + field + '.npy',
                    getattr(self, field + '_np'))


class FireBatch(object):
    """Class describing a batch of fire cases."""

    def __init__(self, cases=None):
        """Initialize the fire case batch.

        Model and run configurations (as defined earlier in the FireCase
        definition) must be identical for all FireCases used to build the
        FireBatch.
        """

        self.batchSize = 0
        self.tensorsBuilt = False

        if cases is not None:
            assert isinstance(cases, (list, tuple)), "Constructor argument must be a list or tuple of cases"
            self.extend(cases)
    

    def append_source(self, source_batch, source):
        """Append source term arrays."""

        # New case has more time steps than batch. Pad batch appropriately.
        if source_batch.shape[1] < source.shape[0]:
            padShape = (1,
                        source.shape[0] - source_batch.shape[1],
                        source_batch.shape[2],
                        source_batch.shape[3])
            pad = np.zeros(padShape, dtype=bool)
            source_batch = np.concatenate([source_batch, pad], axis=1)

        # New case has fewer time steps than batch. Pad new case appropriately.
        elif source_batch.shape[1] > source.shape[0]:
            padShape = (source_batch.shape[1] - source.shape[0],
                        source_batch.shape[2],
                        source_batch.shape[3])
            pad = np.zeros(padShape, dtype=bool)
            source = np.concatenate([source, pad], axis=0)

        return np.concatenate([source_batch,
                               source[np.newaxis,:]], axis=0)

    
    def __append_zeroth(self, case):
        """Append a case onto the empty batch."""

        for key in vars(case).keys():

            # Set shared params
            if key in BATCH_SHARED_PARAMS:
                setattr(self, key, getattr(case, key))
            
            # Set relevant numpy arrays - will be used to construct tensors
            elif key in PRE_TENSORS + ['lit_source_np']:
                setattr(self, key, np.expand_dims(getattr(case, key), axis=0))
            
            # Create lists of other attributes
            else:
                setattr(self, key, [getattr(case, key)])
        
        self.batchSize = 1
        self.tensorsBuilt = False


    def append(self, case):
        """Append a single case onto the batch."""
        
        if self.batchSize == 0:
            self.__append_zeroth(case)
        
        else:
            for key in vars(case).keys():

                # Verify configurations match
                if key in BATCH_SHARED_PARAMS:
                    if getattr(self, key) != getattr(case, key):
                        raise ValueError(key + " of FireCase to append does not match the value for this FireBatch")

                elif key == 'lit_source_np':
                    self.lit_source_np = self.append_source(self.lit_source_np, case.lit_source_np)
                
                elif key in PRE_TENSORS:
                    setattr(self,
                            key,
                            np.concatenate((getattr(self, key),
                                            np.expand_dims(getattr(case, key), axis=0)),
                                           0))

                # Append other key values to corresponding lists
                else:
                    getattr(self, key).append(getattr(case, key))

            self.batchSize += 1
        self.tensorsBuilt = False


    def extend(self, cases):
        """Extend batch with a list of cases."""

        for case in cases:
            self.append(case)
        
        self.tensorsBuilt = False
    

    def append_params(self, params1, params2):
        """Append two sets of batch parameters."""

        params = dict()
        for key in EXPORT_PARAMS + ['batchSize']:
            if key in BATCH_SHARED_PARAMS:
                assert params1[key] == params2[key], key + " of params to append must be identical"
                params[key] = params1[key]
            else:
                params[key] = params1[key] + params2[key]
        
        return params


    def export_params(self, append=False):
        """Export model parameters to json file in output directory."""

        params = dict()
        for key in EXPORT_PARAMS + ['batchSize']:
            params[key] = getattr(self, key)
            if isinstance(params[key], tuple):
                params[key] = list(params[key])

        jsonPath = self.outDir + 'params.json'
        append = append and os.path.exists(jsonPath)

        if append:
            with open(jsonPath, 'r+') as outFile:
                params_old = json.load(outFile)
                outFile.seek(0, 0)
                outFile.truncate()
                params = self.append_params(params_old, params)
                json.dump(params, outFile)
        else:
            with open(jsonPath, 'w') as outFile:
                json.dump(params, outFile)


    def export_fields(self):
        """Export field parameters to files in output directory."""

        for i in range(self.batchSize):
            for field in EXPORT_FIELDS:
                fieldData = getattr(self, field + '_np')
                np.save(self.outDir + field + '_{0}.npy'.format(i + self.initCaseID),
                        fieldData[i])


    def initialize_tensors(self):
        """Initialize the TensorFlow tensors."""

        self.epochDuration = tf.Variable(0)
        self.fireDuration_init = tf.zeros([self.batchSize], dtype=np.int32)
        self.fireDuration = self.fireDuration_init

        self.dynKernel = tf.constant(self.dynKernel_np.astype(FLOAT_TYPE))
        self.lit_source = tf.constant(self.lit_source_np.astype(np.bool))

        # Add extra False timestep
        pad = tf.zeros_like(self.lit_source[:,0], dtype=np.bool)
        pad = tf.expand_dims(pad, 1)
        self.lit_source = tf.concat([self.lit_source, pad], axis=1)

        self.tensorShape = [self.batchSize, self.fieldShape[0], self.fieldShape[1], 1]

        self.lit_init = tf.Variable(tf.reshape(self.lit_np.astype(np.bool), self.tensorShape))
        self.heat_init = tf.Variable(tf.reshape(self.heat_np.astype(FLOAT_TYPE), self.tensorShape))
        self.fire_init = tf.Variable(tf.reshape(self.fire_np.astype(FLOAT_TYPE), self.tensorShape))

        self.ignitionHeat = tf.Variable(tf.reshape(self.ignitionHeat_np.astype(FLOAT_TYPE), self.tensorShape))
        self.burnDuration = tf.Variable(tf.reshape(self.burnDuration_np.astype(FLOAT_TYPE), self.tensorShape))
        self.lit = self.lit_init
        self.heat = self.heat_init
        self.fire = self.fire_init

        self.tensorsBuilt = True


    def build_graph(self):
        """Construct while_loop graph."""

        if not self.tensorsBuilt:
            raise Exception("Tensors for this FireBatch have not yet been initialized.")


        def burn_step(lit, heat, fire, epochDuration, fireDuration):
            """Propagate the fire for one time step."""

            # Increment fire where lit
            fire = tf.where(lit, tf.add(fire, 1), fire)

            # Find where burn duration has been reached
            burnt = tf.greater_equal(fire, self.burnDuration)

            # Get lit source for current step if still available, else False tensor
            step = tf.reduce_min([tf.reduce_max(fireDuration)+1,
                                  self.lit_source.shape[1]-1])
            lit_source_step = self.lit_source[:,step]

            # Add lit source term
            lit = tf.logical_or(lit,
                                tf.expand_dims(lit_source_step, -1))

            # Prepare image tensor
            lit_float = tf.dtypes.cast(lit, tf.float32)

            # Pad tensor according to boundary condition
            lit_float_padded = tensor_utils.pad_tensor(lit_float,
                                                       self.tensorShape,
                                                       self.kernelShape,
                                                       self.boundaryCondition)

            # Compute dynamic kernel convolution to compute heat added at each point
            # Using VALID padding removes the initial padding automatically
            heatAdded = tensor_utils.conv2d_dynamic(lit_float_padded,
                                                    self.dynKernel,
                                                    self.kernelShape,
                                                    padding='VALID')

            # Add new heat
            heat = tf.add(heat, heatAdded)

            # Ignite cells where heat is sufficient, as long as they are not already burnt
            lit = tf.logical_and(tf.greater_equal(heat, self.ignitionHeat),
                                 tf.logical_not(burnt))

            # Increment step count for cases where fire is still active
            fireDuration = tf.where(fire_active(lit, self.termConditions, self.termWalls),
                                    tf.add(fireDuration, 1),
                                    fireDuration)
            epochDuration = epochDuration + 1

            return [lit, heat, fire, epochDuration, fireDuration]


        def fire_active_batch(lit, heat, fire, epochDuration, fireDuration): #pylint: disable=unused-argument
            """Determine whether at least one of the fires in the batch is active."""

            if any(cond in ['exhausted', 'walls'] for cond in self.termConditions):
                active = tf.reduce_any(fire_active(lit, self.termConditions, self.termWalls))
            else:
                active = True
            
            if 'step' in self.termConditions:
                timeout = tf.reduce_any(tf.greater_equal(fireDuration, self.termStep))
            else:
                timeout = False

            if self.outInterval is None:
                output = False
            else:
                output = tf.greater_equal(epochDuration, self.outInterval)
            
            activeAndNotTimeout = tf.logical_and(active, tf.logical_not(timeout))
            return tf.logical_and(activeAndNotTimeout, tf.logical_not(output))


        def construct_graph(lit, heat, fire, epochDuration, fireDuration):
            """Graph constructor function for use with tpu.rewrite."""

            return tf.while_loop(fire_active_batch, burn_step, [lit,
                                                                heat,
                                                                fire,
                                                                epochDuration,
                                                                fireDuration])


        if self.useTPU:
            [self.lit,
             self.heat,
             self.fire,
             self.epochDuration,
             self.fireDuration] = tf.compat.v1.tpu.rewrite(construct_graph,
                                                           [self.lit,
                                                            self.heat,
                                                            self.fire,
                                                            self.epochDuration,
                                                            self.fireDuration])
        else:
            [self.lit,
             self.heat,
             self.fire,
             self.epochDuration,
             self.fireDuration] = construct_graph(self.lit,
                                                  self.heat,
                                                  self.fire,
                                                  self.epochDuration,
                                                  self.fireDuration)


def append_memmap(file, array):

    if array.shape[0] == 0:
        return
    
    # Load file as memory-mapped array
    current_map = np.load(file, mmap_mode='r')
    fileShape = current_map.shape

    assert fileShape[1] == array.shape[1], "File and array have incompatible shapes"
    assert fileShape[2] == array.shape[2], "File and array have incompatible shapes"

    # Create new file as memory-mapped array with new shape
    new_map = open_memmap(file + '.tmp',
                          mode='w+',
                          dtype=array.dtype,
                          shape=(fileShape[0] + array.shape[0],
                                 fileShape[1],
                                 fileShape[2]))

    # Add new array to new file
    new_map[:fileShape[0], :, :] = current_map
    new_map[-array.shape[0]:, :, :] = array

    del current_map
    del new_map

    os.rename(file + '.tmp', file)


def end_step(step, currentIter, fireDuration):
    """Determine index of final step to write"""

    endStep = -(step-currentIter-fireDuration) - 1
    endStep[endStep < 0] = 0
    return endStep


def write_output(lit_arr, fire_arr, endStep, outDir, initCaseID, overwrite=False):
    """Write lit and fire arrays to output files."""

    for j in range(endStep.shape[0]):

        litFile = outDir + 'lit_{0}.npy'.format(j + initCaseID)
        fireFile = outDir + 'fire_{0}.npy'.format(j + initCaseID)

        if overwrite:
            np.save(litFile, lit_arr[:endStep[j], j])
            np.save(fireFile, fire_arr[:endStep[j], j])
        else:
            append_memmap(litFile, lit_arr[:endStep[j], j])
            append_memmap(fireFile, fire_arr[:endStep[j], j])


def run_batch(batch, sess):
    """Run given FireBatch in given TensorFlow session."""

    # Build graph
    print("Building graph...")
    startBuildTime = time.time()
    batch.build_graph()
    buildTime = time.time() - startBuildTime

    # Evaluate model
    print("Evaluating model...")

    if batch.outInterval is None:

        # Run in single sweep. (Far better performance, no output)
        startRunTime = time.time()
        fireDuration = sess.run(batch.fireDuration)
        runTime = time.time() - startRunTime

    else:

        # Write field matrices to disk
        batch.export_fields()

        # Initialization variables
        fireDuration = np.zeros(batch.batchSize, dtype=np.int16)
        lit = np.reshape(batch.lit_np.astype(np.bool), batch.tensorShape)
        heat = np.reshape(batch.heat_np.astype(FLOAT_TYPE), batch.tensorShape)
        fire = np.reshape(batch.fire_np.astype(FLOAT_TYPE), batch.tensorShape)

        # Compute write interval from memory limit
        floatSize = np.dtype(FLOAT_TYPE).itemsize
        fireSize = batch.batchSize * batch.fieldShape[0] * batch.fieldShape[1] * floatSize
        writeInterval = int(np.floor(MEMORY_LIMIT / fireSize))

        # Array to temporarily store output in memory
        arrShape = (writeInterval,
                    batch.tensorShape[0],
                    batch.tensorShape[1],
                    batch.tensorShape[2])
        lit_arr = np.empty(arrShape, dtype=bool)
        fire_arr = np.empty(arrShape, dtype=FLOAT_TYPE)

        lit_arr[0] = np.squeeze(lit, axis=3)
        fire_arr[0] = np.squeeze(fire, axis=3)

        i = 1
        writeIters = 0
        step = 0
        active = True
        timeout = False

        startRunTime = time.time()
        while active and not timeout:

            print("Iteration {0}...".format(step + 1))

            # In intervals, post-process and write output
            if i >= writeInterval:

                print("Writing intermediate results to disk...")
                endStep = end_step(step, i, fireDuration)
                overwrite = (writeIters == 0)
                write_output(lit_arr, fire_arr, endStep, batch.outDir, batch.initCaseID, overwrite)
                
                # Reset temp array
                lit_arr = np.empty_like(lit_arr)
                fire_arr = np.empty_like(fire_arr)
                i = 0
                writeIters += 1

            # Evaluate all relevant tensors, initializing with the previous values
            feedDict = {batch.fireDuration_init: fireDuration,
                        batch.lit_init: lit,
                        batch.heat_init: heat,
                        batch.fire_init: fire}
            fireDuration_new = sess.run(batch.fireDuration, feed_dict=feedDict)
            lit_new = sess.run(batch.lit, feed_dict=feedDict)
            heat_new = sess.run(batch.heat, feed_dict=feedDict)
            fire_new = sess.run(batch.fire, feed_dict=feedDict)

            # Update initialization variables
            fireDuration = fireDuration_new
            lit = lit_new
            heat = heat_new
            fire = fire_new

            # Write output to memory
            lit_arr[i] = np.squeeze(lit, axis=3)
            fire_arr[i] = np.squeeze(fire, axis=3)

            i += 1
            step += batch.outInterval

            # Check whether any fires are still active
            active = np.any(fire_active_np(np.squeeze(lit, axis=3), batch.termConditions, batch.termWalls))
            
            # Check for timeout
            if 'step' in batch.termConditions:
                timeout = step >= batch.termStep
            else:
                timeout = False

        print("Cleaning up...")
        
        # Process remaining output and write to disk
        # Iterate over cases
        endStep = end_step(step, i, fireDuration)
        overwrite = (writeIters == 0)
        write_output(lit_arr, fire_arr, endStep, batch.outDir, batch.initCaseID, overwrite)
        
        # Compute run time
        runTime = time.time() - startRunTime

    # Display results
    print("Finished!")
    print("")
    print("Results:")
    if batch.batchSize == 1:
        print("Fire duration: {0:d} steps".format(fireDuration[0]))
    print("Build time: {0:5f} s".format(buildTime))
    print("Execution time: {0:5f} s".format(runTime))
    print("")

    return (buildTime, runTime, fireDuration)


def initiate_session(graph, useTPU):
    """Configure TensorFlow Session, depending on processor."""

    if useTPU:
        cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu=os.environ['TPU_NAME'],
            zone='us-central1-a',
            project='crypto-will-254123')
        config = tf.compat.v1.estimator.tpu.RunConfig(cluster=cluster)
        sess = tf.compat.v1.Session(cluster.get_master(),
                                    config=config.session_config,
                                    graph=graph)
        sess.run(tf.compat.v1.tpu.initialize_system())
    else:
        sess = tf.compat.v1.Session(graph=graph)

    return sess


def terminate_session(sess):
    """Terminate TPU session."""

    sess.run(tf.compat.v1.tpu.shutdown_system())


def append_perfStats(perfStats1, perfStats2):
    """Append two sets of perfStats"""

    perfStats = dict()
    for key in perfStats1:
        perfStats[key] = perfStats1[key] + perfStats2[key]
    
    return perfStats


def export_perfStats(buildTime, runTime, fireDuration,
                     outDir=DEFAULTS['outDir'], append=False):
    """Export the performance statistics to perfStats.json"""

    fireDuration = [i.tolist() for i in fireDuration]
    perfStats = {'buildTime': buildTime,
                 'runTime': runTime,
                 'fireDuration': fireDuration}

    jsonPath = outDir + 'perfStats.json'
    append = append and os.path.exists(jsonPath)

    if append:
        with open(jsonPath, 'r+') as outFile:
            perfStats_old = json.load(outFile)
            outFile.seek(0, 0)
            outFile.truncate()
            perfStats = append_perfStats(perfStats_old, perfStats)
            json.dump(perfStats, outFile)
    else:
        with open(jsonPath, 'w') as outFile:
            json.dump(perfStats, outFile)


def simulate(batch):
    """Configure session and run given FireBatch."""

    if not isinstance(batch, FireBatch):
        raise TypeError("Expected FireBatch object")

    # Begin graph construction
    fireGraph = tf.compat.v1.get_default_graph()
    with fireGraph.as_default():

        # Evaluate graph
        print("Configuring session...")
        sess = initiate_session(fireGraph, batch.useTPU)
        print("Initializing variables...")
        sess.run(tf.compat.v1.global_variables_initializer())

        buildTime = [None] * batch.runIters
        runTime = [None] * batch.runIters

        for i in range(batch.runIters):
            (buildTime[i], runTime[i], fireDuration) = run_batch(batch, sess)

        if batch.useTPU:
            terminate_session(sess)

    return (buildTime, runTime, fireDuration)
