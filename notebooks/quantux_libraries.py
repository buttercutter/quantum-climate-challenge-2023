# # 1. Import libraries and qiskit functions

# 0. Data science & ML libraries
# =============================
import pandas as pd
import numpy as np
import warnings, os, time
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')

from functools import wraps
from qiskit.utils import algorithm_globals

from functools import partial
from scipy.optimize import minimize

_SEED = 5
np.random.seed(_SEED)
algorithm_globals.random_seed = _SEED

####

# All are post-migration imports, ie. qiskit_nature == 0.5.0
import qiskit, qiskit_nature
v_qiskit = qiskit.__version__
v_qiskit_nature = qiskit_nature.__version__

if (v_qiskit >= '0.22') and (v_qiskit_nature >= '0.5'):
    
    # 1. Load qiskit account & providers
    # =============================
    from qiskit import IBMQ
    IBMQ.load_account()
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    deloitte_provider = provider
    # deloitte_simulators = []
    from qiskit.providers.fake_provider import FakeProvider
    # from qiskit.providers.fake_provider import FakeVigo
    # from qiskit.test.mock import FakeProvider
    fake_provider = FakeProvider()
    
    # provider.backends() # List all backend providers
    qasm_simulator = provider.get_backend('ibmq_qasm_simulator')
    statevector_simulator = provider.get_backend('simulator_statevector')
    
    from qiskit.providers.aer import StatevectorSimulator
    from qiskit import Aer
    from qiskit_aer.noise import NoiseModel
    
    from qiskit_ibm_runtime import QiskitRuntimeService, Options, Session
    

    # 2. Molecule & backend definition
    # =============================
    from qiskit_nature.second_q.formats.molecule_info import MoleculeInfo
    from qiskit_nature.units import DistanceUnit
    from qiskit.utils import QuantumInstance
    from qiskit_nature.second_q.drivers import PySCFDriver

    # VQE Client seem can't be used, use VQE instead
    # from qiskit_nature.runtime import VQEClient
    
    
    # 3. Estimators & Sampler Primitives
    # =============================
    from qiskit.primitives import Estimator, Sampler # Noiseless
    from qiskit_aer.primitives import Estimator as AerEstimator # Noisy
    
    
    # 4. Ansatz, initial circuit libraries, initial points (Ansatz parameters)
    # =============================
    from qiskit_nature.second_q.circuit.library import UCC, UCCSD, PUCCD, SUCCD, UVCCSD # Ansatz
    from qiskit_nature.second_q.circuit.library import HartreeFock # Initial State
    from qiskit.circuit.library import TwoLocal, PauliTwoDesign, EfficientSU2, RealAmplitudes, ExcitationPreserving # Initial Circuit
    from qiskit.quantum_info import Pauli
    #from qiskit.circuit.library.initial_states import hartree_fock, fermionic_gaussian_state, vscf
    #from qiskit.circuit.library.initial_points import mp2_initial_point, hf_initial_point, mp2_initial_point, vscf_initial_point
    
    
    # 5. Qubit Mappers & transformers
    # =============================
    from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitConverter, BravyiKitaevMapper, ParityMapper
    from qiskit_nature.second_q.transformers import ActiveSpaceTransformer, FreezeCoreTransformer
    
    
    # 6. Algorithms & optimizers
    # =============================
    from qiskit_nature.second_q.algorithms import NumPyMinimumEigensolverFactory, VQEUCCFactory, VQEUVCCFactory
    from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
    from qiskit_nature.second_q.algorithms.ground_state_solvers import GroundStateEigensolver
    from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA, QNSPSA

    from qiskit.algorithms import HamiltonianPhaseEstimation, PhaseEstimation
    from qiskit.opflow import StateFn, PauliExpectation, CircuitSampler, PauliTrotterEvolution

else:
    print('Qiskit version not mostly updated - v2. Need updates before continue.')
    pass

####

# 2. Decorators 

# Timeit decorator
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} took {total_time:.4f} seconds to run\n')
        return result
    return timeit_wrapper

####

# Try except decorator
def get_err_msg_silent(value):
    def decorate(f):
        def applicator(_silent = False, *args, **kwargs):
            try:
                if not _silent:
                    print('{}: Loading...'.format(f.__name__))
                
                res = f(*args,**kwargs)
                
                if not _silent:
                    print('Success in loading {}'.format(f.__name__))
                return res
                
            except:
                if not _silent:
                    print('Fail in loading {}'.format(f.__name__))
                return value
        return applicator
    return decorate

####

# Try except decorator
def get_err_msg(value):
    def decorate(f):
        def applicator(*args, **kwargs):
            try:
                print('{}: Loading...'.format(f.__name__))
                res = f(*args,**kwargs)
                return res
            except:
                return value
        return applicator
    return decorate

####

from qiskit_ibm_runtime import (QiskitRuntimeService, Session as RuntimeSession,
                                Estimator as RuntimeEstimator, Options)
from qiskit import Aer
service = QiskitRuntimeService(
    channel='ibm_quantum',
    instance='ibm-q/open/main',
)

backend_real = 'ibm_lagos'

import signal, time

from qiskit.providers import JobStatus

def timeout_handler(signum, frame):
    raise Exception('Iteration timed out')

from qiskit.algorithms import MinimumEigensolver, VQEResult

# Define a custome VQE class to orchestra the ansatz, classical optimizers, 
# initial point, callback, and final result
class CustomVQE(MinimumEigensolver):
    
    def __init__(self, estimator, circuit, optimizer, callback=None, init_data=[]):
        self._estimator = estimator
        self._circuit = circuit
        self._optimizer = optimizer
        self._callback = callback
        self._init_data = init_data
        
    def compute_minimum_eigenvalue(self, operators, aux_operators=None):
                
        # Define objective function to classically minimize over
        def objective(x):
            # Execute job with estimator primitive
            job = self._estimator.run([self._circuit], [operators], [x])
            #print("EstimatorJob:", job.job_id())
            # Get results from jobs
            est_result = job.result()
            # Get the measured energy value
            value = est_result.values[0]
            #print("Job Value:", job.job_id(), value)
            # Save result information using callback function
            if self._callback is not None:
                self._callback(value)
            return value
            
        # Select an initial point for the ansatzs' parameters
        if len(self._init_data) > 0:
            x0 = init_data
        else:
            x0 = np.pi/4 * np.random.rand(self._circuit.num_parameters)
        
        # Run optimization
        res = self._optimizer.minimize(objective, x0=x0)
        
        # Populate VQE result
        result = VQEResult()
        result.cost_function_evals = res.nfev
        result.eigenvalue = res.fun
        result.optimal_parameters = res.x
        return result


class RetryEstimator(RuntimeEstimator):
    """RuntimeRetryEstimator class.
    
    This class inherits from Qiskit IBM Runtime's Estimator and overwrites its run method such that it retries calling it
    a maximum of 'max_retries' consecutive times, if it encounters one of the following randomly occuring errors:
    
    * An Estimator error (in this case "Job.ERROR" is printed, and the job is cancelled automatically)
    * A timeout error where the job either remains running or completes but does not return anything, for a time larger 
      than 'timeout' (in this case the job is cancelled by the patch and "Job.CANCELLED" is printed)
    * A creation error, where the job fails to be created because connection is lost between the runtime server and the
      quantum computer (in this case "Failed to create job." is printed). If this error occurs, the patch connects the user
      to a new Session (to be handled with care! also, this will unfortunately put the next job in the queue). 
    """
    
    def __init__(self, *args, max_retries: int = 5, timeout: int = 3600, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries
        self.timeout = timeout
        self.backend = super().session._backend
        signal.signal(signal.SIGALRM, timeout_handler)
    
    def run(self, circuits, observables, parameter_values, **kwargs):
        result = None
        for i in range(self.max_retries):
            try:
                job = super().run(circuits, observables, parameter_values, **kwargs)
                while job.status() in [JobStatus.INITIALIZING, JobStatus.QUEUED, JobStatus.VALIDATING]:
                    time.sleep(5) # Check every 5 seconds whether job status has changed
                signal.alarm(self.timeout) # Once job starts running, set timeout to 1 hour by default
                result = job.result()
                if result is not None:
                    signal.alarm(0) # reset timer
                    return job
            except Exception as e:
                print("\nSomething went wrong...")
                print(f"\n\nERROR MESSAGE:\n{e}\n\n")
                if 'job' in locals(): # Sometimes job fails to create
                    print(f"Job ID: {job.job_id}. Job status: {job.status()}.")
                    if job.status() not in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]:
                        job.cancel()
                else:
                    print("Failed to create job.")
                print(f"Starting trial number {i+2}...\n")
                print(f"Creating new session...\n")
                signal.alarm(0) # reset timer
                super().session.close()
                self._session = RuntimeSession(backend=self.backend)
        if result is None:
            raise RuntimeError(f"Program failed! Maximum number of retries ({self.max_retries}) exceeded")

intermediate_info_real_backend = []
def callback_real(value):
        intermediate_info_real_backend.append(value)
        
####

# 2-1. Define modular functions

# Get a random FakeProvider from IBM runtime
def _0_get_min_runtime_provider(provider, min_qubits):
    
    # fake_provider = FakeProvider()
    # deloitte_provider = IBMProvider(instance="deloitte-event23/level-1-access/quantux")
    try:
        provider_list = [[b.configuration().n_qubits, b.name()] for b in provider.backends() if b.configuration().n_qubits >= min_qubits] # Usable in Aer
    except:
        provider_list = [[b.configuration().n_qubits, b.name] for b in provider.backends() if b.configuration().n_qubits >= min_qubits] # Usable in FakeProvider

    from collections import defaultdict
    provider_dict = defaultdict(list)
    for k, v in provider_list:
        provider_dict[k].append(v)

    provider_dict = dict(provider_dict)

    min_qubit = min(i for i in list((provider_dict.keys())) if i > min_qubits)

    provider_list_chosen = provider_dict[min_qubit]
    idx = np.random.randint(len(provider_list_chosen)) # Random choose provider with equivalent qubit computing size
    provider_out=provider_list_chosen[idx]

    print('Noisy Aer provider with min_qubit chosen: {}, {}'.format(min_qubit, provider_out))

    return provider_out

#_0_get_min_runtime_provider(provider, min_qubits = 5) # Deloitte provider / IBM main provider
#_0_get_min_runtime_provider(fake_provider, min_qubits = 5)

####

# 1. Get problem driver
#@get_err_msg('')
#@timeit
def _1_get_problem_driver(display_dict, input_value, basis = 'sto3g', input_type = 'molecule'):
    
    # =============================
    # 1. Define Molecule dictionary and return problem from PySCFDriver
    # =============================
    
    # =============================
    if input_type == 'molecule':
        moleculeinfo = MoleculeInfo(symbols = input_value['symbols'], coords = input_value['coords'], masses = input_value['masses'], 
                                    charge = input_value['charge'], multiplicity = input_value['multiplicity'])
    
    elif input_type == 'moleculeinfo':
        moleculeinfo = input_value
    
    driver = PySCFDriver.from_molecule(moleculeinfo, basis=basis)
    problem = driver.run()
    
    
    if display_dict is not None and input_type != 'moleculeinfo':
        display_dict['molecule'] = '{}'.format(input_value['symbols'])
        display_dict['charge'] = '{}'.format(input_value['charge'])
        display_dict['multiplicity'] = '{}'.format(input_value['multiplicity'])
        
        display_dict['reference_energy'] = '{}'.format(problem.reference_energy)
        display_dict['num_spin_orbitals'] = '{}'.format(problem.num_spin_orbitals)
        display_dict['num_spatial_orbitals'] = '{}'.format(problem.num_spatial_orbitals)
        display_dict['num_particles'] = '{}'.format(problem.num_particles)
        display_dict['nuclear_repulsion_energy'] = '{}'.format(problem.nuclear_repulsion_energy)
        display_dict['num_alpha'] = '{}'.format(problem.num_alpha)
        display_dict['num_beta'] = '{}'.format(problem.num_beta)
        
        display_dict_new = display_dict
    else:
        display_dict_new = display_dict
    
    return problem, display_dict_new

####

# 1. Get transform problem
#@get_err_msg('')
#@timeit
def _2_get_problem_transform(display_dict, problem, input_value, reduced):
    
    # =============================
    # 2. Transform the problem to reduce simulation space
    # =============================
    
    # =============================
    # If enter gas_molecules info, can transform; else if enter molecularinfo info, can not transform (should have previously transformed)
    
    # Problem reduction
    if reduced == 'FreezeCore':
        try:
            fc_transformer = FreezeCoreTransformer(freeze_core = input_value['fc_transformer']['fc_freeze_core'], 
                                                   remove_orbitals = input_value['fc_transformer']['fc_remove_orbitals'])
            problem = fc_transformer.transform(problem)
            display_chosen = input_value['fc_transformer']['fc_remove_orbitals']
        except:
            #print('FreezeCore Transformer did not succeed.')
            if display_dict is not None:
                display_dict['FreezeCoreTransformer'] = 'error'
                display_chosen = 'error'
            pass
        
        # How to determine which orbitals to be removed
        # https://quantumcomputing.stackexchange.com/questions/17852/use-one-body-integrals-to-know-which-orbitals-to-freeze-in-electronicstructurepr
        # https://www.youtube.com/watch?v=3B04KB0pDwE&t=667s
        
    elif reduced == 'ActiveSpace':
        #max_num_spatial_orbitals = problem.num_spatial_orbitals
        #max_num_electrons = problem.num_electrons
        #max_active_orbitals = itertools.combinations(max_num_spatial_orbitals, 2)
        # Use optuna to setup objective
        
        try:
            as_transformer = ActiveSpaceTransformer(num_electrons = input_value['as_transformer']['as_num_electrons'], 
                                                    num_spatial_orbitals = input_value['as_transformer']['as_num_spatial_orbitals'], 
                                                    active_orbitals = input_value['as_transformer']['as_active_orbitals'])
            problem = as_transformer.transform(problem)
            display_chosen = input_value['as_transformer']['as_active_orbitals']
        except:
            #print('ActiveSpace Transformer did not succeed.')
            if display_dict is not None:
                display_dict['ActiveSpaceTransformer'] = 'error'
                display_chosen = 'error'
            pass
    else:
        # print('Expect lengthy simulation if can not succeed in reducing orbitals using FreezeCoreTransformer or ActiveSpaceTransformer.')
        display_chosen = ''
        pass
    
    
    if display_dict is not None:
        display_dict['reduction_method'] = '{}'.format(reduced)
        display_dict['orbitals_removed'] = '{}'.format(display_chosen)
        display_dict_new = display_dict
        #print("_2_get_problem_transform", display_dict_new)
    else:
        display_dict_new = display_dict
    
    return problem, display_dict_new

####

# 3. Get qubit operator
#@get_err_msg('')
#@timeit
def _3_get_qubit_operator(display_dict, problem, hyperparam, mapper_type):
    
    # =============================
    # 3. Define qubit mapping and convert to qubit operator
    # =============================
    
    # =============================
    # Qubit mapping
    if mapper_type == 'ParityMapper':
        mapper = ParityMapper()
    elif mapper_type == 'JordanWignerMapper':
        mapper = JordanWignerMapper()
    elif mapper_type == 'BravyiKitaevMapper':
        mapper = BravyiKitaevMapper()
    
    fermionic_hamiltonian = problem.hamiltonian
    second_q_op = fermionic_hamiltonian.second_q_op()
    
    num_particles = problem.num_particles
        
    qubit_converter = QubitConverter(mapper, 
                                     two_qubit_reduction = hyperparam['two_qubit_reduction'], 
                                     z2symmetry_reduction = hyperparam['z2symmetry_reduction'])
    qubit_op = qubit_converter.convert(second_q_op, num_particles = num_particles, sector_locator = problem.symmetry_sector_locator)
    
    if display_dict is not None:
        display_dict['second_q_op'] = '{}'.format("\n".join(str(second_q_op).splitlines()[:10] + ["..."]))
        
        display_dict_new = display_dict
    else:
        display_dict_new = display_dict
        
    return qubit_op, qubit_converter, display_dict_new

####

# 4. Get quantum problem solver
#@get_err_msg('')
#@timeit
def _4_get_ansatz(display_dict, problem, hyperparam, qubit_op, qubit_converter, seed, randomize = False):
    
    # =============================
    # 4. Define various solver types with initalizing the ansatz or initial circuit
    # =============================
    
    # =============================
    num_spin_orbitals = problem.num_spin_orbitals
    num_spatial_orbitals = problem.num_spatial_orbitals
    num_particles = problem.num_particles
    num_qubits = qubit_op.num_qubits
    
    # i. List of ansatz
    # [UCCSD(), UCC(), PUCCD(), SUCCD(), UVCCSD()] # Others like PUCCD needs alpha == beta, ie. lots of restrictions thus all except UCCSD are left unused
    ansatz_hf = HartreeFock(num_qubits, num_particles, qubit_converter)
    
    ansatz_uccsd = UCCSD(num_spatial_orbitals,
                         num_particles,
                         qubit_converter,
                     initial_state=ansatz_hf)
    
    ansatz_lst = [ansatz_uccsd]
    ansatz_name = ['UCCSD']
    
    # =============================
    # ii. For QNSPSA, must match num_qubits to the circuit observable, currently H2 is 4, use this constraint to subordinate for num_qubits' design from initial_circuits
    
    circuit_tl = TwoLocal(num_qubits, rotation_blocks = ['h', 'rx'], entanglement_blocks = 'cz', entanglement='full', reps=hyperparam['reps'], parameter_prefix = 'y')
    circuit_su2 = EfficientSU2(num_qubits, reps=hyperparam['reps'], entanglement="full")
    circuit_p2d = PauliTwoDesign(num_qubits, reps=hyperparam['reps'], seed=seed)
    circuit_ra = RealAmplitudes(num_qubits, reps=hyperparam['reps'])
    circuit_ep = ExcitationPreserving(num_qubits, entanglement = 'full', reps=hyperparam['reps'], mode = 'iswap')
    
    initial_circuit_lst = [circuit_tl, circuit_su2, circuit_p2d, circuit_ra, circuit_ep]
    initial_circuit_name = ['TwoLocal', 'EfficientSU2', 'PauliTwoDesign', 'RealAmplitudes', 'ExcitationPreserving']
    
    if randomize:
        idx_ansatz = np.random.randint(len(ansatz_lst))
        ansatz_chosen = ansatz_lst[idx_ansatz]
        
        idx_initial_circuit = np.random.randint(len(initial_circuit_lst))
        initial_circuit_chosen = initial_circuit_lst[idx_initial_circuit]
    else:
        idx_ansatz = ansatz_name.index(hyperparam['ansatz'])
        an = hyperparam['ansatz']
        an_val = np.where(np.array(ansatz_name) == an)[0][0]
        ansatz_chosen = ansatz_lst[an_val]
        
        ic = hyperparam['initial_circuit']
        ic_val = np.where(np.array(initial_circuit_name) == ic)[0][0]
        initial_circuit_chosen = initial_circuit_lst[ic_val]
        idx_initial_circuit = ic_val
        #print(ansatz_chosen, initial_circuit_chosen)
    
    if display_dict is not None:
        display_dict['ansatz_chosen'] = ansatz_name[idx_ansatz]
        display_dict['initial_circuit_chosen'] = initial_circuit_name[idx_initial_circuit]
        
        display_dict_new = display_dict
    else:
        display_dict_new = display_dict
    
    return ansatz_chosen, initial_circuit_chosen, display_dict_new

####

# 5. Get optimizer
#@get_err_msg('')
#@timeit
def _5_get_solver_optimizer(display_dict, problem, hyperparam, qubit_op, qubit_converter, solver_type, ansatz_chosen, initial_circuit_chosen, optimizer, method):
    
    # =============================
    # 5. Define VQE solvers to solve for PES in quantum chemistry
    # =============================
    
    # a. Optimizer set
    # [SPSA(maxiter=100), SLSQP(maxiter=100)] + [partial(minimize, method=i) for i in method_lst]
    # =============================
    method_lst = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'trust-constr']
    opt = [partial(minimize, method=i) for i in method_lst]
    iterations = 1
    if hyperparam.get('iterations') != None:
        iterations = hyperparam['iterations']
    
    optimizer_lst = [SPSA(maxiter=iterations), SLSQP(maxiter=iterations), COBYLA(maxiter=iterations)] + opt
    optimizer_name = ['SPSA', 'SLSQP', 'COBYLA'] + ['minimize_{}'.format(i) for i in method_lst]
    
    optimizer_dict = dict(zip(optimizer_name, optimizer_lst))
    optimizer_chosen = optimizer_dict[optimizer]
    # print("Iterations:", iterations, optimizer_chosen)
    
    # Fixed solver and optimizers
    # ['vqe_qnspsa', 'vqe_simulator_noiseless', 'vqe_simulator_noisy_0_0', 'vqe_runtime_noiseless', 'vqe_runtime_noisy_0_0', 'vqe_fake_runtime_noiseless', 
    # 'vqe_fake_runtime_noisy_0_0', 'vqe_ansatz', 'numpy_solver_with_filter', 'numpy_solver', 'vqe_initial_circuit']
    # =============================
    if solver_type == 'vqe_qnspsa':
        ans = initial_circuit_chosen
        obs = qubit_op
        initial_point = np.random.random(ans.num_parameters)

        # loss function
        est = Estimator()

        def loss(x):
            result = est.run([ans], [obs], [x]).result()
            return np.real(result.values[0])

        # fidelity for estimation of the geometric tensor
        sam = Sampler()
        fidelity = QNSPSA.get_fidelity(ans, sam)

        # run QN-SPSA
        solver = QNSPSA(fidelity, maxiter=3000, perturbation = 0.7, learning_rate = 0.01)
        ground_state = solver.optimize(ans.num_parameters, loss, initial_point=initial_point)
        energy = ground_state[1]

    elif solver_type == 'vqe_simulator_noiseless':
        options = {'optimization_level': 3, 'resilience_level': 3}

        ans = initial_circuit_chosen
        opt = optimizer_chosen
        est = Estimator(options = options)

        solver = VQE(ansatz = ans, estimator = est, optimizer = opt)
        ground_state = solver.compute_minimum_eigenvalue(qubit_op)
        energy = ground_state.eigenvalue

    elif solver_type == 'vqe_simulator_noisy_0_0':
        options = {'optimization_level': 0, 'resilience_level': 0}

        ans = initial_circuit_chosen
        opt = optimizer_chosen
        est = Estimator(options = options)

        solver = VQE(ansatz = ans, estimator = est, optimizer = opt)
        ground_state = solver.compute_minimum_eigenvalue(qubit_op)
        energy = ground_state.eigenvalue

    elif solver_type == 'vqe_runtime_noiseless':
        # Other options available: options = {'optimization_level': 0-3, 'resilience_level': 0-3}
        ans = initial_circuit_chosen
        opt = optimizer_chosen

        device = provider.get_backend(_0_get_min_runtime_provider(provider = provider, min_qubits = qubit_op.num_qubits))
        coupling_map = device.configuration().coupling_map
        noise_model = None

        noisy_est = AerEstimator(backend_options={"method": "statevector", "coupling_map": coupling_map, "noise_model": noise_model,},
                                 run_options={"seed": _SEED, "shots": 1024, 'optimization_level': 3, 'resilience_level': 3},
                                 transpile_options={"seed_transpiler": _SEED},
                                )
        solver = VQE(noisy_est, ans, optimizer=opt)
        ground_state = solver.compute_minimum_eigenvalue(qubit_op)
        energy = ground_state.eigenvalue

    elif solver_type == 'vqe_runtime_noisy_0_0':
        # Other options available: options = {'optimization_level': 0-3, 'resilience_level': 0-3}
        ans = initial_circuit_chosen
        opt = optimizer_chosen

        device = provider.get_backend(_0_get_min_runtime_provider(provider = provider, min_qubits = qubit_op.num_qubits))
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)

        noisy_est = AerEstimator(backend_options={"method": "density_matrix", "coupling_map": coupling_map, "noise_model": noise_model,},
                                 run_options={"seed": _SEED, "shots": 1024, 'optimization_level': 0, 'resilience_level': 0},
                                 transpile_options={"seed_transpiler": _SEED},
                                )
        solver = VQE(noisy_est, ans, optimizer=opt)
        ground_state = solver.compute_minimum_eigenvalue(qubit_op)
        energy = ground_state.eigenvalue

    elif solver_type == 'vqe_fake_runtime_noiseless':
        # Other options available: options = {'optimization_level': 0-3, 'resilience_level': 0-3}
        provider = fake_provider
        ans = initial_circuit_chosen
        opt = optimizer_chosen

        device = provider.get_backend(_0_get_min_runtime_provider(provider = provider, min_qubits = qubit_op.num_qubits))
        coupling_map = device.configuration().coupling_map
        noise_model = None #NoiseModel.from_backend(device)

        noisy_est = AerEstimator(backend_options={"method": "density_matrix", "coupling_map": coupling_map, "noise_model": noise_model,},
                                 run_options={"seed": _SEED, "shots": 1024, 'optimization_level': 3, 'resilience_level': 3},
                                 transpile_options={"seed_transpiler": _SEED},
                                )
        solver = VQE(noisy_est, ans, optimizer=opt)
        ground_state = solver.compute_minimum_eigenvalue(qubit_op)
        energy = ground_state.eigenvalue

    elif solver_type == 'vqe_fake_runtime_noisy_0_0':
        # Other options available: options = {'optimization_level': 0-3, 'resilience_level': 0-3}
        provider = fake_provider
        ans = initial_circuit_chosen
        opt = optimizer_chosen

        device = provider.get_backend(_0_get_min_runtime_provider(provider = provider, min_qubits = qubit_op.num_qubits))
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)

        noisy_est = AerEstimator(backend_options={"method": "density_matrix", "coupling_map": coupling_map, "noise_model": noise_model,},
                                 run_options={"seed": _SEED, "shots": 1024, 'optimization_level': 0, 'resilience_level': 0},
                                 transpile_options={"seed_transpiler": _SEED},
                                )
        solver = VQE(noisy_est, ans, optimizer=opt)
        ground_state = solver.compute_minimum_eigenvalue(qubit_op)
        energy = ground_state.eigenvalue
        
    elif solver_type == 'vqe_runtime_test':
        ansatz = ansatz_chosen
        optimizer = SPSA(1)
        backend = hyperparam['backend']
        init_data = []
        #service_test = QiskitRuntimeService(channel='ibm_quantum')
        
        with RuntimeSession(service=service, backend=backend) as session:
            # Prepare extended primitive
            rt_estimator = RetryEstimator(session=session)
            # set up algorithm
            custom_vqe = CustomVQE(rt_estimator, ansatz, optimizer, callback=callback_real, init_data=init_data)
            # run algorithm
            result = custom_vqe.compute_minimum_eigenvalue(qubit_op)
            ground_state = result
            energy = result.eigenvalue

    elif solver_type == 'vqe_runtime_real':
        options_ZNE = Options()
        options_ZNE.execution.shots = 8900
        options_ZNE.optimization_level = 0 # no optimization
        options_ZNE.resilience_level = 2 # ZNE
        options_ZNE.resilience.noise_factors = [np.pi/2, (np.pi/2)*2, (np.pi/2)*3]
        options_ZNE.resilience.noise_amplifier = "LocalFoldingAmplifier"
        options_ZNE.resilience.extrapolator = "QuadraticExtrapolator"

        options = options_ZNE
        #options = None
        init_data = []
        ansatz = initial_circuit_chosen
        #optimizer = SPSA(1)
        
        #start = time.time()
        with RuntimeSession(service=service, backend=backend_real) as session:
            # Prepare extended primitive
            rt_estimator = RetryEstimator(session=session, options=options)
            # set up algorithm
            custom_vqe = CustomVQE(rt_estimator, ansatz, optimizer, callback=callback_real, init_data=init_data)
            # run algorithm
            result = custom_vqe.compute_minimum_eigenvalue(qubit_op)
            ground_state = result
            energy = result.eigenvalue
        #end = time.time()
        #print(f'execution time (s): {end - start:.2f}')
        
    # Customizable solver and optimizers
    # =============================
    # Solvers
    elif solver_type == 'vqe_ansatz':
        est = Estimator()
        ans = ansatz_chosen
        opt = optimizer_chosen

        solver = VQEUCCFactory(est, ans, opt)

    elif solver_type == 'numpy_solver_with_filter':
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)

    elif solver_type == 'numpy_solver':
        solver = NumPyMinimumEigensolverFactory()

    elif solver_type == 'vqe_initial_circuit':
        est = Estimator()
        ic = initial_circuit_chosen
        opt = optimizer_chosen

        solver = VQE(est, ic, opt) 

    # Optimizers
    if method == 'gses':
        calc = GroundStateEigensolver(qubit_converter, solver)
        ground_state = calc.solve(problem)

        energy = ground_state.total_energies[0].real

    elif method == 'qpe':
        # Quantum Phase Estimation
        quantum_instance = QuantumInstance(backend = Aer.get_backend('aer_simulator_statevector'))
        evolution = PauliTrotterEvolution('trotter', reps = hyperparam['qpe_num_time_slices'])

        qpe = HamiltonianPhaseEstimation(hyperparam['qpe_n_ancilliae'], quantum_instance=quantum_instance)

        state_preparation = None
        # state_preparation = 
        ground_state = qpe.estimate(qubit_op, state_preparation, evolution=evolution)

        energy = ground_state.most_likely_eigenvalue


    
    # =============================
    
    if display_dict is not None:
        display_dict['solver'] = '{}'.format(solver_type)
        display_dict['optimizer'] = '{}'.format(optimizer)
        display_dict['solution_method'] = '{}'.format(method)
        
        display_dict_new = display_dict
    else:
        display_dict_new = display_dict
        
    return ground_state, energy, display_dict_new

####

# 2-2. Build construct pipeline

# Create construct problem
#@get_err_msg('')
#@timeit
def get_construct_problem(input_value, hyperparam, input_type, display_report, reduced, basis, mapper_type, solver_type, method, optimizer, seed, randomize=False):
    
    # =============================
    # 0. Display Report Dict
    # =============================
    if display_report == True:
        display_dict = {}
    else:
        display_dict = None
    
    # =============================
    # A. The quantum solver pipeline
    # =============================
    
    # Specify input_type = 'molecule' or 'moleculeinfo'
    problem, display_dict_1 = _1_get_problem_driver(display_dict, input_value, basis = basis, input_type = input_type)
    
    problem, display_dict_2 = _2_get_problem_transform(display_dict_1, problem, input_value, reduced)
    
    qubit_op, qubit_converter, display_dict_3 = _3_get_qubit_operator(display_dict_2, problem, hyperparam, mapper_type)
    
    ansatz_chosen, initial_circuit_chosen, display_dict_4 = _4_get_ansatz(display_dict_3, problem, hyperparam, qubit_op, qubit_converter, seed, randomize)
    
    ground_state, energy, display_dict_5 = _5_get_solver_optimizer(display_dict_4, problem, hyperparam, qubit_op, qubit_converter, 
                                                                   solver_type, ansatz_chosen, initial_circuit_chosen, optimizer, method)
    if display_report == True:
        display_df = pd.DataFrame.from_dict([display_dict_5])
    else:
        display_df = None
    
    return ground_state, energy, display_df

####

# 3-1. Calculate molecule energy by BOPES calculations with regressed distance

def get_a_molecule_perturbation_list(moleculeinfo, hyperparam, perturbation_steps):
    
    from copy import deepcopy
    
    EPSILON = 1e-3
    
    atom_pair = hyperparam['atom_pair']
    pair_0, pair_1 = atom_pair
    
    x0, y0, z0 = moleculeinfo.coords[pair_0 - 1]
    x1, y1, z1 = moleculeinfo.coords[pair_1 - 1]
    # Get a straight line connecting the 2 interacting atom pair, y = mx + p
    m = 0
    p = y0
    if abs(x1 - x0) > EPSILON:
        m = (y1 - y0)/(x1 - x0)
        p = y0 - m*x0

    #print("coords:", len(moleculeinfo.coords), " ", x0, y0, z0, " - ", x1, y1, z1)
    # This perturbation assumes lying on the same plane
    size = len(perturbation_steps)
    energy_lst = np.empty(size)
    
    perturbation_lst = []
    for k in range(size):
        # print("Step: ", k)
        
        if (abs(x0) < EPSILON and abs(y0) < EPSILON):
            z0_new = z0 + perturbation_steps[k]
            
            coords_new = []
            for l in range(len(moleculeinfo.coords)):
                #print("Add Z ",l)
                if l == atom_pair[0]:
                    coords_new.append((0.0, 0.0, z0_new))
                else:
                    coords_new.append(moleculeinfo.coords[l])
        
        elif (abs(z0) < EPSILON and abs(z1) < EPSILON):
            x0_new = x0 + perturbation_steps[k]
            y0_new = m*x0_new + p
            
            coords_new = []
            for l in range(len(moleculeinfo.coords)):
                #print("Add XY ",l)
                if l == atom_pair[0]:
                    coords_new.append((x0_new, y0_new, 0.0))
                else:
                    coords_new.append(moleculeinfo.coords[l])
                    
        else:
            print("bopes - Error: unsupported molecule geometry, atom pairs must be in the same line or in the same plane")
            return perturbation_steps, 0
    
        moleculeinfo_new = deepcopy(moleculeinfo)
        moleculeinfo_new.coords = coords_new
        
        perturbation_lst.append(moleculeinfo_new)
        
    return perturbation_lst

####

# Get BOPES energy curve

def get_a_molecule_bopes(input_value, hyperparam, input_type = 'molecule',
                       reduced = 'ActiveSpace', basis = 'sto3g', mapper_type = 'JordanWignerMapper', solver_type = 'numpy_solver', method = 'gses', 
                       perturbation_steps = np.linspace(0.5, 3, 250), optimizer = 'SLSQP', display_report = False, seed = _SEED, randomize = False):
    
    # Obtain molecule coordinates
    # ===========================
    EPSILON = 1e-3
    
    if input_type == 'molecule':
        moleculeinfo = MoleculeInfo(symbols = input_value['symbols'], coords = input_value['coords'], masses = input_value['masses'], 
                                    charge = input_value['charge'], multiplicity = input_value['multiplicity'])
    
    elif input_type == 'moleculeinfo':
        moleculeinfo = input_value
    
    perturbation_lst = get_a_molecule_perturbation_list(moleculeinfo, hyperparam, perturbation_steps)
    
    energy_lst = []
    for k in tqdm(range(len(perturbation_lst)), position = 0, leave = True):
        
        tmp_moleculeinfo = perturbation_lst[k]
        #print("tmp_moleculeinfo", tmp_moleculeinfo)
        input_value["coords"] = tmp_moleculeinfo.coords
        # Get ground state energy from construct problem
        # ===========================
        _, tmp_energy, _ = get_construct_problem(input_value, hyperparam, input_type, display_report, reduced, basis, mapper_type, solver_type, method, optimizer, seed, randomize)
        energy_lst.append(tmp_energy)
        
    if display_report:
        plt.rcParams["figure.figsize"] = [10, 10]
        plt.rcParams["figure.autolayout"] = True
        
        df = pd.DataFrame(list(zip(energy_lst, perturbation_steps)), columns = ['Hartree energy', 'step'])
        fig, ax = plt.subplots(facecolor='lightslategray')
        ax.set_clip_on(False)
        df.plot(kind='scatter', x='step', y='Hartree energy', ax=ax, color='black', linewidth=0, label = 'VQE Energy (HF)')
        
        e_min = min(energy_lst)
        p_min = perturbation_steps[energy_lst == e_min]
        print('Molecule energy:{} with atoms separated at {} apart'.format(e_min, p_min))
        
    return perturbation_steps, energy_lst

####

# Load Quantistry coordinates for comparison

import pandas as pd
import numpy as np
from zipfile import *
import warnings, os, gzip
warnings.filterwarnings('ignore')
pth = os.getcwd() + r'/quantistry'

def get_coord(zip_file):
    all_res = []

    with ZipFile(zip_file, 'r') as f:
        coord_list = [x for x in f.namelist() if 'csv' not in x]
        for i in coord_list:
            tmp = f.extract(i)
            df = pd.read_csv(tmp)

            new_df = []
            for j in range(len(df)):
                df.iloc[j] = df.iloc[j].str.replace('  ', ' ', regex = True)
                new_df.append(df.iloc[j].str.split(' ', expand = True))

            new_df = pd.concat(new_df)
            
            new_df0 = pd.DataFrame(['{}'.format(i).replace('.xyz', '')] * len(new_df), columns = ['step'])
            
            new_df1 = new_df.iloc[:, 0].reset_index(drop = True)
            
            new_df2 = new_df.iloc[:, 1:]
            out = pd.DataFrame(new_df2.iloc[:,::-1].apply(lambda x: x.dropna().tolist(), axis=1).tolist(), columns=new_df2.columns[::-1]).iloc[:,::-1]
            out.replace('', float('NaN'), inplace = True)
            out.dropna(axis = 1, how = 'all', inplace = True)

            final = pd.concat([new_df0, new_df1, out], axis = 1)
            final.columns = ['step', 'atom', 'x', 'y', 'z']
            bohr2ang = 0.52917721092            
            final[['x', 'y', 'z']] = final[['x', 'y', 'z']].astype(float) * bohr2ang
            all_res.append(final)

        f.close()
        
    return pd.concat(all_res)

####

def getCoordinate(data, step, size):
    mol_start = step*size
    mol = data.get("atom").values[mol_start:mol_start + size]
    #print(data, dir(mol), mol)
    coords = []
    for size_start in range(mol_start,mol_start + size):
        coordx = float(data.get("x")[size_start:size_start+1].values[:1].real[0])
        coordy = float(data.get("y")[size_start:size_start+1].values[:1].real[0])
        coordz = float(data.get("z")[size_start:size_start+1].values[:1].real[0])

        coords.append([coordx,coordy,coordz])
    return mol, coords

####

def getCoordinateQuantistryToBOPES(coords):
    for i in range(len(coords)):
        coords[i][2] = 0
    return coords

####

# 3-2. Calculate molecule energy by BOPES calculations with Variable MOF atomic distance relative to gas molecule

def run_VQE_BOPES_Energies(molecule_name, multiplicity, charge, masses, atom_pair = (1, 0), freeze_remove=None, run_real=False, perturbation_steps = np.linspace(0.5, 3.0, 150), optimizer = 'COBYLA'):
    # ===
    step = 0
    molecule_data = get_coord(pth + r'/'+molecule_name+'.zip')
    mol, coords = getCoordinate(molecule_data, step, len(masses))
    bopesCoords = getCoordinateQuantistryToBOPES(coords)
    
    gas_molecules = {
        molecule_name: {'symbols': mol,
               'coords': coords,
               'multiplicity': multiplicity,
               'charge': charge,
               'units': DistanceUnit.ANGSTROM,
               'masses': masses,
               'atom_pair': atom_pair, # not available after migration
               'fc_transformer': {
                   'fc_freeze_core': True, 
                   'fc_remove_orbitals': freeze_remove, 
                   },
    #           'as_transformer': {
    #               'as_num_electrons': 0,
    #               'as_num_spatial_orbitals': 2,
    #               'as_active_orbitals': [1, 1],
    #               }
               }
    }

    hyperparameters = {
        molecule_name: {'reps': 2,
               'two_qubit_reduction': True,
               'z2symmetry_reduction': 'auto',
               'perturbation_steps': perturbation_steps,
               'qpe_num_time_slices': 1,
               'qpe_n_ancilliae': 3,
               'atom_pair': atom_pair, # index start from 1
               'optimizer': optimizer,
               'ansatz': 'UCCSD',
               'initial_circuit': 'PauliTwoDesign',
              },

    }

    # ===
    input_value = gas_molecules[molecule_name]
    hyperparam = hyperparameters[molecule_name]
    input_type = 'molecule'
    display_report = True
    display_dict = {}
    reduced = 'FreezeCore'
    basis = 'sto3g'
    mapper_type = 'JordanWignerMapper'
    solver_type = 'numpy_solver'
    method = 'gses'
    seed = _SEED
    _, energy, display_df = get_construct_problem(input_value, hyperparam, input_type, display_report, 
                                                  reduced, basis, mapper_type, solver_type, method, optimizer, seed)
    display(display_df)
    perturb_steps, energy_lst = get_a_molecule_bopes(input_value = input_value, hyperparam = hyperparam, 
                                                     input_type = input_type, reduced = reduced, basis = basis, 
                                                     mapper_type = mapper_type, 
                                                     solver_type = solver_type, method = method, 
                                                     perturbation_steps = perturbation_steps, 
                                                     optimizer = optimizer, display_report = True, seed = _SEED)
    return perturb_steps, energy_lst

####

def create_optuna(molecule_name, multiplicity, charge, masses, atom_pair = (1, 0), freeze_remove=None, run_real=False, perturbation_steps = np.linspace(0.5, 3.0, 150), optimizer = 'COBYLA'):
    # ===
    step = 0
    molecule_data = get_coord(pth + r'/'+molecule_name+'.zip')
    mol, coords = getCoordinate(molecule_data, step, len(masses))
    bopesCoords = getCoordinateQuantistryToBOPES(coords)
    
    gas_molecules = {
        molecule_name: {'symbols': mol,
               'coords': coords,
               'multiplicity': multiplicity,
               'charge': charge,
               'units': DistanceUnit.ANGSTROM,
               'masses': masses,
               'atom_pair': atom_pair, # not available after migration
               'fc_transformer': {
                   'fc_freeze_core': True, 
                   'fc_remove_orbitals': freeze_remove, 
                   },
                #'as_transformer': {
                #    'as_num_electrons': 8,
                #    'as_num_spatial_orbitals': 4,
                #}
               }
    }

    hyperparameters = {
        molecule_name: {'reps': 2,
               'two_qubit_reduction': True,
               'z2symmetry_reduction': None,
               'perturbation_steps': perturbation_steps,
               'qpe_num_time_slices': 1,
               'qpe_n_ancilliae': 3,
               'backend': 'qasm_simulator',
               'atom_pair': atom_pair, # index start from 1
               'optimizer': optimizer,
               'ansatz': 'UCCSD',
               'initial_circuit': 'RealAmplitudes',
              },

    }

    # ===
    input_value = gas_molecules[molecule_name]
    hyperparam = hyperparameters[molecule_name]
   
    return input_value, hyperparam

####

import optuna
# ==========
# Input Dict
# ==========

dict_search_space = {
    'lst_mapper': ['JordanWignerMapper', 'BravyiKitaevMapper', 'ParityMapper'],
    'optimizer': ['COBYLA', 'SLSQP', 'SPSA'],
    'min_reps': 1, 'max_reps': 3,
    'lst_initial_circuit': ['EfficientSU2', 'TwoLocal', 'PauliTwoDesign', 'RealAmplitudes', 'ExcitationPreserving'],
}

# ==========
# Search space
# ==========
def get_search_space(trial, h):
    # Use trial to create your hyper parameter space based 
    # based on any conditon or loops !!
    # 'atom_pair'?
    
    search_space = {
        'sp_qubit_mapper': {'qm_mapper': trial.suggest_categorical('qm_mapper', h['lst_mapper'])},
        'sp_optimizer': {'qm_optimizer': trial.suggest_categorical('optimizer_name', h['optimizer'])},        
        'sp_ansatz': {'reps': trial.suggest_int('reps', h['min_reps'], h['max_reps']),
                      'initial_circuit': trial.suggest_categorical('initial_circuit', h['lst_initial_circuit']),
                     },
        }
    
    return search_space

# ==========
# EarlyStoppingCallback
# ==========
class EarlyStoppingCallback(object):
    """Early stopping callback for Optuna."""

    def __init__(self, early_stopping_rounds: int, direction: str = "minimize") -> None:
        self.early_stopping_rounds = early_stopping_rounds

        self._iter = 0

        if direction == "minimize":
            self._operator = operator.lt
            self._score = np.inf
        elif direction == "maximize":
            self._operator = operator.gt
            self._score = -np.inf
        else:
            ValueError(f"invalid direction: {direction}")

    def __call__(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Do early stopping."""
        if self._operator(study.best_value, self._score):
            self._iter = 0
            self._score = study.best_value
        else:
            self._iter += 1

        if self._iter >= self.early_stopping_rounds:
            study.stop()


import logging, sys, operator     

#study = optuna.create_study(pruner=optuna.pruners.PatientPruner(None, patience=9), direction='minimize')
#early_stopping = EarlyStoppingCallback(early_stopping_rounds=10, direction='minimize')

optuna_steps = []
optuna_energies = []
optuna_labels = []

def objective(trial): # To be replaced by returning min of get_a_molecule_bopes function
    global optuna_steps, optuna_energies, optuna_labels
    training_param_search_space = get_search_space(trial, dict_search_space)
    # Variable Optuna parameters
    mapper = training_param_search_space['sp_qubit_mapper']['qm_mapper']
    optimizer = training_param_search_space['sp_optimizer']['qm_optimizer']
    repetitions = training_param_search_space['sp_ansatz']['reps']
    initial_circuit = training_param_search_space['sp_ansatz']['initial_circuit']
    # Static parameters
    solver_type = 'vqe_simulator_noiseless'
    iterations = 10 #training_param_search_space['sp_iteration']['maxiter']
    perturbation_steps = np.linspace(0.3, 2, 50)
    freeze_remove=range(13,30)
    
    molecule, hyperparameter = create_optuna(molecule_name='Fe2_CO2',multiplicity=1,charge=2,masses=[56, 16, 12, 16], 
                       atom_pair = (0,1), freeze_remove=freeze_remove, run_real = False, 
                       perturbation_steps = perturbation_steps, optimizer = optimizer);
    # Override generated parameters
    hyperparameter['iterations'] = iterations
    hyperparameter['reps'] = repetitions
    hyperparameter['initial_circuit'] = initial_circuit
    label = f" VQE HF Energy {mapper} {optimizer}({iterations}) {repetitions} {initial_circuit}"
            
    print("Optuna running mapper: ", label)
    perturb_steps, energy_lst = get_a_molecule_bopes(input_value = molecule, hyperparam = hyperparameter, input_type = 'molecule',
                                         reduced = 'FreezeCore', 
                                         basis = 'sto3g', 
                                         mapper_type = mapper, 
                                         solver_type = solver_type, method = 'gses',
                                         perturbation_steps = perturbation_steps,
                                         optimizer = optimizer, display_report = False, seed = _SEED)

    optuna_steps.append(perturb_steps)
    optuna_energies.append(energy_lst)
    optuna_labels.append(label)

    return min(energy_lst)

####

# 4. Deployment

def runVQEEnergies(molecule_name, multiplicity, charge, masses, freeze_remove, run_real = False, max_range = 23, optimizer = 'COBYLA', iterations = 100, reps = 1, initial_circuit = 'RealAmplitudes'):
    # ===
    exact_energies = []
    vqe_energies = []

    steps = range(0,max_range)
    for step in steps:
        #print("Starting step:", step, " VQE result:", vqe_energies)
        molecule_data = get_coord(pth + r'/'+molecule_name+'.zip')
        mol, coords = getCoordinate(molecule_data, step, len(masses))
        #print(mol, coords)

        gas_molecules = {
            molecule_name: {'symbols': mol,
                   'coords': coords,
                   'multiplicity': multiplicity,
                   'charge': charge,
                   'units': DistanceUnit.ANGSTROM,
                   'masses': masses,
                   #'atom_pair': (1, 2), # not available after migration
                   'fc_transformer': {
                       'fc_freeze_core': True, 
                       'fc_remove_orbitals': freeze_remove,
                       },
                    #'as_transformer': {
                    #    'as_num_electrons': 8,
                    #    'as_num_spatial_orbitals': 4,
                    #}
            }
        }

        hyperparameters = {
            molecule_name: {'reps': reps,
                   'two_qubit_reduction': True,
                   'z2symmetry_reduction': None,
                   'backend': 'qasm_simulator', #qasm_simulator
                   'optimizer': optimizer,
                   'iterations': iterations,
                   'ansatz': 'UCCSD',
                   'initial_circuit': initial_circuit
                  },

        }


        # ===
        molecule = gas_molecules[molecule_name]
        hyperparam = hyperparameters[molecule_name]
        print(molecule,hyperparam)
        optimizer = optimizer
        # display_report = False
        display_dict = {}

        _, energy, display_df = get_construct_problem(input_value = molecule, hyperparam = hyperparam, 
                                                      input_type = 'molecule', display_report = False, reduced = 'FreezeCore', 
                                                      basis = 'sto3g', mapper_type = 'JordanWignerMapper', 
                                                      solver_type = 'numpy_solver', method = 'gses', optimizer = optimizer, seed = _SEED)
        #print("Classical:", _, energy, display_df )
        exact_energies.append(energy)

        if run_real == False:
            _, energy, display_df = get_construct_problem(input_value = molecule, hyperparam = hyperparam, 
                                                          input_type = 'molecule', display_report = False, reduced = 'FreezeCore', 
                                                          basis = 'sto3g', mapper_type = 'JordanWignerMapper', 
                                                          solver_type = 'vqe_simulator_noisy_0_0', method = 'gses', optimizer = optimizer, seed = _SEED)
        else:
            _, energy, display_df = get_construct_problem(input_value = molecule, hyperparam = hyperparam, 
                                                          input_type = 'molecule', display_report = False, reduced = 'FreezeCore', 
                                                          basis = 'sto3g', mapper_type = 'JordanWignerMapper', 
                                                          solver_type = 'vqe_runtime_real', method = None, optimizer = optimizer, seed = _SEED)
        vqe_energies.append(energy)

    plt.plot(steps, exact_energies, label=molecule_name + " Exact Energy")
    plt.plot(steps, vqe_energies, label=molecule_name + " VQE Hartree Energy")
    plt.xlabel('Step')
    plt.ylabel('Energy')
    plt.legend()
    display(plt.show())
    
####


