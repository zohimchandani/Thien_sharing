
# !apt-get update
# !apt-get install sudo
# !echo cuda-quantum | sudo -S apt-get install -y cuda-toolkit-11.8 && python3 -m pip install cupy
# !pip install mpi4py


#Run with: mpirun -n 4 --allow-run-as-root python3 qsvm.py

import cudaq
import cupy as cp 
import mpi4py
from mpi4py import MPI

# print(cudaq.__version__)

# cupy_version = cp.__version__
# print("CuPy Version:", cupy_version)

# mpi4py_version = mpi4py.__version__
# print("mpi4py Version:", mpi4py_version)

#CUDA Quantum Version latest (https://github.com/NVIDIA/cuda-quantum 1c434b15044e9fe1df32af4a00d662aa270aa69e)
#CuPy Version: 13.0.0
#mpi4py Version: 3.1.5

cp.random.seed(5)

cudaq.set_target('nvidia')

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_ranks = comm.Get_size()


m = 4                #number of data samples 
n = 2                #number of variational params 

gpus_per_node = 4
num_of_nodes = 1
num_of_gpus = num_of_nodes * gpus_per_node
qubit_count = 2

# Make sure each rank uses a different GPU
# cp.cuda.runtime.setDevice(rank % gpus_per_node)
# print(f'rank {rank} running with GPU {cp.cuda.Device().pci_bus_id}')


kernel, parameters = cudaq.make_kernel(list)
qubits = kernel.qalloc(qubit_count)
kernel.h(qubits)
kernel.rx(parameters[0], qubits[0])
kernel.ry(parameters[1], qubits[1])



if rank == 0: 
    
    # define the parameters
    param_vals = cp.random.rand(m,n)
    states = cp.zeros((m, 2**qubit_count), dtype = complex)
    
    # split the parameters into nested lists 
    param_vals = cp.array_split(param_vals, num_ranks)
    states = cp.array_split(states, num_ranks)
    
    assert len(param_vals) == len(states) == num_ranks
    

else: 
    
    param_vals = None
    states = None 
   
# distribute the parameters across the available gpus 
param_vals_split = comm.scatter(param_vals, root = 0) 
states_split = comm.scatter(states, root = 0) 

print('rank', rank, 'has' , param_vals_split.shape[0], 'data sample and', states_split.shape[0], 'state' )

assert param_vals_split.shape[0] == states_split.shape[0]


#quantum computation 
for i in range(param_vals_split.shape[0]): 
    
    states_split[i] = cp.array(cudaq.get_state(kernel, param_vals_split[i]))

# gather the results from the different ranks 
results = comm.gather(states_split, root = 0)

#post processing to calculate the gram matrix which is the input to SVM 

if rank == 0: 
    
    kets = cp.concatenate(results)

    assert kets.shape == (m, 2**qubit_count), 'gathered data does not match dimensions of problem'

    bras = cp.transpose(cp.conj(kets))       #bras for all the states corresponding to each data input 
    gram_matrix = cp.zeros((m,m))

    for i in range(m): 
        for j in range(m):
        
            ith_bra = bras[:,i].reshape(1,-1)   

            jth_ket = kets[j].reshape(-1,1)  

            gram_matrix[i][j] = cp.abs(cp.matmul(ith_bra, jth_ket))**2

            
    print(gram_matrix)





