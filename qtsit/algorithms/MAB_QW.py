import numpy as np
import qiskit
# Initializing random walks
import matplotlib as mpl
import numpy as np
from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister, execute
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit import Aer, transpile
import qiskit.quantum_info as qi
from qiskit import IBMQ



class walkq2:
    def __init__(self, size, initial_pos, initial_theta, steps,j,k,gt):
        self.size = size
        self.n = int(np.ceil(np.log2(size)))
        self.j = j
        self.k = k
        self.result = ""
        self.steps = steps
        self.toss_val = 0.1
        self.walking_space = []
        self.initial_theta = initial_theta
        for i in range(size):
            self.walking_space.append(initial_theta)
        self.pos = []
        self.pos_index = initial_pos
        self.ground_truth = gt
        self.emp_truth=[]
        for i in range(size):
          self.emp_truth.append(0.0)
        self.happiness=[0 for i in range(self.size)]
        self.hits=[0 for i in range(self.size)]
        self.cumsum = 0
        self.init_state = str(bin(initial_pos)[2:].zfill(5))
        # self.qnodes = QuantumRegister(self.n,'qc')
        # self.qsubnodes = QuantumRegister(2,'qanc')
        # self.csubnodes = ClassicalRegister(2,'canc')
        # self.cnodes = ClassicalRegister(self.n,'cr')
        self.q_walk = QuantumCircuit(self.n+2, self.n)

    def incremment_gate(self): # n = number of walking qubits
      qc = QuantumCircuit(self.n+1, name = "Inc")
      qc.cx(0,1)
      qc.ccx(0,1,2)
      for i in range(self.n+1):
        if (i > 2):
          qc.mcx([j for j in range(i)], i)
      return qc.to_gate()

    def decrement_two_coins(self):
      qc = QuantumCircuit(self.n+2, name = "dec")
      for i in range(self.n, 0, -1):
        if (i > 1):
          qc.x([j for j in range(2,i+1)])
          qc.mcx([j for j in range(i+1)], i+1)
          qc.x([j for j in range(2,i+1)])
      qc.ccx(0,1,2)
      #print(qc)
      return qc.to_gate()

    def walkk(self): #prepares the walk circuit for a given j'
      self.q_walk.clear()
      self.q_walk.prepare_state(self.init_state,[i+2 for i in range(self.n)])
      for i in range(self.steps): # makes the circuit for steps number of steps
        self.q_walk.u(self.walking_space[self.pos_index],0,0,0)
        self.q_walk.u(self.walking_space[self.pos_index],0,0,1)
        self.q_walk.append(self.incremment_gate(), [i+1 for i in range(self.n+1)])
        self.q_walk.x(1)
        self.q_walk.append(self.decrement_two_coins(), [i for i in range(self.n+2)])
        self.q_walk.x(1)
      for i in range(self.n):
        self.q_walk.measure(i+2,i)
        
      
      backend = Aer.backends(name = 'qasm_simulator')[0]
      result = execute(self.q_walk, backend, shots=1).result().get_counts(self.q_walk)
      self.pos_index = int(max(result).replace(" ", ""),2)

    def slotMachine(self):
        x = np.random.binomial( n=1, p= self.ground_truth[self.pos_index])
        return x

    def explore(self):
        max=-1
        a=5
        b=6
        for i in range(self.j):
            self.walkk()
            x = self.slotMachine()
            if(x):
                self.happiness[self.pos_index]+=1
            self.hits[self.pos_index]+=1
            self.emp_truth[self.pos_index] = self.happiness[self.pos_index]/self.hits[self.pos_index]
            for j in range(self.size):

              self.walking_space[j] = self.initial_theta*np.exp(-a*np.power(self.emp_truth[j],b))
            self.pos_index = np.array(self.emp_truth).argmax()
            self.init_state = str(bin(self.pos_index)[2:].zfill(5))

    def main(self):

      self.explore()

      return sum(self.happiness)


T = 16 #32
result = []
ground_truth = []
for i in range(32):
  if((i+1)%2==0):
    ground_truth.append(0.7)
  if((i+1)%2!=0):
    ground_truth.append(0.5)
ground_truth[13] = 0.9
ground_truth[14] = 0.1
k=500 #500
import concurrent.futures
initial_theta = 1.257
def process_iteration(j):
    walker = walkq2(32, 0, initial_theta, i, 1000, k, ground_truth) #(size, initial_pos, initial_theta, steps,j,k,gt)
    return walker.main()

for i in range(2,T+1,2):#this shud als

  cumsum=0
  with concurrent.futures.ThreadPoolExecutor() as executor:
      results = executor.map(process_iteration, range(k))
  # results=walker.main()
  cumsum = sum(results)
  result.append(cumsum / k) #k = 500 actually
  print(result)
T_array = [i for i in range(2,T+1,2)]

