# Import the required libraries
from jax import numpy as jnp, random, value_and_grad, jit, grad, jacrev
import matplotlib.pyplot as plt
from pytket.circuit import Circuit
from sympy import Symbol
import numpy as np
import qujax
from pytket.extensions.qujax import tk_to_qujax#, tk_to_qujax_symbolic
from pytket.circuit.display import render_circuit_jupyter
from pytket.extensions.qiskit import AerStateBackend
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ipywidgets import interactive, HBox, VBox
import scipy as sp
random_key = random.PRNGKey(0)
import networkx as nx
import jax as jax
import matplotlib.pyplot as plt
from pytket.utils import probs_from_counts 
from pytket.extensions.qiskit import AerBackend, AerStateBackend

# Function definitions
# All the information about the ansatz can be found on the presentation paper.
def Ansatz(num_qubits : int, exp : int, layer : int):
  """
  This function creates a quantum circuit ansatz base on the
  three levels of expressability.
  Str:
    num_qubits(int) : number of qubits for the ansatz.
    exp(int) : expressability circuit.
    layer(int) : number of layers for the ansatz.

  Returns:
    qc: quantum circuit.
    para(list): list with all the parameters in the ansatz.
  """

  para = []                              # storage all the parameters.

  # 1st ansatz.
  if exp == 0:
    qc = Circuit(num_qubits)
    p = 0
    for j in range(layer):
      for i in range(0, num_qubits):
        para.append(Symbol(f'p_{p}'))
        qc.Rx(para[p]/np.pi, i)
        p += 1
      for i in range(0, num_qubits):
        para.append(Symbol(f'p_{p}'))
        qc.Rz(para[p]/np.pi, i)
        p += 1
  # 2nd ansatz.
  elif exp == 1:
    p = 0
    qc = Circuit(num_qubits)
    for i in range(0,num_qubits):
      para.append(Symbol(f'p_{p}'))
      qc.Rx(para[p]/np.pi, i)
      p += 1

    for i in range(0,num_qubits):
      para.append(Symbol(f'p_{p}'))
      qc.Ry(para[p]/np.pi, i)
      p += 1

    for i in range(num_qubits):
      for j in range(num_qubits):
        if i == j:
          p = p 
        else:
          para.append(Symbol(f'p_{p}'))
          qc.CRx(para[p]/np.pi, i, j)
          p += 1
      
    for i in range(0,num_qubits):
      para.append(Symbol(f'p_{p}'))
      qc.Rx(para[p]/np.pi, i)
      p += 1

    for i in range(0,num_qubits):
      para.append(Symbol(f'p_{p}'))
      qc.Ry(para[p]/np.pi, i)
      p += 1
  # 3rd ansatz.
  if exp == 2:
    p = 0
    qc = Circuit(num_qubits)
    for j in range(layer):
      for i in range(0, num_qubits):
        para.append(Symbol(f'p_{p}'))
        qc.Rx(para[p]/np.pi, i)
        p += 1
      for i in range(num_qubits):
        if i+1 == num_qubits:
          qc.CX(i, 0)
        else:
          qc.CX(i, i+1)
      for i in range(0, num_qubits):
        para.append(Symbol(f'p_{p}'))
        qc.Rx(para[p]/np.pi, i)
        p += 1

  return qc, para

# Definition of the cost function for FVQE.
def cost_fun(st, st_old, f_tau, f_tau_2):
    """
    This function return the value of the cost function
    Args:
      st: quantum state
      st_old: old quantum state
      f_tau: value for the filter tau.
      f_tau_2: value for the filter tau square.
    Return:
      cost function.
    """
    st_old_conj = jnp.conj(st_old)
    filt_st = jnp.multiply(f_tau, st)
    aux = jnp.multiply(st_old_conj, filt_st).sum()
    exp_f_2_old = jnp.multiply(st_old_conj, jnp.multiply(f_tau_2, st_old)).sum()
    return (1 - aux.real/jnp.sqrt(exp_f_2_old)).real

def cost(E, bitstr):
    """
    Cost function for the MaxCut problem.
    """
    cost = 0
    bitstr = "0" + bitstr
    for i in range(len(E)):
        cost += -1*E[i][2]*((int(bitstr[E[i][0]]) + int(bitstr[E[i][1]]))%2)
    return cost

def cost_mod(E, bitstr, psuedo_low):
    """
    Cost function modified. (range in [0,1])
    """
    new = -(cost(E, bitstr)-psuedo_low)/psuedo_low
    return new

def Graph(num_qubits):
  """
  Definition of the graph base on the number of qubits.
  """
  deg_seq = [3]*(num_qubits+1)
  G = nx.random_degree_sequence_graph(deg_seq, seed = 350)
  weights = jax.random.uniform(random_key, [len(G.edges())])
  i = 0
  for e in G.edges():
      G[e[0]][e[1]]['weight'] = weights[i]
      i+=1
  E = []
  for i in range(len(G.edges)):
      E.append([list(G.edges)[i][0], list(G.edges)[i][1], float(weights[i])])
  return E, G

def expectation_value_n(parameter, circ, filter_func, tau, backend, E, type, p_keys, pseudo_low, num_shots):
  """
  Calculate the expectation value for the n value.
  """
  s_map = {Symbol(f'p_{k}'): float(parameter[k]) for k in range(len(p_keys))}
  new_circ = circ.copy()
  new_circ.measure_all()
  new_circ.symbol_substitution(s_map)
  compiled_circ = backend.get_compiled_circuit(new_circ)
  handle = backend.process_circuit(compiled_circ, num_shots)
  counts = backend.get_result(handle).get_counts()
  probs  = probs_from_counts(counts)
  exp_val = 0
  for j in probs.keys():
    stri = ""
    for x in j:
      stri += str(x)
    e = cost_mod(E, stri, pseudo_low)
    exp_val += filter_func(e, tau)*probs[j]
  return exp_val

def expectation_value_pm(parameter, circ, filter_func, tau, backend, E, typeo, p_keys, pseudo_low, num_shots):
  """
  Calculate the expectation value.
  """
  grad = []
  for i in range(len(parameter)):
    if typeo == "plus":
      s_map = {Symbol(f'p_{k}'): float(parameter[k]) if k!= i else (float(parameter[k]) + np.pi/2) for k in range(len(p_keys))}
    elif typeo == "minus":
      s_map = {Symbol(f'p_{k}'): float(parameter[k]) if k!= i else (float(parameter[k]) - np.pi/2) for k in range(len(p_keys))}
    new_circ = circ.copy()
    new_circ.measure_all()
    new_circ.symbol_substitution(s_map)
    compiled_circ = backend.get_compiled_circuit(new_circ)
    handle = backend.process_circuit(compiled_circ, num_shots)
    counts = backend.get_result(handle).get_counts()
    probs  = probs_from_counts(counts)
    exp_val = 0
    for j in probs.keys():
      stri = ""
      for x in j:
        stri += str(x)
      e = cost_mod(E, stri, pseudo_low)
      exp_val += filter_func(e, tau)*probs[j]
    grad.append(exp_val)
  return grad

def paramater_shift_rule(parameter, circ, filter_func, filter_2, tau, backend, E, p_keys, pseudo_low, num_shots):
  """
  Calculate the gradient using the parameter shift rule.
  """
  exp_plus = expectation_value_pm(parameter, circ, filter_func, tau, backend, E, "plus", p_keys, pseudo_low, num_shots)
  exp_minus = expectation_value_pm(parameter, circ, filter_func, tau, backend, E, "minus", p_keys, pseudo_low, num_shots)
  exp_lower = expectation_value_n(parameter, circ, filter_2, tau, backend, E, "normal", p_keys, pseudo_low, num_shots)
  gradient = [exp_plus[t] - exp_minus[t] for t in range(len(exp_plus))]
  gradient = gradient/(4*np.sqrt(exp_lower)) # normalize
  return gradient

# Definition of different filters.
def filter_inv(e, tau): return np.power(e, -tau)
def filter_inv2(e, tau):  return np.power(e, -2*tau)

# Hyperparameters

def FVQE(tau, lr, steps, num_qubits, expressability, filter, psr, num_shots, backend, info, layer):

  circ, p_keys = Ansatz(num_qubits, expressability, layer)
  param_to_st = tk_to_qujax(circ, {p_keys[i]: i for i in range(len(p_keys))})
  param_to_st_pi = lambda params: param_to_st(params/jnp.pi)
  params = np.random.random(len(p_keys))
  params = jnp.array(list(params))
  param_to_st(params)

  param_to_cost = lambda param, old_param, f_tau, f_tau_2: cost_fun(param_to_st_pi(param).flatten(), 
                                                                  param_to_st_pi(old_param).flatten(), f_tau, f_tau_2)

  grad_jit = jit(jacrev(param_to_cost))

  E, G = Graph(num_qubits)

  min_cost = 0
  min_str = []
  min_value = []

  for binum in range(2**(num_qubits)):
      bitstr = str(bin(binum)[2:])
      while len(bitstr) < (num_qubits):
          bitstr = '0' + bitstr
      if cost(E, bitstr) == min_cost:
          min_str.append(bitstr)
          min_value.append(binum)
      if cost(E, bitstr) < min_cost:
          min_cost = cost(E, bitstr)
          min_str = [bitstr]
          min_value = [binum]
 
  psuedo_low = min_cost - 0.05

  min_cost = 1
  min_str = []
  f_one = []
  f_two = []

  for binum in range(2**(num_qubits)):
      bitstr = str(bin(binum)[2:])
      while len(bitstr) < (num_qubits):
          bitstr = '0' + bitstr
      if cost_mod(E, bitstr, psuedo_low) == min_cost:
          min_str.append(bitstr)
      if cost_mod(E, bitstr, psuedo_low) < min_cost:
          min_cost = cost_mod(E, bitstr, psuedo_low)
          min_str = [bitstr]

      ### Define the filter function: 
      if filter == 0:
        f_one.append(np.power(cost_mod(E, bitstr, psuedo_low), -tau))
        f_two.append(np.power(cost_mod(E, bitstr, psuedo_low), -2*tau))
      elif filter == 1:
        f_one.append(np.power(np.cos(cost_mod(E, bitstr, psuedo_low)), tau))
        f_two.append(np.power(np.cos(cost_mod(E, bitstr, psuedo_low)), 2*tau))
      elif filter == 2:
        f_one.append(np.power(cost_mod(E, bitstr, psuedo_low), -tau))
        f_two.append(np.power(cost_mod(E, bitstr, psuedo_low), -2*tau))

  #### INITIAL PARAMETERS
  if expressability == 0:
    params = jnp.array([np.pi/2]*num_qubits+[0.0]*(len(p_keys)-num_qubits))
  elif expressability == 1:
    params = jnp.array([0.0]*(len(p_keys)-num_qubits)+[np.pi/2]*num_qubits)
  elif expressability == 2:
    params = jnp.array([0.0]*(len(p_keys)-num_qubits)+[np.pi/2]*num_qubits)

  f_tau = jnp.array(list(f_one))
  f_tau_2 = jnp.array(list(f_two))

  params_evol = [params]

  prob = []

  ### Information about the circuit:
  if info == True:
    print("Number of Parameters: {}".format(len(p_keys)))
    print("Initial Parameters: {}".format(params))
    print("Graph information: {}".format(E))
    print("Graph: ")
    colors = ['r' for node in G.nodes()]
    default_axes = plt.axes(frameon=True)
    pos = nx.spring_layout(G)
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=nx.get_edge_attributes(G, 'weight'))
    nx.draw_networkx(G, node_color=colors, node_size=1600, ax=default_axes, pos=pos)

    render_circuit_jupyter(circ)

  for i in range(steps):
      # Get the state
      # Compute the gradients
      if psr == 0:
        g = grad_jit(params, params, f_tau, f_tau_2)
      elif psr == 1:
        g = -paramater_shift_rule(params, circ, filter_inv, filter_inv2, tau, backend, E, p_keys, psuedo_low, num_shots)

      # we normalize the gradient. Any constant factor can be absorved by the learning rate parameter
      g = g / jnp.linalg.norm(g)

      # update params using gradient descent
      params -= lr * g
      params_evol.append(params)

      state = param_to_st_pi(params)
      prob.append(np.linalg.norm(np.array(param_to_st_pi(params).flatten())[min_value]) ** 2)

  return prob, len(p_keys)