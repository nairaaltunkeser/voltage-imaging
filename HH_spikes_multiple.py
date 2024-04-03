import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
C_m = 1.0   # membrane capacitance (uF/cm^2)
g_Na = 120.0  # maximum sodium conductance (mS/cm^2)
g_K = 36.0    # maximum potassium conductance (mS/cm^2)
g_L = 0.3     # leak conductance (mS/cm^2)
E_Na = 50.0   # sodium reversal potential (mV)
E_K = -77.0   # potassium reversal potential (mV)
E_L = -54.387 # leak reversal potential (mV)


def I_inject(t):
    return 10*(t>100) - 10*(t>150) + 35*(t>300) - 35*(t>400)+ 10*(t>500) - 10*(t>550)

def rateChange(X, t):
    V1, m1, h1, n1, V2, m2, h2, n2 = X
    
    # Alpha and beta 
    alpha_m = lambda V: 0.1*(V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    beta_m = lambda V: 4.0*np.exp(-(V + 65.0) / 18.0)
    alpha_h = lambda V: 0.07*np.exp(-(V + 65.0) / 20.0)
    beta_h = lambda V: 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    alpha_n = lambda V: 0.01*(V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    beta_n = lambda V: 0.125*np.exp(-(V + 65.0) / 80.0)
    
    # Currents for compartment 1
    I_Na_1 = g_Na * m1**3 * h1 * (V1 - E_Na)
    I_K_1 = g_K * n1**4 * (V1 - E_K)
    I_L_1 = g_L * (V1 - E_L)
    
    # Currents for compartment 2
    I_Na_2 = g_Na * m2**3 * h2 * (V2 - E_Na)
    I_K_2 = g_K * n2**4 * (V2 - E_K)
    I_L_2 = g_L * (V2 - E_L)
    
    # Coupling---Voltage of compartment 1 affecting the current of compartment 2
    coupling = 0.1 * (V1 - V2)  
    
    # Rate of change for compartment 1
    dV1dt = (I_inject(t) - I_Na_1 - I_K_1 - I_L_1 ) / C_m
    dmdt_1 = alpha_m(V1) * (1 - m1) - beta_m(V1) * m1
    dhdt_1 = alpha_h(V1) * (1 - h1) - beta_h(V1) * h1
    dndt_1 = alpha_n(V1) * (1 - n1) - beta_n(V1) * n1
    
    # Rate of change for compartment 2
    dV2dt = (-I_Na_2 - I_K_2 - I_L_2 + coupling) / C_m
    dmdt_2 = alpha_m(V2) * (1 - m2) - beta_m(V2) * m2
    dhdt_2 = alpha_h(V2) * (1 - h2) - beta_h(V2) * h2
    dndt_2 = alpha_n(V2) * (1 - n2) - beta_n(V2) * n2
    
    return [dV1dt, dmdt_1, dhdt_1, dndt_1, dV2dt, dmdt_2, dhdt_2, dndt_2]

# Time 
t = np.arange(0, 850, 0.01)

# Initial conditions
X0_1 = [-65.0, 0.05, 0.05, 0.2]
X0_2 = [-65.0, 0.05, 0.05, 0.2]
initial_conditions = X0_1 + X0_2

# Solve ODE
X = odeint(rateChange, initial_conditions, t)

# Extract variables for both compartments
V1, m1, h1, n1, V2, m2, h2, n2 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5], X[:, 6], X[:, 7]

fig, axs = plt.subplots(4, 1, figsize=(10, 12))

axs[0].plot(t, V1, label='Compartment 1')
axs[0].plot(t, V2, label='Compartment 2')
axs[0].set_title('Membrane Potential (Vm)')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Membrane Potential (Vm)')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t, m1, label='m (Compartment 1)')
axs[1].plot(t, h1, label='h (Compartment 1)')
axs[1].plot(t, n1, label='n (Compartment 1)')
axs[1].set_title('Gating Variables (Compartment 1)')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Value')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(t, m2, label='m (Compartment 2)')
axs[2].plot(t, h2, label='h (Compartment 2)')
axs[2].plot(t, n2, label='n (Compartment 2)')
axs[2].set_title('Gating Variables (Compartment 2)')
axs[2].set_xlabel('Time (ms)')
axs[2].set_ylabel('Value')
axs[2].legend()
axs[2].grid(True)

axs[3].plot(t, I_inject(t), 'r', label='Injected Current')
axs[3].set_title('Injected Current')
axs[3].set_xlabel('Time (ms)')
axs[3].set_ylabel('Current (mA/cm$^2$)')
axs[3].legend()
axs[3].grid(True)

plt.tight_layout()
plt.show()