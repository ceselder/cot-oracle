from audioop import cross
import numpy as np

T_CoT, D = 10, 512
T_prompt = 20
L = 12
cot = np.empty((T_CoT, D))
oracle_prompt = np.empty((T_prompt, D))
h_cot = cot
h_orac = oracle_prompt

for i_l in range(L):
    h_cot = self_att(h_cot, W_q, W_k, W_v)
    h_orac = self_att(h_orac, W_q, W_k, W_v)
    
    h_cot = MLP(h_cot)
    h_orac = MLP(h_orac)
    
    h_orac = cross_att(h_orac, h_cot, W_q_cross, W_k_cross, W_v_cross)

