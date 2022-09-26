#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import casadi as ca


# In[14]:
def read():

    data = pd.read_csv('staircase.changes')


    # In[15]:


    ddx = data.to_numpy()


    # In[18]:


    fn_str = '' 
    cnt = 0
    for ti, tj, v, dep in zip(ddx[:-1,0], ddx[1:,0], ddx[:-1, 1], ddx[:-1,2]):
        depx = f'+{dep}*(t-{ti})' if not np.isnan(dep) else '' 
        frag = f'ca.if_else(t<{tj}, {v}{depx}, '
        cnt += 1
        fn_str += frag
    fn_str += f'{ddx[-1, 1]}'
    fn_str += ')' * cnt


    # In[26]:


    fn_full = f"ca.Function('vfn', [t], [{fn_str}])"
    return fn_full

def expanded_fn(N):
    fn_str = read()
    
    t = ca.SX.sym('t')
    fn_fn = eval(fn_str)
    fn_fn_n = fn_fn.map(N).expand()
    return fn_fn_n

# In[ ]:




