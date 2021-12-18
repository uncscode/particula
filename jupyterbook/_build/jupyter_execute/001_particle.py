#!/usr/bin/env python
# coding: utf-8

# # 001 `Particle` class

# In[1]:


get_ipython().run_line_magic('cd', '..')


# In[67]:


import imp
from aerosol_dynamics import particle


# In[68]:


imp.reload(particle)


# In[69]:


get_ipython().run_line_magic('pinfo', 'particle.Particle.diffusive_knudsen_number')


# In[70]:


particle_a = particle.Particle(
    name='a', radius=1.0e-9, density=1.0e3, charge=0
)


# In[71]:


particle_b = particle.Particle(
    name='b', radius=2.0e-9, density=1.1e3, charge=-1
)


# In[72]:


particle_a.name()


# In[73]:


particle_a.mass()


# In[74]:


particle_a.radius()


# In[75]:


particle_a.density()


# In[76]:


particle_a.charge()


# In[77]:


particle_a.knudsen_number()


# In[78]:


particle_a.slip_correction_factor()


# In[79]:


particle_a.friction_factor()


# In[80]:


particle_a.reduced_mass(particle_b)


# In[81]:


particle_a.reduced_friction_factor(particle_b)


# In[82]:


from aerosol_dynamics import physical_parameters as pp


# In[83]:


imp.reload(pp)


# In[84]:


pp.BOLTZMANN_CONSTANT


# In[85]:


particle_a.coulomb_potential_ratio(particle_b)


# In[86]:


particle_a.coulomb_enhancement_kinetic_limit(particle_b)


# In[87]:


particle_a.coulomb_enhancement_continuum_limit(particle_b)


# In[88]:


particle_a.diffusive_knudsen_number(particle_b)


# In[ ]:





# In[89]:


particle.Particle.diffusive_knudsen_number(particle_a, particle_b)


# In[90]:


particle_a.dimensionless_coagulation_kernel_hard_sphere(particle_b)


# In[91]:


particle_a.collision_kernel_continuum_limit(particle_b)


# In[11]:


particle_a.density()


# In[ ]:




