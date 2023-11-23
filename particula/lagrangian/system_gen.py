"""Exploring the pytorch package for lagrangian particle systems."""

from networkx import neighbors
import torch
import numpy as np





TOTAL_NUMBER_OF_PARTICLES = 100
TIME_STEP = 0.001
RADIUS = 10
DOMAIN_RADIUS = 100

position = torch.rand(3, TOTAL_NUMBER_OF_PARTICLES, dtype=torch.float16) * DOMAIN_RADIUS
velocity = torch.rand(3, TOTAL_NUMBER_OF_PARTICLES, dtype=torch.float16) * 500
force = torch.zeros(3, TOTAL_NUMBER_OF_PARTICLES, dtype=torch.float16)

mass = torch.ones(TOTAL_NUMBER_OF_PARTICLES, dtype=torch.float16) * 1
charge = torch.ones(TOTAL_NUMBER_OF_PARTICLES, dtype=torch.float16) * 1
density = torch.ones(TOTAL_NUMBER_OF_PARTICLES, dtype=torch.float16) * 1

position.requires_grad = True
velocity.requires_grad = True
mass.requires_grad = True

gravity = torch.tensor([0, -9.81, 0]).repeat(TOTAL_NUMBER_OF_PARTICLES, 1).transpose(0, 1)




# update position from velocity using leapfrog
position = position + TIME_STEP * velocity + 0.5 * TIME_STEP * TIME_STEP * force / mass
# update velocity from force using leapfrog
velocity = velocity + 0.5 * TIME_STEP * force / mass

# calculate force for gravity
force = force + mass * gravity

# calculate collisions of particles with each other, Find nearest neighbours




shape = position.shape
print('done')