# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy
import sys
import pandas as pd

# Initialize the compositional order parameter
def InitializeComposition(N, phi_zero):	
	# Initialize the phi array taking values between
	# -1 (oil) and 1 (water). 
	phi = np.random.randn(N,N) * 0.1 + np.ones((N,N)) * (phi_zero)
	M = 0.1
	dt = 2
	dx = 1
	a = 0.1
	k = 0.1
	return phi, a, k, M, dt, dx
	
# Find the laplacian of the grid for the required 
# time evolution
def Laplacian(grid, M, dt, dx):
	# Apply the required formula for 2D space using the 
	# roll function from numpy library
	help_grid = np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) + np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) - 4 * grid
	
	# Now implement the space and time discretization for working
	# out the laplacian operator
	laplacian = (help_grid) / (dx**2) 
	return laplacian 

# Find the squared gradient of the grid	
def SquareGradient(grid):
	# Calculate the gradient on the axis 0 firstly by
	# using the roll function from numpy
	grad_x = 0.5 * ( np.roll(grid, -1, axis=0) - np.roll(grid, 1, axis=0) )
	
	# Perform the same operation for the gradient on 
	# y-axis with the roll function
	grad_y = 0.5 * ( np.roll(grid, -1, axis=1) - np.roll(grid, 1, axis=1) )
	
	# Work out the overall square gradient and return it
	grad_sq = grad_x**2 + grad_y**2
	return grad_sq
	
# Find the free energy in the grid
def FreeEnergy(grid, a, k):

	# Return the free energy using the formula depending on
	# the grid itself and the squared gradient of the grid
	F = (-a/2) * grid**2 + (a/4) * grid**4 + (k/2) * SquareGradient(grid)
	return F
	
# Create function for finding the chemical potential of the system
def ChemicalPotential(grid, a, k, M, dt, dx):
	# Apply the formula for working out the chemical potential
	# Use the laplacian operator for this
	mu = -a * grid + a * grid**3 - k * Laplacian(grid, M, dt, dx)
	return mu
	
def UpdateComposition(grid, a, k, M, dt, dx):
	# Apply the time evolution using the chemical potential
	# formula
	mu = ChemicalPotential(grid, a, k, M, dt, dx)
	grid += M * dt * Laplacian(mu, M, dt, dx)
	return grid
	
def PlotComposition(N, phi_zero):
	# Initialize the composition
	emulsion, a, k, M, dt, dx = InitializeComposition(N, phi_zero)
	
	# Iterate until the limit of 100,000 iterations is reached.
	# Create lists for the time passed and the free energy at one
	# particular time frame
	count = 0
	time = 0
	no_iterations = 100000
	T = np.arange(no_iterations)
	F = np.zeros(no_iterations)
	
	while(time < no_iterations):
		
		# Perform an animation of the emulsion phenomenon between oil and water.
		# Sum the free energy elements in the whole grid and store all the values
		# in a .csv file which will be later used for plotting
		if((time % 100) == 0):
			plt.cla()
			image = plt.imshow(emulsion, animated=True, vmin=-1, vmax=1)
			
			# Add the colorbar
			if(time == 0):
				plt.colorbar()
				
			plt.draw()
			plt.pause(0.001)
			
			print("Free energy of the system: " + str(np.sum(FreeEnergy(emulsion, a, k))))
			print(time)
		
		F[time] = np.sum(FreeEnergy(emulsion, a, k))	
		time += 1
		
		# Update the mixture
		emulsion = UpdateComposition(emulsion, a, k, M, dt, dx)
		
	# Save the data into a .csv file
	df = pd.DataFrame({"Time " : T, "Free Energy " : F})
	df.to_csv("FreeEnergy_phi_minus_half_N_100.csv")
		
	
def main():

	# Read the values from the terminal first of all
	N = int(sys.argv[1])
	phi_zero = float(sys.argv[2])
	
	PlotComposition(N, phi_zero)
	
main()
