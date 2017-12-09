from scipy import constants as cs #constants of physics
import emcee #MCMC
import matplotlib.pyplot as plt #Plotting
import math #Basic maths
import numpy as np #More maths
from astropy import constants as asc #Astronomical constants
import datetime #Timing calculations
import corner #Corner plots

def sini_mcmc(period=2.425,eperiod=0.003,vsini=50.9,evsini=0.8,radii_bounds=[0.8,1.2],nMonte=int(1e4),nwalkers=12,nburnin=int(1e3),nThreads=4,figurename='corner.png'):
	#Period in hours
	#v sin i in km/s
	#Radii in R_Jup
	
	#Determine bounds, parameters and scaling for MCMC exploration
	parameters_estimates = [0.0,period,0.0]
	parameters_scale = [0.0,eperiod/2.0,0.0]
	parameters_labels = ['Radius (RJup)','Period (h)','Inclination (deg)']
	
	#Create some initial chains with random noise + estimates
	ndim = len(parameters_estimates)
	initial_noise = np.random.randn(ndim*nwalkers).reshape((nwalkers,ndim))
	initial_cube = initial_noise*(np.tile(parameters_scale,(nwalkers,1))) + np.tile(parameters_estimates,(nwalkers,1))
	#Force periods to positive values
	
	#Change radius random to uniform
	initial_cube[:,0] = np.random.uniform(low=radii_bounds[0],high=radii_bounds[1],size=nwalkers)
	#Change inclination random to uniform
	initial_cube[:,2] = np.random.uniform(low=0.,high=90.,size=nwalkers)
	initial_cube[:,1] = abs(initial_cube[:,1])
	
	#Initiate the MCMC sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnp_sini_mcmc, args=[vsini,evsini, period, eperiod, radii_bounds], threads=nThreads)
	
	#Launch an initial burn-in phase
	print('Initiating burn-in phase ('+str(int(nburnin))+' steps)...')
	before_burnin = datetime.datetime.now()
	pos, prob, state = sampler.run_mcmc(initial_cube, int(nburnin))
	after_burnin = datetime.datetime.now()
	print('Time to complete burn-in phase: ', after_burnin-before_burnin)
	
	#Reset sampler properties
	sampler.reset()
	
	#Run MCMC
	print('Initiating main phase ('+str(int(nMonte))+' steps) ...')
	before_main = datetime.datetime.now()
	sampler.run_mcmc(pos, int(nMonte))
	after_main = datetime.datetime.now()
	print('Time to complete main phase: ', after_main-before_main)
	
	#Create a corner plot for parameters
	flat_chains = sampler.chain.reshape(nwalkers*int(nMonte),ndim)
	corner.corner(flat_chains,labels=parameters_labels)
	plt.savefig(figurename)
	avg_radius = np.average(flat_chains[:,0])
	std_radius = np.std(flat_chains[:,0])
	avg_period = np.average(flat_chains[:,1])
	std_period = np.std(flat_chains[:,1])
	avg_inclination = np.average(flat_chains[:,2])
	std_inclination = np.std(flat_chains[:,2])
	print('Radius : '+'{0:.2f}'.format(avg_radius)+' +/- '+'{0:.2f}'.format(std_radius)+' R_Jup')
	print('Period : '+'{0:.4f}'.format(avg_period)+' +/- '+'{0:.4f}'.format(std_period)+' h')
	print('Inclination : '+'{0:.1f}'.format(avg_inclination)+' +/- '+'{0:.1f}'.format(std_inclination)+' degrees')
	
def lnp_sini_mcmc(parameters,measured_vsini,error_vsini,measured_period,error_period, radii_bounds):
	
	#Reject unphysical inclinations or periods
	if parameters[2] < 0.0 or parameters[2] > 90.0:
		return -float('inf')
	if parameters[1] < 0.0:
		return -float('inf')
	#Reject radii outside of bounds
	if parameters[0] < radii_bounds[0] or parameters[0] > radii_bounds[1]:
		return -float('inf')
	
	#Read model parameters and convert units
	radius = parameters[0]*6.9911e4 #Radius in km
	period = parameters[1]*3600.0 #Period in seconds
	inclination = parameters[2]
	
	#Compute geometric Bayesian prior on inclination
	sini = math.sin(math.radians(inclination))
	prior_i = sini
	
	#Predict vsini from model parameters (in km/s)
	model_vsini = 2.0*math.pi*radius*sini/period
	
	#Calculate N-sigma deviations from measurement
	#Treat asymmetric error bars separately
	if np.size(error_vsini) == 1:
		nsigma_vsini = (model_vsini - measured_vsini)/error_vsini
	if np.size(error_vsini) == 2:
		if model_vsini <= measured_vsini:
			nsigma_vsini = (model_vsini - measured_vsini)/error_vsini[0]
		else:
			nsigma_vsini = (model_vsini - measured_vsini)/error_vsini[1]
	
	
	nsigma_period = (parameters[1] - measured_period)/error_period
	
	#Convert to posterior ln probability assuming Gaussian error bars
	lnP = -nsigma_vsini**2/2.0 - nsigma_period**2/2.0 - math.log(prior_i)
	
	return lnP