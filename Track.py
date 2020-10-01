import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from EM_Tools import *
import os

#####################
# Mock Data 		#
#####################
L = np.load('data/LORs_K4.npy')	# LORs
# L[:N,:1+2*3]		: float		: lines array
#						L[0,:]		--> times
#						L[1:4,:]	--> feature points
#						L[4:,:]		--> directions
X = np.load('data/Pos_K4.npy')	# Particles trajectories
# X[:,:Kp,:4]		: float		: Positions array
#						X[:,:,0]	--> times
#						X[:,:,1:]	--> 3D coordinates
Kp = X.shape[1]	#	: integer	: number of particles

#########################
# PEPT-EM parameters	#
#########################
order=1	# 			: integer	: order of the centroïds
alpha=0 # 			: float		: threshold 
Kc=6	# 			: integer	: number of clusters
n_step_init = 50 # 	: integer	: number of EM steps for the itialization
n_step_track = 10 # : integer	: number of EM steps for tracking  

##########################
# Time frames parameters #
##########################
frames_duration = 40 #	: float		: frames' duration or "shutter speed "
frames_overlap = 20	#	: float		: overlaping time between successive frames
nframes = np.int_((L[-1,0]-L[0,0])//(frames_duration-frames_overlap)-1)	
# 						: integer	: number of frames

#########################
# Ploting parameters	#
#########################
toPlot=True	#			: logical	: flag enabling ploting on the fly
bounds = (-150,150,-150,150,-150,150)
#(xmin,xmax,ymin,ymax,zmin,zmax): tuple : boundaries of the ploting frame
#####################
# Storage file		#
#####################
ToSave = True #			: logical	: flag enabling saving of the results
Recompute = True #		: logical	: flag enabling computating again
fid='data/Tracked_Kp%d' %Kp
fid+='_Kc%d' %Kc
fid+='_ord%d' %order
fid+='.npz' #			: char		: saving file name
if os.path.isfile(fid) and not Recompute:
	x = np.load(fid)['x']
	# x[:nframes,:Kc,:3*(o+1)]	: float		: inferred centroids coordinates array
	# 							x[:,:,:3]		--> positions
	# 							x[:,:,3:6]		--> velocities
	# 							x[:,:,6:]		--> acceleration, ect.	
	s = np.load(fid)['s']
	# s[:nframes,:Kc]			: float		: inferred clusters variances
	r = np.load(fid)['r']
	# r[:nframes,:Kc]			: float		: inferred clusters weights
	t = np.load(fid)['t']
	# t[:nframes]				: float		: times
else:
#############
# Main loop	#
#############
	# Initialize arrays
	x	= np.zeros((nframes,Kc,3+3*order))
	s	= np.zeros((nframes,Kc))
	r	= np.zeros((nframes,Kc))
	t	= np.arange(nframes)*(frames_duration-frames_overlap)+frames_duration/2
	# Loop over frames
	for iframe in tqdm(np.arange(nframes)):
		# Extract specific frame's data
		Lframe = L[np.abs(L[:,0]-t[iframe])<frames_duration/2]
		if toPlot:
			# Frame's mean particles positions
			Xframe = np.mean(X[np.abs(X[:,0,0]-t[iframe])<frames_duration/2],axis=0)[:,1:]
		if iframe==0: # Initial frame
			# Initialize clusters
			x0,s0 =	Initial_centroid(Lframe)
			x_m=x0[np.newaxis,:]+np.random.normal(0,s0,(Kc,3))
			if order>0:
				x_m=np.concatenate((x_m,np.zeros((Kc,3*order))),axis=-1)
			s_m = np.ones(Kc)*s0
			r_m = np.ones(Kc)/Kc
			if toPlot:
				# Initialize the "on the fly" ploting window
				fig,P=InitPlot(x=x_m[:,:3],s=s_m,X=Xframe,lim=bounds)
			d2=None # distances matrix to be initialized to "None" when changing frames
			# Actual EM-algorithm
			for istep in np.arange(n_step_init):
				x_m,s_m,r_m,d2=EM_Single_Step(x_m,s_m,Lframe,alpha=alpha,parallel=True,d2=d2)
				if toPlot:
					# Update ploting window
					P=UpdatePlot(fig,P,x_m,X=Xframe,s=s_m)
		else: # Tracking
			d2=None
			for istep in np.arange(n_step_track):
				x_m,s_m,r_m,d2=EM_Single_Step(x_m,s_m,Lframe,alpha=alpha,parallel=True,d2=d2)
			if toPlot:
				P=UpdatePlot(fig,P,x_m,X=Xframe,s=s_m)
		# Store results then move to next frame
		x[iframe]	=	x_m
		s[iframe]	=	s_m
		r[iframe]	=	r_m
	# Save results
	np.savez(fid,t=t,x=x,s=s,r=r)
	if toPlot:
		plt.close('all')
		plt.ioff()

#############################################
# Extract and process inferred trajectories #
# and compare them with actual ones			#
#############################################
Track=SplitTraj(t,x,s,r,4,minframes=200)
fig1 		=	plt.figure(figsize=[15,15])
ax1			=	fig1.add_subplot(221, projection='3d')
for i in np.arange(len(Track)):
	ax1.plot(Track[i][1][:-2,0],Track[i][1][:-2,1],Track[i][1][:-2,2])
for i in np.arange(Kp):
	ax1.plot(X[:,i,1],X[:,i,2],X[:,i,3],'k--',linewidth=0.8)
ax1.set_title(r'3D view')
ax2			=	fig1.add_subplot(222)
for i in np.arange(len(Track)):
	ax2.plot(Track[i][0][:-5],Track[i][1][:-5,0])
for i in np.arange(Kp):
	ax2.plot(X[:,i,0],X[:,i,1],'k--',linewidth=0.8)
ax2.set_title(r'$x$')
ax3			=	fig1.add_subplot(223)
for i in np.arange(len(Track)):
	ax3.plot(Track[i][0][:-5],Track[i][1][:-5,1])
for i in np.arange(Kp):
	ax3.plot(X[:,i,0],X[:,i,2],'k--',linewidth=0.8)
ax3.set_title(r'$y$')
ax4			=	fig1.add_subplot(224)
for i in np.arange(len(Track)):
	ax4.plot(Track[i][0][:-5],Track[i][1][:-5,2])
for i in np.arange(Kp):
	ax4.plot(X[:,i,0],X[:,i,3],'k--',linewidth=0.8)
ax4.set_title(r'$z$')
plt.show()

