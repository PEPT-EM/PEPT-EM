def dist_matrix(x,L):
	#################################################################
	# Computation of the squared distances between the clusters  	#
	# centroids and the lines										#
	#################################################################
	## input ##
	# x[:3*(o+1),:K]	: float		: centroids coordinates array
	# L[:N,:1+2*3]		: float		: lines array
	#						L[:,0]	--> times
	#						L[:,1:4]--> feature points
	#						L[:,4:]	--> directions
	# o 				: integer	: order of the centroids
	# K					: integer	: number of centroids
	# N					: integer	: number of lines
	## output ##
	# d2[:N,:K]			: float		: array of squared distances from  
	#								  centroids to lines
	import numpy as np
	from numpy import newaxis as nx
	from scipy.special import factorial
	o = x.shape[-1]//3-1
	X = x[nx,:,:3]-L[:,nx,1:4]	# vetcor centroids-->feature points
	if o > 0: 					# higher order correction of the centroids positions
		DT = L[:,0]-L[:,0].mean()
		for i in np.arange(o)+1:
			X += x[nx,:,3*i:3*(i+1)]*DT[:,nx,nx]**i/factorial(i)  # Taylor expansion about the mean btach time
	d2 = np.sum(X**2,axis=-1)-np.sum(X*L[:,nx,4:],axis=-1)**2	  # Pythagor theorem
	return d2

def Latent_weights(d2,s,alpha=0,r=None,low_thresh=10**(-20)):
	#############################################################
	# Computation of latent weigths								#
	#############################################################
	## input ##
	# d2[:N,:K]			: float 	: array of squared distances from  
	#								  centroids to lines
	# s[:K]				: float		: variances array
	# K					: integer	: number of centroids
	# N					: integer	: number of lines
	# alpha 			: float		: outlier cluster parameter
	# r[:K]				: float		: cluster weights array
	# low_thresh		: float		: lower bound to ensure non-zero 
	#								  contribution
	## output ##
	# w[:N,:K]			: float		: latent weights array	
	import numpy as np
	from numpy import newaxis as nx
	if r is None: 			# assume equal cluster weights
		w = np.exp(-d2/2/s**2)/s**2+10**(-20) 
		w /= np.sum(w,axis=-1)[:,nx]+alpha
	else:
		w = np.exp(-d2/2/s**2)*r/s**2+10**(-20)
		w/= np.sum(w,axis=-1)[:,nx]+alpha*(1-np.sum(r))
	return w

def Centroid(L,w,o=0):
	#################################################################
	# Computation of the centroid of a weighted batch of lines 		#
	#################################################################
	## input ##
	# L[:N,:1+2*3]		: float		: lines array
	#						L[:,0]	--> times
	#						L[:,1:4]--> feature points
	#						L[:,4:]	--> directions
	# w[:N]				: float		: weights array
	# o 				: integer	: order of the centroids
	# N					: integer	: number of lines
	## output ##
	# x[:3*(o+1)]		: float		: centroids coordinates array
	import numpy as np
	from numpy import newaxis as nx
	from scipy.special import factorial
	m = (np.identity(3)[nx,:,:]-L[:,nx,4:]*L[:,4:,nx])*w[:,nx,nx] # weighted projectors
	if o==0:
		M = np.sum(m,axis=0)
		V = np.sum(np.sum(m*L[:,nx,1:4],axis=-1),axis=0)
	else:
		DT = L[:,0]-L[:,0].mean()
		M = np.zeros((3*(o+1),3*(o+1)))
		V = np.zeros(3*(o+1))
		for i in np.arange(o+1):
			V[i*3:(i+1)*3] = np.sum(np.sum(m*L[:,nx,1:4]*DT[:,nx,nx]**i/factorial(i),axis=-1),axis=0)
			for j in np.arange(o-i+1)+i:
				M[i*3:(i+1)*3,j*3:(j+1)*3] = np.sum(m*DT[:,nx,nx]**(i+j)/(factorial(i)*factorial(j)),axis=0)
		for i in np.arange(o+1):
			for j in np.arange(i):
				M[i*3:(i+1)*3,j*3:(j+1)*3] = M[j*3:(j+1)*3,i*3:(i+1)*3]	
	x = np.matmul(np.linalg.inv(M),V)
	return x

def Centroid_Multi(L,w,o=0,parallel=False):
	#################################################################
	# Computation of the centroids of multiple weighted batches 	#
	#################################################################
	## input ##
	# L[:N,:1+2*3]		: float		: lines array
	#						L[:,0]	--> times
	#						L[:,1:4]--> feature points
	#						L[:,4:]	--> directions
	# w[:N]				: float		: weights array
	# o 				: integer	: order of the centroids
	# K					: integer	: number of centroids
	# N					: integer	: number of lines
	# parallel			: logical	: flag enabling parallel computation
	## output ##
	# x[:3*(o+1),:K]	: float		: centroids coordinates array
	import numpy as np
	if parallel:
		import joblib as jl
		x = np.asarray(jl.Parallel(n_jobs=-1)(jl.delayed(Centroid)(L,w[:,i],o=o) for i in np.arange(w.shape[1])))
	else:
		x = np.asarray([Centroid(L,w[:,i],o=o) for i in np.arange(w.shape[1])])
	return x

def Initial_centroid(L):
	#################################################
	# Initialize a unique cluster zero-th order  	#
	# centroid and variance 						#
	#################################################
	## input ##
	# L[:N,:1+2*3]		: float		: lines array
	#						L[:,0]	--> times
	#						L[:,1:4]--> feature points
	#						L[:,4:]	--> directions
	# N					: integer	: number of lines
	## output ##
	# x[:3]				: float		: global centroid coordinates array
	# s					: float		: variance array
	import numpy as np
	from numpy import newaxis as nx
	w = np.ones(L.shape[0])
	x = Centroid(L,w)
	d2	= dist_matrix(x[nx,:],L)
	s=np.sqrt(np.mean(d2))
	return x,s

def EM_Single_Step(x,s,L,alpha=0,parallel=False,r=None,d2=None):
	#################################################
	# Single Expactation-maximization step 			#
	#################################################
	## input ##
	# x[:3*(o+1),:K]	: float		: centroids coordinates array
	# s[:K]				: float		: variances array
	# L[:N,:1+2*3]		: float		: lines array
	#						L[:,0]	--> times
	#						L[:,1:4]--> feature points
	#						L[:,4:]	--> directions
	# o 				: integer	: order of the centroids
	# K					: integer	: number of centroids
	# N					: integer	: number of lines
	# alpha 			: float		: outlier cluster parameter
	# parallel			: logical	: flag enabling parallel computation
	# r[:K]				: float		: cluster weights array
	# d2[:N,:K]			: float		: precomputed array of squared distances   
	#								  from centroids to lines
	## output ##
	# x[:3,K]			: float		: updated  centroids coordinates array
	# s[:K]				: float		: updated variances array
	# r[:K]				: float		: updated cluster weights array
	# d2[:N,:K]			: float		: updated array of squared distances   
	#								  from centroids to lines	
	import numpy as np
	if d2 is None:
		d2 = dist_matrix(x,L)
	w=Latent_weights(d2,s,alpha=alpha,r=r)
	r = np.mean(w,axis=0)
	x = Centroid_Multi(L,w,o=x.shape[-1]//3-1,parallel=parallel)
	d2 = dist_matrix(x,L)
	s = np.sqrt(np.mean(d2*w,axis=0)/r/2)
	return x,s,r,d2



def InitPlot(L=None,x=None,X=None,s=None,lim=None):
	#################################################
	# Initialize the animated plot					#
	#################################################
	## input ##
	# L[:N,:1+2*3]		: float		: lines array
	#						L[:,0]	--> times
	#						L[:,1:4]--> feature points
	#						L[:,4:]	--> directions
	# x[:3,K]			: float		: inferred centroids coordinates array
	# x0[:3,K']			: float		: actual centroids coordinates array
	# s[:K]				: float		: variances array
	# K					: integer	: number of inferred centroids
	# K'				: integer	: actual number of centroids
	# N					: integer	: number of lines
	# lim(xm,xM,ym,yM,zm,zM)
	#					: float		: bounds of the ploting volume	
	## output ##
	# fig				: plt.fig	: figure object
	# P					: list		: list of markers to be updated live
	import matplotlib.pyplot as plt
	import numpy as np
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.cm as cm
	cmap = cm.get_cmap('nipy_spectral')
	def axisEqual3D(ax):
		extents		=	np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
		sz			= 	extents[:,1] - extents[:,0]
		centers		=	np.mean(extents, axis=1)
		maxsize		=	max(abs(sz))
		r			=	maxsize/2
		for ctr, dim in zip(centers, 'xyz'):
			getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
	plt.ion()
	fig 		=	plt.figure(figsize=[10,10])
	ax			=	fig.add_subplot(111, projection='3d')
	if L is not None:
		if len(L)==2:
			x1,x2=L
			for i in np.arange(x1.shape[0]):
				ax.plot([x1[i,0],x2[i,0]],[x1[i,1],x2[i,1]],[x1[i,2],x2[i,2]],'k-',linewidth=0.3,alpha=0.1)
		elif len(L)==3:
			x1,x2,w=L
			colors=cmap(np.linspace(0,1,w.shape[1]+1)[1:-1])
			colorsx=np.concatenate((np.copy(colors),np.array([[0.,0.,0.,1.]])),axis=0)
			colorsx[:,-1]/=5
			rgba_colors=np.minimum(1,np.sum(colorsx[np.newaxis,:,:]*w[:,:,np.newaxis],axis=1))
			for i in np.arange(x1.shape[0]):
				ax.plot([x1[i,0],x2[i,0]],[x1[i,1],x2[i,1]],[x1[i,2],x2[i,2]],'-',color=rgba_colors[i],linewidth=0.4)
	P=[]
	if x is not None:
		colors=cmap(np.linspace(0,1,x.shape[0]+2)[1:-1])
		if s is not None:
			for ip in np.arange(x.shape[0]):
				pp,=ax.plot([x[ip,0]],[x[ip,1]],[x[ip,2]],'o',color=colors[ip],markersize=np.maximum(s[ip],5),alpha=np.maximum(0.1,np.minimum(1,5/s[ip])),markeredgewidth=1,markeredgecolor='k')
				P.append(pp)
		else:
			for ip in np.arange(x.shape[0]):
				pp,=ax.plot([x[ip,0]],[x[ip,1]],[x[ip,2]],'o',color=colors[ip],markersize=5)
				P.append(pp)
	if X is not None:
		pp,=ax.plot(X[:,0],X[:,1],X[:,2],'kx',markersize=5)
		P.append(pp)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')
	if lim is not None:
		xm,xM,ym,yM,zm,zM=lim
		ax.set_xlim(xm,xM)
		ax.set_ylim(ym,yM)
		ax.set_zlim(zm,zM)
	axisEqual3D(ax)
	fig.canvas.draw()
	fig.canvas.flush_events()
	return fig,P
	
def UpdatePlot(fig,P,x,X=None,s=None):
	#################################################
	# Update the animated ploting					#
	#################################################
	## input ##
	# fig				: plt.fig	: figure object
	# P					: list		: list of markers to be updated live
	# x[:3,:K]			: float		: inferred centroids coordinates array
	# s[:K]				: float		: variances array
	# K					: integer	: number of inferred centroids
	## output ##
	# P					: list		: list of upadted markers
	import matplotlib.pyplot as plt
	import numpy as np
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.cm as cm
	cmap = cm.get_cmap('nipy_spectral')
	plt.ion()
	colors=cmap(np.linspace(0,1,x.shape[0]+2)[1:-1])
	colors[:,-1]=np.maximum(0.1,np.minimum(1,5/s))
	if s is not None:
		for ip in np.arange(x.shape[0]):
			P[ip].set_xdata([x[ip,0]])
			P[ip].set_ydata([x[ip,1]])
			P[ip].set_3d_properties([x[ip,2]])
			P[ip].set_markersize(np.maximum(s[ip],5))
			P[ip].set_markerfacecolor(colors[ip])
			P[ip].set_alpha(np.maximum(0.1,np.minimum(1,5/s[ip])))
	else:
		for ip in np.arange(x.shape[0]):
			P[ip].set_xdata([x[ip,0]])
			P[ip].set_ydata([x[ip,1]])
			P[ip].set_3d_properties([x[ip,2]])
	if X is not None:
		P[-1].set_xdata(X[:,0])
		P[-1].set_ydata(X[:,1])
		P[-1].set_3d_properties(X[:,2])
	fig.canvas.draw()
	fig.canvas.flush_events()
	return P

def SplitTraj(t,x,s,r,threshold_s,threshold_r=0,minframes=10):
	#################################################
	# Extract continuous trajectories				#
	#################################################
	## input ##
	# t[:nframes]				: float		: frames time
	# x[:nframes,:K,:3*(o+1)]	: float		: inferred centroids coordinates array
	# s[:nframes,:K]			: float		: inferred clusters variances
	# r[:nframes,:K]			: float		: inferred clusters weights
	# nframes					: integer	: number of frames
	# K							: integer	: number of clusters
	# o							: integer	: order of the centroids
	# threshold_s				: float		: threshold in variance
	# threshold_r				: float		: threshold in weights
	# minframes					: integer	: minimum number of frames per trajectory
	## output ##
	# Traj[:ntraj][:4]			: list		: list of trajectories
	#				Traj[n][0][:nframe_n] 			--> times  
	#				Traj[n][1][:nframe_n,:3*(o+1)]	--> centroid's coordinates 
	#				Traj[n][2][:nframe_n]			--> variance 
	#				Traj[n][3][:nframe_n]			--> weights
	#				nframe_n						--> number of frames for the n-th trajectory
	# ntraj						: integer	: number of trajectories
	import numpy as np
	nframes=t.size
	K=x.shape[1]
	Traj=[]
	for ip in np.arange(K):
		index=np.arange(nframes)[(s[:,ip]<threshold_s)&(r[:,ip]*K>threshold_r)]
		if (index.size<t.size) & (index.size>0):
			index=np.split(index,np.arange(index.size-1)[np.diff(index)>1]+1)
			for i in np.arange(len(index)):
				if t[index[i]].size>minframes-1:
					Traj.append([t[index[i]],x[:,ip][index[i]],s[:,ip][index[i]],r[:,ip][index[i]]])
		elif index.size==t.size:
			Traj.append([t[index],x[:,ip][index],s[:,ip][index],r[:,ip][index]])
	return Traj