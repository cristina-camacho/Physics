# -*- coding: utf-8 -*-

import numpy as np
import PolarimetricFunctions as pf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc


import glob
import os

#import pymueller
#from pymueller.decomposition import lu_chipman

import Lu_Chip as lc

import matplotlib.animation as animation
from matplotlib.widgets import Button


import warnings

# Suppress warnings for invalid values in sqrt
warnings.filterwarnings("ignore", category=RuntimeWarning)


import dash
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State


plt.rcParams.update({
    "font.family": "serif", 
    "mathtext.fontset": "cm", 
    "mathtext.rm": "serif",
    "axes.labelsize": 15,
    "font.size": 11,
    "xtick.labelsize": 10, 
    "ytick.labelsize": 10,
	"figure.titlesize": 11
})

 



#%% Functions

list_parameters = ['ret', 'diat', 'pol', 'pdelta', 'ps', 'coeff', 'ipps']
subtitles = {'Retardance': r'$\Delta$', 'Diattenuation': r'$D$', 'Polarizance': r'$\mathcal{P}$', 'Pdelta': r'$P_\Delta$', 'Ps': r'$P_S$', 'Coefficients': r'$c$'}

#%%% Plots

def plot_IPPS(ipps, title='',  x=None, y=None, xtitle=None, ytitle=None, name=None, lims=[0,1], num=3, origin='lower',rot=False, color='viridis'):
    """
    Plots a heatmap of the IPPS.

    Parameters:
        ipps (3xNxN array): Values of the ipps P1, P2, P3
        title (string): Title of the plot
        x (N,) array: X axis values
        y (N,) array: Y axis value

    Returns:
        Shows a plot
    """
    
    if x is None:
        extent = None
        axis = 'off'
        heatmap_width = 3.3
        heatmap_height = 3.1
    else:
        extent = [x.min(), x.max(), y.min(), y.max()]
        axis='on'
        heatmap_width = 3.5
        heatmap_height = 3.1
    
    

    if rot==True:
        ipps=ipps.transpose(0,2,1)


    if num == 3:
        fig, axes = plt.subplots(1, 3, figsize=(heatmap_width * 3, heatmap_height))
        
    elif num == 2:
        fig, axes = plt.subplots(1, 2, figsize=(heatmap_width * 2, heatmap_height))
 
        

    # Plot each colormap
    for i, ax in enumerate(axes.flat):  
        vmin=lims[0] if lims is not None else None
        vmax=lims[1] if lims is not None else 1 if np.max(ipps[i])>1 else None
        
    
        
        im = ax.imshow(
            ipps[i], aspect="auto", extent=extent,
            origin=origin, cmap=color, vmin=vmin, vmax=vmax
        )
        ax.set_title(f"$P_{i+1}$")  # Set title for each subplot
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.axis(axis)
        fig.colorbar(im, ax=ax)  # Add colorbar for each subplot

    fig.suptitle(title)
    
    #Adjust layout
    if title!='':
        plt.tight_layout(rect=[0, 0, 1, 1.08])
    else:
        plt.tight_layout(rect=[0, 0, 1, 1])
        

        
    if name is not None:
        plt.savefig(name, dpi=300, bbox_inches='tight')
    
    plt.show()



def plot_parameter(parameter, title='', x=None, y=None, xtitle=None, ytitle=None, subtitle='M',lims=[0,1], name=None, num=3, rot=False, origin='lower', color='viridis'):
    """
    Plots the specified parameter for M0, M1, M2, M3

    Parameters:
        parameter (mxNxN): Parameter to plot
        
        title (string) : Title of the plot
        
        x (N,) array: x axis values
        
        y (N,) array: y axis values
        
        xtitle (string): x axis title
        
        ytitle (string): y axis title
        
        
        
    Returns:
        Shows a plot

    """

    if x is None:
        extent = None
        axis = 'off'
        heatmap_width = 3.3
        heatmap_height = 3.1
    else:
        extent = [x.min(), x.max(), y.min(), y.max()]
        axis='on'
        heatmap_width = 3.5
        heatmap_height = 3.1
        
         
    
    if num == 3:
        fig, axes = plt.subplots(1, 3, figsize=(heatmap_width * 3, heatmap_height))
        
    elif num == 4:
        fig, axes = plt.subplots(1, 4, figsize=(heatmap_width * 4, heatmap_height))
        
    elif num == 2:
        fig, axes = plt.subplots(1, 2, figsize=(heatmap_width * 2, heatmap_height))
        
    elif num == 1:
        fig, ax = plt.subplots(1, 1, figsize=(heatmap_width, heatmap_height))
        axes=np.array([[ax]])
        parameter = parameter[np.newaxis, :, :] 
    else: 
        print("Number not valid. It must be 1-4")
        return
    
    if rot==True:
        parameter=parameter.transpose(0,2,1)

    


    vmin=lims[0] if lims is not None else None
    vmax=lims[1] if lims is not None else None  
        
    for i, ax in enumerate(axes.flat):  # Iterate over the subplots
        
        im = ax.imshow(
            parameter[i], aspect="auto", extent=extent, 
            origin=origin, cmap=color, vmin=vmin, vmax=vmax
        )
        
        ax.set_title(subtitle+f"$_{i}$" if num!=1 else '')  # Set title for each subplot
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.axis(axis)
        fig.colorbar(im, ax=ax)


    if num==1:
         ax.set_title(title, loc='left')
    else:
         fig.suptitle(title, ha='center')

    #Adjust layout
    if num==3:
        plt.tight_layout(rect=[0, 0, 1, 1] if title=='' else [0, 0, 1, 1.08])
        
    if num==2:
        plt.tight_layout(rect=[0, 0, 1, 1])
    
    if num==4:
        plt.tight_layout(rect=[0, 0, 1, 1.02])
    if num==1:
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
   
    
    if name is not None:
        plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.show()
    
    
def plot_dif_IPPS(ipps, title='',  x=None, y=None, xtitle=None, ytitle=None, name=None, lims=[0,1], origin='lower',rot=False, color='viridis'):
    """
    Plots a heatmap of the difference between IPPS.

    Parameters:
        ipps (3xNxN array): Values of the ipps P1, P2, P3
        title (string): Title of the plot
        x (N,) array: X axis values
        y (N,) array: Y axis value

    Returns:
        Shows a plot
    """
    
    heatmap_width = 3.3
    heatmap_height = 2.7
    

    if rot==True:
        ipps=ipps.transpose(0,2,1)
    
    if x is None:
        extent = None
        axis = 'off'
    else:
        extent = [x.min(), x.max(), y.min(), y.max()]
        axis='on'

    fig, axes = plt.subplots(1, 2, figsize=(heatmap_width * 2, heatmap_height))
   
        

    # Plot each colormap
    for i, ax in enumerate(axes.flat):  
        vmin=lims[0] if lims is not None else None
        vmax=lims[1] if lims is not None else 1 if np.max(ipps[i])>1 else None
        
    
        im = ax.imshow(
            ipps[i+1]-ipps[i], aspect="auto", extent=extent,
            origin=origin, cmap='viridis', vmin=vmin, vmax=vmax
        )
        ax.set_title(f"P{i+2}-P{i+1}")  # Set title for each subplot
        ax.set_xlabel(xtitle)
        ax.axis(axis)
        ax.set_ylabel(ytitle)
        fig.colorbar(im, ax=ax)  # Add colorbar for each subplot

    fig.suptitle(title)
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 1.06])
    if name is not None:
        plt.savefig(name, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    
#%%% Computation of parameters   
def parameters_ChDecomp(data, height, width, parameters):
    """
    Computes (if selected) the IPP values of the introduced matrices, as well 
    as the coefficients from its characteristic decomposition, and the 
    parameters of the matrices obtained from said decomposition of the 
    introduced Mueller matrices.

    Parameters
    ----------
    data (4x4xN array): Collection of Mueller matrices
    
    height (int) : Height of the image
    
    width (int) : Width of the image
    
    parameters (array): List of parameters we want to compute, to select between: 
        ret, diat, pol, pdelta, ps, coeff, ipps

    Returns
    -------
    Dictionary of parameters, each entry is a (3 x height x width)

    """
    
    ret=1 if 'ret' in parameters else None
    diat=1 if 'diat' in parameters else None
    polar=1 if 'pol' in parameters else None
    pd=1 if 'pdelta' in parameters else None
    coeffs=1 if 'coeff' in parameters else None
    ipp=1 if 'ipps' in parameters else None
    ps=1 if 'ps' in parameters else None
    
    
    dim =np.shape(data)[2]
    if height*width!=  dim:
        print("Height and width do not correspond with the dimension of the data")
        return None

    retardance = np.zeros((3, height, width))
    polarizance = np.zeros((3, height, width))
    diattenuation = np.zeros((3, height, width))
    pdelta = np.zeros((3, height, width))
    sphericalpty = np.zeros((3, height, width))

    coefficients = np.zeros((4, height, width))
    IPPS = np.zeros((3, height, width))

    for j in range(dim):
        M_total = data[:,:, j]
        M_total = M_total.reshape(4,4,1)
        
        y = j % width
        x = j // width
        
        if ipp==1:
            ipps = pf.IPPs(M_total)
            IPPS[0, y, x] = ipps[0][0]
            IPPS[1, y, x] = ipps[1][0]
            IPPS[2, y, x] = ipps[2][0]
        
        if 1 in [ret, diat, polar, pd, coeffs, ps]:
            coefs, matrius = pf.Characteristic_decomposition(M_total)
            
            coefficients[3, y, x] = coefs[3,0] if coeffs==1 else None
            
            for i in range(3):
                matriu = matrius[i]
                
                coefficients[i, y, x] = coefs[i,0] if coeffs==1 else None
                #retards[i, nt, npy] = pf.Retardance(matriu)[0]
                
                if ret==1:
                    try:
                        r=lc.Lu_Chip(matriu)[2][0,0]
                    except:
                        r=float('NaN')
                    
                    retardance[i, y, x] = r*180/np.pi
                
                if polar==1:
                    polarizance[i, y, x] = pf.Polarizance(matriu/matriu[0,0,0])[1][0]
                if diat==1:
                    diattenuation[i, y, x] = pf.Diattenuation(matriu/matriu[0,0,0])[1][0] 
                if pd==1:
                    pdelta[i, y, x] = pf.Pdelta(matriu)[0] 
                if ps==1:
                    sphericalpty[i,y,x] = pf.Ps(matriu/matriu[0,0,0])[0] 
                
            
    result = {}

    if ret == 1:
        result["Retardance"] = retardance
    if polar == 1:
        result["Polarizance"] = polarizance
    if diat == 1:
        result["Diattenuation"] = diattenuation
    if pd == 1:
        result["Pdelta"] = pdelta
    if coeffs == 1:
        result["Coefficients"] = coefficients
    if ipp == 1:
        result["IPPS"] = IPPS
    if ps == 1:
        result['Ps'] = sphericalpty
        
    return result     
    

def parameters_MM (data, height, width, parameters):
    
    dim =np.shape(data)[2]
    if height*width!=  dim:
        print("Height and width do not correspond with the dimension of the data")
        return None
    
    ret=1 if 'ret' in parameters else None
    diat=1 if 'diat' in parameters else None
    polar=1 if 'pol' in parameters else None
    pd=1 if 'pdelta' in parameters else None
    ps=1 if 'ps' in parameters else None
    
    retardance = np.zeros((height, width))
    polarizance = np.zeros((height, width))
    diattenuation = np.zeros((height, width))
    pdelta = np.zeros((height, width))
    sphericalpty = np.zeros((height, width))
    
    for j in range(dim):
        M_total = data[:,:, j]
        matriu = M_total.reshape(4,4,1)
        
        y = j % width
        x = j // width
    
        if ret==1:
            try:
                r=lc.Lu_Chip(matriu)[2][0,0]
            except:
                r=float('NaN')
            
            retardance[y, x] = r*180/np.pi
        
        if polar==1:
            polarizance[y, x] = pf.Polarizance(matriu/matriu[0,0,0])[1][0]
        if diat==1:
            diattenuation[y, x] = pf.Diattenuation(matriu/matriu[0,0,0])[1][0] 
        if pd==1:
            pdelta[y, x] = pf.Pdelta(matriu)[0] 
        if ps==1:
            sphericalpty[y,x] = pf.Ps(matriu/matriu[0,0,0])[0] 

    
    result ={}
    
    if ret == 1:
        result["Retardance"] = retardance
    if polar == 1:
        result["Polarizance"] = polarizance
    if diat == 1:
        result["Diattenuation"] = diattenuation
    if pd == 1:
        result["Pdelta"] = pdelta

    if ps == 1:
        result['Ps'] = sphericalpty
    
    return result
    
#%%% Simulations

def simulation_diattenuators(p, px, N=300, N_Thetas=100, N_Py=100, Maxsigma_theta=np.pi, norm=1):
    """
    Parameters
    ----------
    p : float between 0 and 1
        Percentage of photons interacting by isotropic scattering
    px : float between 0 and 1
        Px fixed value for each diattenuator.
    N : int, optional 
        Number of interactions. The default is 300.
    N_Thetas : int, optional
        Number of variations for sigma Theta. The default is 100.
    N_Py : int, optional
        Number of variations for Py. The default is 100.
    Maxsigma_theta: float, optional
        Maximum deviation (in radians)
    norm: int, optional
        Normalization of the diattenuators interacting (1 is normalized, 0 not)
    Returns
    -------
    MMD: (N_Thetas, N_Py, 4, 4) array
        Matrices resulting from the simulation

    """    
    
    
    # Definition of parameters
    N_B = int(p * N)  # Number of interactions via isotropic scattering (interaction B)
    N_A = N - N_B     # Number of interactions via other means (interaction A)
    
    sigmaTheta = np.linspace(0, Maxsigma_theta, N_Thetas)  # Deviation values
    ThetaM = (np.pi / 180) * 60  # Mean value of theta (set to 60 degrees)
    
    # Parameters of the diattenuator (px, py)
    

    pyM = 1 * np.linspace(0, 1, N_Py)   #Mean values for py
    #pyM = 0.2 * np.ones(N_Py) 
    sigmaPy = 0*np.linspace(0, 1, N_Py) 
    
    px = px*np.ones((N_A)) 
        
    # Calculation
    
    # Initialize matrices
    M_dis = np.zeros((4, 4))  
    M_dis[0, 0] = 1  
    
    A = np.zeros((4, 4, N_A))  
    MMD = np.zeros((N_Thetas, N_Py, 4, 4))  
    
    k = 0
    
    for nt in range(N_Thetas):  
        theta = ThetaM + sigmaTheta[nt] * np.random.randn(N_A)
    
        RN = pf.rotator_MM(-theta)  
        RP = pf.rotator_MM(theta)  
    
    
        for npy in range(N_Py):  
            k += 1  
    
            py = pyM[npy] + sigmaPy[npy] * np.random.randn(N_A)
    
            M1 = pf.diattenuator_MM(px, py, norm=norm)  
    
            A = np.moveaxis(RN, -1, 0) @ np.moveaxis(M1, -1, 0) @ np.moveaxis(RP, -1, 0)  # The matrix multiplication assumes the matrices are in the 2nd and 3rd dimension
            # A has shape (N_A, 4, 4)
            
            # --------------------------- TOTAL MATRIX ---------------------------
            A_T = np.sum(A, axis=0)  # Incoherent sum of all interaction matrices A
            B_T = N_B * M_dis  # Matrices corresponding to interaction B
            M_total = A_T + B_T  
    
            MMD[nt, npy, :, :] = M_total  
            
            
    return MMD, pyM, sigmaTheta*180/np.pi


def simulation_retarders(p,  N=300, N_Thetas=100, N_Phis=100, Maxsigma_theta=np.pi):
    """
    Parameters
    ----------
    p : float between 0 and 1
        Percentage of photons interacting by isotropic scattering
    px : float between 0 and 1
        Px fixed value for each diattenuator.
    N : int, optional 
        Number of interactions. The default is 300.
    N_Thetas : int, optional
        Number of variations for sigma Theta. The default is 100.
    N_Py : int, optional
        Number of variations for Py. The default is 100.
    Maxsigma_theta: float, optional
        Maximum deviation (in radians)

    Returns
    -------
    MMD: (N_Thetas, N_Py, 4, 4) array
        Matrices resulting from the simulation

    """    
    
        
    # Definition of parameters
    N_B = int(p * N)  # Number of interactions via isotropic scattering (interaction B)
    N_A = N - N_B     # Number of interactions via other means (interaction A)
    
    sigmaTheta = np.linspace(0, Maxsigma_theta, N_Thetas)  # Deviation values
    ThetaM = (np.pi / 180) * 20  # Mean value of theta (set to 60 degrees)

    
    # Parameters of the retarder (phi)
    Maxsigma_phi = 2*np.pi
    sigmaPhi = 0*np.linspace(0, Maxsigma_phi, N_Phis)
    PhiM = 1*np.linspace(0, np.pi*2, N_Phis) 
    
    # Calculation
    
    # Initialize matrices
    M_dis = np.zeros((4, 4))  
    M_dis[0, 0] = 1  
    
    A = np.zeros((4, 4, N_A)) 
     
    MMR = np.zeros((N_Thetas, N_Phis, 4, 4))  
    
    k = 0
    for nt in range(N_Thetas):  
        
        theta = ThetaM + sigmaTheta[nt] * np.random.randn(N_A)  
    
        RN = pf.rotator_MM(-theta)  
        RP = pf.rotator_MM(theta)  
    
        for nph in range(N_Phis):  
            k += 1  
    
            phi = PhiM[nph] + sigmaPhi[nph] * np.random.randn(N_A)
            
            M1 = pf.retarder_MM(phi)  
    
            A = np.moveaxis(RN, -1, 0) @ np.moveaxis(M1, -1, 0) @ np.moveaxis(RP, -1, 0)  # The matrix multiplication assumes the matrices are in the 2nd and 3rd dimension
            # A has shape (N_A, 4, 4)
            
            # --------------------------- TOTAL MATRIX ---------------------------
            A_T = np.sum(A, axis=0)  # Incoherent sum of all interaction matrices A
            B_T = N_B * M_dis  # Matrices corresponding to interaction B
            M_total = A_T + B_T  
    
            MMR[nt, nph, :, :] = M_total
        
    return MMR, PhiM*180/np.pi, sigmaTheta*180/np.pi


def simulation_both(proportion_diatt, beta, pX, sigTheta, norm=1, N=300, N_Phis=100, N_Py=100):
    
    N = 300       # Number of interactions
    p = beta       # Percentage of photons interacting by isotropic scattering
    alpha = proportion_diatt # Proportion of diattenuators
    N_B = int(p * N)  # Number of interactions via isotropic scattering (interaction B)
    N_C = int(alpha * (N-N_B))
    N_A = N - N_B - N_C    # Number of interactions via other means (interaction A)
    
    sigmaTheta = (np.pi / 180) * sigTheta
    
    
    ThetaM = (np.pi / 180) * 60  # Mean value of theta (set to 60 degrees)
    
    # Parameters of the retarder (phi)
    
    Maxsigma_phi = np.pi
    sigmaPhi = 0*np.linspace(0, Maxsigma_phi, N_Phis)
    PhiM = 1*np.linspace(0, np.pi*2, N_Phis) #(np.pi / 180) * 60
    
    # Parameters of the diattenuator (px, py)
    
    pyM = 1 * np.linspace(0, 1, N_Py)   #Mean values for py
    sigmaPy = 0 * np.abs(np.random.randn(N_C))
    
    px = pX * np.ones((N_C)) 
    
    # Calculation
    
    # Initialize matrices
    M_dis = np.zeros((4, 4))  
    M_dis[0, 0] = 1  
    
     
    MMDR = np.zeros((N_Phis, N_Py, 4, 4)) 
    
       
    theta1 = ThetaM + sigmaTheta * np.random.randn(N_A)  
    theta2 = ThetaM + sigmaTheta * np.random.randn(N_C)
    
    RN_A = pf.rotator_MM(-theta1)  
    RP_A = pf.rotator_MM(theta1)  
    
    RN_C = pf.rotator_MM(-theta2)  
    RP_C = pf.rotator_MM(theta2)
    
    for nph in range(N_Phis):  
    
        phi = PhiM[nph] + sigmaPhi[nph] * np.random.randn(N_A)
        MRe = pf.retarder_MM(phi)  
        
        for npy in range(N_Py):
            
            py = pyM[npy] + sigmaPy
            MDi = pf.diattenuator_MM(px, py, norm=norm)
    
            A = np.moveaxis(RN_A, -1, 0) @ np.moveaxis(MRe, -1, 0) @ np.moveaxis(RP_A, -1, 0)  # The matrix multiplication assumes the matrices are in the 2nd and 3rd dimension
            C = np.moveaxis(RN_C, -1, 0) @ np.moveaxis(MDi, -1, 0) @ np.moveaxis(RP_C, -1, 0)
            # A has shape (N_A, 4, 4)
        
            # --------------------------- TOTAL MATRIX ---------------------------
            A_T = np.sum(A, axis=0)  # Incoherent sum of all interaction matrices A
            C_T = np.sum(C, axis=0)
            B_T = N_B * M_dis  # Matrices corresponding to interaction B
            M_total = A_T + B_T + C_T 
    
            MMDR[nph, npy, :, :] = M_total 
            
    return MMDR, PhiM*180/np.pi, pyM

#%%% Processing
def process_data(MM, params_list, height, width, x=None, y=None, xtitle=None, ytitle=None, general=False, chdecomp=True, save=False, filename=None, plotsdir=''):
    """
    Processes the given MMs and returns the plots of the specified parameters

    Parameters
    ----------
    MM : (n, m, 4, 4) 
        DESCRIPTION.
    params_list : array
        List of parameters we want to calculate to choose between:
            'diat','ret', 'pol', 'pdelta', 'ps', 'coeff', 'ipps'
    filename: string, optional
        If we want to save the plot, name by which we want it to be saved
    save: bool, optional
        Whether we want to save txt files with the data obtained or not
        The default is False.
    plotsdir : string, optional
        If we want to specify a route to where the files should be saved. The default is ''.

    Returns
    -------
    None.

    """
    
    if MM.shape[-2:] == (4, 4):
        if len(MM.shape) == 4:
            MM = MM.reshape(height*width, 4, 4).transpose(1, 2, 0)
        elif len(MM.shape) != 3:
            print("Unexpected shape")
    else:
        print("Not a 4x4 matrix in the last dimensions")
            
    
    if chdecomp==True:
        params_chd = parameters_ChDecomp(MM, height, width, params_list)     
    
        if filename==None:
            for param in params_chd.keys():
                if param=='IPPS':
                    plot_IPPS(params_chd[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle)
                elif param=='Retardance':
                    plot_parameter(params_chd[param], subtitle=subtitles[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,180])
                else:
                    plot_parameter(params_chd[param], subtitle=subtitles[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle)
    
        else:
            for param in params_chd.keys():
                if param=='IPPS':
                    plot_IPPS(params_chd[param], subtitle=subtitles[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle, name=plotsdir+filename+'_'+param)
                elif param=='Retardance':
                    plot_parameter(params_chd[param],subtitle=subtitles[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,180], name=plotsdir+filename+'_'+param)
                else:
                    plot_parameter(params_chd[param], subtitle=subtitles[param], x=x, y=y, xtitle=xtitle, ytitle=ytitle, name=plotsdir+filename+'_'+param)

        
        #If specified, save the IPP data    
        if save==True:
            np.save("Data/"+filename, params_chd['IPPS'])
            np.save("Data/Inc_PhiM", y)
            np.save("Data/Inc_PyM", x)
            
    if general==True:
        params_gen = parameters_MM(MM, width, height, params_list)
        
        #Plot of the parameters of the original MM
        for key in params_gen.keys():
            if filename==None:
                plot_parameter(params_gen[param], param, x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1)
            
            else:
                plot_parameter(params_gen[param], param, x=x, y=y, xtitle=xtitle, ytitle=ytitle, name=plotsdir+filename+'_'+param, num=1)
                
    return None

#%% Diattenuators 


xtitle=r'$p_y$'
ytitle = r'$\sigma_\theta$'


filename="DNotNorm_p0_px05"

MMD, x, y = simulation_diattenuators(0, 0.5, norm=0)
height=len(y)
width=len(x)
MMD = MMD.reshape(height*width, 4, 4).transpose(1, 2, 0)
data_chd_di = parameters_ChDecomp(MMD, len(y), len(x), ['diat'])
data_gen_di = parameters_MM(MMD, len(y), len(x), ['diat', 'pdelta', 'ps','pol'])


#%%%% Plots Ch Decomp

save=False
    
# plot_IPPS(data_chd_di['IPPS'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, rot=True, name = filename +'_IPPS' if save is True else None)
#plot_parameter(data_chd_di['Retardance'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$\Delta$', lims=[0,180], rot=True, name= filename +'_DRet' if save is True else None)
plot_parameter(data_chd_di['Diattenuation'], x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$D$',  rot=True, name= filename +'_DDiat' if save is True else None)
# plot_parameter(data_chd_di['Polarizance'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$\mathcal{P}$',  rot=True, name= filename +'_DPol' if save is True else None)
#plot_parameter(data_chd_di['Pdelta'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_\Delta$', rot=True, name= filename +'_DPdelta' if save is True else None)
# plot_parameter(data_chd_di['Ps'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$P_S$',  rot=True, name= filename +'_DPs' if save is True else None)
# plot_parameter(data_chd_di['Coefficients'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, rot=True, name= filename +'_Coefs' if save is True else None)
  
#%%%% Plots General

save = False
    
#plot_parameter(data_gen_di['Retardance']*180/np.pi, title=r'Retardance ($\Delta$)', x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,180], num=1, rot=True, name= filename +'_Ret' if save is True else None)
plot_parameter(data_gen_di['Diattenuation'], title=r'Diattenuation ($D$)',  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  num=1, rot=True, name= filename +'_Diat' if save is True else None)
plot_parameter(data_gen_di['Polarizance'], title=r'Polarizance ($\mathcal{P}$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Pol' if save is True else None)
plot_parameter(data_gen_di['Pdelta'], title=r'Polarimetric Purity ($P_\Delta$)',  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  num=1,  rot=True, name= filename +'_Pdelta' if save is True else None)
plot_parameter(data_gen_di['Ps'], title=r'Spherical Purity ($P_S$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Ps' if save is True else None)

#%%%% Save data

# np.save("Data/"+filename, data_chd_di['IPPS'])
# np.save("Data/DNorm_PyM", x)
# np.save("Data/DNorm_sigmaTheta", y)

#%%%% Iteration


ps={'0':0, '30':0.3}
pxs={'02':0.2, '05': 0.5, '1':1}

xtitle=r'$p_y$'
ytitle = r'$\sigma_\theta$'
save=False


for ip in ps.keys():
    for ipx in pxs.keys():
        filename="DNorm_p"+ip+"_px"+ipx
        
        MMD, x, y = simulation_diattenuators(ps[ip], pxs[ipx])
        height=len(y)
        width=len(x)
        MMD = MMD.reshape(height*width, 4, 4).transpose(1, 2, 0)
        #data_chd_di = parameters_ChDecomp(MMD, len(y), len(x), ['diat', 'pol', 'coeff', 'ps', 'ipps'])
        data_gen_di = parameters_MM(MMD, len(y), len(x), ['diat', 'pdelta', 'ps','pol'])

        # plot_IPPS(data_chd_di['IPPS'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, rot=True, name = filename +'_IPPS' if save is True else None)
        # #plot_parameter(data_chd_di['Retardance'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$\Delta$', lims=[0,180], rot=True, name= filename +'_DRet' if save is True else None)
        # plot_parameter(data_chd_di['Diattenuation'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$D$',  rot=True, name= filename +'_DDiat' if save is True else None)
        # plot_parameter(data_chd_di['Polarizance'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$\mathcal{P}$',  rot=True, name= filename +'_DPol' if save is True else None)
        # #plot_parameter(data_chd_di['Pdelta'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_\Delta$', rot=True, name= filename +'_DPdelta' if save is True else None)
        # plot_parameter(data_chd_di['Ps'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$P_S$',  rot=True, name= filename +'_DPs' if save is True else None)
        # plot_parameter(data_chd_di['Coefficients'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, rot=True, name= filename +'_Coefs' if save is True else None)
        
        plot_parameter(data_gen_di['Diattenuation'], title=r'Diattenuation ($D$)',  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  num=1, rot=True, name= filename +'_Diat' if save is True else None)
        plot_parameter(data_gen_di['Polarizance'], title=r'Polarizance ($\mathcal{P}$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Pol' if save is True else None)
        plot_parameter(data_gen_di['Pdelta'], title=r'Polarimetric Purity ($P_\Delta$)',  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  num=1,  rot=True, name= filename +'_Pdelta' if save is True else None)
        plot_parameter(data_gen_di['Ps'], title=r'Spherical Purity ($P_S$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Ps' if save is True else None)



            

#%% Retarders

filename="R_p0"
xtitle=r'$\phi$'
ytitle = r'$\sigma_\theta$'

MMR,x,y=simulation_retarders(0)

height=len(y)
width=len(x)
MMR = MMR.reshape(height*width, 4, 4).transpose(1, 2, 0)
data_chd_re = parameters_ChDecomp(MMR, len(y), len(x), ['diat', 'ret', 'pol', 'coeff', 'ps', 'ipps'])
data_gen_re = parameters_MM(MMR, len(y), len(x), ['diat', 'ret', 'pdelta', 'ps','pol'])

#%%%% Plots Ch Decomp

save = True
    
plot_IPPS(data_chd_re['IPPS'],  x=x, y=y, num=2, xtitle=xtitle, ytitle=ytitle, rot=True, name = filename +'_IPPS2' if save is True else None)
#plot_parameter(data_chd_re['Retardance'],  x=x, y=y,  xtitle=xtitle, ytitle=ytitle, subtitle=r'$\Delta$', lims=[0,180], rot=True, name= filename +'_DRet' if save is True else None)
# plot_parameter(data_chd_re['Diattenuation'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$D$',  rot=True, name= filename +'_DDiat' if save is True else None)
# plot_parameter(data_chd_re['Polarizance'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$\mathcal{P}$',  rot=True, name= filename +'_DPol' if save is True else None)
#plot_parameter(data_chd_re['Pdelta'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_\Delta$', rot=True, name= filename +'_DPdelta' if save is True else None)
#plot_parameter(data_chd_re['Ps'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_S$',  rot=True, name= filename +'_DPs' if save is True else None)
#plot_parameter(data_chd_re['Coefficients'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, rot=True, name= filename +'_Coefs' if save is True else None)
  
#%%%% Plots General

save = True
    
plot_parameter(data_gen_re['Retardance'], title=r'Retardance ($\Delta$)', x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,180], num=1, rot=True, name= filename +'_Ret' if save is True else None)
# plot_parameter(data_gen_re['Diattenuation'], title=r'Diattenuation ($D$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Diat' if save is True else None)
# plot_parameter(data_gen_re['Polarizance'], title=r'Polarizance ($\mathcal{P}$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Pol' if save is True else None)
plot_parameter(data_gen_re['Pdelta'], title=r'Polarimetric Purity ($P_\Delta$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1,  rot=True, name= filename +'_Pdelta' if save is True else None)
plot_parameter(data_gen_re['Ps'], title=r'Spherical Purity ($P_S$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Ps' if save is True else None)

#%%%% Save data

# np.save("Data/"+filename, data_chd_re['IPPS'])
# np.save("Data/Re_Phi", x)
# np.save("Data/Re_sigmaTheta", y)
#%%%% Iteration

ps={'0':0, '30':0.3}
xtitle=r'$\phi$'
ytitle = r'$\sigma_\theta$'

save=True

for ip in ps.keys():
    filename="R_p"+ip
    MMR,x,y=simulation_retarders(ps[ip])

    height=len(y)
    width=len(x)
    MMR = MMR.reshape(height*width, 4, 4).transpose(1, 2, 0)
    data_chd_re = parameters_ChDecomp(MMR, len(y), len(x), ['diat', 'ret', 'pol', 'coeff', 'ps', 'ipps'])
    data_gen_re = parameters_MM(MMR, len(y), len(x), ['diat', 'ret', 'pdelta', 'ps','pol'])

    plot_IPPS(data_chd_re['IPPS'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, rot=True, name = filename +'_IPPS' if save is True else None)
    plot_parameter(data_chd_re['Retardance'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$\Delta$', lims=[0,180], rot=True, name= filename +'_DRet' if save is True else None)
    # plot_parameter(data_chd_re['Diattenuation'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$D$',  rot=True, name= filename +'_DDiat' if save is True else None)
    # plot_parameter(data_chd_re['Polarizance'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$\mathcal{P}$',  rot=True, name= filename +'_DPol' if save is True else None)
    #plot_parameter(data_chd_re['Pdelta'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_\Delta$', rot=True, name= filename +'_DPdelta' if save is True else None)
    plot_parameter(data_chd_re['Ps'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_S$',  rot=True, name= filename +'_DPs' if save is True else None)
    plot_parameter(data_chd_re['Coefficients'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, rot=True, name= filename +'_Coefs' if save is True else None)
      
    plot_parameter(data_gen_re['Retardance'], title=r'Retardance ($\Delta$)', x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,180], num=1, rot=True, name= filename +'_Ret' if save is True else None)
    # plot_parameter(data_gen_re['Diattenuation'], title=r'Diattenuation ($D$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Diat' if save is True else None)
    # plot_parameter(data_gen_re['Polarizance'], title=r'Polarizance ($\mathcal{P}$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Pol' if save is True else None)
    plot_parameter(data_gen_re['Pdelta'], title=r'Polarimetric Purity ($P_\Delta$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1,  rot=True, name= filename +'_Pdelta' if save is True else None)
    plot_parameter(data_gen_re['Ps'], title=r'Spherical Purity ($P_S$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, rot=True, name= filename +'_Ps' if save is True else None)


#%% Diattenuators and retarders 

xtitle=r"$\phi$"
ytitle=r"$p_y$"


MMDR, x, y = simulation_both(0.5, 0, 0.5, 90)
height=len(y)
width=len(x)
MMDR = MMDR.reshape(height*width, 4, 4).transpose(1, 2, 0)
data_chd_dr = parameters_ChDecomp(MMDR, len(y), len(x), ['ipps'])
#data_gen_dr = parameters_MM(MMD, len(y), len(x), ['diat', 'pdelta', 'ps','pol'])

#%%% Plots Ch Decomp

save=True
    
plot_IPPS(data_chd_dr['IPPS'], num=2, x=x, y=y, xtitle=xtitle, ytitle=ytitle, name = filename +'_IPPS2' if save is True else None)
#plot_parameter(data_chd_di['Retardance'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$\Delta$', lims=[0,180], name= filename +'_DRet' if save is True else None)
# plot_parameter(data_chd_di['Diattenuation'], x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$D$',  name= filename +'_DDiat' if save is True else None)
# plot_parameter(data_chd_di['Polarizance'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$\mathcal{P}$',  name= filename +'_DPol' if save is True else None)
# #plot_parameter(data_chd_di['Pdelta'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_\Delta$', name= filename +'_DPdelta' if save is True else None)
# plot_parameter(data_chd_di['Ps'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$P_S$',  name= filename +'_DPs' if save is True else None)
# plot_parameter(data_chd_di['Coefficients'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, name= filename +'_Coefs' if save is True else None)
  
#%%% Plots General

save = False
    
#plot_parameter(data_gen_di['Retardance']*180/np.pi, title=r'Retardance ($\Delta$)', x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,180], num=1, name= filename +'_Ret' if save is True else None)
plot_parameter(data_gen_di['Diattenuation'], title=r'Diattenuation ($D$)',  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  num=1, name= filename +'_Diat' if save is True else None)
plot_parameter(data_gen_di['Polarizance'], title=r'Polarizance ($\mathcal{P}$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, name= filename +'_Pol' if save is True else None)
plot_parameter(data_gen_di['Pdelta'], title=r'Polarimetric Purity ($P_\Delta$)',  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  num=1,  name= filename +'_Pdelta' if save is True else None)
plot_parameter(data_gen_di['Ps'], title=r'Spherical Purity ($P_S$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, name= filename +'_Ps' if save is True else None)


#%%%% Save data

# np.save("Data/"+filename, data_chd_dr['IPPS'])
# np.save("Data/Inc_PyM", pyM)
# np.save("Data/Inc_Phi", Phi)
             
#%%% Iteration

alphas = dict(zip(['0', '25', '50', '75','100'], [0, 0.25, 0.50, 0.75, 1]))
ps = dict(zip(['0','30','50'], [0, 0.3, 0.5]))
pxs = dict(zip(['02','05','1'], [0.2, 0.5, 1]))
sigmas = dict(zip(['10','20','30', '40', '60', '90', '150'], [10, 20, 30, 40, 60, 90, 150]))
norms = dict(zip(['No', 'Si'], [0, 1]))

ialph = '50'
ip = '0'
ipx = '1'
isigma = '150'
inorm = 'Si'

xtitle=r"$\phi$"
ytitle=r"$p_y$"
save=False

for ialph in ['25', '50', '75']:
    details = r"$\beta=$"+f"{ps[ip]}\t"+r"$\sigma_\theta=$"+f"{isigma}\t"+r"$\alpha=$"+f"{alphas[ialph]}\t"+r"$p_x$"+f"={pxs[ipx]}\t"
    filename = "Inc_p"+f"{ip}"+"_px"+f"{ipx}"+"_sth"+f"{isigma}"+"_alpha"+f"{ialph}"+f"_{inorm}"+"norm"
    MMDR, x, y = simulation_both(alphas[ialph], ps[ip], pxs[ipx], sigmas[isigma])
    height=len(y)
    width=len(x)
    MMDR = MMDR.reshape(height*width, 4, 4).transpose(1, 2, 0)
    data_chd_dr = parameters_ChDecomp(MMDR, len(y), len(x), ['ret'])
    #data_gen_dr = parameters_MM(MMD, len(y), len(x), ['diat', 'pdelta', 'ps','pol'])
    
    #plot_IPPS(data_chd_dr['IPPS'], title=details, x=x, y=y, xtitle=xtitle, ytitle=ytitle, name = filename +'_IPPS' if save is True else None)
    plot_parameter(data_chd_dr['Retardance'], title=details, x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$\Delta$', lims=[0,180], name= filename +'_DRet' if save is True else None)
    #plot_parameter(data_chd_dr['Diattenuation'], title=details, x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$D$',  name= filename +'_DDiat' if save is True else None)
    # plot_parameter(data_chd_dr['Polarizance'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$\mathcal{P}$',  name= filename +'_DPol' if save is True else None)
    # #plot_parameter(data_chd_dr['Pdelta'],  x=x, y=y, xtitle=xtitle, ytitle=ytitle, subtitle=r'$P_\Delta$', name= filename +'_DPdelta' if save is True else None)
    # plot_parameter(data_chd_dr['Ps'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  subtitle=r'$P_S$',  name= filename +'_DPs' if save is True else None)
    # plot_parameter(data_chd_dr['Coefficients'],  x=x, y=y,xtitle=xtitle, ytitle=ytitle, subtitle=r'$c$', num=4, name= filename +'_Coefs' if save is True else None)
      
    #plot_parameter(data_gen_dr['Retardance']*180/np.pi, title=r'Retardance ($\Delta$)', x=x, y=y, xtitle=xtitle, ytitle=ytitle, lims=[0,180], num=1, name= filename +'_Ret' if save is True else None)
    # plot_parameter(data_gen_dr['Diattenuation'], title=r'Diattenuation ($D$)',  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  num=1, name= filename +'_Diat' if save is True else None)
    # plot_parameter(data_gen_dr['Polarizance'], title=r'Polarizance ($\mathcal{P}$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, name= filename +'_Pol' if save is True else None)
    # plot_parameter(data_gen_dr['Pdelta'], title=r'Polarimetric Purity ($P_\Delta$)',  x=x, y=y,xtitle=xtitle, ytitle=ytitle,  num=1,  name= filename +'_Pdelta' if save is True else None)
    # plot_parameter(data_gen_dr['Ps'], title=r'Spherical Purity ($P_S$)',  x=x, y=y, xtitle=xtitle, ytitle=ytitle, num=1, name= filename +'_Ps' if save is True else None)




                        
#%% Experimental data

#%%% Data processing
# THIS TAKES A LONG TIME
# Run only the first time, then the data is saved and can be reloaded

files = glob.glob("MM experimentales/*.mat")

for file in files[3:5]:
    name = file.removeprefix("MM experimentales\\").removesuffix(".mat")
    filename = 'Plots Experimentals/'+name
    
    MM, M00, Nx, Ny = pf.read_file(file)[:4]
    
    # Plot of intensity
    M=M00.reshape(Nx,Ny).T
    plt.imshow(M, cmap='RdBu_r', aspect="auto")
    plt.colorbar() 
    plt.axis('off')
    plt.title(name)
    plt.savefig(filename+'_Int', dpi=300, bbox_inches='tight')
    plt.show()
    
    #Calculation and saving of the parameters for the matrices of the characteristic decomposition
    params = parameters_ChDecomp(MM, Ny, Nx, ['diat', 'ret', 'pol', 'pdelta', 'ps', 'ipps'])
    np.save('Data Mostres/'+name+'_ChD', params)
    
    #Plot of the parameters of the characteristic decomposition    
    plot_IPPS(params['IPPS'], 'IPPS', lims=None,  origin='upper', name= filename +'_IPPS')
    plot_parameter(params['Retardance']*180/np.pi, 'Retardance', lims=None, origin='upper', name= filename +'_DRet')
    plot_parameter(params['Diattenuation'], 'Diattenuation', lims=None,  origin='upper', name= filename +'_DDiat')
    plot_parameter(params['Polarizance'], 'Polarizance', lims=None,  origin='upper', name= filename +'_DPol')
    plot_parameter(params['Pdelta'], r'$P_\Delta$', lims=None,  origin='upper', name= filename +'_DPdelta')
    plot_parameter(params['Ps'], r'$P_S$', lims=None,  origin='upper', name= filename +'_DPs')
    
    #Calculation and saving of the parameters of the original MM
    params_gen = parameters_MM(MM, Nx, Ny, ['diat', 'ret', 'pdelta', 'ps','pol'])
    np.save('Data Mostres/'+name+'_gen', params_gen)
    
    #Plot of the parameters of the original MM
    for key in params_gen.keys():
        plt.imshow(params_gen[key], cmap='viridis', aspect="auto")
        plt.colorbar()  # Show color scale
        plt.axis('off')
        plt.title(key)
        plt.savefig(filename+'_'+key, dpi=300, bbox_inches='tight')
        plt.show()
        
        
#%%% Plots

translations = {
    'Chicken3_Tendon1': 'Chicken tendon',
    'heart5': 'Heart',
    'rabbit_leg2': 'Rabbit leg',
    'rabbit_leg3': 'Rabbit leg',
    'rinon_debajo': 'Kidney (bottom)',
    'traquea_cordero': 'Lamb trachea'
}

format_names = {}

for file in files:
    name = file.replace('MM experimentales\\', '').replace('.mat', '')
    base, wavelength = name.replace('MM_', '').rsplit('_', 1)
    label = translations.get(base, base.replace('_', ' ').capitalize())
    formatted = f"{label} {wavelength}"
    format_names[name] = formatted
    
    
#%%%% Plot intensity

files = glob.glob("MM experimentales/*.mat")

save=True
for file in files[0:7]:
    name = file.removeprefix("MM experimentales\\").removesuffix(".mat")
    title = format_names[name]
    filename = 'Plots Experimentals/'+name
    
    MM, M00, Nx, Ny = pf.read_file(file)[:4]
    vmax=np.percentile(M00, 99.9)
    vmin=np.min(M00)
    M00=M00.reshape(Nx,Ny).T

    #plot_parameter(M00, title, num=1, lims=None, origin='upper',  name= filename +'_Int' if save is True else None)
    plot_parameter(M00, title+'*', num=1, lims=[vmin, vmax], origin='upper',  name= filename +'_Int' if save is True else None)

#%%%% Plots Ch Decomp

files = glob.glob("MM experimentales/*.mat")

save = False
for file in files[4:5]:
    name = file.removeprefix("MM experimentales\\").removesuffix(".mat")
    filename = 'Plots Experimentals/'+name
    data_chd = np.load('Data Mostres 2/'+name+'_ChD.npy', allow_pickle=True).item()
    
    # plot_IPPS(data_chd['IPPS'], lims=None,  origin='upper', name = filename +'_IPPS' if save is True else None)
    plot_parameter(data_chd['Retardance']*180/np.pi, subtitle=r'$\Delta$', lims=None, origin='upper', name= filename +'_DRet' if save is True else None)
    # plot_parameter(data_chd['Diattenuation'], subtitle=r'$D$', lims=None,  origin='upper', name= filename +'_DDiat' if save is True else None)
    # plot_parameter(data_chd['Polarizance'], subtitle=r'$\mathcal{P}$', lims=None,  origin='upper', name= filename +'_DPol' if save is True else None)
    #plot_parameter(data_chd['Pdelta'], subtitle=r'$P_\Delta$', lims=None,  origin='upper', name= filename +'_DPdelta' if save is True else None)
    plot_parameter(data_chd['Ps'], subtitle=r'$P_S$', lims=None,  origin='upper', name= filename +'_DPs' if save is True else None)
    

#%%%% Plots General

files = glob.glob("MM experimentales/*.mat")

save = True
for file in files:
    name = file.removeprefix("MM experimentales\\").removesuffix(".mat")
    filename = 'Plots Experimentals/'+name
    data_gen = np.load('Data Mostres/'+name+'_gen.npy', allow_pickle=True).item()
    
    plot_parameter(data_gen['Retardance']*180/np.pi, title=r'Retardance ($\Delta$)', lims=None, num=1, origin='upper', name= filename +'_Ret' if save is True else None)
    plot_parameter(data_gen['Diattenuation'], title=r'Diattenuation ($D$)', lims=None, num=1, origin='upper', name= filename +'_Diat' if save is True else None)
    plot_parameter(data_gen['Polarizance'], title=r'Polarizance ($\mathcal{P}$)', lims=None,  num=1, origin='upper', name= filename +'_Pol' if save is True else None)
    plot_parameter(data_gen['Pdelta'], title=r'Polarimetric Purity ($P_\Delta$)', lims=None, num=1,  origin='upper', name= filename +'_Pdelta' if save is True else None)
    plot_parameter(data_gen['Ps'], title=r'Spherical Purity ($P_S$)', lims=None, num=1, origin='upper', name= filename +'_Ps' if save is True else None)
    

