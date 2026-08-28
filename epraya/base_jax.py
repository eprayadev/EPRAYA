import io
import numpy as np
import scipy as sci
import scipy.constants as scic
import scipy.integrate as scii
import scipy.signal as scs
import scipy.constants as scc
from functools import cmp_to_key
from scipy.interpolate import CubicSpline as cubichers
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull
from typing import Union, Any, List
from dataclasses import dataclass, replace
from dataclasses import field as dcfield
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets, Label
from ipywidgets import VBox, interactive_output, IntSlider,fixed, IntText, Dropdown, RadioButtons,interact_manual, Layout, Button, FloatText,Output, HBox
from IPython.display import display, Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.ticker import FormatStrFormatter,EngFormatter
from pandas import DataFrame, concat, read_csv
import warnings
from copy import deepcopy
import jax as jx
import jax.numpy as jxn
from chex import dataclass as jaxdatclass
import optax
import jax.nn as jnn
import jax.scipy.signal as jsig
import jax.scipy.special as jsp
from numba import njit
from functools import partial
from scipy.optimize import least_squares as leasts
from scipy.stats import linregress
import concurrent.futures
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
import re
from itertools import product as iterproduct
from .base_powd import *
from .base_ham import *
from matplotlib import cm

jx.config.update("jax_debug_nans", True)
jx.config.update("jax_enable_x64", True)

@jaxdatclass
class JHval:
    '''
    Container jax-class for hamiltonial's parameters. Must be initialized with a variable like ``Ham``. To change one parameter use the sintaxis ``Ham.S=1/2``.
    
    Parameters
    ----------
    
    S : float 
        Spin value ex. (1/2,0,3/2).
    g : array_like or float

        g value of system, can be float, for isotropic case or array for anisotropic.
    I : float
        Nuclear spin value

    L : float
        Angular momentum
    A : array_like or float 
        Hyperfine constant, float for isotropic and array for anisotropic

    Q : array_like or float 
        Quadrupole nuclear interaction constant, float for isotropic and array for anisotropic
    D : array_like
        Zero field interaction constants D and E, two value array [0,0]

    Bk2 : array_like
        Stevens k=-/+2 constants
    Bk4 : array_like

        Stevens k=-/+4 constants    
    Bk6 : array_like
        Stevens k=-/+6 constants      
    lc : float

        Spin-orbit interaction constant
    Hpp : array_like
        Peak to peak distance for the voigtian function using [Hg,Hl], for gaussian and lorentzian distance

    eta : float
        weight of the gaussian contribution to the voigtian function, from 0 to 1. If eta is 0, the function is lorentzian and if eta is 1, the function is gaussian.
    weight: float

        Dummy variable by the moment
    Nucl : str
        Isotope of the sample. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
        

    Example
    --------
    
    >>> import epraya as epr 
    >>> Ham=epr.JHval()
    >>> Ham.S=1
    >>> Ham.I=1/2
    >>> Ham.g=[2.003,1.8,1.5]
    >>> print(Ham)
    JHval(S=1, g=[2.003, 1.8, 1.5], I=0.5, L=0.0, A=0.0,
    Q=Array([0, 0, 0], dtype=int32), 
    D=Array([0, 0], dtype=int32), 
    Bk2=[0, 0, 0, 0, 0], Bk4=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    Bk6=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    lc=0.0, Hpp=Array([0, 1], dtype=int32), eta=0.5, weight=0.0)

    '''
    S: Union[float,int]=1/2   # Spin
    g: Union[List[float],float,int]=dcfield(default_factory=lambda: 2.003)  # g value
    I: float=0.0   # Nuclear spin
    L: float=0.0   # Angular momentum
    A: Union[List[float], float]=dcfield(default_factory=lambda:0.0)     # Hyperfine constant
    Q: Union[list[float], float]=dcfield(default_factory=lambda:jxn.array([0,0,0]))    # Quadrupole interaction constant
    D: Union[list[float], float]=dcfield(default_factory=lambda:jxn.array([0,0]))     # Zero field interaction D and E constants
    Bk2: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0])
    Bk4: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0,0,0,0,0])
    Bk6: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0,0,0,0,0,0,0,0,0])
    lc: float=0.0                          # Spin-orbit interaction constant
    Hpp: List=dcfield(default_factory=lambda:jxn.array([0,1]))
    eta: float=0.5
    weight: float=0.0

@jaxdatclass
class JEco:
    '''
    Container jax-class for the experimental parameters. Must be initialized with a variable like ``Exp``. To change one parameter use the sintaxis ``Exp.Freq=9.40``.
    
    Parameters
    ----------
    
    Freq : float
        Frequency of the microwave in the EPR spectrometer in GHz.
    Points  : int
        Number of points used to take the EPR spectrum.
    Temperature : float
        Temperature of the sample during the measurement in Kelvin.

    Fdirection : list=[0,0,1]
        Direction of incidence of the magnetic field in relation with the lab frame. By deafult it's [0,0,1], the Z direction.
    Mwdirection : list=[1,0,0]
        Direction of incidence of the microwave radiation in relation with the lab frame. By deafult it's [1,0,0], the X direction.
    Frange : list=[0,1]
        Field range used in the EPR spectrum in mT.
    Sampleframe : list==[0,0,0]
        Three Euler angles of the orientation of the sample in relation to the lab frame.
    Molframe : list=[0,0,0]
        Three Euler angles of the orientation of the  paramagnetic molecule in relation to the sample frame.
    gframe : list=[0,0,0]
        Three Euler angles of the orientation of the tensor g in the molecular frame.
    Aframe : list=[0,0,0]
        Three Euler angles of the orientation of the tensor A in the molecular frame.    
    Dframe : list=[0,0,0]
        Three Euler angles of the orientation of the tensor Q in the molecular frame.    
    Qframe : list=[0,0,0]
        Three Euler angles of the orientation of the tensor D in the molecular frame.   
        
    Example
    --------
    
    >>> import epraya as epr 
    >>> Exp=epr.JEco()
    >>> Exp.Freq=9.43
    >>> Exp.Points=2046
    >>> Exp.Temperature=306
    >>> Exp.Frange=[0,400]
    >>> Exp.Mwdirection=[0,1,0]
    >>> print(Exp)
    JEco(Freq=9.43, Points=2046, Temperature=306, 
    Fdirection=[0, 0, 1], Mwdirection=[0, 1, 0], 
    Frange=[0, 400], Sampleframe=[0, 0, 0], 
    Molframe=[0, 0, 0], gframe=[0, 0, 0], 
    Aframe=[0, 0, 0], Dframe=[0, 0, 0], Qframe=[0, 0, 0])
    '''
    Freq: float=9.433
    Points: int=4096
    Temperature: float=295.15
    Fdirection: list[float]=dcfield(default_factory=lambda:[0,0,1])
    Mwdirection: list[float]=dcfield(default_factory=lambda:[1,0,0])
    Frange: list[float]=dcfield(default_factory=lambda:[0,1])
    Sampleframe: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Molframe: list[float]=dcfield(default_factory=lambda:[0,0,0])
    gframe: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Aframe: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Dframe: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Qframe: list[float]=dcfield(default_factory=lambda:[0,0,0])


@jaxdatclass
class JEva:
    '''
    Container jax-class for the range of variation of the hamiltonian parameters by defining the minimum and then the maximum value of each variable. Must be initialized with a variable like ``Vary``. To change one parameter use the sintaxis ``Vary.g=[1.5,2.5,1.0,2.0,0.0,1.0]``.
    
    Parameters
    ----------
    
    g : list[float]
        Range to vary the three values of the g tensor.
    A : list[float]  
        Range to vary the three values of the A tensor.
    Q : list[float]  
        Range to vary the three values of the Q tensor.
    D : list[float]
        Range to vary the two values of the D tensor.
    Hpp : list[float]
        Range to vary the two values of the peak to peak distance.
    weight : float
        Dummy variable by the moment.
    
    Example
    --------
    
    >>> import epraya as epr
    >>> Vary=epr.JEva()
    >>> Vary.g=[1.5,2.5,1.0,2.0,0.0,1.0]
    >>> Vary.A=[100,300,200,400,200,250]
    >>> Vary.Hpp=[0,20,10,15]
    >>> print(Vary)
    JEva(g=[1.5, 2.5, 1.0, 2.0, 0.0, 1.0],
    A=[100, 300, 200, 400, 200, 250], Q=0.0, 
    D=0.0, Hpp=[0, 20, 10, 15], weight=0.0)
    '''
    g: Union[list[float],float]=0.0
    A: Union[list[float],float]=0.0     # Hyperfine constant
    Q: Union[list[float],float]=0.0     # Quadrupole interaction constant
    D: Union[list[float],float]=0.0
    Hpp: List=dcfield(default_factory=lambda:jxn.array([0,0]))
    weight: float=0.0

def Jstart():
    '''
    Creates the three JAX containers ``Ham``, ``Exp`` and ``Vary`` for one system.
    
    Returns
    -------
    
    Ham : Class
        Jax container for the hamiltonian parameters.
    Exp : Class
        Jax container for the experimental conditions.
    Vary : Class
        Jax container for the range and parameters to vary.
    
    Example
    -------
    
    >>> import epraya as epr
    >>> Ham, Exp, Vary=epr.Jstart()
    >>> print(Ham,Exp,Vary)
    JHval(S=0.5, g=2.003, I=0.0, L=0.0, A=0.0, 
    Q=Array([0, 0, 0], dtype=int32), D=Array([0, 0], dtype=int32),
    Bk2=[0, 0, 0, 0, 0], Bk4=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    Bk6=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], lc=0.0,
    Hpp=Array([0, 1], dtype=int32), eta=0.5, weight=0.0) 
    JEco(Freq=9.433, Points=4096, Temperature=295.15, 
    Fdirection=[0, 0, 1], Mwdirection=[1, 0, 0], Frange=[0, 1],
    Sampleframe=[0, 0, 0], Molframe=[0, 0, 0], gframe=[0, 0, 0],
    Aframe=[0, 0, 0], Dframe=[0, 0, 0], Qframe=[0, 0, 0]) 
    JEva(g=0.0, A=0.0, Q=0.0, D=0.0, 
    Hpp=Array([0, 0], dtype=int32), weight=0.0)
    '''
    global Exp,Vary,Ham
    Ham,Exp,Vary=JHval(),JEco(),JEva()
    return Ham,Exp,Vary

def JKronecker(a0,b0):
    '''
    Implementation of the Kronecker delta.
    
    Parameters
    ----------

    
    a0 : int
        First value to compare.
    b0 : int
        Second value to compare
    Returns
    -------
    
    1 : int
        If the two values are equal
    0 : int
        If the two values differ
        
    Example
    -------
    
    >>> import epraya as epr
    >>> a, b = 2, 3
    >>> c, d = 2, 4
    >>> print(epr.JKronecker(a,b))
    >>> print(epr.JKronecker(a,c))
    >>> print(epr.JKronecker(b,d))
    0.0
    1.0
    0.0
    '''
    return jxn.where(a0==b0,1.0,0.0)

def JPauli(s):
    '''
    Defines the Pauli matrices for the spin s.

    Parameters
    ----------

    s : float
        Spin operator value of the system.

    Returns
    -------

    sx : jax.numpy.array
        Pauli matrix of the spin x component.

    sy : jax.numpy.array
        Pauli matrix of the spin y component.
    sz : jax.numpy.array
        Pauli matrix of the spin z component.


    Example
    -------

    >>> import epraya as epr
    >>> s = 1
    >>> sx, sy, sz = epr.JPauli(s)
    >>> print(sx, sy, sz)
    [[0.        +0.j 0.70710677+0.j 0.        +0.j]
    [0.70710677+0.j 0.        +0.j 0.70710677+0.j]
    [0.        +0.j 0.70710677+0.j 0.        +0.j]] 
    [ 0.+0.j         -0.-0.70710677j  0.+0.j        ]
    [ 0.+0.70710677j  0.+0.j         -0.-0.70710677j]
    [ 0.+0.j          0.+0.70710677j  0.+0.j        ]] 
    [ 1.  0.  0.]
    [ 0.  0.  0.]
    [ 0.  0. -1.]]
    '''
    #Defines the pauli matrix for all s values:
    ms=jxn.linspace(s,-s,int(2*s+1))
    z1=0.5j
    r,g=0,0
    sz=jxn.diag(ms)
    sx,sy=jxn.zeros([int(2*s+1),int(2*s+1)],dtype=complex),jxn.zeros([int(2*s+1),int(2*s+1)],dtype=complex)
    for r in range (0,int(2*s+1)):
        for g in range (0,int(2*s+1)):
            yn=JKronecker(r+1,g+1)
            ym=JKronecker(r+1,g+2)
            yl=JKronecker(r+2,g+1)
            sx=sx.at[r,g].set(0.5*(ym+yl)*(jxn.sqrt((s+1)*(r+g+1)-((r+1)*(g+1)))))
            sy=sy.at[r,g].set(z1*((ym-yl)*(jxn.sqrt((s+1)*(r+g+1)-((r+1)*(g+1))))))
    return sx,sy,sz

#Spin Orbit term
def JLorbit(sx,sy,sz,lamda,dim,l=0):
    '''
    Function for the spin orbit interaction, with an isotropic coupling constant.
    *It's recommended to use the Stevens Operators instead of this function*

    Parameters
    ----------

    sx : jax.np.array
        Pauli matrix of the spin x component.
    sy : jax.np.array
        Pauli matrix of the spin y component.
    sz : jax.np.array
        Pauli matrix of the spin z component.
    lamda : float
        Isotropic coupling constant.
    dim : int
        Dimension of the total hamiltonian.
    l : float
        Angular momentum operator

    Returns
    -------
    
    orbe : jax.np.array
        Matrix of the spin orbit interaction with dimension dim.

    Example
    -------
    
    >>> import epraya as epr 
    >>> Ham, Exp, Vary=epr.Jstart()
    >>> Ham.S=1/2
    >>> Ham.L=1/2
    >>> Ham.lc=20
    >>> sx,sy,sz=epr.JPauli(Ham.S)
    >>> dim=int((2*Ham.S+1)*(2*Ham.L+1))
    >>> print(epr.JLorbit(sx,sy,sz,Ham.lc,dim,Ham.L))
    [[ 5.+0.j  0.+0.j  0.+0.j  0.+0.j]
    [ 0.+0.j -5.+0.j 10.+0.j  0.+0.j]
    [ 0.+0.j 10.+0.j -5.+0.j  0.+0.j]
    [ 0.+0.j  0.+0.j  0.+0.j  5.+0.j]]
    '''
    if l!=0:
        lx,ly,lz=JPauli(l)
        orbe=lamda*(jxn.kron(lx,sx)+jxn.kron(ly,sy)+jxn.kron(lz,sz))
        orbe=jxn.kron(orbe,jxn.eye(int(dim/(orbe).shape[1])))
        return orbe
    else:
        return np.zeros(dim)

def JHfi(ssx,ssy,ssz,iix,iiy,iiz,at,dim):
    '''
    Function for the hiperfine interaction.
    
    Parameters
    ----------
    ssx : jax.np.array
        Pauli matrix of the spin x component.
    ssy : jax.np.array
        Pauli matrix of the spin y component.
    ssz : jax.np.array
        Pauli matrix of the spin z component.
    iix : jax.np.array
        Pauli matrix of the nuclear spin x component.
    iiy : jax.np.array
        Pauli matrix of the nuclear spin y component.
    iiz : jax.np.array
        Pauli matrix of the nuclear spin z component.
    at : jax.np.array
        Array for the hiperfine interaction constant.
    dim : int
        Dimension of the total hamiltonian.
        
    Returns
    -------
    
    ta : jax.np.array
        Matrix of the hiperfine interaction with dimension dim.


    Example
    -------

    >>> import epraya as epr
    >>> Ham,Exp,_=epr.Jstart()
    >>> Ham.S=1/2
    >>> Ham.I=1/2
    >>> Ham.A=[200,300,200]
    >>> Ham=epr.Jchaframe(Ham,Exp)
    >>> sx,sy,sz=epr.JPauli(Ham.S)
    >>> ix,iy,iz=epr.JPauli(Ham.S)
    >>> dim=int((2*Ham.S+1)*(2*Ham.I+1))
    >>> print(epr.JHfi(sx,sy,sz,ix,iy,iz,Ham.A,dim))
    [[ 50.+0.j   0.+0.j   0.+0.j -25.+0.j]
    [  0.+0.j -50.+0.j 125.+0.j   0.+0.j]
    [  0.+0.j 125.+0.j -50.+0.j   0.+0.j]
    [-25.+0.j   0.+0.j   0.+0.j  50.+0.j]]
    '''
    ta=(at[0,0]*jxn.kron(ssx,iix))+(at[0,1]*jxn.kron(ssx,iiy))+(at[0,2]*jxn.kron(ssx,iiz))+(at[1,0]*jxn.kron(ssy,iix))+(at[1,1]*jxn.kron(ssy,iiy))+(at[1,2]*jxn.kron(ssy,iiz))+(at[2,0]*jxn.kron(ssz,iix))+(at[2,1]*jxn.kron(ssz,iiy))+(at[2,2]*jxn.kron(ssz,iiz))
    ta=jxn.kron(ta,jxn.eye(int(dim/(ta).shape[1])))
    return ta

def JHze(ssx,ssy,ssz,g,biel,dim):
    '''

    Function for the Zeeman interaction.
    
    Parameters
    ----------
    
    ssx : jax.np.array
        Pauli matrix of the spin x component.
    ssy : np.array
        Pauli matrix of the spin y component.
    ssz : jax.np.array
        Pauli matrix of the spin z component.
    g : jax.np.array
        Array for the g factor.
    biel : jax.np.array
        Direction of incidence of the magnetic field.

    dim : int
        Dimension of the total hamiltonian.
        
    Returns

    -------
    thz : jax.np.array
        Matrix of the Zeeman interaction with dimension dim.

    Example
    -------

    >>> import epraya as epr
    >>> Ham,Exp,_=epr.Jstart()
    >>> Ham.S=1/2
    >>> Ham.g=[2.003,3.0,2.003]
    >>> Ham=epr.Jchaframe(Ham,Exp)
    >>> sx,sy,sz=epr.JPauli(Ham.S)
    >>> dim=int((2*Ham.S+1))
    >>> print(epr.JHze(sx,sy,sz,Ham.g,[0,0,1],dim))
    [[ 1.0015+0.j  0.    +0.j]
    [ 0.    +0.j -1.0015+0.j]]

    '''
    hze=biel[0]*(g[0,0]*ssx+g[0,1]*ssy+g[0,2]*ssz)+biel[1]*(g[1,0]*ssx+g[1,1]*ssy+g[1,2]*ssz)+biel[2]*(g[2,0]*ssx+g[2,1]*ssy+g[2,2]*ssz)
    thz=jxn.kron(hze,jxn.eye(int(dim/(hze).shape[1])))
    return thz

def JIee(ssx1,ssy1,ssz1,ssx2,ssy2,ssz2,X,dim):
    '''
    Function for the electron-electron interaction.

    Parameters
    ----------
    
    ssx1 : jax.np.array
        Pauli matrix of the first spin x component.

    ssy1 : jax.np.array
        Pauli matrix of the first spin y component.
        
    ssz1 : jax.np.array
        Pauli matrix of the first spin z component.
        
    ssx2 : np.array
        Pauli matrix of the second spin x component.
    ssy2 : np.array
        Pauli matrix of the second spin y component.
    ssz2 : np.array
        Pauli matrix of the second spin z component.
    
    X : jax.np.array
        Electron-electron interaction constant.

    dim : int
        Dimension of the total hamiltonian.
    
    Returns
    -------

    eet : jax.np.array
        Matrix of the Electron-Electron interaction with dimension dim.
    Example

    -------
    >>> import epraya as epr
    >>> import numpy as np
    >>> Ham,_,_=epr.Jstart()
    >>> Ham.S1=1
    >>> Ham.S2=1/2
    >>> sx1,sy1,sz1=epr.JPauli(Ham.S1)
    >>> sx2,sy2,sz2=epr.JPauli(Ham.S2)
    >>> Ham.X1_2=[200,500,200]
    >>> Ham.X1_2=(Ham.X1_2)*np.eye(3)
    >>> dim=int((2*Ham.S1+1)*(2*Ham.S2+1))
    >>> print(epr.JIee(sx1,sy1,sz1,sx2,sy2,sz2,Ham.X1_2,dim))
    [[ 100.     +0.j    0.     +0.j    0.     +0.j -106.06602+0.j
    0.     +0.j    0.     +0.j]
    [   0.     +0.j -100.     +0.j  247.48737+0.j    0.     +0.j
    0.     +0.j    0.     +0.j]
    [   0.     +0.j  247.48737+0.j    0.     +0.j    0.     +0.j
    0.     +0.j -106.06602+0.j]
    [-106.06602+0.j    0.     +0.j    0.     +0.j    0.     +0.j
    247.48737+0.j    0.     +0.j]
    [   0.     +0.j    0.     +0.j    0.     +0.j  247.48737+0.j
    -100.     +0.j    0.     +0.j]
    [   0.     +0.j    0.     +0.j -106.06602+0.j    0.     +0.j
    0.     +0.j  100.     +0.j]]
    '''
    eet=(X[0,0]*np.kron(ssx1,ssx2))+(X[0,1]*np.kron(ssx1,ssy2))+(X[0,2]*np.kron(ssx1,ssz2))+(X[1,0]*np.kron(ssy1,ssx2))+(X[1,1]*np.kron(ssy1,ssy2))+(X[1,2]*np.kron(ssy1,ssz2))+(X[2,0]*np.kron(ssz1,ssx2))+(X[2,1]*np.kron(ssz1,ssy2))+(X[2,2]*np.kron(ssz1,ssz2))
    eet=jxn.kron(eet,jxn.eye(int(dim/(eet).shape[1])))
    return eet

def JQii(iix,iiy,iiz,q,dim):
    '''

    Function for the nuclear quadrupolar interaction.
    
    Parameters
    ----------
    iix : jax.np.array
        Pauli matrix of the nuclear spin x component.
    iiy : jax.np.array
        Pauli matrix of the nuclear spin y component.
    iiz : jax.np.array
        Pauli matrix of the nuclear spin z component.
    q : jax.np.array
        Array for the quadrupolar interaction constant.
    dim : int
        Dimension of the total hamiltonian.
        
    Returns
    -------
    tql : jax.np.array
        Matrix of the nuclear quadrupolar interaction with dimension dim.

    Example
    -------

    >>> import epraya as epr 
    >>> Ham,Exp,_=epr.Jstart()
    >>> Ham.I=1
    >>> Ham.Q=[0.5,10.0,0]
    >>> Ham=epr.Jchaframe(Ham,Exp)
    >>> ix,iy,iz=epr.JPauli(Ham.I)
    >>> dim=int((2*Ham.I+1))
    >>> print(epr.JQii(ix,iy,iz,Ham.Q,dim))
    [[ 0.  +0.j         -4.75+0.49999997j  0.  +0.j        ]
    [-4.75-0.49999997j  0.  +0.j         -4.75+0.49999997j]
    [ 0.  +0.j         -4.75-0.49999997j  0.  +0.j        ]]
    '''
    hql=(q[0,0]*iix*iix)+(q[1,1]*iiy*iiy)+(q[2,2]*iiz*iiz)+(q[0,1]*(iix*iiy)-(iiy*iix))+(q[1,2]*(iiy*iiz)-(iiz*iiy))
    +(q[2,0]*(iiz*iix)-(iix*iiz))
    tql=jxn.kron(hql,jxn.eye(int(dim/(hql).shape[1])))
    return tql


def JNhze(I,iix,iiy,iiz,dim,gn,direction=[0,0,1]):
    '''
    Function for the nuclear Zeeman interaction.
    
    Parameters
    ----------
    
    I : float
        Nuclear spin value.
    iix : jax.np.array
        Pauli matrix of the nuclear spin x component.
    iiy : jax.np.array
        Pauli matrix of the nuclear spin y component.
    iiz : jax.np.array
        Pauli matrix of the nuclear spin z component.
    dim : int
        Dimension of the total hamiltonian.
    gn : float
        Nuclear g factor of the element of the paramagnetic center.
    direction : jax.np.array
        Direction of incidence of the magnetic field.

    Returns
    -------
    
    nhz : jax.np.array
        Matrix of the Zeeman nuclear interaction with dimension dim.

    Example
    -------

    >>> import epraya as epr 
    >>> Ham,Exp,_=epr.Jstart()
    >>> Ham.I=1
    >>> gn=epr.gnfactor('55Mn')
    >>> ix,iy,iz=epr.JPauli(Ham.I)
    >>> dim=int((2*Ham.I+1))
    >>> print(epr.JNhze(Ham.I,ix,iy,iz,dim,gn,[0,0,1]))
    [[ 1.3813+0.j  0.    +0.j  0.    +0.j]
    [ 0.    +0.j  0.    +0.j  0.    +0.j]
    [ 0.    +0.j  0.    +0.j -1.3813+0.j]]
    '''
    direct=direction[0]*iix+direction[1]*iiy+direction[2]*iiz
    nhz=gn*direct
    nhz=jxn.kron(nhz,jxn.eye(int(dim/(nhz).shape[1])))
    return nhz

def gnfactor(Nucl='None'):
    '''
    Function to read the nuclear g factor from the added table.
    
    Parameter
    ---------
    
    Nucl : str
        Isotope of the sample. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    
    Returns
    -------
    
    gn : float
        Nuclear g factor for the Nucl element.
    '''
    if Nucl!='None':
        route=resources.files(__package__).joinpath("nucleardaat.txt")
        krle=read_csv(route,header=0,sep='\t')
        deq=krle[krle['Symbol']==Nucl]
        return float(deq['gN_factor'].values[0])
    else:
        return 0.0

def Jchaframe(Ham,Exp):
    '''
    Adds the rotations for the constants g, A, Q and D, described by the conditions gframe, Aframe, Qframe and D.frame.
    
    Parameters
    ----------
    
    Ham : Class
        Container of the Hamiltonian parameters.

    Exp : Class
        Container of the Experimental conditions.

    Returns
    -------
    
    Ham : Class
        Container of the Hamiltonian parameters, with the corrected values.
    
    Example
    -------

    >>> import epraya as epr 
    >>> Ham,Exp,_=epr.Jstart()
    >>> Ham.g=[2.0003,2.5,2.5]
    >>> Ham.Q=[0.5,10.0,0]
    >>> print(Ham)
    >>> Ham=epr.Jchaframe(Ham,Exp)
    >>> print(Ham)
    JHval(S=0.5, g=[2.0003, 2.5, 2.5], I=0.0, L=0.0, A=0.0,
    Q=[0.5, 10.0, 0], D=Array([0, 0], dtype=int32), 
    Bk2=[0, 0, 0, 0, 0], Bk4=[0, 0, 0, 0, 0, 0, 0, 0, 0], 
    Bk6=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], lc=0.0,
    Hpp=Array([0, 1], dtype=int32), eta=0.5, weight=0.0)
    JHval(S=0.5, g=Array([[ 2.1690652 ,  0.16222095,  0.20332736],
    [ 0.16222097,  2.7222438 , -0.09966913],
    [ 0.20332736, -0.09966913,  2.3750749 ]], dtype=float32),
    I=0.0, L=0.0, A=Array([[0., 0., 0.],[0., 0., 0.],[0., 0., 0.]],
    dtype=float32), Q=Array([[ 0.5,  0. ,  0. ],[ 0. , 10. ,  0. ],
    [ 0. ,  0. ,  0. ]], dtype=float32), D=Array([[0., 0., 0.],
    [0., 0., 0.],[0., 0., 0.]], dtype=float32), Bk2=[0, 0, 0, 0, 0], 
    Bk4=[0, 0, 0, 0, 0, 0, 0, 0, 0], 
    Bk6=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], lc=0.0, 
    Hpp=Array([0, 1], dtype=int32), eta=0.5, weight=0.0)
    '''
    Ham=JConvtarray(Ham)
    tA=jxn.eye(3)*Ham.A
    tQ=jxn.eye(3)*Ham.Q
    tg=jxn.eye(3)*Ham.g
    D2s=jxn.asarray([-Ham.D[0]/3+Ham.D[1],-Ham.D[0]/3-Ham.D[1],2*Ham.D[0]/3])
    tD=jxn.eye(3)*D2s
    rot=JRotationmat(Exp)
    A1=(JRotmatrix(Exp.Aframe[0],Exp.Aframe[1],Exp.Aframe[2])@rot).T@tA@(JRotmatrix(Exp.Aframe[0],Exp.Aframe[1],Exp.Aframe[2])@rot)
    g1=(JRotmatrix(Exp.gframe[0],Exp.gframe[1],Exp.gframe[2])@rot).T@tg@(JRotmatrix(Exp.gframe[0],Exp.gframe[1],Exp.gframe[2])@rot)
    D2=(JRotmatrix(Exp.Dframe[0],Exp.Dframe[1],Exp.Dframe[2])@rot).T@tD@(JRotmatrix(Exp.Dframe[0],Exp.Dframe[1],Exp.Dframe[2])@rot)
    Q1=(JRotmatrix(Exp.Qframe[0],Exp.Qframe[1],Exp.Qframe[2])@rot).T@tQ@(JRotmatrix(Exp.Qframe[0],Exp.Qframe[1],Exp.Qframe[2])@rot)
    return Ham.replace(A=A1,g=g1,D=D2,Q=Q1)

def JRotationmat(Exp):
    '''
    Creates the rotation matrix to pass from the Lab frame to the molecular frame.
    
    Parameters
    ---------
    
    Exp : Class
        Container for the experimental conditions.

    Returns
    -------
    
    RRmatrix: jax.np.array
        Matrix for the rotation.

    
    Example
    -------
    >>> import epraya as epr 
    >>> _, Exp, _=epr.Jstart()
    >>> Exp.Sampleframe=[30,20,0]
    >>> Exp.Molframe=[60,0,0]
    >>> print(epr.JRotationmat(Exp))
    [[ 0.49999997 -0.86602545  0.        ]
    [ 0.          0.49999997  0.        ]
    [ 0.          0.          1.        ]]
    '''
    RRmatrix=JRotmatrix(Exp.Sampleframe[0],Exp.Sampleframe[1],Exp.Sampleframe[2])@JRotmatrix(Exp.Molframe[0],Exp.Molframe[1],Exp.Molframe[2])
    return RRmatrix.T

def JRotmatrix(alfa,beta,gamma):
    '''
    Creates the rotation matrix based in the Euler angles in degrees.


    Parameters
    ----------

    alfa : float
        Euler angle for rotations around the z axis
    beta : float

        Euler angle for rotations around the y' axis
    gamma : float
        Euler angle for rotations around the z'' axis


    Returns
    -------

    Reuler : np.array
        Rotation matrix

    Example
    -------

    >>> import epraya as epr
    >>> alfa,beta,gamma=20,30,40
    >>> print(epr.JRotmatrix(alfa,beta,gamma))
    [[ 0.40355885  0.96394587 -0.38302222]
    [-0.78510165  0.5294539   0.3213938 ]
    [ 0.4698463   0.17101006  0.8660254 ]]
    '''
    alfa,beta,gamma=jxn.radians(alfa),jxn.radians(beta),jxn.radians(gamma)
    cosg,sing=jxn.cos(gamma),jxn.sin(gamma)
    cosa,sina=jxn.cos(alfa),jxn.sin(alfa)
    cosb,sinb=jxn.cos(beta),jxn.sin(beta)
    eps=10**-9
    cosg=jxn.where(jxn.abs(cosg)<eps,0.0,cosg)
    sing=jxn.where(jxn.abs(sing)<eps,0.0,sing)
    cosa=jxn.where(jxn.abs(cosa)<eps,0.0,cosa)
    sina=jxn.where(jxn.abs(sina)<eps,0.0,sina)
    cosb=jxn.where(jxn.abs(cosb)<eps,0.0,cosb)
    sinb=jxn.where(jxn.abs(sinb)<eps,0.0,sinb)
    Reuler=jxn.array([[(cosg*cosa*cosb)-(sing*sina),(cosg*cosa*sinb)+(sing*cosa),-cosg*sinb],
    [-(sing*cosb*cosa)-(cosg*sina),-(sing*cosb*sina)+(cosg*cosa),sing*sinb],
    [sinb*cosa,sina*sinb,cosb]])
    return Reuler

def JConvtarray(Ham):
    '''
    Conditional function to assure that the hamiltonian parameters have the right dimension.

    Parameters
    ----------

    Ham : Class

        Container of the Hamiltonian parameters.

    Returns
    -------

    Ham : Class
        Container of the Hamiltonian parameters with the corrected parameters.
    '''
    def formatj(val,variable):
        iti=jxn.asarray(val,dtype=float)
        if iti.ndim==0:
            return iti*jxn.array([1.0,1.0,1.0])
        elif iti.ndim==1:
            if iti.shape[0]==3:
                return iti
            elif iti.shape[0]==2:
                return jxn.array([iti[0],iti[0],iti[1]])
            else:
                raise RuntimeError(f"Not enough/too many values in {variable}, expected 2 or 3, got {iti.shape[0]}.")

        elif iti.ndim==2:
            if iti.shape!=(3,3):
                raise RuntimeError(f"Matrix {variable} doesn't have the correct dimensions, must be (3,3).")
            return iti

        else:
            raise RuntimeError(f"Wrong data dimensions in {variable}.")
    hg=formatj(Ham.g,"Hval.g")
    hA=formatj(Ham.A,"Hval.A")
    hD=jxn.asarray(Ham.D,dtype=float)
    if hD.ndim!=1:
        raise RuntimeError("Wrong data type/dimensions in Hval.D, must be 1D list or array.")
    if hD.shape[0]==2:
        hD=hD
    elif hD.shape[0]==3:
        hD=jxn.array([3.0*hD[2]/2.0,(hD[0]-hD[1])/2.0])
    else:
        raise RuntimeError(f"Wrong number of values in Hval.D, expected 2 or 3, got {hD.shape[0]}.")

    hQ=jxn.asarray(Ham.Q,dtype=float)
    if hQ.ndim!=1 or hQ.shape[0]!=3:
        raise RuntimeError(f"Wrong values in Hval.Q, expected exactly 3, got {hQ.shape[0] if hQ.ndim==1 else 'matrix'}.")
    hQ=hQ
    return Ham.replace(g=hg,A=hA,D=hD,Q=hQ)

#Stevens Operators
#Rule: k<=2s
def JStevensO(ssx,ssy,ssz,s,Ham,dim):
    '''
    Expanded Stevens operators following the rule k<=2s and the definition by Rudowicz and Chung.


    Parameters
    ----------
    
    ssx : jax.np.array
        Pauli matrix of the spin x component.
    ssy : jax.np.array
        Pauli matrix of the spin y component.
    ssz : jax.np.array
        Pauli matrix of the spin z component.
    s : float
        Spin operator value.
    Ham : Class
        Hamiltonian parameters container.
    dim : int
        Dimension of the total hamiltonian.
        
    Returns
    -------
    
    totales : jax.np.array
        Matrix with the contribution of the relevant Stevens operators.
        
    Example
    -------
    >>> import epraya as epr
    >>> Ham, _, _ = epr.Jstart()
    >>> Ham.S=1
    >>> Ham.D=[700,200]
    >>> Ham=epr.Jchaframe(Ham,Exp)
    >>> dim=int(2*Ham.S+1)
    >>> sx,sy,sz=epr.JPauli(Ham.S)
    >>> print(epr.JStevensO(sx,sy,sz,Ham.S,Ham,dim))
    [[ 233.33333+0.j    0.     +0.j  199.99998+0.j]
    [   0.     +0.j -466.66666+0.j    0.     +0.j]
    [ 199.99998+0.j    0.     +0.j  233.33333+0.j]]
    '''
    k=int(2*s)
    B22,B21,B20,Bq21,Bq22=Ham.Bk2
    B20,B22=3*Ham.D[2,2]/2,(Ham.D[0,0]-Ham.D[1,1])*0.5
    B21=Ham.D[0,2]
    Bq21=Ham.D[1,2]
    Bq22=Ham.D[0,1]
    B44,B43,B42,B41,B40,Bq41,Bq42,Bq43,Bq44=Ham.Bk4
    B66,B65,B64,B63,B62,B61,B60,Bq61,Bq62,Bq63,Bq64,Bq65,Bq66=Ham.Bk6
    sour=(int(k-k%2))
    if sour==0:
        return jxn.kron(jxn.zeros((int(2*s+1),int(2*s+1))),jxn.eye(int(dim/(jxn.zeros((int(2*s+1),int(2*s+1)))).shape[1])))
    else:
        k=np.arange(0,sour+1,2)[1:]
        sxminus=ssx-(ssy*1.0j)
        sxsum=ssx+(ssy*1.0j)
        sxminus2=jxn.linalg.matrix_power(sxminus,2)
        sxsum2=jxn.linalg.matrix_power(sxsum,2)
        ssz2=jxn.linalg.matrix_power(ssz,2)
        xs=s*(s+1)
        eye=jxn.eye(int(2*s+1),dtype=complex)
        if k[-1]>=4:
            ssz3=jxn.linalg.matrix_power(ssz,3)
            ssz4=jxn.linalg.matrix_power(ssz,4)
            sxminus3=jxn.linalg.matrix_power(sxminus,3)
            sxsum3=jxn.linalg.matrix_power(sxsum,3)
            sxminus4=jxn.linalg.matrix_power(sxminus,4)
            sxsum4=jxn.linalg.matrix_power(sxsum,4)
        if k[-1]>=6:
            ssz6=jxn.linalg.matrix_power(ssz,6)
            ssz5=jxn.linalg.matrix_power(ssz,5)
            sxminus5=jxn.linalg.matrix_power(sxminus,5)
            sxsum5=jxn.linalg.matrix_power(sxsum,5)
            sxminus6=jxn.linalg.matrix_power(sxminus,6)
            sxsum6=jxn.linalg.matrix_power(sxsum,6)
        total=0
        for ris in range(0,len(k)):
            if (k[ris])==2:
                #q=0
                temp=B20*(3*ssz2-xs*eye)/3
                #q=1
                temp+=B21*(ssx@ssz+ssz@ssx)
                #q=-1
                temp+=Bq21*(ssy@ssz+ssz@ssy)
                #q=2
                temp+=B22*0.5*(sxminus2+sxsum2)
                #q=-2
                temp+=(1j)*(Bq22*0.5*(sxminus2-sxsum2))
                total+=temp
            if (k[ris])==4:
                #q=0
                temp=B40*((35*ssz4)-(((30*xs)-25)*ssz2)+(eye*((-6*xs)+(3*xs**2))))
                #q=1
                temp+=B41*(0.25)*((35*ssz3-(30*xs*ssz)+25*ssz)@(sxsum+sxminus)+(sxsum+sxminus)@(35*ssz3-(30*xs*ssz)+25*ssz))
                #q=-1
                temp+=(-1j)*Bq41*(0.25)*((35*ssz3-(30*xs*ssz)+25*ssz)@(sxsum-sxminus)+(sxsum-sxminus)@(35*ssz3-(30*xs*ssz)+25*ssz))
                #q=2
                temp+=B42*0.25*(((7*ssz2-xs*eye-5*eye)@(sxsum2+sxminus2))+((sxsum2+sxminus2)@(7*ssz2-xs*eye-5*eye)))
                #q=-2
                temp+=Bq42*0.25*(-1j)*(((7*ssz2-xs*eye-5*eye)@(sxsum2-sxminus2))+((sxsum2-sxminus2)@(7*ssz2-xs*eye-5*eye)))
                #q=3
                temp+=B43*0.25*((ssz@(sxminus3+sxsum3))+((sxminus3+sxsum3)@ssz))
                #q=-3
                temp+=(-1j)*Bq43*0.25*((ssz@(sxsum3-sxminus3))+((sxsum3-sxminus3)@ssz))
                #q=4
                temp+=B44*0.5*(sxminus4+sxsum4)
                #q=-4
                temp+=(-1j)*Bq44*0.5*(sxsum4-sxminus4)
                total+=temp
            if (k[ris])==6:
                #q=0
                temp=B60*((231*ssz6)-(315*xs*ssz4)+735*ssz4+(105*xs*xs*ssz2)-(525*xs*ssz2)+(294*ssz2)-(5*xs**3*eye)+(40*xs**2*eye)-60*xs*eye)
                #q=1
                temp+=B61*0.25*(((231*ssz5)-(315*xs-735)*ssz3+(105*xs**2-525*xs+294)*ssz)@(sxsum+sxminus)+(sxsum+sxminus)@((231*ssz5)-(315*xs-735)*ssz3+(105*xs**2-525*xs+294)*ssz))
                #q=-1
                temp+=(-1j)*Bq61*0.25*(((231*ssz5)-(315*xs-735)*ssz3+(105*xs**2-525*xs+294)*ssz)@(sxsum-sxminus)+(sxsum-sxminus)@((231*ssz5)-(315*xs-735)*ssz3+(105*xs**2-525*xs+294)*ssz))
                #q=2
                temp+=0.25*B62*((33*ssz4-18*xs*ssz2-123*ssz2+(xs**2*eye)+10*xs*eye+102*eye)@(sxminus2+sxsum2)+(sxminus2+sxsum2)@(33*ssz4-18*xs*ssz2-123*ssz2+(xs**2*eye)+10*xs*eye+102*eye))
                #q=-2
                temp+=(-1j)*0.25*Bq62*((33*ssz4-18*xs*ssz2-123*ssz2+(xs**2*eye)+10*xs*eye+102*eye)@(sxminus2-sxsum2)+(sxminus2-sxsum2)@(33*ssz4-18*xs*ssz2-123*ssz2+(xs**2*eye)+10*xs*eye+102*eye))
                #q=3
                temp+=B63*0.25*(((11*ssz3-(3*xs*ssz)-(59*ssz))@(sxminus3+sxsum3))+((sxminus3+sxsum3)@(11*ssz3-(3*xs*ssz)-(59*ssz))))
                #q=-3
                temp+=(-1j)*Bq63*0.25*(((11*ssz3-(3*xs*ssz)-(59*ssz))@(sxsum3-sxminus3))+((sxsum3-sxminus3)@(11*ssz3-(3*xs*ssz)-(59*ssz))))
                #q=4
                temp+=B64*0.25*(((11*ssz2-xs*eye-38*eye)@(sxminus4+sxsum4))+((sxminus4+sxsum4)@(11*ssz2-xs*eye-38*eye)))
                #q=-4
                temp+=(-1j)*Bq64*0.25*(((11*ssz2-xs*eye-38*eye)@(sxsum4-sxminus4))+((sxsum4-sxminus4)@(11*ssz2-xs*eye-38*eye)))
                #q=5
                temp+=B65*0.25*(ssz@(sxsum5+sxminus5)+(sxsum5+sxminus5)@ssz)
                #q=-5
                temp+=(-1j)*Bq65*0.25*(ssz@(sxsum5-sxminus5)+(sxsum5-sxminus5)@ssz)
                #q=6
                temp+=B66*0.5*(sxminus6+sxsum6)
                #q=-6
                temp+=(-1j)*Bq66*0.5*(sxsum6-sxminus6)
                total+=temp
        totales=jxn.kron(total,jxn.eye(int(dim/(total).shape[1])))
        return totales


def JMsmi(I,S,L=0):
    '''
    Determinates the quantum numbers for the spin, nuclear spin and angular momentum.

    Parameters
    ----------
    I : float
        Nuclear spin operator value.
    S : float
        Spin operator value.
    L : float
        Angular momentum operator value.

    Returns
    -------

    sl : jax.np.array
        Quantum numbers of the spin operator.
    nl : jax.np.array
        Quantum numbers of the nuclear spin operator.
    ll : jax.np.array
        Quantum numbers of the angular momentum operator

    Example
    -------

    >>> import epraya as epr
    >>> Ham, _, _ = epr.Jstart()
    >>> Ham.S=1
    >>> Ham.I=1
    >>> Ham.L=0
    >>> print(epr.JMsmi(Ham.I,Ham.S,Ham.L))
    (Array([ 1.,  1.,  1.,  0.,  0.,  0., -1., -1., -1.], dtype=float32),
    Array([ 1.,  0., -1.,  1.,  0., -1.,  1.,  0., -1.], dtype=float32), 
    Array([0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32))
    '''
    if I<0 or S<0 or L<0:
        raise ValueError('Spin values cannot be negative')
    dim=int(2*S+1)*int(2*I+1)*int(2*L+1)
    poss=jxn.linspace(S,-S,int(2*S+1))
    posi=jxn.linspace(I,-I,int(2*I+1))
    posl=jxn.linspace(L,-L,int(2*L+1))
    ll,sl,il=jxn.meshgrid(posl,poss,posi,indexing='ij')
    return sl.flatten(),il.flatten(),ll.flatten()



#For powder
@jx.jit
def JLorentzp(field,Int,rfield,Hpp):
    '''
    Defines the lorentzian profile of the spectrum, using the resonant fields and calculated intensities.
    
    Parameters
    ----------
    
    field : jax.np.array
        Array of the field values of the spectrum.

    Int : jax.np.array
        Array of the intensity calculated in the *field* values 
    rfield : jax.np.array
        List of the resonant fields of the system.

    Hpp : float
        Peak to peak distance in mT.

    Returns
    -------
    
    espec : jax.np.array

        EPR spectrum
    
    Example
    -------
    .. code-block:: python
    
       import matplotlib.pyplot as plt
       import epraya as epr
       import numpy as np
       Ham,Exp,_=epr.Jstart()
       Ham.Hpp=[0,2]
       Exp.Frange=[200,500]
       Exp.Points=2000
       fieldr=np.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
       mu=350
       sigma=50
       rsonant=np.array([300,350,320])
       inten=np.random.normal(mu,sigma,len(rsonant))
       spc=epr.JLorentzp(fieldr,inten,rsonant,Ham.Hpp[1])

       plt.plot(fieldr,spc)
       plt.grid()
       plt.show()    
       
    .. image:: /_static/jlorentz.png
       :alt: Plot of the JLorentzp function
       :align: center
    '''
    espec=jxn.zeros(len(field),dtype=float)
    gamma=jxn.sqrt(3.0)*Hpp
    gamma2=gamma/2.0
    dif=field[:,None]-rfield[None,:]
    ert=(-dif*gamma/jxn.pi)/((dif**2+gamma2**2)**2)
    espec=jxn.sum(Int*ert,axis=1)
    return espec
@jx.jit
def JGaussp(field,Int,rfield,Hpp):
    '''
    Defines the gaussian profile of the spectrum, using the resonant fields and calculated intensities.
    
    Parameters
    ----------
    
    field : jax.np.array
        Array of the field values of the spectrum.
    Int : jax.np.array
        Array of the intensity calculated in the *field* values 
    rfield : jax.np.array
        List of the resonant fields of the system.
    Hpp : float
        Peak to peak distance in mT.
    
    Returns
    -------
    
    espec : jax.np.array
        EPR spectrum
    
    Example
    -------
    .. code-block:: python
       
       import matplotlib.pyplot as plt
       import epraya as epr
       import numpy as np
       Ham,Exp,_=epr.Jstart()
       Ham.Hpp=[2,0]
       Exp.Frange=[200,500]
       Exp.Points=2000
       fieldr=np.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
       mu=350
       sigma=50
       rsonant=np.array([300,350,320])
       inten=np.random.normal(mu,sigma,len(rsonant))
       spc=epr.JGaussp(fieldr,inten,rsonant,Ham.Hpp[0])

       plt.plot(fieldr,spc)
       plt.grid()
       plt.show()  
       
    .. image:: /_static/Jgauss.png
       :alt: Plot of the JGaussp function
       :align: center
    '''

    espec=jxn.zeros(len(field),dtype=float)
    gamma=jxn.sqrt(jxn.log(2.0)/2.0)*Hpp
    ymax=jxn.sqrt(jxn.log(2.0)/jxn.pi)*(1/gamma)
    dif=field[:,None]-rfield[None,:]
    ert=((-ymax*2.0*jxn.log(2.0)*dif)/gamma**2)*jxn.exp(-jxn.log(2.0)*dif**2/gamma**2)
    espec=jxn.sum(Int*ert,axis=1)
    return espec

@jx.jit
def JVoigtp(field,Int,rfield,Hpp,eta):
    '''
    Determinates the voigtian profile of the spectrum as a lineal combination of gaussian and lorentzian profiles.
    
    Parameters
    ----------

    field : jax.np.array
        Array of the field values of the spectrum.

    Int : jax.np.array
        Array of the intensity calculated in the *field* values.
    rfield : jax.np.array
        List of the resonant fields of the system.
    Hpp : list
        Peak to peak distance in mT, for gaussian and lorentzian profiles.
    eta : float
        Percentage of the voigitan profile that corresponds to a gaussian profile, from 0 to 1.
    
    Returns
    -------

    espec : jax.np.array
        EPR spectrum
    
    Example
    -------
    .. code-block:: python
    
       import matplotlib.pyplot as plt
       import epraya as epr
       import numpy as np
       Ham,Exp,_=epr.Jstart()
       Ham.Hpp=np.array([10,20])
       Ham.eta=0.5
       Exp.Frange=[200,500]
       Exp.Points=2000

       fieldr=np.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
       mu,sigma=350,50
       rsonant=np.array([300,350,320])
       inten=np.random.normal(mu,sigma,len(rsonant))

       spc=epr.JVoigtp(fieldr,inten,rsonant,Ham.Hpp,Ham.eta)
       plt.plot(fieldr,spc)
       plt.grid()
       plt.show()
       
    .. image:: /_static/jvoigt.png
       :alt: Plot of the Voigtp function
       :align: center
    '''
    hppg=jxn.where(Hpp[0]==0.0,1e-10,Hpp[0])
    hppl=jxn.where(Hpp[1]==0.0,1e-10,Hpp[1])
    gas=JGaussp(field,Int,rfield,hppg)
    lor=JLorentzp(field,Int,rfield,hppl)
    espec=(lor*eta)+(gas*(1.0-eta))
    return espec

#Find energy values in function of field
@jx.jit
def JPadaptarray(espac,h1,hx,hy,hz,nx,ny,nz):
    '''
    Constructs the Zeeman hamiltonian, adds it to the complete one and finds the energy values and eigenvectors.
    
    Parameters
    ----------
    
    espac : np.array
        Array with the values of the magnetic field.
    h1 : np.array
        Hamiltonian matrix that contains all no Zeeman interactions.
    hx : np.array
        Hamiltonian matrix of the Zeeman interaction in the x direction.
    hy : np.array
        Hamiltonian matrix of the Zeeman interaction in the y direction.
    hz : np.array
        Hamiltonian matrix of the Zeeman interaction in the z direction.
    nx : float
        Weight coefficient for the interaction in the x direction.
    ny : float
        Weight coefficient for the interaction in the y direction.
    nz : float
        Weight coefficient for the interaction in the z direction.
        
    Returns
    -------

    Elist : np.array
        Array of the energy values of the hamiltonian.
    Vlist : np.array
        Array of the eigenvectors of the hamiltonian.
    h2 : np.array
        Total Zeeman interaction matrix.
    '''
    h2=nx*hx+ny*hy+nz*hz
    h3=h1[None,:,:]+h2[None,:,:]*espac[:,None,None]
    Elist,Vlist=jxn.linalg.eigh(h3)
    return Elist,Vlist,h2
    
# Makes the approximation by the assigment problem solution
def Hungarian(cost):
    '''
    Solves the assigment problem with the J-V method implemented in scipy, to organize the eigenvectors and energies of the hamiltonian and relate them with the quantum numbers, depending of the change in the cost.
    
    Parameteres
    -----------
    
    cost : list
        Cost matrix of the eigenvectors and energies configuration
        
    Returns 
    -------
    
    novo : array
        Sorted eigenvectors and energies of the hamiltonian
    '''
    rowidx,colidx=sci.optimize.linear_sum_assignment(np.array(cost))
    novo=colidx[np.argsort(rowidx)]
    return novo.astype(np.int32)

def Jungarian(cost):
    '''
    Wrap function to call the Scipy J-V method outside of JAX using ShapeDtypeStruct and pure_callback, in order to not interfere in the ADAM implementation.
    '''
    shake=jx.ShapeDtypeStruct((cost.shape[0],),jxn.int32)
    return jx.pure_callback(Hungarian,shake,cost,vmap_method='sequential')

@jx.jit
def JPretrack(Enegria, Vector):
    '''
    Organize the eigenvectors and energies to relate them with the quantum numbers of the system, taking as references the values at high field. It use the laxscan function from JAX to calculate the best configuration.
    
    Parameters
    ----------
    
    Enegria : jax.np.array
        Array of the energy values of the hamiltonian.
    Vector : jax.np.array
        Array of the eigenvectors of the hamiltonian.
    

    Returns
    -------
    tE : jax.numpy.vstack
        Concatenate array of the energy values of the Hamiltonian.
        
    tV : jax.numpy.vstack
        Concatenate array of the eigenvectors values of the Hamiltonian.
    '''
    def tmback(cara,vals):
        oldE,oldV=cara
        actE,actV=vals
        supermatrix=jxn.abs(jxn.dot(oldV.conj().T,actV))
        cost=1.0-supermatrix
        Edif=jxn.abs(oldE[:,None]-actE[None,:])
        maxEdif=jxn.max(Edif)
        maxEdif=jxn.where(maxEdif==0.0,1e-10,maxEdif)
        coste=jxn.where(maxEdif>0,Edif/maxEdif,0.0)
        tot=cost+0.1*coste
        idx=Jungarian(jx.lax.stop_gradient(tot))
        sorE=actE[idx]
        sorV=actV[:,idx]
        return (sorE,sorV),(sorE,sorV)

    eincar=(Enegria[-1],Vector[-1])
    restof=(Enegria[:-1],Vector[:-1])
    _,(tre,trv)=jx.lax.scan(tmback,eincar,restof,reverse=True)
    tE=jxn.vstack([tre,Enegria[-1:]])
    tV=jxn.vstack([trv,Vector[-1:]])
    return tE,tV

@jx.jit
def JBoltfactor(Eghz,di,dj,Temp):
    '''

    Calculates the Boltzmann distribution for the intensity, using states *di* and *dj* and their related energies.
    
    Parameters
    ----------

    Eghz : float
        Approximated energy value of hamiltonian.
    di : float
        State di of the system.
    dj : float
        State dj of the system.
    Temp : float
        Temperature of the system.
    
    Returns
    -------
    
    popui-popuj : jax.np.array
        Temperature dependent Boltzmann distribution.

    '''
    h=scc.h
    kb=scc.k
    conver=1e9*h 
    Ej=Eghz*conver
    Temp=jxn.where(Temp<=0.0,1.0,Temp)
    beta=1.0/(kb*Temp)
    Emin=jxn.min(Ej)
    boltz=jxn.exp(-beta*(Ej-Emin))
    Z=jxn.sum(boltz)
    popui=boltz[di]/Z
    popuj=boltz[dj]/Z
    return (popui-popuj)

@partial(jx.jit,static_argnames=['dim'])
def JNresina(Blist,Elist,Vlist,dim,Freq,isx,isy,isz,nx,ny,nz,Tem,Hpp,h2):
    '''
    Determinates the resonant fields and intensities of the spectrum using the expression for the first order perturbation limit. The intensity is calculated as the product of the transition rate (probability of transition), the Boltzmann factor (Boltzmann distribution) and a frecuency to field conversion factor. 
    
    Initially, the function calculates the intensities for all the block of orientations, then finds the resonant fields searching for the values that comply with the resonant equation and have a crossing. Finally it relates the resonant fields and intensity values depending in the probability of transitions, producing the arrays of intensity, fields and the total number of transitions.
    
    Parameters
    ----------
    
    Blist : jax.np.array
        Magnetic field values that are evaluated to find the resonant fields.
    Elist : jax.np.array
        List of the energies from the hamiltonian.
    Vlist : jax.np.array
        List of the eigenvectors of the hamiltonian.
    dim : float
        Dimension of the total hamiltonian matrix.
    Freq : float
        Frequency of operation of the microwave radiation in GHz.
    isx : jax.np.array
        Pauli matrix of the total spin (electronic, nuclear spin and angular momentum) contribution in the x direction.
    isy : jax.np.array
        Pauli matrix of the total spin (electronic, nuclear spin and angular momentum) contribution in the y direction.
    isz : jax.np.array
        Pauli matrix of the total spin (electronic, nuclear spin and angular momentum) contribution in the z direction.
    nx : jax.np.array 
        Vectors (points in the grid) to consider in the calculation of the spectrum in the x direction.
    ny : jax.np.array 
        Vectors (points in the grid) to consider in the calculation of the spectrum in the y direction.
    nz : jax.np.array 
        Vectors (points in the grid) to consider in the calculation of the spectrum in the z direction.
    Tem : float
        Temperature of the system.
    h2 : jax.np.array
        Zeeman interaction matrix of the system.
        
    Returns
    -------
    ffres : jax.np.array 
        List of the resonant fields of the system.
    ffint : jax.np.array 
        List of the intensity of the spectrum, evaluated in the resonant fields.
    ntrans : jax.np.array
        Zero dimensional array with the number of possible transitions related to the resonant fields.
    '''
    iidx,jidx=jxn.triu_indices(dim,k=1)
    Vdag=Vlist.conj().swapaxes(-1,-2)
    Tx=Vdag@isx@Vlist
    Ty=Vdag@isy@Vlist
    Tz=Vdag@isz@Vlist
    #Interpolate for intensities
    Txij=Tx[:,iidx,jidx]
    Tyij=Ty[:,iidx,jidx]
    Tzij=Tz[:,iidx,jidx]
    M2=jxn.real(Txij*jxn.conj(Txij))+jxn.real(Tyij*jxn.conj(Tyij))+jxn.real(Tzij*jxn.conj(Tzij))
    Mn=nx*Txij+ny*Tyij+nz*Tzij
    prob=M2-jxn.abs(Mn)**2
    #Frecuency to field
    h22=jxn.real(Vdag@h2@Vlist)
    h2diag=jxn.diagonal(h22,axis1=1,axis2=2)
    dert=h2diag[:,iidx]
    izrt=h2diag[:,jidx]
    gma=jxn.abs(izrt-dert)
    gma=jxn.where(gma<1e-4,1e-4,gma)
    gema=1.0/gma   
    #Boltzmann distribution
    conver=1e9*scc.h
    Ej=Elist*conver
    Temp=jxn.where(Tem<=0.0,1.0,Tem)
    beta=1.0/(scc.k*Temp)
    Emin=jxn.min(Ej,axis=-1,keepdims=True)
    boltz=jxn.exp(-beta*(Ej-Emin))
    Z=jxn.sum(boltz,axis=-1,keepdims=True)
    boltz=boltz/Z
    popui=boltz[:,iidx]
    popuj=boltz[:,jidx]
    boltzm=popui-popuj
    intensy=prob*gema*boltzm
    #FInd resonant fields
    diffv=jxn.abs(Elist[:,jidx]-Elist[:, iidx])-Freq
    dEl=diffv[:-1,:]
    dEr=diffv[1:,:]
    #Search for crossings
    cross=(dEl*dEr<=0.0)&(dEl!=dEr)
    denom=dEr-dEl
    denom=jxn.where(denom== 0.0,1e-10,denom)
    t=-dEl/denom
    t=jxn.where(cross,t,0.0)
    Bl=Blist[:-1,None]
    Br=Blist[1:,None]
    res=Bl+t*(Br-Bl)
    intle=intensy[:-1,:]
    intri=intensy[1:,:]
    eintensy=intle+t*(intri-intle)
    fres=jxn.where(cross,res,0.0).flatten()
    fint=jxn.where(cross,eintensy,0.0).flatten()
    cross=cross.flatten()
    ntrans=jxn.sum(cross).astype(jxn.float64)
    Ktra=200
    #Scores for transition possibility
    scores=jxn.where(cross,1.0+fint,-1.0)
    topones,toponesind=jx.lax.top_k(scores,Ktra)
    ffres=fres[toponesind]
    ffint=fint[toponesind]
    maski=topones>0.0
    ffres=jxn.where(maski,ffres,0.0)
    ffint=jxn.where(maski,ffint,0.0)
    return ffres,ffint,ntrans

def Meshtriangle():
    '''
    Creates the baricentral mesh of triangles for the JCaltriangle, that calculates the contribution of the resonant fields and their intensities in the final spectrum. It is defined as an independent function because it can interfere with the ADAM algorithm monitoring of the variables.
    
    Returns
    -------
    
    w1 : jax.np.array
        Vector of the points in the x direction of the grid.
    w2 : jax.np.array
        Vector of the points in the y direction of the grid.
    w3 : jax.np.array
        Vector of the points in the z direction of the grid.
    '''
    numd=20
    tpoints=(numd*(numd+1))/2.0
    w1,w2,w3=[],[],[]
    for i in range(numd):
        for j in range(numd-i):
            k=numd-1-i-j
            w1.append(i/(numd-1))
            w2.append(j/(numd-1))
            w3.append(k/(numd-1))
    w1=jxn.array(w1)[:,None]
    w2=jxn.array(w2)[:,None]
    w3=jxn.array(w3)[:,None]
    return w1,w2,w3,tpoints


@partial(jx.jit, static_argnames=['points'])
#@jx.checkpoint
def JCaltriangle(Bmin,dB,allres,allint,transi,hulk,weight,points):
    '''
    Creates a sketch spectrum using a barycentral mesh of triangles to calculate the contribution of the resonant fields and their intensities in the final spectrum. The result of this process is later convolute to generate the corrected EPR spectrum. Using the JAX framework, calculates multiple points at the same time with the function lax_scan and the Takeonetriangle wrap function.
    
    Parameters
    ----------
    
    Bmin : float
        Minimun value of the magnetic field.
    dB : float
        Delta difference between the magnetic field values.
    allres: jax.np.array
        List of the resonant fields of the system.
    allint : jax.np.array
        List of the intensity of the spectrum, evaluated in the resonant fields.
    transi : jax.np.array
        List of the possible transitions between the energy levels, related to the resonant fields.
    hulk : jax.np.arraynp.array
        Smaller convex poligon tha contains all the points require for the simulation.
    weight : jax.np.arraynp.array
        Normalized value of the weight of the points to evaluate it's contribution to the spectrum.
        
    Returns
    -------
    
    estotal : jax.np.array
        Array that contains the resulting spectrum sketch.
    
    '''
    def Takeonetriangle(trindex):
        i1,i2,i3=trindex[0],trindex[1],trindex[2]
        B1,B2,B3=allres[i1],allres[i2],allres[i3]
        I1,I2,I3=allint[i1],allint[i2],allint[i3]
        weig=(weight[i1]+weight[i2]+weight[i3])/3.0
        pointweg=weig/tpoints
        Bin=ww1*B1+ww2*B2+ww3*B3
        Iint=(ww1*I1+ww2*I2+ww3*I3)*pointweg
        diferb=(Bin-Bmin)/dB
        fdiferb=diferb.flatten()
        fIint=Iint.flatten()
        # Find the neighbours
        vecl=jxn.floor(fdiferb).astype(jxn.int32)
        vecr=vecl+1
        #Divides the intensity between the neighbours, by percentages 
        fracr=fdiferb-vecl
        fracl=1.0-fracr
        Ilef=fIint*fracl
        Irig=fIint*fracr
        # Is in range?
        rleft=(vecl>=0)&(vecl<points)
        rright=(vecr>=0)&(vecr<points)
        vecl=jxn.where(rleft,vecl,0)
        vecr=jxn.where(rright,vecr,0)
        Ilef=jxn.where(rleft,Ilef,0.0)
        Irig=jxn.where(rright,Irig,0.0)
        sketch1=jxn.zeros(points)
        sketch1=sketch1.at[vecl].add(Ilef)
        sketch1=sketch1.at[vecr].add(Irig)
        n1,n2,n3=transi[i1],transi[i2],transi[i3]
        taketrian=(n1==n2)&(n2==n3)&(n1>0)
        return jxn.where(taketrian,sketch1,jxn.zeros_like(sketch1))
    csize=500
    pad=(csize-(hulk.shape[0]%csize))%csize
    hulkpad=jxn.pad(hulk,((0,pad),(0,0)),constant_values=0)
    batch=hulkpad.shape[0]//csize
    hulkbatch=hulkpad.reshape((batch,csize,3))
    def EWsize(carry,batch):
        bspc=jx.vmap(Takeonetriangle)(batch)
        return carry+jxn.sum(bspc,axis=0),None
    estotal,_=jx.lax.scan(EWsize,jxn.zeros(points),hulkbatch)
    return estotal
    
ww1,ww2,ww3,tpoints=Meshtriangle()

def JPowder(Hamer,Expe,Nucl='None',graph=True):
    '''
    Wrap function for the simulation of the EPR spectrum for powder samples. It also use the SOPHE modified method to create the triangular grid and its weights.
    
    Parameters
    ----------
    
    Hamer : Class
        Container for the hamiltonian parameters of the system.
    
    Expe : Class
        Container for the experimental conditions.
        
    Nucl : str
        Isotope of the sample. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 

    graph : Bool
        Plots the resulting spectrum.

    Returns
    -------
    
    Blist : jax.np.array
        Array of the magnetic field.
    epc : jax.np.array
        Array of the counts of the spectrum.
    
    Example
    -------
    
    .. code-block:: python
    
       import matplotlib.pyplot as plt
       import epraya as epr
       import numpy as np
       Ham,Exp,_=epr.Jstart()
       Ham.S=3/2
       Ham.I=1
       Ham.g=np.array([2.003, 2, 2])
       Ham.A=np.array([200, 200, 200])  #Hyperfine constant
       Ham.D=np.array([800,200])      #Zero field D and E
       Ham.Hpp=[0,10]
       Exp.Freq=9.4
       Exp.Points=4096
       Exp.Temperature=300
       Exp.Frange=[0,800]
       B,spc=epr.JPowder(Ham,Exp)
       
    .. image:: /_static/jaxp.PNG
       :alt: Plot of the spectrum of the Jpowder function
       :align: center
    
    '''
    iwas,jwas,kwas,weight,hulk=Delaunay(Expe)
    iwas,jwas,kwas,weight,hulk=jxn.array(iwas),jxn.array(jwas),jxn.array(kwas),jxn.array(weight),jxn.array(hulk)
    Blist,epc=JCalpowder(Hamer,Expe,iwas,jwas,kwas,weight,hulk,Nucl)
    if graph:
        plt.figure(figsize=(10,6))
        plt.plot(Blist,epc,color='navy',label='Spectrum')
        plt.xlabel('Magnetic Field [mT]')
        formatter=EngFormatter(sep='') 
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.ylabel('Counts [A. U.]')
        plt.xlim(Expe.Frange[0],Expe.Frange[1])
        plt.grid()
        plt.legend()
        plt.show(block=False)
    return Blist,epc

def JCalpowder(Hamer,Expe,iwas,jwas,kwas,weight,hulk,Nucl='None'):
    '''
    Function for the simulation of the EPR spectrum for powder samples using the JAX functions. Instead of making the calculations with fors cycles, uses the vmap function for the calculations, dividing the points of calculations in blocks, that pass using the Oneori (One orientation) and Processvmap functions.
    
    Parameters
    ----------
    
    Hamer : Class
        Container for the hamiltonian parameters of the system.
    
    Expe : Class
        Container for the experimental conditions.
        
    iwas : jax.np.array
        Vectors (points in the grid) to consider in the calculation of the spectrum in the x direction.
    
    jwas : jax.np.array
        Vectors (points in the grid) to consider in the calculation of the spectrum in the y direction.
        
    kwas : jax.np.array
        Vectors (points in the grid) to consider in the calculation of the spectrum in the z direction.

    weight : jax.np.array
        Normalized value of the weight of the points to evaluate it's contribution to the spectrum.
    
    hulk : jax.np.array
        Smaller convex poligon tha contains all the points require for the simulation.
    Nucl : str
        Isotope of the sample. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    Returns

    -------
    Blist2 : jax.np.array
        Array of the magnetic field.


    espectotal : jax.np.array
        Array of the counts of the spectrum.
    
    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> import epraya as epr
    >>> import numpy as np
    >>> import jax.numpy as jxn
    >>> Ham,Exp,_=epr.Jstart()
    >>> iwas,jwas,kwas,weight,hulk=epr.Delaunay(Exp)
    >>> iwas,jwas,kwas,weight,hulk=jxn.array(iwas),jxn.array(jwas),jxn.array(kwas),jxn.array(weight),jxn.array(hulk)
    >>> Ham.S=3/2
    >>> Ham.I=1
    >>> Ham.g=np.array([2.003, 2, 2])
    >>> Ham.A=np.array([200, 200, 200])  #Hyperfine constant
    >>> Ham.D=np.array([800,200])      #Zero field D and E
    >>> Ham.Hpp=[0,10]
    >>> Exp.Freq=9.4
    >>> Exp.Points=4096
    >>> Exp.Temperature=300
    >>> Exp.Frange=[0,800]
    >>> print(epr.JCalpowder(Ham,Exp,iwas,jwas,kwas,weight,hulk))
    (Array([0.00000000e+00, 1.95360195e-01, 3.90720391e-01, ...,
       7.99609280e+02, 7.99804640e+02, 8.00000000e+02], dtype=float64),
    Array([ 1.48600800e-22,  1.06329540e-21,  6.49499460e-22, ...,
        2.06205909e-21,  5.04622598e-22, -1.50266650e-21], dtype=float64))   
    '''
    frange0=jxn.where(Expe.Frange[0]<0.0,1e-4,Expe.Frange[0])
    Ham=Hamer.replace(A=jxn.asarray(Hamer.A)/1000.0,D=jxn.asarray(Hamer.D)/1000.0,Hpp=jxn.asarray(Hamer.Hpp)/1.0,Q=jxn.asarray(Hamer.Q)/1000.0,
                     Bk2=jxn.asarray(Hamer.Bk2)/1000.0,Bk4=jxn.asarray(Hamer.Bk4)/1000.0,Bk6=jxn.asarray(Hamer.Bk6)/1000.0)
    etas=Ham.eta
    etas=jxn.where(Ham.Hpp[1]==0.0,0.0,etas)
    etas=jxn.where(Ham.Hpp[0]==0.0,1.0,etas)
    Ham=Ham.replace(eta=etas)
    dim=int(2*Ham.S+1)*int(2*Ham.I+1)*int(2*Ham.L+1)
    Exp=Expe
    Ham=Jchaframe(Ham,Exp)
    sx,sy,sz=JPauli(Ham.S)
    ix,iy,iz=JPauli(Ham.I)
    isx=jxn.kron(sx,jxn.eye(int(2*Ham.I+1)))
    isx=jxn.kron(jxn.eye(int(2*Ham.L+1)),isx)
    isx=jxn.asarray(isx,dtype=jxn.complex64)
    isy=jxn.kron(sy,jxn.eye(int(2*Ham.I+1)))
    isy=jxn.kron(jxn.eye(int(2*Ham.L+1)),isy)
    isy=jxn.asarray(isy,dtype=jxn.complex64)
    isz=jxn.kron(sz,jxn.eye(int(2*Ham.I+1)))
    isz=jxn.kron(jxn.eye(int(2*Ham.L+1)),isz)
    isz=jxn.asarray(isz,dtype=jxn.complex64)
    E=Exp.Freq
    beta=(scic.physical_constants["Bohr magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
    betan=(scic.physical_constants["nuclear magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
    hzex=jxn.asarray(beta*JHze(sx,sy,sz,Ham.g,[1,0,0],dim),dtype=complex)
    hzey=jxn.asarray(beta*JHze(sx,sy,sz,Ham.g,[0,1,0],dim),dtype=complex)
    hzez=jxn.asarray(beta*JHze(sx,sy,sz,Ham.g,[0,0,1],dim),dtype=complex)
    h1=jxn.zeros((dim,dim),dtype='complex64')
    if Ham.S>=1:
        h1=h1+JStevensO(sx,sy,sz,Ham.S,Ham,dim)
    if Ham.L!=0:
        h1=h1+JLorbit(sx,sy,sz,Ham.lc,dim,Ham.L)
    if Ham.I!=0:
        h1=h1+JHfi(sx,sy,sz,ix,iy,iz,Ham.A,dim)
        h1=h1+JQii(ix,iy,iz,Ham.Q,dim)
        gnk=gnfactor(Nucl)
        nhzex=jxn.asarray(betan*JNhze(Ham.I,ix,iy,iz,dim,gnk,[1,0,0]),dtype=complex)
        nhzey=jxn.asarray(betan*JNhze(Ham.I,ix,iy,iz,dim,gnk,[0,1,0]),dtype=complex)
        nhzez=jxn.asarray(betan*JNhze(Ham.I,ix,iy,iz,dim,gnk,[0,0,1]),dtype=complex)
        hzex-=nhzex
        hzey-=nhzey
        hzez-=nhzez
    h1=jxn.asarray(h1,dtype=complex)
    Blist1=jxn.linspace(Exp.Frange[0],Exp.Frange[1],1000)
    dB=(Exp.Frange[1]-Exp.Frange[0])/(Exp.Points-1)
    Bmin=Exp.Frange[0]
    
    @jx.jit
    def Oneori(nx,ny,nz):
        '''
        Wrap function to use JPadaptarray and JNresina with the JAX.vmap implementation. Vmap vectorizes this two functions, making them work with block of data instead of individual process in a loop. The blocks are divided in parts of csize size, and process with the function Processvmap, that calculates the total values of resonant fields, intenitys and transitions.
        
        Parameters
        ----------
        
        nx : jax.np.array
            Vectors (points in the grid) to consider in the calculation of the spectrum in the x direction.
        ny : jax.np.array
            Vectors (points in the grid) to consider in the calculation of the spectrum in the y direction.
        nz : jax.np.array
            Vectors (points in the grid) to consider in the calculation of the spectrum in the z direction.
        
        Returns
        -------
        resfield : jax.np.array 
            List of the resonant fields of the system.
        intensy : jax.np.array 
            List of the intensity of the spectrum, evaluated in the resonant fields.
        ntrans : jax.np.array
            Zero dimensional array with the number of possible transitions related to the resonant fields.
        
    
        '''
        Elist,Vlist,h2=JPadaptarray(Blist1,h1,hzex,hzey,hzez,nx,ny,nz)
        resfield,intensy,ntrans=JNresina(Blist1,Elist,Vlist,dim,Exp.Freq,isx,isy,isz,nx,ny,nz,Exp.Temperature,Ham.Hpp,h2)
        return resfield,intensy,ntrans
    voneori=jx.vmap(Oneori,in_axes=(0,0,0))
    csize=50 #Divides the orientations blocks so the RAM doesn't explote
    tlen=len(weight)
    plen=(csize-(tlen%csize))%csize
    pdw=jxn.pad(weight,(0,plen))
    pdi=jxn.pad(iwas,(0, plen))
    pdj=jxn.pad(jwas,(0,plen))
    pdk=jxn.pad(kwas,(0,plen))
    nparts=len(pdi)//csize
    bati=pdi.reshape(nparts,csize)
    batj=pdj.reshape(nparts,csize)
    batk=pdk.reshape(nparts,csize)
    @jx.checkpoint
    def Processvmap(curspect,bat):
        bnx,bny,bnz=bat
        batres,batint,batntras=voneori(bnx,bny,bnz)
        return curspect,(batres,batint,batntras)
    _,(allres,allint,ntrans)=jx.lax.scan(Processvmap,None,(bati,batj,batk))
    allres=allres.reshape(-1,allres.shape[-1])[:tlen]
    allint=allint.reshape(-1,allint.shape[-1])[:tlen]
    ntrans=ntrans.reshape(-1)[:tlen]
    sketch=JCaltriangle(Bmin,dB,allres,allint,ntrans,hulk,weight,Exp.Points)
    maxlenght=jxn.max(jxn.array(Ham.Hpp))*10
    kpoints=1000
    kaxis=jxn.arange(-kpoints//2+1,kpoints//2+1)*dB
    kvoigt=JVoigtp(kaxis,jxn.array([1.0]),jxn.array([0.0]),Ham.Hpp,etas)
    espectotal=jsig.fftconvolve(sketch,kvoigt,mode='same')*dB
    Blist2=jxn.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
    return Blist2,espectotal
    
def Jresonant(Hamer,Expe,graph=True,table=True,Nucl='None'):
    '''
    Wrap function for the simulation of the EPR spectrum for monocrystal samples. Also creates the table of transitions and energy diagrams of the system.
    
    Parameters
    ----------
    
    Hamer : Class
        Container for the hamiltonian parameters of the system.
    
    Expe : Class
        Container for the experimental conditions.
    graph : Bool
        Plots the resulting spectrum.
    table : Bool
        Creates the table of transitions.
    Nucl : str
        Isotope of the sample. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 


        
    Returns
    -------
    
    Blist : jax.np.array

        Array of the magnetic field.
    epc : jax.np.array
        Array of the counts of the spectrum.
    

    Example
    -------
    
    .. code-block:: python
    
       import matplotlib.pyplot as plt
       import epraya as epr
       import numpy as np
       Ham,Exp,_=epr.Jstart()
       Ham.S=3/2
       Ham.I=1
       Ham.g=np.array([2.003, 2, 2])
       Ham.A=np.array([200, 200, 200])  #Hyperfine constant
       Ham.D=np.array([800,200])      #Zero field D and E
       Ham.Hpp=[0,10]
       Exp.Freq=9.4
       Exp.Points=4096
       Exp.Temperature=300
       Exp.Frange=[0,800]
       B,spc=epr.Jresonant(Ham,Exp)
       
     .. image:: /_static/tabaj.PNG
       :alt: Table of transitions of the Jresonant function
       :align: center
       
    .. image:: /_static/jreso.PNG
       :alt: Plot of the spectrum of the Jresonant function
       :align: center
       
    .. image:: /_static/diaj.PNG
       :alt: Energy diagram of the Jresonant function
       :align: center    
    '''
    slit,nlit,llit,transitions=Msmi(Hamer.I,Hamer.S,Hamer.L)
    Blist,epc,Elist,Vlist=Calresonant(Hamer,Expe,Nucl,diagram=True)
    Blist=np.array(Blist)
    Elist=np.array(Elist)
    Vlist=np.array(Vlist)
    #For the energy diagrams
    Elist,Vlist=JPretrack(Elist,Vlist)
    splines=cubichers(Blist,Elist,axis=0)
    targettr=set()
    targettr.update(tuple(sorted(p)) for p in transitions["allowed"])
    targettr.update(tuple(sorted(p)) for p in transitions["for Dms2"])
    maxvector=Vlist[-1]
    curvebasis=Assingstatestobasis(maxvector)
    dim=Elist.shape[1]
    resfield=[]
    resonants=[]
    for i in range(dim):
        for j in range(i+1, dim):
            basis1=curvebasis[i]
            basis2=curvebasis[j]
            pair=tuple(sorted((basis1,basis2)))
            if pair not in targettr:
                continue
            diffv=np.abs(Elist[:,j]-Elist[:,i])-Expe.Freq
            signch=np.where(np.diff(np.signbit(diffv)))[0]
            for k in signch:
                bstart,bend=Blist[k],Blist[k+1]
                def deltaE(b):
                    return np.real(np.abs(splines(b)[j]-splines(b)[i]))-Expe.Freq
                try:
                    res=sci.optimize.root_scalar(deltaE,bracket=[bstart,bend],method='brentq')
                    if res.converged:
                        ms1,ms2=slit[basis1],slit[basis2]
                        mi1,mi2=nlit[basis1],nlit[basis2]
                        dms=np.abs(ms1-ms2)
                        dmi=np.abs(mi1-mi2)
                        if np.isclose(dms,1) and np.isclose(dmi,0):
                            ttyp="Allowed"
                        elif np.isclose(dms,2):
                            ttyp="Forbidden (2)"
                        elif not np.isclose(dmi,0):
                            ttyp="Forbidden (N)"
                        else:
                            ttyp="Forbidden"
                        state1=Getlabel(basis1,slit,nlit,llit,Hamer.L,Hamer.I)
                        state2=Getlabel(basis2,slit,nlit,llit,Hamer.L,Hamer.I)
                        resonants.append({'field': res.root,'inx': (i, j),'bainx': (basis1,basis2),'type': ttyp,'transition': f"{state1} <-> {state2}"})
                        resfield.append(res.root)
                except ValueError:
                    pass
    if len(resfield)>0:
        if table:
            df=DataFrame(data=resonants)
            dfdis=df[['field', 'transition', 'type']].copy()
            dfl=dfdis.iloc[::2].reset_index(drop=True)
            dfr=dfdis.iloc[1::2].reset_index(drop=True)
            dfdis=concat([dfl, dfr],axis=1)
            dfdis.columns=['Field (mT)','Transition','Type','Field (mT)','Transition','Type']
            dfdis['Field (mT)']=dfdis['Field (mT)'].round(3)
            if is_notebook():
                from IPython.display import display
                display(dfdis)
            else:
                print(dfdis)
    else:
        print("No resonant fields detected in selected range")
    if graph:
        plt.figure(figsize=(10,6))
        plt.plot(Blist,epc,color='navy',label='Spectrum')
        plt.xlabel('Magnetic field [mT]')
        plt.ylabel('Counts [A. U.]')
        formatter=EngFormatter(sep='') 
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.xlim(Expe.Frange[0],Expe.Frange[1])
        plt.grid()
        plt.legend()
        plt.show(block=False)
        
        fig2,ax2=plt.subplots(figsize=(10,6))
        numlevels=Elist.shape[1]
        colenergy= cm.viridis(np.linspace(0,1,numlevels))
        coljet=cm.jet(np.linspace(0,1,numlevels))
        for elk in range(numlevels):
            basidx=curvebasis[elk]
            labelr=Getlabel(basidx,slit,nlit,llit,Hamer.L,Hamer.I)
            ax2.plot(Blist,Elist[:,elk],color=colenergy[elk],label=labelr)

        for r in resonants:
            fv=r['field']
            idi,idj =r['inx']
            eni=splines(fv)[idi]
            enj=splines(fv)[idj]
            if r['type']=='Allowed':
                ax2.plot([fv,fv],[eni,enj],color=coljet[idi],marker='o',markersize=4,linestyle='-')
            else:
                ax2.plot([fv,fv],[eni,enj],color='gray',marker='o',markersize=4,linestyle='-')

        ax2.set_title('Energy VS Field',fontsize=18)
        ax2.set_xlabel('Field [mT]')
        ax2.set_ylabel('Energy [GHz]')
        ax2.set_xlim(Blist[0],Blist[-1]+5)
        ax2.grid(True,color='black',alpha=0.3,linestyle='-')
        ax2.legend(bbox_to_anchor=(1.02,1),loc='upper left')
        plt.tight_layout()
        plt.show()

    return Blist,epc

def Calresonant(Hamer,Expe,Nucl='None',diagram=False):
    '''
    Function for the calculation of the EPR cw spectrum of monocristal systems. Uses a formulation similar to the *Eresonant* function, to calculate the resonant fields and intensities, but calculates de absorption curve (first integral) of the spectrum that is numerically  derived, producing the final spectrum.
    
    Parameters
    ----------
    
    Hamer : Class
        Container for the hamiltonian parameters of the system.
    
    Expe : Class
        Container for the experimental conditions.
    Nucl : str
        Isotope of the sample. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    diagram : Bool
        To pass the Energy data.
        
    Returns
    -------
    
    Blist : jax.np.array
        Array of the magnetic field.
    epc : jax.np.array
        Array of the counts of the spectrum.
    
    Example
    -------

    >>> import matplotlib.pyplot as plt
    >>> import epraya as epr
    >>> import numpy as np
    >>> Ham,Exp,_=epr.Jstart()
    >>> Ham.S=3/2
    >>> Ham.I=1
    >>> Ham.g=np.array([2.003, 2, 2])
    >>> Ham.A=np.array([200, 200, 200])  #Hyperfine constant
    >>> Ham.D=np.array([800,200])      #Zero field D and E
    >>> Ham.Hpp=[0,10]
    >>> Exp.Freq=9.4
    >>> Exp.Points=4096
    >>> Exp.Temperature=300
    >>> Exp.Frange=[0,800]
    >>> print(epr.Calresonant(Ham,Exp))
    (Array([0.00000000e+00, 1.95360195e-01, 3.90720391e-01, ...,
       7.99609280e+02, 7.99804640e+02, 8.00000000e+02], dtype=float64),
    Array([-1.64840330e-07, -1.10427100e-07, -8.03249933e-08, ...,
    -8.85167972e-08, -8.84171157e-08, -8.83673152e-08], dtype=float64))
    '''
    frange0=jxn.where(Expe.Frange[0]<0.0,1e-4,Expe.Frange[0])
    Ham=Hamer.replace(A=jxn.asarray(Hamer.A)/1000.0,D=jxn.asarray(Hamer.D)/1000.0,Hpp=jxn.asarray(Hamer.Hpp)/1.0,Q=jxn.asarray(Hamer.Q)/1000.0,
                     Bk2=jxn.asarray(Hamer.Bk2)/1000.0,Bk4=jxn.asarray(Hamer.Bk4)/1000.0,Bk6=jxn.asarray(Hamer.Bk6)/1000.0)
    etas=Ham.eta
    etas=jxn.where(Ham.Hpp[1]==0.0,0.0,etas)
    etas=jxn.where(Ham.Hpp[0]==0.0,1.0,etas)
    Ham=Ham.replace(eta=etas)
    slit,nlit,llit=JMsmi(Ham.I,Ham.S,Ham.L)
    dim=int(2*Ham.S+1)*int(2*Ham.I+1)*int(2*Ham.L+1)
    Exp=Expe
    ndir=jxn.array(Expe.Fdirection,dtype=jxn.float32)
    ndir=ndir/jxn.linalg.norm(ndir)
    nx,ny,nz=ndir[0],ndir[1],ndir[2]
    mdir=jxn.array(Expe.Mwdirection,dtype=jxn.float32)
    mdir=mdir/jxn.linalg.norm(mdir)
    mx,my,mz=mdir[0],mdir[1],mdir[2]
    Ham=Jchaframe(Ham,Exp)
    sx,sy,sz=JPauli(Ham.S)
    ix,iy,iz=JPauli(Ham.I)
    isx=jxn.kron(sx,jxn.eye(int(2*Ham.I+1)))
    isx=jxn.kron(jxn.eye(int(2*Ham.L+1)),isx)
    isx=jxn.asarray(isx,dtype=jxn.complex64)
    isy=jxn.kron(sy,jxn.eye(int(2*Ham.I+1)))
    isy=jxn.kron(jxn.eye(int(2*Ham.L+1)),isy)
    isy=jxn.asarray(isy,dtype=jxn.complex64)
    isz=jxn.kron(sz,jxn.eye(int(2*Ham.I+1)))
    isz=jxn.kron(jxn.eye(int(2*Ham.L+1)),isz)
    isz=jxn.asarray(isz,dtype=jxn.complex64)
    E=Exp.Freq
    espac1=jxn.linspace(frange0,Exp.Frange[1],500)
    beta=(scic.physical_constants["Bohr magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
    betan=(scic.physical_constants["nuclear magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
    hzex=jxn.asarray(beta*JHze(sx,sy,sz,Ham.g,[1,0,0],dim),dtype=complex)
    hzey=jxn.asarray(beta*JHze(sx,sy,sz,Ham.g,[0,1,0],dim),dtype=complex)
    hzez=jxn.asarray(beta*JHze(sx,sy,sz,Ham.g,[0,0,1],dim),dtype=complex)
    hwm=jxn.asarray(beta*JHze(sx,sy,sz,Ham.g,Exp.Mwdirection,dim),dtype=complex)
    h1=jxn.zeros((dim,dim),dtype='complex64')
    if Ham.S>=1:
        h1=h1+JStevensO(sx,sy,sz,Ham.S,Ham,dim)
    if Ham.L!=0:
        h1=h1+JLorbit(sx,sy,sz,Ham.lc,dim,Ham.L)
    if Ham.I!=0:
        h1=h1+JHfi(sx,sy,sz,ix,iy,iz,Ham.A,dim)
        h1=h1+JQii(ix,iy,iz,Ham.Q,dim)
        gnk=gnfactor(Nucl)
        nhzex=jxn.asarray(betan*JNhze(Ham.I,ix,iy,iz,dim,gnk,[1,0,0]),dtype=complex)
        nhzey=jxn.asarray(betan*JNhze(Ham.I,ix,iy,iz,dim,gnk,[0,1,0]),dtype=complex)
        nhzez=jxn.asarray(betan*JNhze(Ham.I,ix,iy,iz,dim,gnk,[0,0,1]),dtype=complex)
        hzex-=nhzex
        hzey-=nhzey
        hzez-=nhzez
    h1=jxn.asarray(h1,dtype=complex)
    hze=nx*hzex+ny*hzey+nz*hzez
    Blist=jxn.linspace(Expe.Frange[0],Expe.Frange[1],Expe.Points)
    def Jdiagop(B):
        h5=h1+B*hze
        Elist,Vlist=jxn.linalg.eigh(h5)
        return Elist,Vlist
    Elist,Vlist=jx.vmap(Jdiagop)(Blist)
    spc=jxn.zeros(Expe.Points)
    dim=Elist.shape[1]
    iidx,jidx=jxn.triu_indices(dim,k=1)
    Vdag=Vlist.conj().swapaxes(-1,-2)
    Tx=Vdag@isx@Vlist
    Ty=Vdag@isy@Vlist
    Tz=Vdag@isz@Vlist
    #Interpolate for intensities
    Txij=Tx[:,iidx,jidx]
    Tyij=Ty[:,iidx,jidx]
    Tzij=Tz[:,iidx,jidx]
    #Probability definition and interpolation
    Mn=mx*Txij+my*Tyij+mz*Tzij
    prob=jxn.abs(Mn)**2
    #Frecuency to field
    h22=jxn.real(Vdag@hze@Vlist)
    h2diag=jxn.diagonal(h22,axis1=1,axis2=2)
    dert=h2diag[:,iidx]
    izrt=h2diag[:,jidx]
    gma=jxn.abs(izrt-dert)
    gma=jxn.where(gma<1e-4,1e-4,gma)
    gema=1.0/gma
    #Boltzmann distribution
    conver=1e9*scc.h
    Ej=Elist*conver
    Temp=jxn.where(Exp.Temperature<=0.0,1.0,Exp.Temperature)
    beta=1.0/(scc.k*Temp)
    Emin=jxn.min(Ej,axis=-1,keepdims=True)
    boltz=jxn.exp(-beta*(Ej-Emin))
    Z=jxn.sum(boltz,axis=-1,keepdims=True)
    boltz=boltz/Z
    popui=boltz[:,iidx]
    popuj=boltz[:,jidx]
    boltzm=popui-popuj
    intensy=prob*gema*boltzm
    deltaE=jxn.abs(Elist[:,jidx]-Elist[:,iidx])
    dfe=deltaE-Exp.Freq
    hppg=Ham.Hpp[0]*gma
    hppl=Ham.Hpp[1]*gma
    hppg=jxn.where(hppg==0.0,1e-10,hppg)
    hpp=jxn.where(hppl==0.0,1e-10,hppl)
    gammag=hppg*jxn.sqrt(jxn.log(2.0)/2.0)
    gbs=jxn.exp(-jxn.log(2.0)*(dfe/gammag)**2)
    gammal=hppl*jxn.sqrt(3.0)
    gamma2l=gammal/2.0
    lbs=(gamma2l**2)/(dfe**2+gamma2l**2)
    voigt=(lbs*etas)+(gbs*(1.0-etas))
    spcint=jxn.sum(intensy*voigt,axis=1)
    dB=Blist[1]-Blist[0]
    spc=jxn.gradient(spcint,dB)
    if diagram:
        return Blist,spc,Elist,Vlist
    else:
        return Blist,spc
    
@jaxdatclass
class Mjhval:
    '''
    Jax container class for the hamiltonial parameters of  two systems. Must be initialized with a variable like ``Ham``. For the nth-system, it can be change using ``Ham.SN=1/2``.
    
    
    *Note: The Hpp is the same for the two systems.*
    
    **Warning: The Stevens operators constants are only defined for the first system.**
    
    Parameters
    ----------
    
    S : float 
        Spin value ex. (1/2,0,3/2).
    g : array_like or float
        g value of system, can be float, for isotropic case or array for anisotropic.
    I : float
        Nuclear spin value
    L : float
        Angular momentum
    A : array_like or float 
        Hyperfine constant, float for isotropic and array for anisotropic
    Q : array_like or float 
        Quadrupole nuclear interaction constant, float for isotropic and array for anisotropic
    D : array_like
        Zero field interaction constants D and E, two value array [0,0]
    Bk2 : array_like
        Stevens k=-/+2 constants
    Bk4 : array_like
        Stevens k=-/+4 constants    
    Bk6 : array_like
        Stevens k=-/+6 constants      
    lc : float
        Spin-orbit interaction constant
    Hpp : array_like
        Peak to peak distance for the voigtian function using [Hg,Hl], for gaussian and lorentzian distance
    eta : float
        weight of the gaussian contribution to the voigtian function, from 0 to 1. If eta is 0, the function is lorentzian and if eta is 1, the function is gaussian.
    weight: float
        Dummy variable by the moment
    Nucl : str
        Isotope of the sample. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    A1_2 : array_like
        Hipefine interaction between the spin 1 and nuclear spin 2.
    X1_2 : array_like
        Electron-Electron interaction between spins 1 and 2.
        
    Example
    -------
    >>> import epraya as epr
    >>> Ham=Mjhval()
    >>> Ham.S1=1
    Ham.I1=1/2
    >>> Ham.g1=[2.003,1.8,1.5]
    >>> Ham.S2=2
    >>> Ham.I2=1
    >>> Ham.g2=[2.003,1.8,1.5]
    >>> Ham.A1_2=[200,300,200]
    >>> Ham.X1_2=[500,1000,500]
    >>> print(Ham)
    Mjhval(S1=1, S2=2, g1=[2.003, 1.8, 1.5], g2=[2.003, 1.8, 1.5],
    I1=0.5, I2=1, L1=0.0, L2=0.0, A1=0.0, A2=0.0, 
    Q1=Array([0, 0, 0], dtype=int32), Q2=Array([0, 0, 0], dtype=int32),
    D1=Array([0, 0], dtype=int32), D2=Array([0, 0], dtype=int32), 
    Bk2=[0, 0, 0, 0, 0], Bk4=[0, 0, 0, 0, 0, 0, 0, 0, 0],
    Bk6=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], lc1=0.0, lc2=0.0,
    A1_2=[200, 300, 200], A2_1=0.0, X1_2=[500, 1000, 500], 
    Hpp=Array([0, 1], dtype=int32), eta=0.5, weight=0.0)
    '''
    S1: Union[float,int]=1/2   # Spin
    S2: Union[float,int]=1/2   # Spin
    g1: Union[List[float],float,int]=dcfield(default_factory=lambda: 2.003)  # g value
    g2: Union[List[float],float,int]=dcfield(default_factory=lambda: 2.003)  # g value
    I1: float=0.0   # Nuclear spin
    I2: float=0.0   # Nuclear spin
    L1: float=0.0   # Angular momentum
    L2: float=0.0   # Angular momentum
    A1: Union[List[float], float]=dcfield(default_factory=lambda:0.0)     # Hyperfine constant
    A2: Union[List[float], float]=dcfield(default_factory=lambda:0.0)     # Hyperfine constant
    Q1: Union[list[float], float]=dcfield(default_factory=lambda:jxn.array([0,0,0]))    # Quadrupole interaction constant
    Q2: Union[list[float], float]=dcfield(default_factory=lambda:jxn.array([0,0,0]))    # Quadrupole interaction constant
    D1: Union[list[float], float]=dcfield(default_factory=lambda:jxn.array([0,0]))     # Zero field interaction D and E constants
    D2: Union[list[float], float]=dcfield(default_factory=lambda:jxn.array([0,0]))     # Zero field interaction D and E constants
    Bk2: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0])
    Bk4: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0,0,0,0,0])
    Bk6: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0,0,0,0,0,0,0,0,0])
    lc1: float=0.0                          # Spin-orbit interaction constant
    lc2: float=0.0                          # Spin-orbit interaction constant
    A1_2: Union[List[float], float]=dcfield(default_factory=lambda:0.0)     # Hyperfine constant
    A2_1: Union[List[float], float]=dcfield(default_factory=lambda:0.0)     # Hyperfine constant
    X1_2: Union[List[float], float]=dcfield(default_factory=lambda:0.0)
    Hpp: List=dcfield(default_factory=lambda:jxn.array([0,1]))
    eta: float=0.5
    weight: float=0.0

@jaxdatclass
class JEmco:
    '''
    Jax container class for the experimental parameters of two systems. Must be initialized with a variable like ``Exp``. To change one parameter use the sintaxis ``Exp.Freq=9.40``.
    
    Parameters
    ----------
    Freq : float
        Frequency of the microwave in the EPR spectrometer in GHz.
    Points  : int
        Number of points used to take the EPR spectrum.
    Temperature : float
        Temperature of the sample during the measurement in Kelvin.

    Fdirection : list=[0,0,1]
        Direction of incidence of the magnetic field in relation with the lab frame. By deafult it's [0,0,1], the Z direction.
    Mwdirection : list=[1,0,0]
        Direction of incidence of the microwave radiation in relation with the lab frame. By deafult it's [1,0,0], the X direction.
    Frange : list=[0,1]
        Field range used in the EPR spectrum in mT.
    Sampleframe : list==[0,0,0]
        Three Euler angles of the orientation of the sample in relation to the lab frame.
    Molframe : list=[0,0,0]
        Three Euler angles of the orientation of the  paramagnetic molecule in relation to the sample frame.
    gframe : list=[0,0,0]
        Three Euler angles of the orientation of the tensor g in the molecular frame.
    Aframe : list=[0,0,0]
        Three Euler angles of the orientation of the tensor A in the molecular frame.    
    Dframe : list=[0,0,0]
        Three Euler angles of the orientation of the tensor Q in the molecular frame.    
    Qframe : list=[0,0,0]
        Three Euler angles of the orientation of the tensor D in the molecular frame.   
        
    Example
    --------
    
    >>> import epraya as epr 
    >>> Exp=epr.JEmco()
    >>> Exp.Freq=9.43
    >>> Exp.Points=2046
    >>> Exp.Temperature=306
    >>> Exp.Frange=[0,400]
    >>> Exp.Mwdirection=[0,1,0]
    >>> Exp.gframe1=[20,30,40]
    >>> Exp.gframe2=[20,30,40]
    >>> print(Exp)
    JEmco(Freq=9.43, Points=2046, Temperature=306, Fdirection=[0, 0, 1],
    Mwdirection=[0, 1, 0], Frange=[0, 400], Sampleframe1=[0, 0, 0],
    Molframe1=[0, 0, 0], gframe1=[20, 30, 40], Aframe1=[0, 0, 0],
    Dframe1=[0, 0, 0], Qframe1=[0, 0, 0], Sampleframe2=[0, 0, 0],
    Molframe2=[0, 0, 0], gframe2=[20, 30, 40], Aframe2=[0, 0, 0], 
    Dframe2=[0, 0, 0], Qframe2=[0, 0, 0])
    '''
    Freq: float=9.433
    Points: int=4096
    Temperature: float=295.15
    Fdirection: list[float]=dcfield(default_factory=lambda:[0,0,1])
    Mwdirection: list[float]=dcfield(default_factory=lambda:[1,0,0])
    Frange: list[float]=dcfield(default_factory=lambda:[0,1])
    Sampleframe1: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Molframe1: list[float]=dcfield(default_factory=lambda:[0,0,0])
    gframe1: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Aframe1: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Dframe1: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Qframe1: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Sampleframe2: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Molframe2: list[float]=dcfield(default_factory=lambda:[0,0,0])
    gframe2: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Aframe2: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Dframe2: list[float]=dcfield(default_factory=lambda:[0,0,0])
    Qframe2: list[float]=dcfield(default_factory=lambda:[0,0,0])

@jaxdatclass
class JEmva:
    '''
    Jax-container class for the range of variation of the hamiltonian parameters by defining the minimum and then the maximum value of each variable. Works for 2 systems and must be initialized with a variable like ``Vary``. To change one parameter use the sintaxis ``Vary.g=[1.5,2.5,1.0,2.0,0.0,1.0]``.
    
    *Note: The Hpp is the same for the two systems.*
    
    Parameters
    ----------
    g : list[float]
        Range to vary the three values of the g tensor.
    A : list[float]  
        Range to vary the three values of the A tensor.
    Q : list[float]  
        Range to vary the three values of the Q tensor.
    D : list[float]
        Range to vary the two values of the D tensor.
    Hpp : list[float]
        Range to vary the two values of the peak to peak distance.
    weight : float
        Dummy variable by the moment.
    
    Example
    --------
    
    >>> import epraya as epr
    >>> Vary=epr.JEmva()
    >>> Vary.g1=[1.5,2.5,1.0,2.0,0.0,1.0]
    >>> Vary.A1=[100,300,200,400,200,250]
    >>> Vary.g2=[1.2,1.5,2.003,2.005,2.5,2.6]
    >>> Vary.A2=[200,250,500,600,200,250]
    >>> Vary.Hpp=[0,20,10,15]
    >>> print(Vary)
    JEmva(g1=[1.5, 2.5, 1.0, 2.0, 0.0, 1.0], 
    A1=[100, 300, 200, 400, 200, 250], Q1=0.0, 
    D1=0.0, g2=[1.2, 1.5, 2.003, 2.005, 2.5, 2.6],
    A2=[200, 250, 500, 600, 200, 250], Q2=0.0, 
    D2=0.0, Hpp=[0, 20, 10, 15], weight=0.0)
    '''
    g1: Union[list[float],float]=0.0
    A1: Union[list[float],float]=0.0     # Hyperfine constant
    Q1: Union[list[float],float]=0.0     # Quadrupole interaction constant
    D1: Union[list[float],float]=0.0
    g2: Union[list[float],float]=0.0
    A2: Union[list[float],float]=0.0     # Hyperfine constant
    Q2: Union[list[float],float]=0.0     # Quadrupole interaction constant
    D2: Union[list[float],float]=0.0
    Hpp: List=dcfield(default_factory=lambda:jxn.array([0,0]))
    weight: float=0.0

def Jmstart():
    '''
    Creates the three JAX containers ``Ham``, ``Exp`` and ``Vary`` for the two systems.
    
    Returns
    -------
    
    Ham : Class
        Jax container for the hamiltonian parameters of the two systems.
    Exp : Class
        Jax container for the experimental conditions of the two systems.
    Vary : Class
        Jax container for the range and parameters to vary of the two systems. 
    
    Example
    -------
    
    >>> import epraya as epr
    >>> Ham, Exp, Vary=epr.Jmstart()
    >>> print(Ham,Exp,Vary)
    Mjhval(S1=0.5, S2=0.5, g1=2.003, g2=2.003, I1=0.0, I2=0.0,
    L1=0.0, L2=0.0, A1=0.0, A2=0.0, Q1=Array([0, 0, 0], dtype=int32),
    Q2=Array([0, 0, 0], dtype=int32), D1=Array([0, 0], dtype=int32),
    D2=Array([0, 0], dtype=int32), Bk2=[0, 0, 0, 0, 0], 
    Bk4=[0, 0, 0, 0, 0, 0, 0, 0, 0], 
    Bk6=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], lc1=0.0, lc2=0.0,
    A1_2=0.0, A2_1=0.0, X1_2=0.0, Hpp=Array([0, 1], dtype=int32),
    eta=0.5, weight=0.0) 
    JEmco(Freq=9.433, Points=4096, Temperature=295.15, 
    Fdirection=[0, 0, 1], Mwdirection=[1, 0, 0], Frange=[0, 1],
    Sampleframe1=[0, 0, 0], Molframe1=[0, 0, 0], gframe1=[0, 0, 0],
    Aframe1=[0, 0, 0], Dframe1=[0, 0, 0], Qframe1=[0, 0, 0], 
    Sampleframe2=[0, 0, 0], Molframe2=[0, 0, 0], gframe2=[0, 0, 0],
    Aframe2=[0, 0, 0], Dframe2=[0, 0, 0], Qframe2=[0, 0, 0]) 
    JEmva(g1=0.0, A1=0.0, Q1=0.0, D1=0.0, g2=0.0, A2=0.0, Q2=0.0,
    D2=0.0, Hpp=Array([0, 0], dtype=int32), weight=0.0)
    '''
    Ham,Exp,Vary=Mjhval(),JEmco(),JEmva()
    return Ham,Exp,Vary

def JMulpol(maham,Expe,Nucl1='None',Nucl2='None',graph=True):
    '''
    Wrap function for the simulation of the EPR spectrum for powder samples of two systems. If there is an interaction between the systems (electron-eletron or hiperfine), solves the total hamiltonian. Otherwise, sums the contributions to the total spectrum.
    
    Parameters
    ----------
    
     maham : Class
        Container for the hamiltonian parameters of the two systems.
    
    Expe : Class
        Container for the experimental conditions.
    Nucl1 : str
        Isotope of the first system. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    Nucl2 : str
        Isotope of the second system. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    graph : Bool
        PLots the spectrum of the multisystem.
    
    Returns
    -------
    
    Blist : jax.np.array
        Array of the magnetic field.
    epc : jax.np.array
        Array of the counts of the spectrum.

    Example
    -------

    .. code-block:: python
    
       import matplotlib.pyplot as plt
       import epraya as epr
       import numpy as np
       Ham,Exp,_=epr.Jmstart()
       Ham.S1=3/2
       Ham.I1=1
       Ham.S2=1
       Ham.I2=1
       Ham.g1=np.array([2.003, 2, 2])
       Ham.g2=np.array([1.5, 1.5, 1.5])
       Ham.A1=np.array([200, 200, 200])  #Hyperfine constant
       Ham.D1=np.array([800,200])      #Zero field D and E
       Ham.Hpp=[0,10]
       Exp.Freq=9.4
       Exp.Points=4096
       Exp.Temperature=300
       Exp.Frange=[0,800]
       B,spc=epr.JMulpol(Ham,Exp)
       
    .. image:: /_static/jpol.PNG
       :alt: Plot of the JMulpol function
       :align: center
    '''
    Blist,epc=Jcalmulta(maham,Expe,Nucl1,Nucl2)
    if graph:
        plt.figure(figsize=(10,6))
        plt.plot(Blist,epc,color='navy',label='Spectrum')
        plt.xlabel('Magnetic field [mT]')
        formatter=EngFormatter(sep='') 
        plt.gca().yaxis.set_major_formatter(formatter)
        plt.ylabel('Counts [A. U.]')
        plt.xlim(Expe.Frange[0],Expe.Frange[1])
        plt.grid()
        plt.legend()
        plt.show(block=False)
    return Blist,epc
    

def Jcalmulta(maham,Expe,Nucl1='None',Nucl2='None'):
    '''
    Determinates the spectrum for a two paramagnetic centers system. Follows the same logic from the function Jpowder, but adapted to the two centers system.
    
    Parameters
    ----------
    
    maham : Class
        Container for the hamiltonian parameters of the two systems.
    Expe : Class
        Container for the experimental conditions.
    Nucl1 : str
        Isotope of the first system. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    Nucl2 : str
        Isotope of the second system. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    graph : Bool
        PLots the spectrum of the multisystem.
    
    Returns
    -------
    
    fielde : jax.np.array
        Array of the magnetic field.
    specs : jax.np.array
        Array of the counts of the spectrum.
    
    Example
    -------
    
    >>> import matplotlib.pyplot as plt
    >>> import epraya as epr
    >>> import numpy as np
    >>> Ham,Exp,_=epr.Jmstart()
    >>> Ham.S1=3/2
    >>> Ham.I1=1
    >>> Ham.S2=1
    >>> Ham.I2=1
    >>> Ham.g1=np.array([2.003, 2, 2])
    >>> Ham.g2=np.array([1.5, 1.5, 1.5])
    >>> Ham.A1=np.array([200, 200, 200])  #Hyperfine constant
    >>> Ham.D1=np.array([800,200])      #Zero field D and E
    >>> Ham.Hpp=[0,10]
    >>> Exp.Freq=9.4
    >>> Exp.Points=4096
    >>> Exp.Temperature=300
    >>> Exp.Frange=[0,800]
    >>> print(epr.Jcalmulta(Ham,Exp))
    (Array([0.00000000e+00, 1.95360195e-01, 3.90720391e-01, ...,
    7.99609280e+02, 7.99804640e+02, 8.00000000e+02], dtype=float64),
    Array([-1.51007828e-21,  1.87293010e-21,  1.91587492e-21, ...,
    3.33966042e-21,  1.25252035e-21, -1.98829450e-21], dtype=float64))
    '''
    maham.X1_2=np.asarray(maham.X1_2)
    maham.A1_2=np.asarray(maham.A1_2)
    maham.A2_1=np.asarray(maham.A2_1)
    Ham1=JHval()
    Ham2=JHval()
    Exp1=JEco()
    Exp2=JEco()
    Ham1.S,Ham1.g,Ham1.I,Ham1.L,Ham1.A,Ham1.Q,Ham1.D,Ham1.Bk2,Ham1.Bk4,Ham1.Bk6,Ham1.lc,Ham1.Hpp,Ham1.eta=maham.S1,maham.g1,maham.I1,maham.L1,maham.A1,maham.Q1,maham.D1,maham.Bk2,maham.Bk4,maham.Bk6,maham.lc1,maham.Hpp,maham.eta
    Ham2.S,Ham2.g,Ham2.I,Ham2.L,Ham2.A,Ham2.Q,Ham2.D,Ham2.lc,Ham2.Hpp,Ham2.eta=maham.S2,maham.g2,maham.I2,maham.L2,maham.A2,maham.Q2,maham.D2,maham.lc2,maham.Hpp,maham.eta
    Exp1.Freq,Exp1.Points,Exp1.Temperature,Exp1.Fdirection,Exp1.Frange,Exp1.Sampleframe,Exp1.Molframe,Exp1.gframe,Exp1.Aframe,Exp1.Dframe,Exp1.Qframe=Expe.Freq,Expe.Points,Expe.Temperature,Expe.Fdirection,Expe.Frange,Expe.Sampleframe1,Expe.Molframe1,Expe.gframe1,Expe.Aframe1,Expe.Dframe1,Expe.Qframe1
    Exp2.Freq,Exp2.Points,Exp2.Temperature,Exp2.Fdirection,Exp2.Frange,Exp2.Sampleframe,Exp2.Molframe,Exp2.gframe,Exp2.Aframe,Exp2.Dframe,Exp2.Qframe=Expe.Freq,Expe.Points,Expe.Temperature,Expe.Fdirection,Expe.Frange,Expe.Sampleframe2,Expe.Molframe2,Expe.gframe2,Expe.Aframe2,Expe.Dframe2,Expe.Qframe2
    iwas,jwas,kwas,weight,hulk=Delaunay(Exp1)
    iwas,jwas,kwas,weight,hulk=jxn.array(iwas),jxn.array(jwas),jxn.array(kwas),jxn.array(weight),jxn.array(hulk)
    if np.allclose(maham.X1_2,0.0) and np.allclose(maham.A1_2,0.0) and np.allclose(maham.A2_1,0.0):
        fielde,specs1=JCalpowder(Ham1,Exp1,iwas,jwas,kwas,weight,hulk,Nucl1)
        _,specs2=JCalpowder(Ham2,Exp2,iwas,jwas,kwas,weight,hulk,Nucl2)
        specs=specs1+specs2
    else:
        frange0=jxn.where(Exp1.Frange[0]<0.0,1e-4,Exp1.Frange[0])
        Ham1=Ham1.replace(A=jxn.asarray(Ham1.A)/1000.0,D=jxn.asarray(Ham1.D)/1000.0,Hpp=jxn.asarray(Ham1.Hpp)/1.0,Q=jxn.asarray(Ham1.Q)/1000.0,
                     Bk2=jxn.asarray(Ham1.Bk2)/1000.0,Bk4=jxn.asarray(Ham1.Bk4)/1000.0,Bk6=jxn.asarray(Ham1.Bk6)/1000.0)
        Ham2=Ham2.replace(A=jxn.asarray(Ham2.A)/1000.0,D=jxn.asarray(Ham2.D)/1000.0,Hpp=jxn.asarray(Ham2.Hpp)/1.0,Q=jxn.asarray(Ham2.Q)/1000.0,
                     Bk2=jxn.asarray(Ham2.Bk2)/1000.0,Bk4=jxn.asarray(Ham2.Bk4)/1000.0,Bk6=jxn.asarray(Ham2.Bk6)/1000.0)
        etas=Ham1.eta
        etas=jxn.where(Ham1.Hpp[1]==0.0,0.0,etas)
        etas=jxn.where(Ham1.Hpp[0]==0.0,1.0,etas)
        Ham1=Ham1.replace(eta=etas)
        Ham2=Ham2.replace(eta=etas)
        dim=int(2*Ham1.S+1)*int(2*Ham1.I+1)*int(2*Ham1.L+1)*int(2*Ham2.S+1)*int(2*Ham2.I+1)*int(2*Ham2.L+1)
        dim1=int(2*Ham1.S+1)*int(2*Ham1.I+1)*int(2*Ham1.L+1)
        dim2=int(2*Ham2.S+1)*int(2*Ham2.I+1)*int(2*Ham2.L+1)
        Ham1=Jchaframe(Ham1,Exp1)
        Ham2=Jchaframe(Ham2,Exp2)
        sx1,sy1,sz1=JPauli(Ham1.S)
        ix1,iy1,iz1=JPauli(Ham1.I)
        sx2,sy2,sz2=JPauli(Ham2.S)
        ix2,iy2,iz2=JPauli(Ham2.I)
        E=Exp.Freq
        espac1=jxn.linspace(frange0,Exp.Frange[1],500)
        beta=(scic.physical_constants["Bohr magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
        betan=(scic.physical_constants["nuclear magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
        h1=np.zeros((dim,dim),dtype='complex')
        hzex=0
        hzey=0
        hzez=0
        hzex1=jxn.asarray(beta*JHze(sx1,sy1,sz1,Ham1.g,[1,0,0],dim1),dtype=complex)
        hzey1=jxn.asarray(beta*JHze(sx1,sy1,sz1,Ham1.g,[0,1,0],dim1),dtype=complex)
        hzez1=jxn.asarray(beta*JHze(sx1,sy1,sz1,Ham1.g,[0,0,1],dim1),dtype=complex)
        hzex2=jxn.asarray(beta*JHze(sx2,sy2,sz2,Ham2.g,[1,0,0],dim2),dtype=complex)
        hzey2=jxn.asarray(beta*JHze(sx2,sy2,sz2,Ham2.g,[0,1,0],dim2),dtype=complex)
        hzez2=jxn.asarray(beta*JHze(sx2,sy2,sz2,Ham2.g,[0,0,1],dim2),dtype=complex)

        hzex+=jxn.kron(hzex1,jxn.eye(dim2,dtype=complex))+jxn.kron(jxn.eye(dim1,dtype=complex),hzex2)
        hzey+=jxn.kron(hzey1,jxn.eye(dim2,dtype=complex))+jxn.kron(jxn.eye(dim1,dtype=complex),hzey2)
        hzez+=jxn.kron(hzez1,jxn.eye(dim2,dtype=complex))+jxn.kron(jxn.eye(dim1,dtype=complex),hzez2)
        if Ham1.S>=1:
            h1=h1+JStevensO(sx1,sy1,sz1,Ham1.S,Ham1,dim)
        if Ham2.S>=1:
            h1=h1+JStevensO(sx2,sy2,sz2,Ham2.S,Ham2,dim)
        if Ham1.L!=0:
            h1=h1+JLorbit(sx1,sy1,sz1,Ham1.lc,dim,Ham1.L)
        if Ham2.L!=0:
            h1=h1+JLorbit(sx2,sy2,sz2,Ham2.lc,dim,Ham2.L)
        if Ham1.I!=0:
            h1=h1+JHfi(sx1,sy1,sz1,ix1,iy1,iz1,Ham1.A,dim)
            h1=h1+JQii(ix1,iy1,iz1,Ham1.Q,dim)
            gnk=gnfactor(Nucl1)
            nhzex=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[1,0,0]),dtype=complex)
            nhzey=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[0,1,0]),dtype=complex)
            nhzez=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[0,0,1]),dtype=complex)
            hzex-=jxn.kron(nhzex,jxn.eye(dim2,dtype=complex))
            hzey-=jxn.kron(nhzey,jxn.eye(dim2,dtype=complex))
            hzez-=jxn.kron(nhzez,jxn.eye(dim2,dtype=complex))
        if Ham2.I!=0:
            h1=h1+JHfi(sx2,sy2,sz2,ix2,iy2,iz2,Ham2.A,dim)
            h1=h1+JQii(ix2,iy2,iz2,Ham2.Q,dim)
            gnk=gnfactor(Nucl2)
            nhzex=jxn.asarray(betan*JNhze(Ham2.I,ix2,iy2,iz2,dim2,gnk,[1,0,0]),dtype=complex)
            nhzey=jxn.asarray(betan*JNhze(Ham2.I,ix2,iy2,iz2,dim2,gnk,[0,1,0]),dtype=complex)
            nhzez=jxn.asarray(betan*JNhze(Ham2.I,ix2,iy2,iz2,dim2,gnk,[0,0,1]),dtype=complex)
            hzex-=jxn.kron(jxn.eye(dim1,dtype=complex),nhzex)
            hzey-=jxn.kron(jxn.eye(dim1,dtype=complex),nhzey)
            hzez-=jxn.kron(jxn.eye(dim1,dtype=complex),nhzez)
        if jxn.any(maham.A1_2):
            Aref=jxn.asarray(maham.A1_2)/1000.0
            Aref=Aref*jxn.eye(3)
            h1+=JHfi(sx1,sy1,sz1,ix2,iy2,iz2,Aref,dim)
        if jxn.any(maham.A2_1):
            Aref=jxn.asarray(maham.A2_1)/1000.0
            Aref=Aref*jxn.eye(3)
            h1+=JHfi(sx2,sy2,sz2,ix1,iy1,iz1,Aref,dim)
        if jxn.any(maham.X1_2):
            Xref=jxn.asarray(maham.X1_2)/1000.0
            Xref=Xref*jxn.eye(3)
            h1+=JIee(sx1,sy1,sz1,sx2,sy2,sz2,Xref,dim)
        h1=jxn.asarray(h1,dtype=complex)
        #For the total magnetic moment of the system
        stodx=np.zeros((dim,dim),dtype='complex')
        stody=np.zeros((dim,dim),dtype='complex')
        stodz=np.zeros((dim,dim),dtype='complex')
        dimwns1=int((2*Ham1.I+1)*(2*Ham1.L+1))
        dimwns2=int((2*Ham2.I+1)*(2*Ham2.L+1))
        if Ham1.S>0:
            sx1r=jxn.kron(sx1,jxn.eye(dimwns1,dtype=complex))
            sy1r=jxn.kron(sy1,jxn.eye(dimwns1,dtype=complex))
            sz1r=jxn.kron(sz1,jxn.eye(dimwns1,dtype=complex))
            stodx+=jxn.kron(sx1r,jxn.eye(dim2,dtype=complex))
            stody+=jxn.kron(sy1r,jxn.eye(dim2,dtype=complex))
            stodz+=jxn.kron(sz1r,jxn.eye(dim2,dtype=complex))
        if Ham2.S>0:
            sx2r=jxn.kron(sx2,jxn.eye(dimwns2,dtype=complex))
            sy2r=jxn.kron(sy2,jxn.eye(dimwns2,dtype=complex))
            sz2r=jxn.kron(sz2,jxn.eye(dimwns2,dtype=complex))
            stodx+=jxn.kron(jxn.eye(dim1,dtype=complex),sx2r)
            stody+=jxn.kron(jxn.eye(dim1,dtype=complex),sy2r)
            stodz+=jxn.kron(jxn.eye(dim1,dtype=complex),sz2r)

        Blist1=jxn.linspace(Exp.Frange[0],Exp.Frange[1],500)
        dB=(Exp.Frange[1]-Exp.Frange[0])/(Exp.Points-1)
        Bmin=Exp.Frange[0]
        
        @jx.jit
        def Oneori(nx,ny,nz):
            Elist,Vlist,h2=JPadaptarray(Blist1,h1,hzex,hzey,hzez,nx,ny,nz)
            resfield,intensy,ntrans=JNresina(Blist1,Elist,Vlist,dim,Exp1.Freq,stodx,stody,stodz,nx,ny,nz,Exp1.Temperature,Ham1.Hpp,h2)
            return resfield,intensy,ntrans
        voneori=jx.vmap(Oneori,in_axes=(0,0,0))
        csize=50 #Divides the orientations blocks so the RAM doesn't explote
        tlen=len(weight)
        plen=(csize-(tlen%csize))%csize
        pdw=jxn.pad(weight,(0,plen))
        pdi=jxn.pad(iwas,(0, plen))
        pdj=jxn.pad(jwas,(0,plen))
        pdk=jxn.pad(kwas,(0,plen))
        nparts=len(pdi)//csize
        bati=pdi.reshape(nparts,csize)
        batj=pdj.reshape(nparts,csize)
        batk=pdk.reshape(nparts,csize)
        @jx.checkpoint
        def Processvmap(curspect,bat):
            bnx,bny,bnz=bat
            batres,batint,batntras=voneori(bnx,bny,bnz)
            return curspect,(batres,batint,batntras)
        _,(allres,allint,ntrans)=jx.lax.scan(Processvmap,None,(bati,batj,batk))
        allres=allres.reshape(-1,allres.shape[-1])[:tlen]
        allint=allint.reshape(-1,allint.shape[-1])[:tlen]
        ntrans=ntrans.reshape(-1)[:tlen]
        sketch=JCaltriangle(Bmin,dB,allres,allint,ntrans,hulk,weight,Exp1.Points)
        maxlenght=jxn.max(jxn.array(Ham1.Hpp))*10
        kpoints=201 
        kaxis=jxn.arange(-kpoints//2+1,kpoints//2+1)*dB
        kvoigt=JVoigtp(kaxis,jxn.array([1.0]),jxn.array([0.0]),Ham1.Hpp,etas)
        espectotal=jsig.fftconvolve(sketch,kvoigt,mode='same')*dB
        Blist2=jxn.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
        fielde,specs=Blist2,espectotal
    return fielde,specs

def JMusic(maham,Expe,Nucl1='None',Nucl2='None',graph=True):
    '''
    Wrap function for the simulation of the EPR spectrum for monocristal samples of two systems. If there is an interaction between the systems (electron-eletron or hiperfine), solves the total hamiltonian. Otherwise, sums the contributions to the total spectrum.
    
    Parameters
    ----------

     maham : Class
        Container for the hamiltonian parameters of the two systems.
    Expe : Class
        Container for the experimental conditions.
    Nucl1 : str
        Isotope of the first system. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    Nucl2 : str
        Isotope of the second system. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    graph : Bool
        PLots the spectrum of the multisystem.
    
    Returns
    -------
    
    Blist : jax.np.array
        Array of the magnetic field.
    epc : jax.np.array
        Array of the counts of the spectrum.

    Example
    -------

    .. code-block:: python
    
       import matplotlib.pyplot as plt
       import epraya as epr
       import numpy as np
       Ham,Exp,_=epr.Jmstart()
       Ham.S1=3/2
       Ham.I1=1
       Ham.S2=1
       Ham.I2=1
       Ham.g1=np.array([2.003, 2, 2])
       Ham.g2=np.array([1.5, 1.5, 1.5])
       Ham.A1=np.array([200, 200, 200])  #Hyperfine constant
       Ham.D1=np.array([800,200])      #Zero field D and E
       Ham.Hpp=[0,10]
       Exp.Freq=9.4
       Exp.Points=4096
       Exp.Temperature=300
       Exp.Frange=[0,800]
       B,spc=epr.JMusic(Ham,Exp)
       
    .. image:: /_static/jmusic.PNG
       :alt: Plot of the Jmusic function
       :align: center
    '''
    Blist,epc=Jcalmusic(maham,Expe,Nucl1,Nucl2)
    if graph:
        plt.figure(figsize=(10,6))
        plt.plot(Blist,epc,color='navy',label='Spectrum')
        plt.xlabel('Magnetic field [mT]')
        plt.ylabel('Counts [A. U.]')
        plt.xlim(Expe.Frange[0],Expe.Frange[1])
        plt.grid()
        plt.legend()
        plt.show(block=False)
    return Blist,epc

def Jcalmusic(maham,Expe,Nucl1='None',Nucl2='None'):
    '''
    Determinates the spectrum for a two paramagnetic centers system. Follows the same logic from the function Jresonant, but adapted to the two centers system.
    
    
    Parameters
    ----------
    
    maham : Class
        Container for the hamiltonian parameters of the two systems.
    Expe : Class
        Container for the experimental conditions.
    Nucl1 : str
        Isotope of the first system. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    Nucl2 : str
        Isotope of the second system. Can be the quantum number and the element or only the element ('55Mn' or 'Mn') 
    graph : Bool
        PLots the spectrum of the multisystem.
    
    Returns
    -------
    
    fielde : jax.np.array
        Array of the magnetic field.
    specs : jax.np.array
        Array of the counts of the spectrum.
    
    Example
    -------
    
    >>> import matplotlib.pyplot as plt
    >>> import epraya as epr
    >>> import numpy as np
    >>> Ham,Exp,_=epr.Jmstart()
    >>> Ham.S1=3/2
    >>> Ham.I1=1
    >>> Ham.S2=1
    >>> Ham.I2=1
    >>> Ham.g1=np.array([2.003, 2, 2])
    >>> Ham.g2=np.array([1.5, 1.5, 1.5])
    >>> Ham.A1=np.array([200, 200, 200])  #Hyperfine constant
    >>> Ham.D1=np.array([800,200])      #Zero field D and E
    >>> Ham.Hpp=[0,10]
    >>> Exp.Freq=9.4
    >>> Exp.Points=4096
    >>> Exp.Temperature=300
    >>> Exp.Frange=[0,800]
    >>> print(epr.Jcalmusic(Ham,Exp))
    (Array([0.00000000e+00, 1.95360195e-01, 3.90720391e-01, ...,
    7.99609280e+02, 7.99804640e+02, 8.00000000e+02], dtype=float64),
    Array([-1.60669060e-07, -9.66774880e-08, -5.69966581e-08, ...,
    -2.01984373e-07, -2.01714437e-07, -2.01579599e-07], dtype=float64))
    '''
    maham.X1_2=np.asarray(maham.X1_2)
    maham.A1_2=np.asarray(maham.A1_2)
    maham.A2_1=np.asarray(maham.A2_1)
    Ham1=JHval()
    Ham2=JHval()
    Ham1.S,Ham1.g,Ham1.I,Ham1.L,Ham1.A,Ham1.Q,Ham1.D,Ham1.Bk2,Ham1.Bk4,Ham1.Bk6,Ham1.lc,Ham1.Hpp,Ham1.eta=maham.S1,maham.g1,maham.I1,maham.L1,maham.A1,maham.Q1,maham.D1,maham.Bk2,maham.Bk4,maham.Bk6,maham.lc1,maham.Hpp,maham.eta
    Ham2.S,Ham2.g,Ham2.I,Ham1.L,Ham2.A,Ham2.Q,Ham2.D,Ham2.lc,Ham2.Hpp,Ham2.eta=maham.S2,maham.g2,maham.I2,maham.L2,maham.A2,maham.Q2,maham.D2,maham.lc2,maham.Hpp,maham.eta
    Exp1=JEco()
    Exp2=JEco()
    Exp1.Freq,Exp1.Points,Exp1.Temperature,Exp1.Fdirection,Exp1.Mwdirection,Exp1.Frange,Exp1.Sampleframe,Exp1.Molframe,Exp1.gframe,Exp1.Aframe,Exp1.Dframe,Exp1.Qframe=Expe.Freq,Expe.Points,Expe.Temperature,Expe.Fdirection,Expe.Mwdirection,Expe.Frange,Expe.Sampleframe1,Expe.Molframe1,Expe.gframe1,Expe.Aframe1,Expe.Dframe1,Expe.Qframe1
    Exp2.Freq,Exp2.Points,Exp2.Temperature,Exp2.Fdirection,Exp2.Mwdirection,Exp2.Frange,Exp2.Sampleframe,Exp2.Molframe,Exp2.gframe,Exp2.Aframe,Exp2.Dframe,Exp2.Qframe=Expe.Freq,Expe.Points,Expe.Temperature,Expe.Fdirection,Expe.Mwdirection,Expe.Frange,Expe.Sampleframe2,Expe.Molframe2,Expe.gframe2,Expe.Aframe2,Expe.Dframe2,Expe.Qframe2

    if np.allclose(maham.X1_2,0.0) and np.allclose(maham.A1_2,0.0) and np.allclose(maham.A2_1,0.0):
        fielde,specs1=Calresonant(Ham1,Exp1,Nucl1)
        _,specs2=Calresonant(Ham2,Exp2,Nucl2)
        specs=specs1+specs2
    else:
        frange0=jxn.where(Exp1.Frange[0]<0.0,1e-4,Exp1.Frange[0])
        Ham1=Ham1.replace(A=jxn.asarray(Ham1.A)/1000.0,D=jxn.asarray(Ham1.D)/1000.0,Hpp=jxn.asarray(Ham1.Hpp)/1.0,Q=jxn.asarray(Ham1.Q)/1000.0,
                     Bk2=jxn.asarray(Ham1.Bk2)/1000.0,Bk4=jxn.asarray(Ham1.Bk4)/1000.0,Bk6=jxn.asarray(Ham1.Bk6)/1000.0)
        Ham2=Ham2.replace(A=jxn.asarray(Ham2.A)/1000.0,D=jxn.asarray(Ham2.D)/1000.0,Hpp=jxn.asarray(Ham2.Hpp)/1.0,Q=jxn.asarray(Ham2.Q)/1000.0,
                     Bk2=jxn.asarray(Ham2.Bk2)/1000.0,Bk4=jxn.asarray(Ham2.Bk4)/1000.0,Bk6=jxn.asarray(Ham2.Bk6)/1000.0)
        etas=Ham1.eta
        etas=jxn.where(Ham1.Hpp[1]==0.0,0.0,etas)
        etas=jxn.where(Ham1.Hpp[0]==0.0,1.0,etas)
        Ham1=Ham1.replace(eta=etas)
        Ham2=Ham2.replace(eta=etas)
        dim=int(2*Ham1.S+1)*int(2*Ham1.I+1)*int(2*Ham1.L+1)*int(2*Ham2.S+1)*int(2*Ham2.I+1)*int(2*Ham2.L+1)
        dim1=int(2*Ham1.S+1)*int(2*Ham1.I+1)*int(2*Ham1.L+1)
        dim2=int(2*Ham2.S+1)*int(2*Ham2.I+1)*int(2*Ham2.L+1)
        ndir=jxn.array(Exp1.Fdirection,dtype=jxn.float32)
        ndir=ndir/jxn.linalg.norm(ndir)
        nx,ny,nz=ndir[0],ndir[1],ndir[2]
        mdir=jxn.array(Exp1.Mwdirection,dtype=jxn.float32)
        mdir=mdir/jxn.linalg.norm(mdir)
        mx,my,mz=mdir[0],mdir[1],mdir[2]
        Ham1=Jchaframe(Ham1,Exp1)
        Ham2=Jchaframe(Ham2,Exp2)
        sx1,sy1,sz1=JPauli(Ham1.S)
        ix1,iy1,iz1=JPauli(Ham1.I)
        sx2,sy2,sz2=JPauli(Ham2.S)
        ix2,iy2,iz2=JPauli(Ham2.I)
        E=Exp1.Freq
        espac1=jxn.linspace(frange0,Exp1.Frange[1],500)
        beta=(scic.physical_constants["Bohr magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
        betan=(scic.physical_constants["nuclear magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
        h1=np.zeros((dim,dim),dtype='complex')
        hzex=0
        hzey=0
        hzez=0
        hzex1=jxn.asarray(beta*JHze(sx1,sy1,sz1,Ham1.g,[1,0,0],dim1),dtype=complex)
        hzey1=jxn.asarray(beta*JHze(sx1,sy1,sz1,Ham1.g,[0,1,0],dim1),dtype=complex)
        hzez1=jxn.asarray(beta*JHze(sx1,sy1,sz1,Ham1.g,[0,0,1],dim1),dtype=complex)
        hzex2=jxn.asarray(beta*JHze(sx2,sy2,sz2,Ham2.g,[1,0,0],dim2),dtype=complex)
        hzey2=jxn.asarray(beta*JHze(sx2,sy2,sz2,Ham2.g,[0,1,0],dim2),dtype=complex)
        hzez2=jxn.asarray(beta*JHze(sx2,sy2,sz2,Ham2.g,[0,0,1],dim2),dtype=complex)

        hzex+=jxn.kron(hzex1,jxn.eye(dim2,dtype=complex))+jxn.kron(jxn.eye(dim1,dtype=complex),hzex2)
        hzey+=jxn.kron(hzey1,jxn.eye(dim2,dtype=complex))+jxn.kron(jxn.eye(dim1,dtype=complex),hzey2)
        hzez+=jxn.kron(hzez1,jxn.eye(dim2,dtype=complex))+jxn.kron(jxn.eye(dim1,dtype=complex),hzez2)
        if Ham1.S>=1:
            h1=h1+JStevensO(sx1,sy1,sz1,Ham1.S,Ham1,dim)
        if Ham2.S>=1:
            h1=h1+JStevensO(sx2,sy2,sz2,Ham2.S,Ham2,dim)
        if Ham1.L!=0:
            h1=h1+JLorbit(sx1,sy1,sz1,Ham1.lc,dim,Ham1.L)
        if Ham2.L!=0:
            h1=h1+JLorbit(sx2,sy2,sz2,Ham2.lc,dim,Ham2.L)
        if Ham1.I!=0:
            h1=h1+JHfi(sx1,sy1,sz1,ix1,iy1,iz1,Ham1.A,dim)
            h1=h1+JQii(ix1,iy1,iz1,Ham1.Q,dim)
            gnk=gnfactor(Nucl1)
            nhzex=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[1,0,0]),dtype=complex)
            nhzey=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[0,1,0]),dtype=complex)
            nhzez=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[0,0,1]),dtype=complex)
            hzex-=jxn.kron(nhzex,jxn.eye(dim2,dtype=complex))
            hzey-=jxn.kron(nhzey,jxn.eye(dim2,dtype=complex))
            hzez-=jxn.kron(nhzez,jxn.eye(dim2,dtype=complex))
        if Ham2.I!=0:
            h1=h1+JHfi(sx2,sy2,sz2,ix2,iy2,iz2,Ham2.A,dim)
            h1=h1+JQii(ix2,iy2,iz2,Ham2.Q,dim)
            gnk=gnfactor(Nucl2)
            nhzex=jxn.asarray(betan*JNhze(Ham2.I,ix2,iy2,iz2,dim2,gnk,[1,0,0]),dtype=complex)
            nhzey=jxn.asarray(betan*JNhze(Ham2.I,ix2,iy2,iz2,dim2,gnk,[0,1,0]),dtype=complex)
            nhzez=jxn.asarray(betan*JNhze(Ham2.I,ix2,iy2,iz2,dim2,gnk,[0,0,1]),dtype=complex)
            hzex-=jxn.kron(jxn.eye(dim1,dtype=complex),nhzex)
            hzey-=jxn.kron(jxn.eye(dim1,dtype=complex),nhzey)
            hzez-=jxn.kron(jxn.eye(dim1,dtype=complex),nhzez)
        if jxn.any(maham.A1_2):
            Aref=jxn.asarray(Ham.A1_2)/1000.0
            Aref=Aref*jxn.eye(3)
            h1+=Hfi(sx1,sy1,sz1,ix2,iy2,iz2,Aref,dim)
        if jxn.any(maham.A2_1):
            Aref=jxn.asarray(maham.A2_1)/1000.0
            Aref=Aref*jxn.eye(3)
            h1+=Hfi(sx2,sy2,sz2,ix1,iy1,iz1,Aref,dim)
        if jxn.any(maham.X1_2):
            Xref=jxn.asarray(maham.X1_2)/1000.0
            Xref=Xref*jxn.eye(3)
            h1+=JIee(sx1,sy1,sz1,sx2,sy2,sz2,Xref,dim)
        h1=jxn.asarray(h1,dtype=complex)
        #For the total magnetic moment of the system
        stodx=np.zeros((dim,dim),dtype='complex')
        stody=np.zeros((dim,dim),dtype='complex')
        stodz=np.zeros((dim,dim),dtype='complex')
        dimwns1=int((2*Ham1.I+1)*(2*Ham1.L+1))
        dimwns2=int((2*Ham2.I+1)*(2*Ham2.L+1))
        if Ham1.S>0:
            sx1r=jxn.kron(sx1,jxn.eye(dimwns1,dtype=complex))
            sy1r=jxn.kron(sy1,jxn.eye(dimwns1,dtype=complex))
            sz1r=jxn.kron(sz1,jxn.eye(dimwns1,dtype=complex))
            stodx+=jxn.kron(sx1r,jxn.eye(dim2,dtype=complex))
            stody+=jxn.kron(sy1r,jxn.eye(dim2,dtype=complex))
            stodz+=jxn.kron(sz1r,jxn.eye(dim2,dtype=complex))
        if Ham2.S>0:
            sx2r=jxn.kron(sx2,jxn.eye(dimwns2,dtype=complex))
            sy2r=jxn.kron(sy2,jxn.eye(dimwns2,dtype=complex))
            sz2r=jxn.kron(sz2,jxn.eye(dimwns2,dtype=complex))
            stodx+=jxn.kron(jxn.eye(dim1,dtype=complex),sx2r)
            stody+=jxn.kron(jxn.eye(dim1,dtype=complex),sy2r)
            stodz+=jxn.kron(jxn.eye(dim1,dtype=complex),sz2r)
        hze=nx*hzex+ny*hzey+nz*hzez
        Blist=jxn.linspace(Exp1.Frange[0],Exp1.Frange[1],Exp1.Points)
        def Jdiagop(B):
            h5=h1+B*hze
            Elist,Vlist=jxn.linalg.eigh(h5)
            return Elist,Vlist
        Elist,Vlist=jx.vmap(Jdiagop)(Blist)
        spc=jxn.zeros(Exp1.Points)
        dim=Elist.shape[1]
        iidx,jidx=jxn.triu_indices(dim,k=1)
        Vdag=Vlist.conj().swapaxes(-1,-2)
        Tx=Vdag@stodx@Vlist
        Ty=Vdag@stody@Vlist
        Tz=Vdag@stodz@Vlist
        #Interpolate for intensities
        Txij=Tx[:,iidx,jidx]
        Tyij=Ty[:,iidx,jidx]
        Tzij=Tz[:,iidx,jidx]
        #Probability definition and interpolation
        M2=jxn.real(Txij*jxn.conj(Txij))+jxn.real(Tyij*jxn.conj(Tyij))+jxn.real(Tzij*jxn.conj(Tzij))
        Mn=mx*Txij+my*Tyij+mz*Tzij
        prob=jxn.abs(Mn)**2
        #Frecuency to field
        h22=jxn.real(Vdag@hze@Vlist)
        h2diag=jxn.diagonal(h22,axis1=1,axis2=2)
        dert=h2diag[:,iidx]
        izrt=h2diag[:,jidx]
        gma=jxn.abs(izrt-dert)
        gma=jxn.where(gma<1e-4,1e-4,gma)
        gema=1.0/gma
        #Boltzmann distribution
        conver=1e9*scc.h
        Ej=Elist*conver
        Temp=jxn.where(Exp1.Temperature<=0.0,1.0,Exp1.Temperature)
        beta=1.0/(scc.k*Temp)
        Emin=jxn.min(Ej,axis=-1,keepdims=True)
        boltz=jxn.exp(-beta*(Ej-Emin))
        Z=jxn.sum(boltz,axis=-1,keepdims=True)
        boltz=boltz/Z
        popui=boltz[:,iidx]
        popuj=boltz[:,jidx]
        boltzm=popui-popuj
        intensy=prob*gema*boltzm
        deltaE=jxn.abs(Elist[:,jidx]-Elist[:,iidx])
        dfe=deltaE-Exp1.Freq
        hppg=Ham1.Hpp[0]*gma
        hppl=Ham1.Hpp[1]*gma
        hppg=jxn.where(hppg==0.0,1e-10,hppg)
        hpp=jxn.where(hppl==0.0,1e-10,hppl)
        gammag=hppg*jxn.sqrt(jxn.log(2.0)/2.0)
        gbs=jxn.exp(-jxn.log(2.0)*(dfe/gammag)**2)
        gammal=hppl*jxn.sqrt(3.0)
        gamma2l=gammal/2.0
        lbs=(gamma2l**2)/(dfe**2+gamma2l**2)
        voigt=(lbs*etas)+(gbs*(1.0-etas))
        spcint=jxn.sum(intensy*voigt,axis=1)
        dB=Blist[1]-Blist[0]
        spc=jxn.gradient(spcint,dB)
        fielde=Blist
        specs=spc
    return fielde,specs
    
def Briggs(Hamer,Exp,Vary,expr,maximal=2000,eps=1e-11,mode='p'):
    '''
    Fitting function for the experimental data using the ADAM Algorithm. Uses the *Optax* (part of the Deepmind proyect) ADAM algorithm implementation with a learning rate of 0.1. The parameters are changed and evaluated using a normalized sigmoid function in the range from the *Vary* container. 
    
    It's recommended to use the function in VS code, Jupyter or Colab, because the process can be stop at any moment using the stop process button of the notebook.
    
    Parameters
    ----------
    Hamer: Class
        Container for the hamiltonian parameters.
    Exp : Class
        Container for the experimental conditions.
    Vary : Class
        Container for the range and parameters to vary.

    exper : np.array
        Experimental spectrum data to fit.
    maximal : int
        Max. number of iterations to evalue the function.
    eps : float
        Tolerance value for the error. Default is 1e-11
    mode : str
        Defines the sample type, 'p' for powder and 'c' for monocristal.
    
    Returns
    -------
    espc : np.array
        Best adjusted spectrum.
        
    Example
    -------
    
    
    '''
    if isinstance(Hamer,JHval):
      Ham=deepcopy(Hamer)
      class StaticHam:
        def replace(self,**kwargs):
          import copy
          nham=copy.copy(self)
          for k,v in kwargs.items():
            setattr(nham,k,v)
          return nham
      class StaticExp:
          pass
      if mode=='p':
          iwas,jwas,kwas,weight,hulk=Delaunay(Exp)
          iwas,jwas,kwas,weight,hulk=jxn.array(iwas),jxn.array(jwas),jxn.array(kwas),jxn.array(weight),jxn.array(hulk)
      SHam=StaticHam()
      SHam.S=float(Ham.S)
      SHam.I=float(Ham.I)
      SHam.L=float(Ham.L)
      SHam.lc=float(Ham.lc)
      SHam.Bk2=np.array(Ham.Bk2).tolist()
      SHam.Bk4=np.array(Ham.Bk4).tolist()
      SHam.Bk6=np.array(Ham.Bk6).tolist()
      SHam.Hpp=np.array(Ham.Hpp)
      SHam.eta=float(Ham.eta)
      SHam.g=np.array(Ham.g)
      SHam.A=np.array(Ham.A)
      SHam.Q=np.array(Ham.Q)
      SHam.D=np.array(Ham.D)
      Vary.Hpp=np.array(Vary.Hpp)
      dExp=StaticExp()
      dExp.Points=int(Exp.Points)
      dExp.Freq=float(Exp.Freq)
      dExp.Temperature=float(Exp.Temperature)
      dExp.Frange=np.array(Exp.Frange)
      dExp.Sampleframe=Exp.Sampleframe
      dExp.Molframe=Exp.Molframe
      dExp.gframe=Exp.gframe
      dExp.Aframe=Exp.Aframe
      dExp.Qframe=Exp.Qframe
      dExp.Dframe=Exp.Dframe
      dExp.Fdirection=Exp.Fdirection
      dExp.Mwdirection=Exp.Mwdirection
      def initpara(Ham,Vara):
          param={}
          def safelog(val,under,over):
              div=jxn.where(over==under,1e-10,over-under)
              frat=(val-under)/div
              safe=jxn.clip(frat,1e-4,1.0-1e-4)
              T=4.0
              return jsp.logit(safe)*T
          if Vara.g!=0.0:
              param['gx']=safelog(Ham.g[0],Vara.g[0],Vara.g[1])
              param['gy']=safelog(Ham.g[1],Vara.g[2],Vara.g[3])
              param['gz']=safelog(Ham.g[2],Vara.g[4],Vara.g[5])
          if Vara.A!=0.0:
              param['Ax']=safelog(Ham.A[0],Vara.A[0],Vara.A[1])
              param['Ay']=safelog(Ham.A[1],Vara.A[2],Vara.A[3])
              param['Az']=safelog(Ham.A[2],Vara.A[4],Vara.A[5])
          if Vara.D!=0.0:
              param['D']=safelog(Ham.D[0],Vara.D[0],Vara.D[1])
              param['E']=safelog(Ham.D[1],Vara.D[2],Vara.D[3])
          if Vara.Q!=0.0:
              param['Qx']=safelog(Ham.Q[0],Vara.Q[0],Vara.Q[1])
              param['Qy']=safelog(Ham.Q[1],Vara.Q[2],Vara.Q[3])
              param['Qz']=safelog(Ham.Q[2],Vara.Q[4],Vara.Q[5])
          if jxn.any(Vara.Hpp):
              param['Hpp1']=safelog(Ham.Hpp[0],Vara.Hpp[0],Vara.Hpp[1])
              param['Hpp2']=safelog(Ham.Hpp[1],Vara.Hpp[2],Vara.Hpp[3])
          return param
      param=initpara(Ham,Vary)
      optimus=optax.adam
      optimus=optax.chain(optax.clip_by_global_norm(1.0),optax.adam(learning_rate=0.1))#,optax.zero_nans(),optax.adam(learning_rate=0.1))
      state=optimus.init(param)

      def Errorcost(params,exper):
          T=4.0
          if 'gx' in params.keys():
              gx=Vary.g[0]+(Vary.g[1]-Vary.g[0])*jnn.sigmoid(params['gx']/T)
              gy=Vary.g[2]+(Vary.g[3]-Vary.g[2])*jnn.sigmoid(params['gy']/T)
              gz=Vary.g[4]+(Vary.g[5]-Vary.g[4])*jnn.sigmoid(params['gz']/T)
              gg=jxn.array([gx,gy,gz])
          else:
              gg=SHam.g
          if 'Ax' in params.keys():
              Ax=Vary.A[0]+(Vary.A[1]-Vary.A[0])*jnn.sigmoid(params['Ax']/T)
              Ay=Vary.A[2]+(Vary.A[3]-Vary.A[2])*jnn.sigmoid(params['Ay']/T)
              Az=Vary.A[4]+(Vary.A[5]-Vary.A[4])*jnn.sigmoid(params['Az']/T)
              AA=jxn.array([Ax,Ay,Az])
          else:
              AA=SHam.A
          if 'D' in params.keys():
              Dx=Vary.D[0]+(Vary.D[1]-Vary.D[0])*jnn.sigmoid(params['D']/T)
              Ey=Vary.D[2]+(Vary.D[3]-Vary.D[2])*jnn.sigmoid(params['E']/T)
              DD=jxn.array([Dx,Ey])
          else:
              DD=SHam.D
          if 'Qx' in params.keys():
              Qx=Vary.Q[0]+(Vary.Q[1]-Vary.Q[0])*jnn.sigmoid(params['Qx']/T)
              Qy=Vary.Q[2]+(Vary.Q[3]-Vary.Q[2])*jnn.sigmoid(params['Qy']/T)
              Qz=Vary.Q[4]+(Vary.Q[5]-Vary.Q[4])*jnn.sigmoid(params['Qz']/T)
              QQ=jxn.array([Qx,Qy,Qz])
          else:
              QQ=SHam.Q
          if 'Hpp1' in params.keys():
              Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(params['Hpp1']/T)
              Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(params['Hpp2']/T)
              HHpp=jxn.array([Hppx,Hppy])
          else:
              HHpp=SHam.Hpp
          SHam.g=gg
          SHam.A=AA
          SHam.D=DD
          SHam.Q=QQ
          SHam.Hpp=HHpp
          Hame=SHam.replace(g=gg,A=AA,D=DD,Q=QQ,Hpp=HHpp)
          if mode=='p':
              _,simul=JCalpowder(Hame,dExp,iwas,jwas,kwas,weight,hulk)
          elif mode=='c':
              _,simul=Calresonant(Hame,dExp)
          maxl=jxn.max(jxn.abs(simul))
          maxl=jxn.where(maxl==0.0,1.0,maxl)
          simul=simul/maxl
          maxe=jxn.max(jxn.abs(exper))
          maxe=jxn.where(maxe==0.0,1.0,maxe)
          exper=exper/maxe
          return jxn.mean((simul-exper)**2)

      Degrad=jx.value_and_grad(Errorcost,argnums=0)
      step=0
      T=4.0

      @jx.jit
      def updatenext(parats,current,exper):
          error,grad=Degrad(parats,exper)
          next,state=optimus.update(grad,current,parats)
          param=optax.apply_updates(parats,next)
          return param,state,error
      try:
          while step<(maximal):
              param,state,error=updatenext(param,state,expr)
              if error<eps:
                  break
              if step%10==0:
                  print(f"Step {step+1:3d} | Error: {error:.5e} |")
                  if 'gx' in param.keys():
                      gxf=Vary.g[0]+(Vary.g[1]-Vary.g[0])*jnn.sigmoid(param['gx']/T)
                      gyf=Vary.g[2]+(Vary.g[3]-Vary.g[2])*jnn.sigmoid(param['gy']/T)
                      gzf=Vary.g[4]+(Vary.g[5]-Vary.g[4])*jnn.sigmoid(param['gz']/T)
                      print(f"| gx: {gxf:.4f} | gy: {gyf:.4f} | gz: {gzf:.4f} |")
                  if 'Ax' in param.keys():
                      Axf=Vary.A[0]+(Vary.A[1]-Vary.A[0])*jnn.sigmoid(param['Ax']/T)
                      Ayf=Vary.A[2]+(Vary.A[3]-Vary.A[2])*jnn.sigmoid(param['Ay']/T)
                      Azf=Vary.A[4]+(Vary.A[5]-Vary.A[4])*jnn.sigmoid(param['Az']/T)
                      print(f"| Ax: {Axf:.4f} | Ay: {Ayf:.4f} | Az: {Azf:.4f} |")
                  if 'D' in param.keys():
                      Dx=Vary.D[0]+(Vary.D[1]-Vary.D[0])*jnn.sigmoid(param['D']/T)
                      Ey=Vary.D[2]+(Vary.D[3]-Vary.D[2])*jnn.sigmoid(param['E']/T)
                      print(f"| D: {Dx:.1f} | E: {Ey:.1f} |")
                  if 'Qx' in param.keys():
                      Qxf=Vary.Q[0]+(Vary.Q[1]-Vary.Q[0])*jnn.sigmoid(param['Qx']/T)
                      Qyf=Vary.Q[2]+(Vary.Q[3]-Vary.Q[2])*jnn.sigmoid(param['Qy']/T)
                      Qzf=Vary.Q[4]+(Vary.Q[5]-Vary.Q[4])*jnn.sigmoid(param['Qz']/T)
                      print(f"| Qx: {Qxf:.4f} | Qy: {Qyf:.4f} | Qz: {Qzf:.4f} |")
                  if 'Hpp1' in param.keys():
                      Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(param['Hpp1']/T)
                      Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(param['Hpp2']/T)
                      print(f"| Hppg: {Hppx:.1f} | Hppl: {Hppy:.1f} |")
              step+=1
      except KeyboardInterrupt:
          print("\n"+"="*50)
          print(f"Process stopped at iteration:{step}")
          print("="*50)
          if 'gx' in param.keys():
              gx=Vary.g[0]+(Vary.g[1]-Vary.g[0])*jnn.sigmoid(param['gx']/T)
              gy=Vary.g[2]+(Vary.g[3]-Vary.g[2])*jnn.sigmoid(param['gy']/T)
              gz=Vary.g[4]+(Vary.g[5]-Vary.g[4])*jnn.sigmoid(param['gz']/T)
              gg=jxn.array([gx,gy,gz])
          else:
              gg=Ham.g
          if 'Ax' in param.keys():
              Ax=Vary.A[0]+(Vary.A[1]-Vary.A[0])*jnn.sigmoid(param['Ax']/T)
              Ay=Vary.A[2]+(Vary.A[3]-Vary.A[2])*jnn.sigmoid(param['Ay']/T)
              Az=Vary.A[4]+(Vary.A[5]-Vary.A[4])*jnn.sigmoid(param['Az']/T)
              AA=jxn.array([Ax,Ay,Az])
          else:
              AA=Ham.A
          if 'D' in param.keys():
              Dx=Vary.D[0]+(Vary.D[1]-Vary.D[0])*jnn.sigmoid(param['D']/T)
              Ey=Vary.D[2]+(Vary.D[3]-Vary.D[2])*jnn.sigmoid(param['E']/T)
              DD=jxn.array([Dx,Ey])
          else:
              DD=Ham.D
          if 'Qx' in param.keys():
              Qx=Vary.Q[0]+(Vary.Q[1]-Vary.Q[0])*jnn.sigmoid(param['Qx']/T)
              Qy=Vary.Q[2]+(Vary.Q[3]-Vary.Q[2])*jnn.sigmoid(param['Qy']/T)
              Qz=Vary.Q[4]+(Vary.Q[5]-Vary.Q[4])*jnn.sigmoid(param['Qz']/T)
              QQ=jxn.array([Qx,Qy,Qz])
          else:
              QQ=Ham.Q
          if 'Hpp1' in param.keys():
              Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(param['Hpp1']/T)
              Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(param['Hpp2']/T)
              HHpp=jxn.array([Hppx,Hppy])
          else:
              HHpp=Ham.Hpp
          Hat=Ham.replace(g=gg,A=AA,D=DD,Q=QQ,Hpp=HHpp)
          print(f"Step {step+1:3d} | Error: {error:.5e} |")
          if 'gx' in param.keys():
              gxf=Vary.g[0]+(Vary.g[1]-Vary.g[0])*jnn.sigmoid(param['gx']/T)
              gyf=Vary.g[2]+(Vary.g[3]-Vary.g[2])*jnn.sigmoid(param['gy']/T)
              gzf=Vary.g[4]+(Vary.g[5]-Vary.g[4])*jnn.sigmoid(param['gz']/T)
              print(f"| gx: {gxf:.4f} | gy: {gyf:.4f} | gz: {gzf:.4f} |")
          if 'Ax' in param.keys():
              Axf=Vary.A[0]+(Vary.A[1]-Vary.A[0])*jnn.sigmoid(param['Ax']/T)
              Ayf=Vary.A[2]+(Vary.A[3]-Vary.A[2])*jnn.sigmoid(param['Ay']/T)
              Azf=Vary.A[4]+(Vary.A[5]-Vary.A[4])*jnn.sigmoid(param['Az']/T)
              print(f"| Ax: {Axf:.4f} | Ay: {Ayf:.4f} | Az: {Azf:.4f} |")
          if 'D' in param.keys():
              Dx=Vary.D[0]+(Vary.D[1]-Vary.D[0])*jnn.sigmoid(param['D']/T)
              Ey=Vary.D[2]+(Vary.D[3]-Vary.D[2])*jnn.sigmoid(param['E']/T)
              print(f"| D: {Dx:.1f} | E: {Ey:.1f} |")
          if 'Qx' in param.keys():
              Qxf=Vary.Q[0]+(Vary.Q[1]-Vary.Q[0])*jnn.sigmoid(param['Qx']/T)
              Qyf=Vary.Q[2]+(Vary.Q[3]-Vary.Q[2])*jnn.sigmoid(param['Qy']/T)
              Qzf=Vary.Q[4]+(Vary.Q[5]-Vary.Q[4])*jnn.sigmoid(param['Qz']/T)
              print(f"| Qx: {QXf:.4f} | Qy: {Qyf:.4f} | Qz: {Qzf:.4f} |")
          if 'Hpp1' in param.keys():
              Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(param['Hpp1']/T)
              Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(param['Hpp2']/T)
              print(f"| Hppg: {Hppx:.1f} | Hppl: {Hppy:.1f} |")

          if mode=='p':
              Blis,espc=JCalpowder(Hat,dExp,iwas,jwas,kwas,weight,hulk)
              plt.figure(figsize=(8,6))
              plt.plot(Blis,expr,label='Data')
              plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
              formatter=EngFormatter(sep='') 
              plt.gca().yaxis.set_major_formatter(formatter)
              plt.xlabel('Field [mT]')
              plt.ylabel('Counts [A. U.]')
              plt.grid()
              plt.legend()
              plt.title('EPR Spectrum')
              plt.show()
              return espc
          elif mode=='c':
              Blis,espc=Calresonant(Hat,dExp)
              plt.figure(figsize=(8,6))
              plt.plot(Blis,expr,label='Data')
              formatter=EngFormatter(sep='') 
              plt.gca().yaxis.set_major_formatter(formatter)
              plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
              plt.xlabel('Field [mT]')
              plt.ylabel('Counts [A. U.]')
              plt.grid()
              plt.legend()
              plt.title('EPR Spectrum')
              plt.show()
              return espc
      if 'gx' in param.keys():
          gx=Vary.g[0]+(Vary.g[1]-Vary.g[0])*jnn.sigmoid(param['gx']/T)
          gy=Vary.g[2]+(Vary.g[3]-Vary.g[2])*jnn.sigmoid(param['gy']/T)
          gz=Vary.g[4]+(Vary.g[5]-Vary.g[4])*jnn.sigmoid(param['gz']/T)
          gg=jxn.array([gx,gy,gz])
      else:
          gg=Ham.g
      if 'Ax' in param.keys():
          Ax=Vary.A[0]+(Vary.A[1]-Vary.A[0])*jnn.sigmoid(param['Ax']/T)
          Ay=Vary.A[2]+(Vary.A[3]-Vary.A[2])*jnn.sigmoid(param['Ay']/T)
          Az=Vary.A[4]+(Vary.A[5]-Vary.A[4])*jnn.sigmoid(param['Az']/T)
          AA=jxn.array([Ax,Ay,Az])
      else:
          AA=Ham.A
      if 'D' in param.keys():
          Dx=Vary.D[0]+(Vary.D[1]-Vary.D[0])*jnn.sigmoid(param['D']/T)
          Ey=Vary.D[2]+(Vary.D[3]-Vary.D[2])*jnn.sigmoid(param['E']/T)
          DD=jxn.array([Dx,Ey])
      else:
          DD=Ham.D
      if 'Qx' in param.keys():
          Qx=Vary.Q[0]+(Vary.Q[1]-Vary.Q[0])*jnn.sigmoid(param['Qx']/T)
          Qy=Vary.Q[2]+(Vary.Q[3]-Vary.Q[2])*jnn.sigmoid(param['Qy']/T)
          Qz=Vary.Q[4]+(Vary.Q[5]-Vary.Q[4])*jnn.sigmoid(param['Qz']/T)
          QQ=jxn.array([Qx,Qy,Qz])
      else:
          QQ=Ham.Q
      if 'Hpp1' in param.keys():
          Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(param['Hpp1']/T)
          Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(param['Hpp2']/T)
          HHpp=jxn.array([Hppx,Hppy])
      else:
          HHpp=Ham.Hpp
      Hat=Ham.replace(g=gg,A=AA,D=DD,Q=QQ,Hpp=HHpp)
      print(f"Step {step+1:3d} | Error: {error:.5e} |")
      if 'gx' in param.keys():
          gxf=Vary.g[0]+(Vary.g[1]-Vary.g[0])*jnn.sigmoid(param['gx']/T)
          gyf=Vary.g[2]+(Vary.g[3]-Vary.g[2])*jnn.sigmoid(param['gy']/T)
          gzf=Vary.g[4]+(Vary.g[5]-Vary.g[4])*jnn.sigmoid(param['gz']/T)
          print(f"| gx: {gxf:.4f} | gy: {gyf:.4f} | gz: {gzf:.4f} |")
      if 'Ax' in param.keys():
          Axf=Vary.A[0]+(Vary.A[1]-Vary.A[0])*jnn.sigmoid(param['Ax']/T)
          Ayf=Vary.A[2]+(Vary.A[3]-Vary.A[2])*jnn.sigmoid(param['Ay']/T)
          Azf=Vary.A[4]+(Vary.A[5]-Vary.A[4])*jnn.sigmoid(param['Az']/T)
          print(f"| Ax: {Axf:.4f} | Ay: {Ayf:.4f} | Az: {Azf:.4f} |")
      if 'D' in param.keys():
          Dx=Vary.D[0]+(Vary.D[1]-Vary.D[0])*jnn.sigmoid(param['D']/T)
          Ey=Vary.D[2]+(Vary.D[3]-Vary.D[2])*jnn.sigmoid(param['E']/T)
          print(f"| D: {Dx:.1f} | E: {Ey:.1f} |")
      if 'Qx' in param.keys():
          Qxf=Vary.Q[0]+(Vary.Q[1]-Vary.Q[0])*jnn.sigmoid(param['Qx']/T)
          Qyf=Vary.Q[2]+(Vary.Q[3]-Vary.Q[2])*jnn.sigmoid(param['Qy']/T)
          Qzf=Vary.Q[4]+(Vary.Q[5]-Vary.Q[4])*jnn.sigmoid(param['Qz']/T)
          print(f"| Qx: {QXf:.4f} | Qy: {Qyf:.4f} | Qz: {Qzf:.4f} |")
      if 'Hpp1' in param.keys():
          Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(param['Hpp1']/T)
          Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(param['Hpp2']/T)
          print(f"| Hppg: {Hppx:.1f} | Hppl: {Hppy:.1f} |")
      if mode=='p':
          Blis,espc=JCalpowder(Hat,dExp,iwas,jwas,kwas,weight,hulk)
          plt.figure(figsize=(8,6))
          plt.plot(Blis,expr,label='Data')
          formatter=EngFormatter(sep='') 
          plt.gca().yaxis.set_major_formatter(formatter)
          plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
          plt.xlabel('Field [mT]')
          plt.ylabel('Counts [A. U.]')
          plt.grid()
          plt.legend()
          plt.title('EPR Spectrum')
          plt.show()
          return espc
      elif mode=='c':
          Blis,espc=Calresonant(Hat,dExp)
          plt.figure(figsize=(8,6))
          plt.plot(Blis,expr,label='Data')
          formatter=EngFormatter(sep='') 
          plt.gca().yaxis.set_major_formatter(formatter)
          plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
          plt.xlabel('Field [mT]')
          plt.ylabel('Counts [A. U.]')
          plt.grid()
          plt.legend()
          plt.title('EPR Spectrum')
          plt.show()
          return espc
    else:
      Ham=deepcopy(Hamer)
      class StaticHam:
        def replace(self,**kwargs):
          import copy
          nham=copy.copy(self)
          for k,v in kwargs.items():
            setattr(nham,k,v)
          return nham
      class StaticExp:
          pass
      SHam=StaticHam()
      SHam.S1,SHam.S2=float(Ham.S1),float(Ham.S2)
      SHam.I1,SHam.I2=float(Ham.I1),float(Ham.I2)
      SHam.L1,SHam.L2=float(Ham.L1),float(Ham.L2)
      SHam.lc1,SHam.lc2=float(Ham.lc1),float(Ham.lc2)
      SHam.Bk2=np.array(Ham.Bk2).tolist()
      SHam.Bk4=np.array(Ham.Bk4).tolist()
      SHam.Bk6=np.array(Ham.Bk6).tolist()
      SHam.Hpp=np.array(Ham.Hpp)
      SHam.eta=float(Ham.eta)
      SHam.g1,SHam.g2=np.array(Ham.g1),np.array(Ham.g2)
      SHam.A1,SHam.A2=np.array(Ham.A1),np.array(Ham.A2)
      SHam.Q1,SHam.Q2=np.array(Ham.Q1),np.array(Ham.Q2)
      SHam.D1,SHam.D2=np.array(Ham.D1),np.array(Ham.D2)
      SHam.A1_2,SHam.A2_1,SHam.X1_2=np.array(Ham.A1_2),np.array(Ham.A2_1),np.array(Ham.X1_2)
      Vary.Hpp=np.array(Vary.Hpp)
      dExp=StaticExp()
      dExp.Points=int(Exp.Points)
      dExp.Freq=float(Exp.Freq)
      dExp.Temperature=float(Exp.Temperature)
      dExp.Frange=np.array(Exp.Frange)
      dExp.Sampleframe1,dExp.Sampleframe2=Exp.Sampleframe1,Exp.Sampleframe2
      dExp.Molframe1,dExp.Molframe2=Exp.Molframe1,Exp.Molframe2
      dExp.gframe1,dExp.gframe2=Exp.gframe1,Exp.gframe2
      dExp.Aframe1,dExp.Aframe2=Exp.Aframe1,Exp.Aframe2
      dExp.Qframe1,dExp.Qframe2=Exp.Qframe1,Exp.Qframe2
      dExp.Dframe1,dExp.Dframe2=Exp.Dframe1,Exp.Dframe2
      dExp.Fdirection=Exp.Fdirection
      dExp.Mwdirection=Exp.Mwdirection
      def initpara(Ham,Vara):
          param={}
          def safelog(val,under,over):
              div=jxn.where(over==under,1e-10,over-under)
              frat=(val-under)/div
              safe=jxn.clip(frat,1e-4,1.0-1e-4)
              T=4.0
              return jsp.logit(safe)*T
          if Vara.g1!=0.0:
              param['gx1']=safelog(Ham.g1[0],Vara.g1[0],Vara.g1[1])
              param['gy1']=safelog(Ham.g1[1],Vara.g1[2],Vara.g1[3])
              param['gz1']=safelog(Ham.g1[2],Vara.g1[4],Vara.g1[5])
          if Vara.g2!=0.0:
              param['gx2']=safelog(Ham.g2[0],Vara.g2[0],Vara.g2[1])
              param['gy2']=safelog(Ham.g2[1],Vara.g2[2],Vara.g2[3])
              param['gz2']=safelog(Ham.g2[2],Vara.g2[4],Vara.g2[5])
          if Vara.A1!=0.0:
              param['Ax1']=safelog(Ham.A1[0],Vara.A1[0],Vara.A1[1])
              param['Ay1']=safelog(Ham.A1[1],Vara.A1[2],Vara.A1[3])
              param['Az1']=safelog(Ham.A1[2],Vara.A1[4],Vara.A1[5])
          if Vara.A2!=0.0:
              param['Ax2']=safelog(Ham.A2[0],Vara.A2[0],Vara.A2[1])
              param['Ay2']=safelog(Ham.A2[1],Vara.A2[2],Vara.A2[3])
              param['Az2']=safelog(Ham.A2[2],Vara.A2[4],Vara.A2[5])
          if Vara.D1!=0.0:
              param['D1']=safelog(Ham.D1[0],Vara.D1[0],Vara.D1[1])
              param['E1']=safelog(Ham.D1[1],Vara.D1[2],Vara.D1[3])
          if Vara.D2!=0.0:
              param['D2']=safelog(Ham.D2[0],Vara.D2[0],Vara.D2[1])
              param['E2']=safelog(Ham.D2[1],Vara.D2[2],Vara.D2[3])
          if Vara.Q1!=0.0:
              param['Qx1']=safelog(Ham.Q1[0],Vara.Q1[0],Vara.Q1[1])
              param['Qy1']=safelog(Ham.Q1[1],Vara.Q1[2],Vara.Q1[3])
              param['Qz1']=safelog(Ham.Q1[2],Vara.Q1[4],Vara.Q1[5])
          if Vara.Q2!=0.0:
              param['Qx2']=safelog(Ham.Q2[0],Vara.Q2[0],Vara.Q2[1])
              param['Qy2']=safelog(Ham.Q2[1],Vara.Q2[2],Vara.Q2[3])
              param['Qz2']=safelog(Ham.Q2[2],Vara.Q2[4],Vara.Q2[5])
          if jxn.any(Vara.Hpp):
              param['Hpp1']=safelog(Ham.Hpp[0],Vara.Hpp[0],Vara.Hpp[1])
              param['Hpp2']=safelog(Ham.Hpp[1],Vara.Hpp[2],Vara.Hpp[3])
          return param
      param=initpara(Ham,Vary)
      optimus=optax.adam
      optimus=optax.chain(optax.clip_by_global_norm(1.0),optax.adam(learning_rate=0.1))#,optax.zero_nans(),optax.adam(learning_rate=0.1))
      state=optimus.init(param)
      def Errorcost(params,exper):
          T=4.0
          if 'gx1' in params.keys():
              gx1=Vary.g1[0]+(Vary.g1[1]-Vary.g1[0])*jnn.sigmoid(params['gx1']/T)
              gy1=Vary.g1[2]+(Vary.g1[3]-Vary.g1[2])*jnn.sigmoid(params['gy1']/T)
              gz1=Vary.g1[4]+(Vary.g1[5]-Vary.g1[4])*jnn.sigmoid(params['gz1']/T)
              gg1=jxn.array([gx1,gy1,gz1])
          else:
              gg1=SHam.g1
          if 'gx2' in params.keys():
              gx2=Vary.g2[0]+(Vary.g2[1]-Vary.g2[0])*jnn.sigmoid(params['gx2']/T)
              gy2=Vary.g2[2]+(Vary.g2[3]-Vary.g2[2])*jnn.sigmoid(params['gy2']/T)
              gz2=Vary.g2[4]+(Vary.g2[5]-Vary.g2[4])*jnn.sigmoid(params['gz2']/T)
              gg2=jxn.array([gx2,gy2,gz2])
          else:
              gg2=SHam.g2
          if 'Ax1' in params.keys():
              Ax1=Vary.A1[0]+(Vary.A1[1]-Vary.A1[0])*jnn.sigmoid(params['Ax1']/T)
              Ay1=Vary.A1[2]+(Vary.A1[3]-Vary.A1[2])*jnn.sigmoid(params['Ay1']/T)
              Az1=Vary.A1[4]+(Vary.A1[5]-Vary.A1[4])*jnn.sigmoid(params['Az1']/T)
              AA1=jxn.array([Ax1,Ay1,Az1])
          else:
              AA1=SHam.A1
          if 'Ax2' in params.keys():
              Ax2=Vary.A2[0]+(Vary.A2[1]-Vary.A2[0])*jnn.sigmoid(params['Ax2']/T)
              Ay2=Vary.A2[2]+(Vary.A2[3]-Vary.A2[2])*jnn.sigmoid(params['Ay2']/T)
              Az2=Vary.A2[4]+(Vary.A2[5]-Vary.A2[4])*jnn.sigmoid(params['Az2']/T)
              AA2=jxn.array([Ax2,Ay2,Az2])
          else:
              AA2=SHam.A2
          if 'D1' in params.keys():
              Dx1=Vary.D1[0]+(Vary.D1[1]-Vary.D1[0])*jnn.sigmoid(params['D1']/T)
              Ey1=Vary.D1[2]+(Vary.D1[3]-Vary.D1[2])*jnn.sigmoid(params['E1']/T)
              DD1=jxn.array([Dx1,Ey1])
          else:
              DD1=SHam.D1
          if 'D2' in params.keys():
              Dx2=Vary.D2[0]+(Vary.D2[1]-Vary.D2[0])*jnn.sigmoid(params['D2']/T)
              Ey2=Vary.D2[2]+(Vary.D2[3]-Vary.D2[2])*jnn.sigmoid(params['E2']/T)
              DD2=jxn.array([Dx2,Ey2])
          else:
              DD2=SHam.D2
          if 'Qx1' in params.keys():
              Qx1=Vary.Q1[0]+(Vary.Q1[1]-Vary.Q1[0])*jnn.sigmoid(params['Qx1']/T)
              Qy1=Vary.Q1[2]+(Vary.Q1[3]-Vary.Q1[2])*jnn.sigmoid(params['Qy1']/T)
              Qz1=Vary.Q1[4]+(Vary.Q1[5]-Vary.Q1[4])*jnn.sigmoid(params['Qz1']/T)
              QQ1=jxn.array([Qx1,Qy1,Qz1])
          else:
              QQ1=SHam.Q1
          if 'Qx2' in params.keys():
              Qx2=Vary.Q2[0]+(Vary.Q2[1]-Vary.Q2[0])*jnn.sigmoid(params['Qx2']/T)
              Qy2=Vary.Q2[2]+(Vary.Q2[3]-Vary.Q2[2])*jnn.sigmoid(params['Qy2']/T)
              Qz2=Vary.Q2[4]+(Vary.Q2[5]-Vary.Q2[4])*jnn.sigmoid(params['Qz2']/T)
              QQ2=jxn.array([Qx2,Qy2,Qz2])
          else:
              QQ2=SHam.Q2
          if 'Hpp1' in params.keys():
              Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(params['Hpp1']/T)
              Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(params['Hpp2']/T)
              HHpp=jxn.array([Hppx,Hppy])
          else:
              HHpp=SHam.Hpp
          SHam.g1,SHam.g2=gg1,gg2
          SHam.A1,SHam.A2=AA1,AA2
          SHam.D1,SHam.D2=DD1,DD2
          SHam.Q1,SHam.Q2=QQ1,QQ2
          SHam.Hpp=HHpp
          Hame=SHam.replace(g1=gg1,g2=gg2,A1=AA1,A2=AA2,D1=DD1,D2=DD2,Q1=QQ1,Q2=QQ2,Hpp=HHpp)
          if mode=='p':
              _,simul=JMulpol(Hame,dExp,graph=False)
          elif mode=='c':
              _,simul=JMusic(Hame,dExp,graph=False)
          maxl=jxn.max(jxn.abs(simul))
          maxl=jxn.where(maxl==0.0,1.0,maxl)
          simul=simul/maxl
          maxe=jxn.max(jxn.abs(exper))
          maxe=jxn.where(maxe==0.0,1.0,maxe)
          exper=exper/maxe
          return jxn.mean((simul-exper)**2)
      Degrad=jx.value_and_grad(Errorcost,argnums=0)
      step=0
      T=4.0

      @jx.jit
      def updatenext(parats,current,exper):
          error,grad=Degrad(parats,exper)
          next,state=optimus.update(grad,current,parats)
          param=optax.apply_updates(parats,next)
          return param,state,error
      try:
          while step<(maximal):
              param,state,error=updatenext(param,state,expr)
              if error<eps:
                  break
              if step%10==0:
                  print(f"Step {step+1:3d} | Error: {error:.5e} |")
                  if 'gx1' in param.keys():
                      gxf1=Vary.g1[0]+(Vary.g1[1]-Vary.g1[0])*jnn.sigmoid(param['gx1']/T)
                      gyf1=Vary.g1[2]+(Vary.g1[3]-Vary.g1[2])*jnn.sigmoid(param['gy1']/T)
                      gzf1=Vary.g1[4]+(Vary.g1[5]-Vary.g1[4])*jnn.sigmoid(param['gz1']/T)
                      print(f"| gx1: {gxf1:.4f} | gy1: {gyf1:.4f} | gz1: {gzf1:.4f} |")
                  if 'gx2' in param.keys():
                      gxf2=Vary.g2[0]+(Vary.g2[1]-Vary.g2[0])*jnn.sigmoid(param['gx2']/T)
                      gyf2=Vary.g2[2]+(Vary.g2[3]-Vary.g2[2])*jnn.sigmoid(param['gy2']/T)
                      gzf2=Vary.g2[4]+(Vary.g2[5]-Vary.g2[4])*jnn.sigmoid(param['gz2']/T)
                      print(f"| gx2: {gxf2:.4f} | gy2: {gyf2:.4f} | gz2: {gzf2:.4f} |")
                  if 'Ax1' in param.keys():
                      Axf1=Vary.A1[0]+(Vary.A1[1]-Vary.A1[0])*jnn.sigmoid(param['Ax1']/T)
                      Ayf1=Vary.A1[2]+(Vary.A1[3]-Vary.A1[2])*jnn.sigmoid(param['Ay1']/T)
                      Azf1=Vary.A1[4]+(Vary.A1[5]-Vary.A1[4])*jnn.sigmoid(param['Az1']/T)
                      print(f"| Ax1: {Axf1:.4f} | Ay1: {Ayf1:.4f} | Az1: {Azf1:.4f} |")
                  if 'Ax2' in param.keys():
                      Axf2=Vary.A2[0]+(Vary.A2[1]-Vary.A2[0])*jnn.sigmoid(param['Ax2']/T)
                      Ayf2=Vary.A2[2]+(Vary.A2[3]-Vary.A2[2])*jnn.sigmoid(param['Ay2']/T)
                      Azf2=Vary.A2[4]+(Vary.A2[5]-Vary.A2[4])*jnn.sigmoid(param['Az2']/T)
                      print(f"| Ax2: {Axf2:.4f} | Ay2: {Ayf2:.4f} | Az2: {Azf2:.4f} |")
                  if 'D1' in param.keys():
                      Dx1=Vary.D1[0]+(Vary.D1[1]-Vary.D1[0])*jnn.sigmoid(param['D1']/T)
                      Ey1=Vary.D1[2]+(Vary.D1[3]-Vary.D1[2])*jnn.sigmoid(param['E1']/T)
                      print(f"| D1: {Dx1:.1f} | E1: {Ey1:.1f} |")
                  if 'D2' in param.keys():
                      Dx2=Vary.D2[0]+(Vary.D2[1]-Vary.D2[0])*jnn.sigmoid(param['D2']/T)
                      Ey2=Vary.D2[2]+(Vary.D2[3]-Vary.D2[2])*jnn.sigmoid(param['E2']/T)
                      print(f"| D2: {Dx2:.1f} | E2: {Ey2:.1f} |")
                  if 'Qx1' in param.keys():
                      Qxf1=Vary.Q1[0]+(Vary.Q1[1]-Vary.Q1[0])*jnn.sigmoid(param['Qx1']/T)
                      Qyf1=Vary.Q1[2]+(Vary.Q1[3]-Vary.Q1[2])*jnn.sigmoid(param['Qy1']/T)
                      Qzf1=Vary.Q1[4]+(Vary.Q1[5]-Vary.Q1[4])*jnn.sigmoid(param['Qz1']/T)
                      print(f"| Qx1: {Qxf1:.4f} | Qy1: {Qyf1:.4f} | Qz1: {Qzf1:.4f} |")
                  if 'Qx2' in param.keys():
                      Qxf2=Vary.Q2[0]+(Vary.Q2[1]-Vary.Q2[0])*jnn.sigmoid(param['Qx2']/T)
                      Qyf2=Vary.Q2[2]+(Vary.Q2[3]-Vary.Q2[2])*jnn.sigmoid(param['Qy2']/T)
                      Qzf2=Vary.Q2[4]+(Vary.Q2[5]-Vary.Q2[4])*jnn.sigmoid(param['Qz2']/T)
                      print(f"| Qx2: {Qxf2:.4f} | Qy2: {Qyf2:.4f} | Qz2: {Qzf2:.4f} |")
                  if 'Hpp1' in param.keys():
                      Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(param['Hpp1']/T)
                      Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(param['Hpp2']/T)
                      print(f"| Hppg: {Hppx:.1f} | Hppl: {Hppy:.1f} |")
              step+=1
      except KeyboardInterrupt:
          print("\n"+"="*50)
          print(f"Process stopped at iteration:{step}")
          print("="*50)
          if 'gx1' in param.keys():
              gx1=Vary.g1[0]+(Vary.g1[1]-Vary.g1[0])*jnn.sigmoid(param['gx1']/T)
              gy1=Vary.g1[2]+(Vary.g1[3]-Vary.g1[2])*jnn.sigmoid(param['gy1']/T)
              gz1=Vary.g1[4]+(Vary.g1[5]-Vary.g1[4])*jnn.sigmoid(param['gz1']/T)
              gg1=jxn.array([gx1,gy1,gz1])
          else:
              gg1=Ham.g1
          if 'gx2' in param.keys():
              gx2=Vary.g2[0]+(Vary.g2[1]-Vary.g2[0])*jnn.sigmoid(param['gx2']/T)
              gy2=Vary.g2[2]+(Vary.g2[3]-Vary.g2[2])*jnn.sigmoid(param['gy2']/T)
              gz2=Vary.g2[4]+(Vary.g2[5]-Vary.g2[4])*jnn.sigmoid(param['gz2']/T)
              gg2=jxn.array([gx2,gy2,gz2])
          else:
              gg2=Ham.g2
          if 'Ax1' in param.keys():
              Ax1=Vary.A1[0]+(Vary.A1[1]-Vary.A1[0])*jnn.sigmoid(param['Ax1']/T)
              Ay1=Vary.A1[2]+(Vary.A1[3]-Vary.A1[2])*jnn.sigmoid(param['Ay1']/T)
              Az1=Vary.A1[4]+(Vary.A1[5]-Vary.A1[4])*jnn.sigmoid(param['Az1']/T)
              AA1=jxn.array([Ax1,Ay1,Az1])
          else:
              AA1=Ham.A1
          if 'Ax2' in param.keys():
              Ax2=Vary.A2[0]+(Vary.A2[1]-Vary.A2[0])*jnn.sigmoid(param['Ax2']/T)
              Ay2=Vary.A2[2]+(Vary.A2[3]-Vary.A2[2])*jnn.sigmoid(param['Ay2']/T)
              Az2=Vary.A2[4]+(Vary.A2[5]-Vary.A2[4])*jnn.sigmoid(param['Az2']/T)
              AA2=jxn.array([Ax2,Ay2,Az2])
          else:
              AA2=Ham.A2
          if 'D1' in param.keys():
              Dx1=Vary.D1[0]+(Vary.D1[1]-Vary.D1[0])*jnn.sigmoid(param['D1']/T)
              Ey1=Vary.D1[2]+(Vary.D1[3]-Vary.D1[2])*jnn.sigmoid(param['E1']/T)
              DD1=jxn.array([Dx1,Ey1])
          else:
              DD1=Ham.D1
          if 'D2' in param.keys():
              Dx2=Vary.D2[0]+(Vary.D2[1]-Vary.D2[0])*jnn.sigmoid(param['D2']/T)
              Ey2=Vary.D2[2]+(Vary.D2[3]-Vary.D2[2])*jnn.sigmoid(param['E2']/T)
              DD2=jxn.array([Dx2,Ey2])
          else:
              DD2=Ham.D2
          if 'Qx1' in param.keys():
              Qx1=Vary.Q1[0]+(Vary.Q1[1]-Vary.Q1[0])*jnn.sigmoid(param['Qx1']/T)
              Qy1=Vary.Q1[2]+(Vary.Q1[3]-Vary.Q1[2])*jnn.sigmoid(param['Qy1']/T)
              Qz1=Vary.Q1[4]+(Vary.Q1[5]-Vary.Q1[4])*jnn.sigmoid(param['Qz1']/T)
              QQ1=jxn.array([Qx1,Qy1,Qz1])
          else:
              QQ1=Ham.Q1
          if 'Qx2' in param.keys():
              Qx2=Vary.Q2[0]+(Vary.Q2[1]-Vary.Q2[0])*jnn.sigmoid(param['Qx2']/T)
              Qy2=Vary.Q2[2]+(Vary.Q2[3]-Vary.Q2[2])*jnn.sigmoid(param['Qy2']/T)
              Qz2=Vary.Q2[4]+(Vary.Q2[5]-Vary.Q2[4])*jnn.sigmoid(param['Qz2']/T)
              QQ2=jxn.array([Qx2,Qy2,Qz2])
          else:
              QQ2=Ham.Q2
          if 'Hpp1' in param.keys():
              Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(param['Hpp1']/T)
              Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(param['Hpp2']/T)
              HHpp=jxn.array([Hppx,Hppy])
          else:
              HHpp=SHam.Hpp
          Hat=Ham.replace(g1=gg1,g2=gg2,A1=AA1,A2=AA2,D1=DD1,D2=DD2,Q1=QQ1,Q2=QQ2,Hpp=HHpp)
          print(f"Step {step+1:3d} | Error: {error:.5e} |")
          if 'gx1' in param.keys():
              gxf1=Vary.g1[0]+(Vary.g1[1]-Vary.g1[0])*jnn.sigmoid(param['gx1']/T)
              gyf1=Vary.g1[2]+(Vary.g1[3]-Vary.g1[2])*jnn.sigmoid(param['gy1']/T)
              gzf1=Vary.g1[4]+(Vary.g1[5]-Vary.g1[4])*jnn.sigmoid(param['gz1']/T)
              print(f"| gx1: {gxf1:.4f} | gy1: {gyf1:.4f} | gz1: {gzf1:.4f} |")
          if 'gx2' in param.keys():
              gxf2=Vary.g2[0]+(Vary.g2[1]-Vary.g2[0])*jnn.sigmoid(param['gx2']/T)
              gyf2=Vary.g2[2]+(Vary.g2[3]-Vary.g2[2])*jnn.sigmoid(param['gy2']/T)
              gzf2=Vary.g2[4]+(Vary.g2[5]-Vary.g2[4])*jnn.sigmoid(param['gz2']/T)
              print(f"| gx2: {gxf2:.4f} | gy2: {gyf2:.4f} | gz2: {gzf2:.4f} |")
          if 'Ax1' in param.keys():
              Axf1=Vary.A1[0]+(Vary.A1[1]-Vary.A1[0])*jnn.sigmoid(param['Ax1']/T)
              Ayf1=Vary.A1[2]+(Vary.A1[3]-Vary.A1[2])*jnn.sigmoid(param['Ay1']/T)
              Azf1=Vary.A1[4]+(Vary.A1[5]-Vary.A1[4])*jnn.sigmoid(param['Az1']/T)
              print(f"| Ax1: {Axf1:.4f} | Ay1: {Ayf1:.4f} | Az1: {Azf1:.4f} |")
          if 'Ax2' in param.keys():
              Axf2=Vary.A2[0]+(Vary.A2[1]-Vary.A2[0])*jnn.sigmoid(param['Ax2']/T)
              Ayf2=Vary.A2[2]+(Vary.A2[3]-Vary.A2[2])*jnn.sigmoid(param['Ay2']/T)
              Azf2=Vary.A2[4]+(Vary.A2[5]-Vary.A2[4])*jnn.sigmoid(param['Az2']/T)
              print(f"| Ax2: {Axf2:.4f} | Ay2: {Ayf2:.4f} | Az2: {Azf2:.4f} |")
          if 'D1' in param.keys():
              Dx1=Vary.D1[0]+(Vary.D1[1]-Vary.D1[0])*jnn.sigmoid(param['D1']/T)
              Ey1=Vary.D1[2]+(Vary.D1[3]-Vary.D1[2])*jnn.sigmoid(param['E1']/T)
              print(f"| D1: {Dx1:.1f} | E1: {Ey1:.1f} |")
          if 'D2' in param.keys():
              Dx2=Vary.D2[0]+(Vary.D2[1]-Vary.D2[0])*jnn.sigmoid(param['D2']/T)
              Ey2=Vary.D2[2]+(Vary.D2[3]-Vary.D2[2])*jnn.sigmoid(param['E2']/T)
              print(f"| D2: {Dx2:.1f} | E2: {Ey2:.1f} |")
          if 'Qx1' in param.keys():
              Qxf1=Vary.Q1[0]+(Vary.Q1[1]-Vary.Q1[0])*jnn.sigmoid(param['Qx1']/T)
              Qyf1=Vary.Q1[2]+(Vary.Q1[3]-Vary.Q1[2])*jnn.sigmoid(param['Qy1']/T)
              Qzf1=Vary.Q1[4]+(Vary.Q1[5]-Vary.Q1[4])*jnn.sigmoid(param['Qz1']/T)
              print(f"| Qx1: {Qxf1:.4f} | Qy1: {Qyf1:.4f} | Qz1: {Qzf1:.4f} |")
          if 'Qx2' in param.keys():
              Qxf2=Vary.Q2[0]+(Vary.Q2[1]-Vary.Q2[0])*jnn.sigmoid(param['Qx2']/T)
              Qyf2=Vary.Q2[2]+(Vary.Q2[3]-Vary.Q2[2])*jnn.sigmoid(param['Qy2']/T)
              Qzf2=Vary.Q2[4]+(Vary.Q2[5]-Vary.Q2[4])*jnn.sigmoid(param['Qz2']/T)
              print(f"| Qx2: {Qxf2:.4f} | Qy2: {Qyf2:.4f} | Qz2: {Qzf2:.4f} |")
          if 'Hpp1' in param.keys():
              Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(param['Hpp1']/T)
              Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(param['Hpp2']/T)
              print(f"| Hppg: {Hppx:.1f} | Hppl: {Hppy:.1f} |")
          if mode=='p':
              Blis,espc=JMulpol(Hat,dExp,graph=False)
              plt.figure(figsize=(8,6))
              plt.plot(Blis,expr,label='Data')
              formatter=EngFormatter(sep='') 
              plt.gca().yaxis.set_major_formatter(formatter)
              plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
              plt.xlabel('Field [mT]')
              plt.ylabel('Counts [A. U.]')
              plt.legend()
              plt.grid()
              plt.title('EPR Spectrum')
              plt.show()
              return espc
          elif mode=='c':
              Blis,espc=JMusic(Hat,dExp,graph=False)
              plt.figure(figsize=(8,6))
              plt.plot(Blis,expr,label='Data')
              formatter=EngFormatter(sep='') 
              plt.gca().yaxis.set_major_formatter(formatter)
              plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
              plt.xlabel('Field [mT]')
              plt.ylabel('Counts [A. U.]')
              plt.legend()
              plt.grid()
              plt.title('EPR Spectrum')
              plt.show()
              return espc
      if 'gx1' in param.keys():
          gx1=Vary.g1[0]+(Vary.g1[1]-Vary.g1[0])*jnn.sigmoid(param['gx1']/T)
          gy1=Vary.g1[2]+(Vary.g1[3]-Vary.g1[2])*jnn.sigmoid(param['gy1']/T)
          gz1=Vary.g1[4]+(Vary.g1[5]-Vary.g1[4])*jnn.sigmoid(param['gz1']/T)
          gg1=jxn.array([gx1,gy1,gz1])
      else:
          gg1=Ham.g1
      if 'gx2' in param.keys():
          gx2=Vary.g2[0]+(Vary.g2[1]-Vary.g2[0])*jnn.sigmoid(param['gx2']/T)
          gy2=Vary.g2[2]+(Vary.g2[3]-Vary.g2[2])*jnn.sigmoid(param['gy2']/T)
          gz2=Vary.g2[4]+(Vary.g2[5]-Vary.g2[4])*jnn.sigmoid(param['gz2']/T)
          gg2=jxn.array([gx2,gy2,gz2])
      else:
          gg2=Ham.g2
      if 'Ax1' in param.keys():
          Ax1=Vary.A1[0]+(Vary.A1[1]-Vary.A1[0])*jnn.sigmoid(param['Ax1']/T)
          Ay1=Vary.A1[2]+(Vary.A1[3]-Vary.A1[2])*jnn.sigmoid(param['Ay1']/T)
          Az1=Vary.A1[4]+(Vary.A1[5]-Vary.A1[4])*jnn.sigmoid(param['Az1']/T)
          AA1=jxn.array([Ax1,Ay1,Az1])
      else:
          AA1=Ham.A1
      if 'Ax2' in param.keys():
          Ax2=Vary.A2[0]+(Vary.A2[1]-Vary.A2[0])*jnn.sigmoid(param['Ax2']/T)
          Ay2=Vary.A2[2]+(Vary.A2[3]-Vary.A2[2])*jnn.sigmoid(param['Ay2']/T)
          Az2=Vary.A2[4]+(Vary.A2[5]-Vary.A2[4])*jnn.sigmoid(param['Az2']/T)
          AA2=jxn.array([Ax2,Ay2,Az2])
      else:
          AA2=Ham.A2
      if 'D1' in param.keys():
          Dx1=Vary.D1[0]+(Vary.D1[1]-Vary.D1[0])*jnn.sigmoid(param['D1']/T)
          Ey1=Vary.D1[2]+(Vary.D1[3]-Vary.D1[2])*jnn.sigmoid(param['E1']/T)
          DD1=jxn.array([Dx1,Ey1])
      else:
          DD1=Ham.D1
      if 'D2' in param.keys():
          Dx2=Vary.D2[0]+(Vary.D2[1]-Vary.D2[0])*jnn.sigmoid(param['D2']/T)
          Ey2=Vary.D2[2]+(Vary.D2[3]-Vary.D2[2])*jnn.sigmoid(param['E2']/T)
          DD2=jxn.array([Dx2,Ey2])
      else:
          DD2=Ham.D2
      if 'Qx1' in param.keys():
          Qx1=Vary.Q1[0]+(Vary.Q1[1]-Vary.Q1[0])*jnn.sigmoid(param['Qx1']/T)
          Qy1=Vary.Q1[2]+(Vary.Q1[3]-Vary.Q1[2])*jnn.sigmoid(param['Qy1']/T)
          Qz1=Vary.Q1[4]+(Vary.Q1[5]-Vary.Q1[4])*jnn.sigmoid(param['Qz1']/T)
          QQ1=jxn.array([Qx1,Qy1,Qz1])
      else:
          QQ1=Ham.Q1
      if 'Qx2' in param.keys():
          Qx2=Vary.Q2[0]+(Vary.Q2[1]-Vary.Q2[0])*jnn.sigmoid(param['Qx2']/T)
          Qy2=Vary.Q2[2]+(Vary.Q2[3]-Vary.Q2[2])*jnn.sigmoid(param['Qy2']/T)
          Qz2=Vary.Q2[4]+(Vary.Q2[5]-Vary.Q2[4])*jnn.sigmoid(param['Qz2']/T)
          QQ2=jxn.array([Qx2,Qy2,Qz2])
      else:
          QQ2=Ham.Q2
      if 'Hpp1' in param.keys():
          Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(param['Hpp1']/T)
          Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(param['Hpp2']/T)
          HHpp=jxn.array([Hppx,Hppy])
      else:
          HHpp=SHam.Hpp
      Hat=Ham.replace(g1=gg1,g2=gg2,A1=AA1,A2=AA2,D1=DD1,D2=DD2,Q1=QQ1,Q2=QQ2,Hpp=HHpp)
      print(f"Step {step+1:3d} | Error: {error:.5e} |")
      if 'gx1' in param.keys():
          gxf1=Vary.g1[0]+(Vary.g1[1]-Vary.g1[0])*jnn.sigmoid(param['gx1']/T)
          gyf1=Vary.g1[2]+(Vary.g1[3]-Vary.g1[2])*jnn.sigmoid(param['gy1']/T)
          gzf1=Vary.g1[4]+(Vary.g1[5]-Vary.g1[4])*jnn.sigmoid(param['gz1']/T)
          print(f"| gx1: {gxf1:.4f} | gy1: {gyf1:.4f} | gz1: {gzf1:.4f} |")
      if 'gx2' in param.keys():
          gxf2=Vary.g2[0]+(Vary.g2[1]-Vary.g2[0])*jnn.sigmoid(param['gx2']/T)
          gyf2=Vary.g2[2]+(Vary.g2[3]-Vary.g2[2])*jnn.sigmoid(param['gy2']/T)
          gzf2=Vary.g2[4]+(Vary.g2[5]-Vary.g2[4])*jnn.sigmoid(param['gz2']/T)
          print(f"| gx2: {gxf2:.4f} | gy2: {gyf2:.4f} | gz2: {gzf2:.4f} |")
      if 'Ax1' in param.keys():
          Axf1=Vary.A1[0]+(Vary.A1[1]-Vary.A1[0])*jnn.sigmoid(param['Ax1']/T)
          Ayf1=Vary.A1[2]+(Vary.A1[3]-Vary.A1[2])*jnn.sigmoid(param['Ay1']/T)
          Azf1=Vary.A1[4]+(Vary.A1[5]-Vary.A1[4])*jnn.sigmoid(param['Az1']/T)
          print(f"| Ax1: {Axf1:.4f} | Ay1: {Ayf1:.4f} | Az1: {Azf1:.4f} |")
      if 'Ax2' in param.keys():
          Axf2=Vary.A2[0]+(Vary.A2[1]-Vary.A2[0])*jnn.sigmoid(param['Ax2']/T)
          Ayf2=Vary.A2[2]+(Vary.A2[3]-Vary.A2[2])*jnn.sigmoid(param['Ay2']/T)
          Azf2=Vary.A2[4]+(Vary.A2[5]-Vary.A2[4])*jnn.sigmoid(param['Az2']/T)
          print(f"| Ax2: {Axf2:.4f} | Ay2: {Ayf2:.4f} | Az2: {Azf2:.4f} |")
      if 'D1' in param.keys():
          Dx1=Vary.D1[0]+(Vary.D1[1]-Vary.D1[0])*jnn.sigmoid(param['D1']/T)
          Ey1=Vary.D1[2]+(Vary.D1[3]-Vary.D1[2])*jnn.sigmoid(param['E1']/T)
          print(f"| D1: {Dx1:.1f} | E1: {Ey1:.1f} |")
      if 'D2' in param.keys():
          Dx2=Vary.D2[0]+(Vary.D2[1]-Vary.D2[0])*jnn.sigmoid(param['D2']/T)
          Ey2=Vary.D2[2]+(Vary.D2[3]-Vary.D2[2])*jnn.sigmoid(param['E2']/T)
          print(f"| D2: {Dx2:.1f} | E2: {Ey2:.1f} |")
      if 'Qx1' in param.keys():
          Qxf1=Vary.Q1[0]+(Vary.Q1[1]-Vary.Q1[0])*jnn.sigmoid(param['Qx1']/T)
          Qyf1=Vary.Q1[2]+(Vary.Q1[3]-Vary.Q1[2])*jnn.sigmoid(param['Qy1']/T)
          Qzf1=Vary.Q1[4]+(Vary.Q1[5]-Vary.Q1[4])*jnn.sigmoid(param['Qz1']/T)
          print(f"| Qx1: {Qxf1:.4f} | Qy1: {Qyf1:.4f} | Qz1: {Qzf1:.4f} |")
      if 'Qx2' in param.keys():
          Qxf2=Vary.Q2[0]+(Vary.Q2[1]-Vary.Q2[0])*jnn.sigmoid(param['Qx2']/T)
          Qyf2=Vary.Q2[2]+(Vary.Q2[3]-Vary.Q2[2])*jnn.sigmoid(param['Qy2']/T)
          Qzf2=Vary.Q2[4]+(Vary.Q2[5]-Vary.Q2[4])*jnn.sigmoid(param['Qz2']/T)
          print(f"| Qx2: {Qxf2:.4f} | Qy2: {Qyf2:.4f} | Qz2: {Qzf2:.4f} |")
      if 'Hpp1' in param.keys():
          Hppx=Vary.Hpp[0]+(Vary.Hpp[1]-Vary.Hpp[0])*jnn.sigmoid(param['Hpp1']/T)
          Hppy=Vary.Hpp[2]+(Vary.Hpp[3]-Vary.Hpp[2])*jnn.sigmoid(param['Hpp2']/T)
          print(f"| Hppg: {Hppx:.1f} | Hppl: {Hppy:.1f} |")
      if mode=='p':
          Blis,espc=JMulpol(Hat,dExp,graph=False)
          plt.figure(figsize=(8,6))
          plt.plot(Blis,expr,label='Data')
          formatter=EngFormatter(sep='') 
          plt.gca().yaxis.set_major_formatter(formatter)
          plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
          plt.xlabel('Magnetic field [mT]')
          plt.ylabel('Counts [A. U.]')
          plt.grid()
          plt.legend()
          plt.title('EPR Spectrum')
          plt.show()
          return espc
      elif mode=='c':
          Blis,espc=JMusic(Hat,dExp,graph=False)
          plt.figure(figsize=(8,6))
          plt.plot(Blis,expr,label='Data')
          formatter=EngFormatter(sep='') 
          plt.gca().yaxis.set_major_formatter(formatter)
          plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
          plt.xlabel('Magnetic field [mT]')
          plt.ylabel('Counts [A. U.]')
          plt.grid()
          plt.legend()
          plt.title('EPR Spectrum')
          plt.show()
          return espc
