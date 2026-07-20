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
from matplotlib.ticker import FormatStrFormatter
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
import threading
from itertools import product as iterproduct
from .base_cris import *
from .base_ham import *
from .base_plot import *
from .base_powd import *
from .base_rotate import *

stopvar=False
result={}
def CostfNM(exper,intens,metric='rmse'):
    norm=np.sum(intens*intens)
    if norm==0 or np.max(intens)<1e-10:
        return 1e6
    maxe=np.max(np.abs(exper))
    maxi=np.max(np.abs(intens))
    exper=exper/maxe
    intens=intens/maxi
    if metric=='rmse':
        # Root Mean Squared Error
        error=np.sqrt(np.mean((intens-exper)**2))
        return error
    elif metric=='correlation':
        corr=np.corrcoef(exper,intens)
        pearson=corr[0, 1]
        if np.isnan(pearson):
            return 1e6
        return 1-pearson

def Nelder1(Hamer,Expe,Vara,exper,eps=1e-10,maximal=5000,dtype='data',mode='p'):
    #global stopvar
    #Based on the code by Hadrien Crassous
    def Fincost(points,funcname):
        if stopvar:
            raise KeyboardInterrupt("Stopped by user")
        jl=0
        if Var.g!=0.0:
            gt=points[jl:jl+3]
            if (gt[0]<Var.g[0] or gt[0]>Var.g[1] or gt[1]<Var.g[2] or gt[1]>Var.g[3] or
            gt[2]<Var.g[4] or gt[2]>Var.g[5]):
                return 1e6
            Ham.g=gt
            jl+=3
        if Var.A!=0.0:
            At=points[jl:jl+3]
            if (At[0]<Var.A[0] or At[0]>Var.A[1] or At[1]<Var.A[2] or At[1]>Var.A[3] or
            At[2]<Var.A[4] or At[2]>Var.A[5]):
                return 1e6
            Ham.A=At
            jl+=3
        if Var.Q!=0.0:
            Qt=points[jl:jl+3]
            if (Qt[0]<Var.Q[0] or Qt[0]>Var.Q[1] or Qt[1]<Var.Q[2] or Qt[1]>Var.Q[3] or
            Qt[2]<Var.Q[4] or Qt[2]>Var.Q[5]):
                return 1e6
            Ham.Q=Qt
            jl+=3
        if Var.D!=0.0:
            Dt=points[jl:jl+2]
            if (Dt[0]<Var.D[0] or Dt[0]>Var.D[1] or Dt[1]<Var.D[2] or Dt[1]>Var.D[3]):
                return 1e6
            Ham.D=Dt
            jl+=2
        if np.any(Var.Hpp):
            Htp=points[jl:jl+2]
            if (Htp[0]<Var.Hpp[0] or Htp[0]>Var.Hpp[1] or Htp[1]<Var.Hpp[2] or Htp[1]>Var.Hpp[3]):
                return 1e6
            Ham.Hpp=Htp
        if funcname in ['Powder']:
            spect=Powder(Ham,Exp,graph='False')[1]
        elif funcname in ['Eresonant']:
            spect=Eresonant(Ham,Exp,graph='False',table='False')[1]
        wcost=CostfNM(exper,spect)
        return wcost
    Ham=deepcopy(Hamer)
    Exp=deepcopy(Expe)
    Var=deepcopy(Vara)
    if mode=='p':
        funtiona=Powder#(Ham,Exp,graph='False')
    elif mode=='c':
        funtiona=Eresonant#(Ham,Exp,graph='False',table='False')
    else:
        raise ValueError (f'Valid functions are powder (p) or cristal (c).')
    funcname=funtiona.__name__
    pointx=[]
    stp=[]
    lowfron=[]
    hifron=[]
    if dtype=='data':
        exper=exper
    elif dtype=='integral':
        fielda=np.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
        exper=scii.cumulative_trapezoid(exper,fielda,initial=0)
    else:
        raise ValueError(f"Data type {dtype} is not supported. Use integral or data instead.")
    iwas,jwas,kwas,weight,hulk=Delaunay(Exp)
    if Var.g!=0.0:
        pointx.extend(Ham.g)
        stp.extend([(Var.g[1]-Var.g[0])/20,(Var.g[3]-Var.g[2])/20,(Var.g[5]-Var.g[4])/20])
        lowfron.extend([Var.g[0],Var.g[2],Var.g[4]])
        hifron.extend([Var.g[1],Var.g[3],Var.g[5]])
    if Var.A!=0.0:
        pointx.extend(Ham.A)
        stp.extend([(Var.A[1]-Var.A[0])/20,(Var.A[3]-Var.A[2])/20,(Var.A[5]-Var.A[4])/20])
        lowfron.extend([Var.A[0],Var.A[2],Var.A[4]])
        hifron.extend([Var.A[1],Var.A[3],Var.A[5]])
    if Var.Q!=0.0:
        pointx.extend(Ham.Q)
        stp.extend([(Var.Q[1]-Var.Q[0])/20,(Var.Q[3]-Var.Q[2])/20,(Var.Q[5]-Var.Q[4])/20])
        lowfron.extend([Var.Q[0],Var.Q[2],Var.Q[4]])
        hifron.extend([Var.Q[1],Var.Q[3],Var.Q[5]])
    if Var.D!=0.0:
        pointx.extend(Ham.D)
        stp.extend([(Var.D[1]-Var.D[0])/20,(Var.D[3]-Var.D[2])/20])
        lowfron.extend([Var.D[0],Var.D[2]])
        hifron.extend([Var.D[1],Var.D[3]])
    if np.any(Var.Hpp):
        pointx.extend(Ham.Hpp)
        stp.extend([(Var.Hpp[1]-Var.Hpp[0])/20,(Var.Hpp[3]-Var.Hpp[2])/20])
        lowfron.extend([Var.Hpp[0],Var.Hpp[2]])
        hifron.extend([Var.Hpp[1],Var.Hpp[3]])
    pointx=np.array(pointx,dtype=float)
    stp=np.array(stp,dtype=float)
    lowfron=np.array(lowfron,dtype=float)
    hifron=np.array(hifron,dtype=float)
    nparams=len(pointx)
    simplex=np.zeros((nparams+1,nparams))
    price=np.zeros(nparams+1)
    simplex[0]=pointx
    price[0]=Fincost(pointx,funcname)

    #Iteration parameters
    alpha=1
    gamma=1+(2/nparams)
    rho=0.75-(1/(2*nparams))
    sigma=1-(1/nparams)

    for la in range(0,nparams):
        npoint=np.copy(pointx)
        change=np.random.uniform(-1,1)
        npoint[la]+=change*stp[la]
        simplex[la+1]=npoint
        price[la+1]=Fincost(npoint,funcname)
    iter=0
    restarts=0
    mrestarts=5
    try:
        while iter<maximal:
            simplex=simplex[np.argsort(price)]
            price=price[np.argsort(price)]
            center=np.mean(simplex[:-1],axis=0)
            melhor=simplex[0]
            segunda=simplex[-2]
            pior=simplex[-1]
            cmelhor=price[0]
            csegunda=price[-2]
            cpior=price[-1]
            if (cpior-cmelhor)<eps:
                if restarts<mrestarts:
                    restarts+=1
                    print(f"Restarting around the best points...")
                    pointx=np.copy(simplex[0])
                    for la in range(1,len(simplex)-2):
                        rad=np.random.uniform(lowfron,hifron)
                        simplex[la]=rad
                        price[la]=Fincost(rad,funcname)
                    continue
                else:
                    print("Data converged.")
                    break
            if iter%10==0:
                print(f"Iteration: {iter} | Best cost: {cmelhor:.5e}")
                if Var.g!=0.0:
                    print(f'gx={Ham.g[0]} | gy={Ham.g[1]} | gz={Ham.g[2]}')
                if Var.A!=0.0:
                    print(f'Ax={Ham.A[0]} | Ay={Ham.A[1]} | Az={Ham.A[2]}')
                if Var.Q!=0.0:
                    print(f'Qx={Ham.Q[0]} | Qy={Ham.Q[1]} | Qz={Ham.Q[2]}')
                if Var.D!=0.0:
                    print(f'D={Ham.D[0]} | E={Ham.D[1]}')
                if np.any(Var.Hpp):
                    print(f'Hppg={Ham.Hpp[0]} | Hppl={Ham.Hpp[1]}')
            vect=np.random.normal(0,1,nparams)
            vnor=np.linalg.norm(vect)
            dist=np.abs(pior-melhor)
            pecenter=center+0.1*(vect*dist)
            #Reflection
            ref=pecenter+alpha*(pecenter-pior)
            cref=Fincost(ref,funcname)
            if cmelhor<=cref<csegunda:
                simplex[-1]=ref
                price[-1]=cref
            elif cref<cmelhor:
                # Expansion
                expan=pecenter+gamma*(ref-pecenter)
                cexpan=Fincost(expan,funcname)
                if cexpan<cref:
                    simplex[-1]=expan
                    price[-1]=cexpan
                else:
                    simplex[-1]=ref
                    price[-1]=cref
            else:
                #Contraction
                if cref<cpior:
                    contra=center+rho*(ref-center) #External
                else:
                    contra=center+rho*(pior-center) #Internal
                ccontra=Fincost(contra,funcname)
                if ccontra<min(cref,cpior):
                    simplex[-1]=contra
                    price[-1]=ccontra
                else:
                    # Shrink
                    for ila in range(1,len(simplex)):
                        simplex[ila]=melhor+sigma*(simplex[ila]-melhor)
                        price[ila]=Fincost(simplex[ila],funcname)
            iter+=1
            if stopvar:
                break
    except KeyboardInterrupt:
        _=Fincost(melhor,funcname)
        print("\n"+"="*50)
        print(f"Process stopped at iteration:{iter} with best cost: {cmelhor:.5e}")
        print("="*50)
        if Var.g!=0.0:
            print(f'gx={Ham.g[0]} | gy={Ham.g[1]} | gz={Ham.g[2]}')
        if Var.A!=0.0:
            print(f'Ax={Ham.A[0]} | Ay={Ham.A[1]} | Az={Ham.A[2]}')
        if Var.Q!=0.0:
            print(f'Qx={Ham.Q[0]} | Qy={Ham.Q[1]} | Qz={Ham.Q[2]}')
        if Var.D!=0.0:
            print(f'D={Ham.D[0]} | E={Ham.D[1]}')
        if np.any(Var.Hpp):
            print(f'Hppg={Ham.Hpp[0]} | Hppl={Ham.Hpp[1]}')
        if funcname in ['Powder']:
            return funtiona(Ham,Exp,graph='False')[1]
        elif funcname in ['Eresonant']:
            return funtiona(Ham,Exp,graph='False',table='False')[1]
    _=Fincost(melhor,funcname)
    print("\n"+"="*50)
    print(f"Process stopped at iteration:{iter} with best cost: {cmelhor:.5e}")
    print("="*50)
    if Var.g!=0.0:
        print(f'gx={Ham.g[0]} | gy={Ham.g[1]} | gz={Ham.g[2]}')
    if Var.A!=0.0:
        print(f'Ax={Ham.A[0]} | Ay={Ham.A[1]} | Az={Ham.A[2]}')
    if Var.Q!=0.0:
        print(f'Qx={Ham.Q[0]} | Qy={Ham.Q[1]} | Qz={Ham.Q[2]}')
    if Var.D!=0.0:
        print(f'D={Ham.D[0]} | E={Ham.D[1]}')
    if np.any(Var.Hpp):
        print(f'Hppg={Ham.Hpp[0]} | Hppl={Ham.Hpp[1]}')
    if funcname in ['Powder']:
        return funtiona(Ham,Exp,graph='False')[1]
    elif funcname in ['Eresonant']:
        return funtiona(Ham,Exp,graph='False',table='False')[1]

def Nelder2(Hamer,Expe,Vara,exper,eps=1e-10,maximal=5000,dtype='data',mode='p'):
    #global stopvar
    #Based on the code by Hadrien Crassous
    def Fincost(points,funcname):
        if stopvar:
            raise KeyboardInterrupt("Stopped by user")
        jl=0
        for i in range(len(Ham.Mulham)):
            if Var.Mvary[i].g!=0.0:
                gt=points[jl:jl+3]
                if (gt[0]<Var.Mvary[i].g[0] or gt[0]>Var.Mvary[i].g[1] or gt[1]<Var.Mvary[i].g[2] or gt[1]>Var.Mvary[i].g[3] or gt[2]<Var.Mvary[i].g[4] or gt[2]>Var.Mvary[i].g[5]):
                    return 1e6
                Ham.Mulham[i].g=gt
                jl+=3
            if Var.Mvary[i].A!=0.0:
                At=points[jl:jl+3]
                if (At[0]<Var.Mvary[i].A[0] or At[0]>Var.Mvary[i].A[1] or At[1]<Var.Mvary[i].A[2] or At[1]>Var.Mvary[i].A[3] or At[2]<Var.Mvary[i].A[4] or At[2]>Var.Mvary[i].A[5]):
                    return 1e6
                Ham.Mulham[i].A=At
                jl+=3
            if Var.Mvary[i].Q!=0.0:
                Qt=points[jl:jl+3]
                if (Qt[0]<Var.Mvary[i].Q[0] or Qt[0]>Var.Mvary[i].Q[1] or Qt[1]<Var.Mvary[i].Q[2] or Qt[1]>Var.Mvary[i].Q[3] or Qt[2]<Var.Mvary[i].Q[4] or Qt[2]>Var.Mvary[i].Q[5]):
                    return 1e6
                Ham.Mulham[i].Q=Qt
                jl+=3
            if Var.Mvary[i].D!=0.0:
                Dt=points[jl:jl+2]
                if (Dt[0]<Var.Mvary[i].D[0] or Dt[0]>Var.Mvary[i].D[1] or Dt[1]<Var.Mvary[i].D[2] or Dt[1]>Var.Mvary[i].D[3]):
                    return 1e6
                Ham.Mulham[i].D=Dt
                jl+=2
            if np.any(Var.Mvary[i].Hpp):
                Htp=points[jl:jl+2]
                if (Htp[0]<Var.Mvary[i].Hpp[0] or Htp[0]>Var.Mvary[i].Hpp[1] or Htp[1]<Var.Mvary[i].Hpp[2] or Htp[1]>Var.Mvary[i].Hpp[3]):
                    return 1e6
                Ham.Mulham[i].Hpp=Htp
                jl+=2
        try:
            if funcname in ['Mulpol']:
                spect=funtiona(Ham,Exp,graph='False')[1]
            elif funcname in ['Music']:
                spect=funtiona(Ham,Exp,graph='False',table='False')[1]
            wcost=CostfNM(exper,spect)
            return wcost
        except Exception:
            return 1e6
    Ham=deepcopy(Hamer)
    Exp=deepcopy(Expe)
    Var=deepcopy(Vara)
    if mode=='p':
        funtiona=Mulpol#(Ham,Exp,graph='False')
    elif mode=='c':
        funtiona=Music#(Ham,Exp,graph='False',table='False')
    else:
        raise ValueError (f'Valid functions are powder (p) or cristal (c).')
    funcname=funtiona.__name__
    pointx=[]
    stp=[]
    lowfron=[]
    hifron=[]
    if dtype=='data':
        exper=exper
    elif dtype=='integral':
        fielda=np.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
        exper=scii.cumulative_trapezoid(exper,fielda,initial=0)
    else:
        raise ValueError(f"Data type {dtype} is not supported. Use integral or data instead.")
    iwas,jwas,kwas,weight,hulk=Delaunay(Exp.Mexp[0])
    numberes=len(Ham.Mulham)
    for lke in range(0,numberes):
        if Var.Mvary[lke].g!=0.0:
            pointx.extend(Ham.Mulham[lke].g)
            stp.extend([(Var.Mvary[lke].g[1]-Var.Mvary[lke].g[0])/20,(Var.Mvary[lke].g[3]-Var.Mvary[lke].g[2])/20,(Var.Mvary[lke].g[5]-Var.Mvary[lke].g[4])/20])
            lowfron.extend([Var.Mvary[lke].g[0],Var.Mvary[lke].g[2],Var.Mvary[lke].g[4]])
            hifron.extend([Var.Mvary[lke].g[1],Var.Mvary[lke].g[3],Var.Mvary[lke].g[5]])
        if Var.Mvary[lke].A!=0.0:
            pointx.extend(Ham.Mulham[lke].A)
            stp.extend([(Var.Mvary[lke].A[1]-Var.Mvary[lke].A[0])/20,(Var.Mvary[lke].A[3]-Var.Mvary[lke].A[2])/20,(Var.Mvary[lke].A[5]-Var.Mvary[lke].A[4])/20])
            lowfron.extend([Var.Mvary[lke].A[0],Var.Mvary[lke].A[2],Var.Mvary[lke].A[4]])
            hifron.extend([Var.Mvary[lke].A[1],Var.Mvary[lke].A[3],Var.Mvary[lke].A[5]])
        if Var.Mvary[lke].Q!=0.0:
            pointx.extend(Ham.Mulham[lke].Q)
            stp.extend([(Var.Mvary[lke].Q[1]-Var.Mvary[lke].Q[0])/20,(Var.Mvary[lke].Q[3]-Var.Mvary[lke].Q[2])/20,(Var.Mvary[lke].Q[5]-Var.Mvary[lke].Q[4])/20])
            lowfron.extend([Var.Mvary[lke].Q[0],Var.Mvary[lke].Q[2],Var.Mvary[lke].Q[4]])
            hifron.extend([Var.Mvary[lke].Q[1],Var.Mvary[lke].Q[3],Var.Mvary[lke].Q[5]])
        if Var.Mvary[lke].D!=0.0:
            pointx.extend(Ham.Mulham[lke].D)
            stp.extend([(Var.Mvary[lke].D[1]-Var.Mvary[lke].D[0])/20,(Var.Mvary[lke].D[3]-Var.Mvary[lke].D[2])/20])
            lowfron.extend([Var.Mvary[lke].D[0],Var.Mvary[lke].D[2]])
            hifron.extend([Var.Mvary[lke].D[1],Var.Mvary[lke].D[3]])
        if np.any(Var.Mvary[lke].Hpp):
            pointx.extend(Ham.Mulham[lke].Hpp)
            stp.extend([(Var.Mvary[lke].Hpp[1]-Var.Mvary[lke].Hpp[0])/20,(Var.Mvary[lke].Hpp[3]-Var.Mvary[lke].Hpp[2])/20])
            lowfron.extend([Var.Mvary[lke].Hpp[0],Var.Mvary[lke].Hpp[2]])
            hifron.extend([Var.Mvary[lke].Hpp[1],Var.Mvary[lke].Hpp[3]])
    pointx=np.array(pointx,dtype=float)
    stp=np.array(stp,dtype=float)
    lowfron=np.array(lowfron,dtype=float)
    hifron=np.array(hifron,dtype=float)
    nparams=len(pointx)
    simplex=np.zeros((nparams+1,nparams))
    price=np.zeros(nparams+1)
    simplex[0]=pointx
    price[0]=Fincost(pointx,funcname)

    #Iteration parameters
    alpha=1
    gamma=1+(2/nparams)
    rho=0.75-(1/(2*nparams))
    sigma=1-(1/nparams)

    for la in range(0,nparams):
        npoint=np.copy(pointx)
        change=np.random.uniform(-1,1)
        npoint[la]+=change*stp[la]
        simplex[la+1]=npoint
        price[la+1]=Fincost(npoint,funcname)
    iter=0
    restarts=0
    mrestarts=5
    try:
        while iter<maximal:
            simplex=simplex[np.argsort(price)]
            price=price[np.argsort(price)]
            center=np.mean(simplex[:-1],axis=0)
            melhor=simplex[0]
            segunda=simplex[-2]
            pior=simplex[-1]
            cmelhor=price[0]
            csegunda=price[-2]
            cpior=price[-1]
            if (cpior-cmelhor)<eps:
                if restarts<mrestarts:
                    restarts+=1
                    print(f"Restarting around the best points...")
                    pointx=np.copy(simplex[0])
                    for la in range(1,len(simplex)-2):
                        rad=np.random.uniform(lowfron,hifron)
                        simplex[la]=rad
                        price[la]=Fincost(rad,funcname)
                    continue
                else:
                    print("Data converged.")
                    break
            if iter%10==0:
                print(f"Iteration: {iter} | Best cost: {cmelhor:.5e}")
                for i in range(len(Ham.Mulham)):
                    print(f"--- System {i+1} ---")
                    if Var.Mvary[i].g!=0.0:
                        print(f'gx={Ham.Mulham[i].g[0]:.4f} | gy={Ham.Mulham[i].g[1]:.4f} | gz={Ham.Mulham[i].g[2]:.4f}')
                    if Var.Mvary[i].A!=0.0:
                        print(f'Ax={Ham.Mulham[i].A[0]:.4f} | Ay={Ham.Mulham[i].A[1]:.4f} | Az={Ham.Mulham[i].A[2]:.4f}')
                    if Var.Mvary[i].Q!=0.0:
                        print(f'Qx={Ham.Mulham[i].Q[0]:.4f} | Qy={Ham.Mulham[i].Q[1]:.4f} | Qz={Ham.Mulham[i].Q[2]:.4f}')
                    if Var.Mvary[i].D!=0.0:
                        print(f'D={Ham.Mulham[i].D[0]:.4f} | E={Ham.Mulham[i].D[1]:.4f}')
                    if np.any(Var.Mvary[i].Hpp):
                        print(f'Hppg={Ham.Mulham[i].Hpp[0]:.4f} | Hppl={Ham.Mulham[i].Hpp[1]:.4f}')
            vect=np.random.normal(0,1,nparams)
            vnor=np.linalg.norm(vect)
            dist=np.abs(pior-melhor)
            pecenter=center+0.1*(vect*dist)
            #Reflection
            ref=pecenter+alpha*(pecenter-pior)
            cref=Fincost(ref,funcname)
            if cmelhor<=cref<csegunda:
                simplex[-1]=ref
                price[-1]=cref
            elif cref<cmelhor:
                # Expansion
                expan=pecenter+gamma*(ref-pecenter)
                cexpan=Fincost(expan,funcname)
                if cexpan<cref:
                    simplex[-1]=expan
                    price[-1]=cexpan
                else:
                    simplex[-1]=ref
                    price[-1]=cref
            else:
                #Contraction
                if cref<cpior:
                    contra=center+rho*(ref-center) #External
                else:
                    contra=center+rho*(pior-center) #Internal
                ccontra=Fincost(contra,funcname)
                if ccontra<min(cref,cpior):
                    simplex[-1]=contra
                    price[-1]=ccontra
                else:
                    # Shrink
                    for ila in range(1,len(simplex)):
                        simplex[ila]=melhor+sigma*(simplex[ila]-melhor)
                        price[ila]=Fincost(simplex[ila],funcname)
            iter+=1
            if stopvar:
                break
    except KeyboardInterrupt:
        _=Fincost(melhor,funcname)
        print("\n"+"="*50)
        print(f"Process stopped at iteration:{iter} with best cost: {cmelhor:.5e}")
        print("="*50)
        for i in range(len(Ham.Mulham)):
            print(f"--- System {i+1} ---")
            if Var.Mvary[i].g!=0.0:
                print(f'gx={Ham.Mulham[i].g[0]:.4f} | gy={Ham.Mulham[i].g[1]:.4f} | gz={Ham.Mulham[i].g[2]:.4f}')
            if Var.Mvary[i].A!=0.0:
                print(f'Ax={Ham.Mulham[i].A[0]:.4f} | Ay={Ham.Mulham[i].A[1]:.4f} | Az={Ham.Mulham[i].A[2]:.4f}')
            if Var.Mvary[i].Q!=0.0:
                print(f'Qx={Ham.Mulham[i].Q[0]:.4f} | Qy={Ham.Mulham[i].Q[1]:.4f} | Qz={Ham.Mulham[i].Q[2]:.4f}')
            if Var.Mvary[i].D!=0.0:
                print(f'D={Ham.Mulham[i].D[0]:.4f} | E={Ham.Mulham[i].D[1]:.4f}')
            if np.any(Var.Mvary[i].Hpp):
                print(f'Hppg={Ham.Mulham[i].Hpp[0]:.4f} | Hppl={Ham.Mulham[i].Hpp[1]:.4f}')
        if funcname in ['Mulpol']:
            return funtiona(Ham,Exp,graph='False')[1]
        elif funcname in ['Music']:
            return funtiona(Ham,Exp,graph='False',table='False')[1]
    _=Fincost(melhor,funcname)
    print("\n"+"="*50)
    print(f"Process stopped at iteration:{iter} with best cost: {cmelhor:.5e}")
    print("="*50)
    for i in range(len(Ham.Mulham)):
        print(f"--- System {i+1} ---")
        if Var.Mvary[i].g!=0.0:
            print(f'gx={Ham.Mulham[i].g[0]:.4f} | gy={Ham.Mulham[i].g[1]:.4f} | gz={Ham.Mulham[i].g[2]:.4f}')
        if Var.Mvary[i].A!=0.0:
            print(f'Ax={Ham.Mulham[i].A[0]:.4f} | Ay={Ham.Mulham[i].A[1]:.4f} | Az={Ham.Mulham[i].A[2]:.4f}')
        if Var.Mvary[i].Q!=0.0:
            print(f'Qx={Ham.Mulham[i].Q[0]:.4f} | Qy={Ham.Mulham[i].Q[1]:.4f} | Qz={Ham.Mulham[i].Q[2]:.4f}')
        if Var.Mvary[i].D!=0.0:
            print(f'D={Ham.Mulham[i].D[0]:.4f} | E={Ham.Mulham[i].D[1]:.4f}')
        if np.any(Var.Mvary[i].Hpp):
            print(f'Hppg={Ham.Mulham[i].Hpp[0]:.4f} | Hppl={Ham.Mulham[i].Hpp[1]:.4f}')
    if funcname in ['Mulpol']:
        return funtiona(Ham,Exp,graph='False')[1]
    elif funcname in ['Music']:
        return funtiona(Ham,Exp,graph='False',table='False')[1]

def Nelder(Hamer,Expe,Vara,exper,eps=1e-10,maximal=5000,dtype='data',mode='p'):
    if type(Hamer)==Multham:
         if Expe.Mexp[0].Points!=len(exper):
            Expe.Mexp[0].Points=len(exper)
         scpe=Nelder2(Hamer,Expe,Vara,exper,eps,maximal,dtype,mode)
    elif type(Hamer)==Hval:
         if Expe.Points!=len(exper):
            Expe.Points=len(exper)
         scpe=Nelder1(Hamer,Expe,Vara,exper,eps,maximal,dtype,mode)

    return scpe

def CostfG(exper,intens,metric='rmse'):
    norm=np.sum(intens*intens)
    if norm==0 or np.max(intens)<1e-10:
        return 1e6
    maxe=np.max(np.abs(exper))
    maxi=np.max(np.abs(intens))
    intens=intens/maxi
    exper=exper/maxe
    if metric=='rmse':
        # Root Mean Squared Error
        error=np.sqrt(np.mean((intens-exper)**2))
        return error
    elif metric=='correlation':
        corr=np.corrcoef(exper,intens)
        pearson=corr[0, 1]
        if np.isnan(pearson):
            return 1e6
        return 1-pearson

def Cross(vat1,vat2):
    ret=len(vat1)
    mask=np.random.randint(0,2,size=ret)
    son1=np.zeros(ret)
    son2=np.zeros(ret)
    for iw in range(0,ret):
        if mask[iw]==1:
            son1[iw]=vat1[iw]
            son2[iw]=vat2[iw]
        else:
            son1[iw]=vat2[iw]
            son2[iw]=vat1[iw]
    return son1,son2

def Mutategauss(fela,lowfron,hifron,proba=0.1,desv=0.05):
    nparams=len(fela)
    mutant=np.copy(fela)
    for ap in range(0,nparams):
        if np.random.rand()<proba:
            rangee=hifron[ap]-lowfron[ap]
            sigma=desv*rangee
            noise=np.random.normal(0,sigma)
            mutant[ap]+=noise
            if mutant[ap]<lowfron[ap]:
                mutant[ap]=lowfron[ap]
            elif mutant[ap]>hifron[ap]:
                mutant[ap]=hifron[ap]
    return mutant

def Genio1(Hamer,Expe,Vara,exper,eps=1e-10,maximal=30,dtype='data',mode='p'):
    def Fincost(points,funcname):
        if stopvar:
            raise KeyboardInterrupt("Stopped by user")
        jl=0
        if Var.g!=0.0:
            gt=points[jl:jl+3]
            if (gt[0]<Var.g[0] or gt[0]>Var.g[1] or gt[1]<Var.g[2] or gt[1]>Var.g[3] or
            gt[2]<Var.g[4] or gt[2]>Var.g[5]):
                return 1e6
            Ham.g=gt
            jl+=3
        if Var.A!=0.0:
            At=points[jl:jl+3]
            if (At[0]<Var.A[0] or At[0]>Var.A[1] or At[1]<Var.A[2] or At[1]>Var.A[3] or
            At[2]<Var.A[4] or At[2]>Var.A[5]):
                return 1e6
            Ham.A=At
            jl+=3
        if Var.Q!=0.0:
            Qt=points[jl:jl+3]
            if (Qt[0]<Var.Q[0] or Qt[0]>Var.Q[1] or Qt[1]<Var.Q[2] or Qt[1]>Var.Q[3] or
            Qt[2]<Var.Q[4] or Qt[2]>Var.Q[5]):
                return 1e6
            Ham.Q=Qt
            jl+=3
        if Var.D!=0.0:
            Dt=points[jl:jl+2]
            if (Dt[0]<Var.D[0] or Dt[0]>Var.D[1] or Dt[1]<Var.D[2] or Dt[1]>Var.D[3]):
                return 1e6
            Ham.D=Dt
            jl+=2
        if np.any(Var.Hpp):
            Htp=points[jl:jl+2]
            if (Htp[0]<Var.Hpp[0] or Htp[0]>Var.Hpp[1] or Htp[1]<Var.Hpp[2] or Htp[1]>Var.Hpp[3]):
                return 1e6
            Ham.Hpp=Htp
        if funcname in ['Powder']:
            spect=Powder(Ham,Exp,graph='False')[1]
        elif funcname in ['Eresonant']:
            spect=Eresonant(Ham,Exp,graph='False',table='False')[1]
        wcost=CostfG(exper,spect)
        return wcost
    #global stopvar
    Ham=deepcopy(Hamer)
    Exp=deepcopy(Expe)
    Var=deepcopy(Vara)
    lowfron=[]
    hifron=[]
    bplayer=0
    if mode=='p':
        funtiona=Powder#(Ham,Exp,graph='False')
    elif mode=='c':
        funtiona=Eresonant#(Ham,Exp,graph='False',table='False')
    else:
        raise ValueError (f'Valid functions are powder (p) or cristal (c).')
    funcname=funtiona.__name__
    iwas,jwas,kwas,weight,hulk=Delaunay(Exp)
    if dtype=='data':
        exper=exper
    elif dtype=='integral':
        fielda=np.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
        exper=scii.cumulative_trapezoid(exper,fielda,initial=0)
    else:
        raise ValueError(f"Data type {dtype} is not supported. Use integral or data instead.")
    if Var.g!=0.0:
        lowfron.extend([Var.g[0],Var.g[2],Var.g[4]])
        hifron.extend([Var.g[1],Var.g[3],Var.g[5]])
    if Var.A!=0.0:
        lowfron.extend([Var.A[0],Var.A[2],Var.A[4]])
        hifron.extend([Var.A[1],Var.A[3],Var.A[5]])
    if Var.Q!=0.0:
        lowfron.extend([Var.Q[0],Var.Q[2],Var.Q[4]])
        hifron.extend([Var.Q[1],Var.Q[3],Var.Q[5]])
    if Var.D!=0.0:
        lowfron.extend([Var.D[0],Var.D[2]])
        hifron.extend([Var.D[1],Var.D[3]])
    if np.any(Var.Hpp):
        lowfron.extend([Var.Hpp[0],Var.Hpp[2]])
        hifron.extend([Var.Hpp[1],Var.Hpp[3]])
    lowfron=np.array(lowfron,dtype=float)
    hifron=np.array(hifron,dtype=float)
    nparams=len(lowfron)
    poles=35*nparams
    if poles%2!=0:
        poles+=1
    population=np.zeros((poles,nparams))
    for ika in range(0,nparams):
        population[:,ika]=np.random.uniform(lowfron[ika],hifron[ika],poles)
    fitprice=np.zeros(poles)
    for jae in range(poles):
        fitprice[jae]=Fincost(population[jae],funcname)
    itea=0
    try:
        while itea<maximal:
            orden=np.argsort(fitprice)
            population=population[orden]
            fitprice=fitprice[orden]
            bcost=fitprice[0]
            bplayer=np.copy(population[0])
            print(f"Generation: {itea} | Best cost: {bcost:.5e}")

            if bcost<eps:
                print("Data converged.")
                break
            pool=np.zeros((poles,nparams))
            for jae in range(poles):
                tchos=np.random.randint(0,poles,size=5)
                costt=fitprice[tchos]
                win=tchos[np.argmin(costt)]
                pool[jae]=population[win]
            npopulation=np.zeros((poles,nparams))
            npopulation[0]=bplayer
            for iq in range(1,poles,2):
                vat1=pool[iq]
                vat2=pool[iq+1] if (iq+1<poles) else pool[0]
                son1,son2=Cross(vat1,vat2)
                son1=Mutategauss(son1,lowfron,hifron)
                son2=Mutategauss(son2,lowfron,hifron)
                npopulation[iq]=son1
                if (iq+1<poles):
                    npopulation[iq+1]=son2
            population=npopulation
            for jae in range(poles):
                if jae==0:
                    fitprice[jae]=bcost
                else:
                    fitprice[jae]=Fincost(population[jae],funcname)
            itea+=1
            if stopvar:
                break
    except KeyboardInterrupt:
        print("\n"+"="*50)
        print(f"Process stopped at iteration:{itea}")
        print("="*50)
        if Var.g!=0.0:
            print(f'gx={Ham.g[0]} | gy={Ham.g[1]} | gz={Ham.g[2]}')
        if Var.A!=0.0:
            print(f'Ax={Ham.A[0]} | Ay={Ham.A[1]} | Az={Ham.A[2]}')
        if Var.Q!=0.0:
            print(f'Qx={Ham.Q[0]} | Qy={Ham.Q[1]} | Qz={Ham.Q[2]}')
        if Var.D!=0.0:
            print(f'D={Ham.D[0]} | E={Ham.D[1]}')
        if np.any(Var.Hpp):
            print(f'Hppg={Ham.Hpp[0]} | Hppl={Ham.Hpp[1]}')
        if funcname in ['Powder']:
            return funtiona(Ham,Exp,graph='False')[1]
        elif funcname in ['Eresonant']:
            return funtiona(Ham,Exp,graph='False',table='False')[1]
    print("\n"+"="*50)
    print(f"Process stopped at iteration:{itea}")
    print("="*50)
    if Var.g!=0.0:
        print(f'gx={Ham.g[0]} | gy={Ham.g[1]} | gz={Ham.g[2]}')
    if Var.A!=0.0:
        print(f'Ax={Ham.A[0]} | Ay={Ham.A[1]} | Az={Ham.A[2]}')
    if Var.Q!=0.0:
        print(f'Qx={Ham.Q[0]} | Qy={Ham.Q[1]} | Qz={Ham.Q[2]}')
    if Var.D!=0.0:
        print(f'D={Ham.D[0]} | E={Ham.D[1]}')
    if np.any(Var.Hpp):
        print(f'Hppg={Ham.Hpp[0]} | Hppl={Ham.Hpp[1]}')
    if funcname in ['Powder']:
        return funtiona(Ham,Exp,graph='False')[1]
    elif funcname in ['Eresonant']:
        return funtiona(Ham,Exp,graph='False',table='False')[1]

def Genio2(Hamer,Expe,Vara,exper,eps=1e-10,maximal=30,dtype='data',mode='p'):
    def Fincost(points,funcname):
        if stopvar:
            raise KeyboardInterrupt("Stopped by user")
        jl=0
        for i in range(len(Ham.Mulham)):
            if Var.Mvary[i].g!=0.0:
                gt=points[jl:jl+3]
                if (gt[0]<Var.Mvary[i].g[0] or gt[0]>Var.Mvary[i].g[1] or gt[1]<Var.Mvary[i].g[2] or gt[1]>Var.Mvary[i].g[3] or gt[2]<Var.Mvary[i].g[4] or gt[2]>Var.Mvary[i].g[5]):
                    return 1e6
                Ham.Mulham[i].g=gt
                jl+=3
            if Var.Mvary[i].A!=0.0:
                At=points[jl:jl+3]
                if (At[0]<Var.Mvary[i].A[0] or At[0]>Var.Mvary[i].A[1] or At[1]<Var.Mvary[i].A[2] or At[1]>Var.Mvary[i].A[3] or At[2]<Var.Mvary[i].A[4] or At[2]>Var.Mvary[i].A[5]):
                    return 1e6
                Ham.Mulham[i].A=At
                jl+=3
            if Var.Mvary[i].Q!=0.0:
                Qt=points[jl:jl+3]
                if (Qt[0]<Var.Mvary[i].Q[0] or Qt[0]>Var.Mvary[i].Q[1] or Qt[1]<Var.Mvary[i].Q[2] or Qt[1]>Var.Mvary[i].Q[3] or Qt[2]<Var.Mvary[i].Q[4] or Qt[2]>Var.Mvary[i].Q[5]):
                    return 1e6
                Ham.Mulham[i].Q=Qt
                jl+=3
            if Var.Mvary[i].D!=0.0:
                Dt=points[jl:jl+2]
                if (Dt[0]<Var.Mvary[i].D[0] or Dt[0]>Var.Mvary[i].D[1] or Dt[1]<Var.Mvary[i].D[2] or Dt[1]>Var.Mvary[i].D[3]):
                    return 1e6
                Ham.Mulham[i].D=Dt
                jl+=2
            if np.any(Var.Mvary[i].Hpp):
                Htp=points[jl:jl+2]
                if (Htp[0]<Var.Mvary[i].Hpp[0] or Htp[0]>Var.Mvary[i].Hpp[1] or Htp[1]<Var.Mvary[i].Hpp[2] or Htp[1]>Var.Mvary[i].Hpp[3]):
                    return 1e6
                Ham.Mulham[i].Hpp=Htp
                jl+=2
        try:
            if funcname in ['Mulpol']:
                spect=funtiona(Ham,Exp,graph='False')[1]
            elif funcname in ['Music']:
                spect=funtiona(Ham,Exp,graph='False',table='False')[1]
            wcost=CostfG(exper,spect)
            return wcost
        except Exception:
            return 1e6
    #global stopvar
    Ham=deepcopy(Hamer)
    Exp=deepcopy(Expe)
    Var=deepcopy(Vara)
    lowfron=[]
    hifron=[]
    bplayer=0
    if mode=='p':
        funtiona=Mulpol#(Ham,Exp,graph='False')
    elif mode=='c':
        funtiona=Music#(Ham,Exp,graph='False',table='False')
    else:
        raise ValueError (f'Valid functions are powder (p) or cristal (c).')
    funcname=funtiona.__name__
    pointx=[]
    lowfron=[]
    hifron=[]
    if dtype=='data':
        exper=exper
    elif dtype=='integral':
        fielda=np.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
        exper=scii.cumulative_trapezoid(exper,fielda,initial=0)
    else:
        raise ValueError(f"Data type {dtype} is not supported. Use integral or data instead.")
    iwas,jwas,kwas,weight,hulk=Delaunay(Exp)
    numberes=len(Ham.Mulham)
    for lke in range(0,numberes):
        if Var.Mvary[lke].g!=0.0:
            pointx.extend(Ham.Mulham[lke].g)
            lowfron.extend([Var.Mvary[lke].g[0],Var.Mvary[lke].g[2],Var.Mvary[lke].g[4]])
            hifron.extend([Var.Mvary[lke].g[1],Var.Mvary[lke].g[3],Var.Mvary[lke].g[5]])
        if Var.Mvary[lke].A!=0.0:
            pointx.extend(Ham.Mulham[lke].A)
            lowfron.extend([Var.Mvary[lke].A[0],Var.Mvary[lke].A[2],Var.Mvary[lke].A[4]])
            hifron.extend([Var.Mvary[lke].A[1],Var.Mvary[lke].A[3],Var.Mvary[lke].A[5]])
        if Var.Mvary[lke].Q!=0.0:
            pointx.extend(Ham.Mulham[lke].Q)
            lowfron.extend([Var.Mvary[lke].Q[0],Var.Mvary[lke].Q[2],Var.Mvary[lke].Q[4]])
            hifron.extend([Var.Mvary[lke].Q[1],Var.Mvary[lke].Q[3],Var.Mvary[lke].Q[5]])
        if Var.Mvary[lke].D!=0.0:
            pointx.extend(Ham.Mulham[lke].D)
            lowfron.extend([Var.Mvary[lke].D[0],Var.Mvary[lke].D[2]])
            hifron.extend([Var.Mvary[lke].D[1],Var.Mvary[lke].D[3]])
        if np.any(Var.Mvary[lke].Hpp):
            pointx.extend(Ham.Mulham[lke].Hpp)
            lowfron.extend([Var.Mvary[lke].Hpp[0],Var.Mvary[lke].Hpp[2]])
            hifron.extend([Var.Mvary[lke].Hpp[1],Var.Mvary[lke].Hpp[3]])
    pointx=np.array(pointx,dtype=float)
    lowfron=np.array(lowfron,dtype=float)
    hifron=np.array(hifron,dtype=float)
    nparams=len(lowfron)
    poles=35*nparams
    if poles%2!=0:
        poles+=1
    population=np.zeros((poles,nparams))
    for ika in range(0,nparams):
        population[:,ika]=np.random.uniform(lowfron[ika],hifron[ika],poles)
    fitprice=np.zeros(poles)
    for jae in range(poles):
        fitprice[jae]=Fincost(population[jae],funcname)
    itea=0
    try:
        while itea<maximal:
            orden=np.argsort(fitprice)
            population=population[orden]
            fitprice=fitprice[orden]
            bcost=fitprice[0]
            bplayer=np.copy(population[0])
            print(f"Generation: {itea} | Best cost: {bcost:.5e}")

            if bcost<eps:
                print("Data converged.")
                break
            pool=np.zeros((poles,nparams))
            for jae in range(poles):
                tchos=np.random.randint(0,poles,size=5)
                costt=fitprice[tchos]
                win=tchos[np.argmin(costt)]
                pool[jae]=population[win]
            npopulation=np.zeros((poles,nparams))
            npopulation[0]=bplayer
            for iq in range(1,poles,2):
                vat1=pool[iq]
                vat2=pool[iq+1] if (iq+1<poles) else pool[0]
                son1,son2=Cross(vat1,vat2)
                son1=Mutategauss(son1,lowfron,hifron)
                son2=Mutategauss(son2,lowfron,hifron)
                npopulation[iq]=son1
                if (iq+1<poles):
                    npopulation[iq+1]=son2
            population=npopulation
            for jae in range(poles):
                if jae==0:
                    fitprice[jae]=bcost
                else:
                    fitprice[jae]=Fincost(population[jae],funcname)
            itea+=1
            if stopvar:
                break
    except KeyboardInterrupt:
        print("\n"+"="*50)
        print(f"Process stopped at iteration:{itea}")
        print("="*50)
        _=Fincost(bplayer, funcname)
        for i in range(len(Ham.Mulham)):
            print(f"--- System {i+1} ---")
            if Var.Mvary[i].g!=0.0:
                print(f'gx={Ham.Mulham[i].g[0]:.4f} | gy={Ham.Mulham[i].g[1]:.4f} | gz={Ham.Mulham[i].g[2]:.4f}')
            if Var.Mvary[i].A!=0.0:
                print(f'Ax={Ham.Mulham[i].A[0]:.4f} | Ay={Ham.Mulham[i].A[1]:.4f} | Az={Ham.Mulham[i].A[2]:.4f}')
            if Var.Mvary[i].Q!=0.0:
                print(f'Qx={Ham.Mulham[i].Q[0]:.4f} | Qy={Ham.Mulham[i].Q[1]:.4f} | Qz={Ham.Mulham[i].Q[2]:.4f}')
            if Var.Mvary[i].D!=0.0:
                print(f'D={Ham.Mulham[i].D[0]:.4f} | E={Ham.Mulham[i].D[1]:.4f}')
            if np.any(Var.Mvary[i].Hpp):
                print(f'Hppg={Ham.Mulham[i].Hpp[0]:.4f} | Hppl={Ham.Mulham[i].Hpp[1]:.4f}')
        if funcname in ['Mulpol']:
            return funtiona(Ham,Exp,graph='False')[1]
        elif funcname in ['Music']:
            return funtiona(Ham,Exp,graph='False',table='False')[1]
    print("\n"+"="*50)
    print(f"Process stopped at iteration:{itea}")
    print("="*50)
    _=Fincost(bplayer,funcname)
    for i in range(len(Ham.Mulham)):
        print(f"--- System {i+1} ---")
        if Var.Mvary[i].g!=0.0:
            print(f'gx={Ham.Mulham[i].g[0]:.4f} | gy={Ham.Mulham[i].g[1]:.4f} | gz={Ham.Mulham[i].g[2]:.4f}')
        if Var.Mvary[i].A!=0.0:
            print(f'Ax={Ham.Mulham[i].A[0]:.4f} | Ay={Ham.Mulham[i].A[1]:.4f} | Az={Ham.Mulham[i].A[2]:.4f}')
        if Var.Mvary[i].Q!=0.0:
            print(f'Qx={Ham.Mulham[i].Q[0]:.4f} | Qy={Ham.Mulham[i].Q[1]:.4f} | Qz={Ham.Mulham[i].Q[2]:.4f}')
        if Var.Mvary[i].D!=0.0:
            print(f'D={Ham.Mulham[i].D[0]:.4f} | E={Ham.Mulham[i].D[1]:.4f}')
        if np.any(Var.Mvary[i].Hpp):
            print(f'Hppg={Ham.Mulham[i].Hpp[0]:.4f} | Hppl={Ham.Mulham[i].Hpp[1]:.4f}')
    if funcname in ['Mulpol']:
        return funtiona(Ham,Exp,graph='False')[1]
    elif funcname in ['Music']:
        return funtiona(Ham,Exp,graph='False',table='False')[1]

def Genio(Hamer,Expe,Vara,exper,eps=1e-10,maximal=100,dtype='data',mode='p'):
    if type(Hamer)==Multham:
         if Expe.Mexp[0].Points!=len(exper):
            Expe.Mexp[0].Points=len(exper)
         scpe=Genio2(Hamer,Expe,Vara,exper,eps,maximal,dtype,mode)
    elif type(Hamer)==Hval:
         if Expe.Points!=len(exper):
            Expe.Points=len(exper)
         scpe=Genio1(Hamer,Expe,Vara,exper,eps,maximal,dtype,mode)

    return scpe

def Costf(exper,intens,metric='rmse'):
    norm=np.sum(intens*intens)
    if norm==0 or np.max(intens)<1e-10:
        return 1e6
    maxe=np.max(np.abs(exper))
    maxi=np.max(np.abs(intens))
    exper=exper/maxe
    intens=intens/maxi
    if metric=='rmse':
        # Root Mean Squared Error
        error=np.sqrt(np.mean((intens-exper)**2))
        return error
    elif metric=='correlation':
        corr=np.corrcoef(exper,intens)
        pearson=corr[0, 1]
        if np.isnan(pearson):
            return 1e6
        return 1-pearson

def Metrostair(Hamer,Exp,Var,date,stepsize,ocos,para,variable,funcname,dtype='data'):
    Ham=deepcopy(Hamer)
    iwas,jwas,kwas,weight,hulk=Delaunay(Exp)
    if 'g' in variable:
        Ham.g[0]+=np.random.normal(0,stepsize['gx'])
        Ham.g[1]+=np.random.normal(0,stepsize['gy'])
        Ham.g[2]+=np.random.normal(0,stepsize['gz'])
        if not (Var.g[0]<=Ham.g[0]<=Var.g[1] and Var.g[2]<=Ham.g[1]<=Var.g[3] and Var.g[4]<=Ham.g[2]<=Var.g[5]):
            return Hamer,ocos,False

    if 'A' in variable:
        Ham.A[0]+=np.random.normal(0,stepsize['Ax'])
        Ham.A[1]+=np.random.normal(0,stepsize['Ay'])
        Ham.A[2]+=np.random.normal(0,stepsize['Az'])
        if not (Var.A[0]<=Ham.A[0]<=Var.A[1] and Var.A[2]<=Ham.A[1]<=Var.A[3] and Var.A[4]<=Ham.A[2]<=Var.A[5]):
            return Hamer,ocos,False

    if 'Q' in variable:
        Ham.Q[0]+=np.random.normal(0,stepsize['Qx'])
        Ham.Q[1]+=np.random.normal(0,stepsize['Qy'])
        Ham.Q[2]+=np.random.normal(0,stepsize['Qz'])
        if not (Var.Q[0]<=Ham.Q[0]<=Var.Q[1] and Var.Q[2]<=Ham.Q[1]<=Var.Q[3] and Var.Q[4]<=Ham.Q[2]<=Var.Q[5]):
            return Hamer,ocos,False

    if 'D' in variable:
        Ham.D[0]+=np.random.normal(0,stepsize['D'])
        Ham.D[1]+=np.random.normal(0,stepsize['E'])
        if not (Var.D[0]<=Ham.D[0]<=Var.D[1] and Var.D[2]<=Ham.D[1]<=Var.D[3]):
            return Hamer,ocos,False

    if 'Hpp' in variable:
        Ham.Hpp[0]+=np.random.normal(0,stepsize['Hpp1'])
        Ham.Hpp[1]+=np.random.normal(0,stepsize['Hpp2'])
        if not (Var.Hpp[0]<=Ham.Hpp[0]<=Var.Hpp[1] and Var.Hpp[2]<=Ham.Hpp[1]<=Var.Hpp[3]):
            return Hamer,ocos,False
    if funcname in ['Calpowder']:
        fielda,intena=Calpowder(Ham,Exp,iwas,jwas,kwas,weight,hulk)
    elif funcname in ['Eresonant']:
        fielda,intena=Eresonant(Ham,Exp,graph='False',table='False')
    if dtype=='data':
        ncos=Costf(date,intena,metric='rmse')
    elif dtype=='integral':
        intenat=scii.cumulative_trapezoid(intena,fielda,initial=0)
        ncos=Costf(date,intenat,metric='rmse')
    else:
        ncos=1e6

    if np.isnan(ncos) or ncos>=1e6:
        return Hamer,ocos,False
    if ncos<ocos:
        return Ham,ncos,True
    else:
        Tsaf=max(para,1e-8)
        prob=np.exp(-(ncos-ocos)/Tsaf)
        if np.random.rand()<prob:
            return Ham,ncos,True
        else:
            return Hamer,ocos,False

def Metro1(Hamer,Exp,Var,dat,maximal,dtype='data',mode='p'):
    #global stopvar
    Ham1=deepcopy(Hamer)
    iwas,jwas,kwas,weight,hulk=Delaunay(Exp)
    if mode=='p':
        funtiona=Calpowder#(Ham,Exp,graph='False')
    elif mode=='c':
        funtiona=Eresonant#(Ham,Exp,graph='False',table='False')
    else:
        raise ValueError (f'Valid functions are powder (p) or cristal (c).')
    funcname=funtiona.__name__
    #First iteration
    if funcname in ['Calpowder']:
        fielda,intena=funtiona(Ham,Exp,iwas,jwas,kwas,weight,hulk)
    elif funcname in ['Eresonant']:
        fielda,intena=funtiona(Ham,Exp,graph='False',table='False')
    if dtype=='data':
        ct1=Costf(dat,intena)
    if dtype=='integral':
        dat=scii.cumulative_trapezoid(dat,fielda,initial=0)
        intena=scii.cumulative_trapezoid(intena,fielda,initial=0)
        ct1=Costf(dat,intena)
    pointx=[]
    stp={}
    lowfron=[]
    hifron=[]
    pg,pa,pq,pdr,php=20,50,50,50,50
    if Var.g!=0.0:
        pointx.extend(Ham.g)
        stp['gx']=(Var.g[1]-Var.g[0])/pg
        stp['gy']=(Var.g[3]-Var.g[2])/pg
        stp['gz']=(Var.g[5]-Var.g[4])/pg
        lowfron.extend([Var.g[0],Var.g[2],Var.g[4]])
        hifron.extend([Var.g[1],Var.g[3],Var.g[5]])
    if Var.A!=0.0:
        pointx.extend(Ham.A)
        stp['Ax']=(Var.A[1]-Var.A[0])/pa
        stp['Ay']=(Var.A[3]-Var.A[2])/pa
        stp['Az']=(Var.A[5]-Var.A[4])/pa
        lowfron.extend([Var.A[0],Var.A[2],Var.A[4]])
        hifron.extend([Var.A[1],Var.A[3],Var.A[5]])
    if Var.Q!=0.0:
        pointx.extend(Ham.Q)
        stp['Qx']=(Var.Q[1]-Var.Q[0])/pq
        stp['Qy']=(Var.Q[3]-Var.Q[2])/pq
        stp['Qz']=(Var.Q[5]-Var.Q[4])/pq
        lowfron.extend([Var.Q[0],Var.Q[2],Var.Q[4]])
        hifron.extend([Var.Q[1],Var.Q[3],Var.Q[5]])
    if Var.D!=0.0:
        pointx.extend(Ham.D)
        stp['D']=(Var.D[1]-Var.D[0])/pdr
        stp['E']=(Var.D[3]-Var.D[2])/pdr
        lowfron.extend([Var.D[0],Var.D[2]])
        hifron.extend([Var.D[1],Var.D[3]])
    if np.any(Var.Hpp):
        pointx.extend(Ham.Hpp)
        stp['Hpp1']=(Var.Hpp[1]-Var.Hpp[0])/php
        stp['Hpp2']=(Var.Hpp[3]-Var.Hpp[2])/php
        lowfron.extend([Var.Hpp[0],Var.Hpp[2]])
        hifron.extend([Var.Hpp[1],Var.Hpp[3]])

    pointx=np.array(pointx,dtype=float)
    lowfron=np.array(lowfron,dtype=float)
    hifron=np.array(hifron,dtype=float)
    securestp=stp.copy()
    metropa=ct1*0.008 #Montecarlo Temperature
    metropas=metropa
    besterror=ct1
    laster=ct1
    defcon=0
    bestHam=deepcopy(Ham1)
    try:
        for gama in range(0,maximal):
            if stopvar:
                raise KeyboardInterrupt("Stopped by user")
            acepv=0
            tries=0
            if Var.g!=0.0:
                hemetro=metropa
                opa=0
                while opa<pg:
                    Ham1,ct1,acep=Metrostair(Ham1,Exp,Var,dat,stp,ct1,hemetro,['g'],funcname,dtype)
                    if acep:
                        acepv+=1
                    tries+=1
                    hemetro=hemetro*0.99
                    opa+=1
                    if ct1<besterror:
                        besterror=ct1
                        bestHam=deepcopy(Ham1)
                Ham1=deepcopy(bestHam)
                ct1=besterror
            if Var.A!=0.0:
                hemetro=metropa
                opa=0
                while opa<pa:
                    Ham1,ct1,acep=Metrostair(Ham1,Exp,Var,dat,stp,ct1,hemetro,['A'],funcname,dtype)
                    if acep:
                        acepv+=1
                    tries+=1
                    hemetro=hemetro*0.99
                    opa+=1
                    if ct1<besterror:
                        besterror=ct1
                        bestHam=deepcopy(Ham1)
                Ham1=deepcopy(bestHam)
                ct1=besterror
            if Var.Q!=0.0:
                hemetro=metropa
                opa=0
                while opa<pq:
                    Ham1,ct1,acep=Metrostair(Ham1,Exp,Var,dat,stp,ct1,hemetro,['Q'],funcname,dtype)
                    if acep:
                        acepv+=1
                    tries+=1
                    hemetro=hemetro*0.99
                    opa+=1
                    if ct1<besterror:
                        besterror=ct1
                        bestHam=deepcopy(Ham1)
                Ham1=deepcopy(bestHam)
                ct1=besterror
            if Var.D!=0.0:
                hemetro=metropa
                opa=0
                while opa<pdr:
                    Ham1,ct1,acep=Metrostair(Ham1,Exp,Var,dat,stp,ct1,hemetro,['D'],funcname,dtype)
                    if acep:
                        acepv+=1
                    tries+=1
                    hemetro=hemetro*0.99
                    opa+=1
                    if ct1<besterror:
                        besterror=ct1
                        bestHam=deepcopy(Ham1)
                Ham1=deepcopy(bestHam)
                ct1=besterror
            if np.any(Var.Hpp):
                hemetro=metropa
                opa=0
                while opa<pdr:
                    Ham1,ct1,acep=Metrostair(Ham1,Exp,Var,dat,stp,ct1,hemetro,['Hpp'],funcname,dtype)
                    if acep:
                        acepv+=1
                    tries+=1
                    hemetro=hemetro*0.99
                    opa+=1
                    if ct1<besterror:
                        besterror=ct1
                        bestHam=deepcopy(Ham1)
                Ham1=deepcopy(bestHam)
                ct1=besterror

            #Last try with all
            varact=[]
            if Var.g!=0.0: varact.append('g')
            if Var.A!=0.0: varact.append('A')
            if Var.D!=0.0: varact.append('D')
            if np.any(Var.Hpp): varact.append('Hpp')
            if len(varact)>1:
                hemetro=metropa*1.5
                opa=0
                while opa<30:
                    Ham1,ct1,acep=Metrostair(Ham1,Exp,Var,dat,stp,ct1,hemetro,varact,funcname,dtype)
                    if acep:
                        acepv+=1
                    tries+=1
                    hemetro=hemetro*0.99
                    opa+=1
                    if ct1<besterror:
                        besterror=ct1
                        bestHam=deepcopy(Ham1)

            metropa=metropa*0.95
            arate=(acepv/tries)*100 if tries> 0 else 0
            if arate>40.0:
                for k in stp.keys():
                    stp[k]*=1.05
            elif arate<20.0:
                for k in stp.keys():
                    stp[k]*=0.95
            #Shake values
            improv=laster-besterror
            if improv<1e-4:
                defcon+=1
            else:
                defcon=0
                laster=besterror
            if defcon>=10:
                print(f"|--- Varying values to avoid stagnation ---|")
                metropa=metropas*5.0
                for k in stp.keys():
                    stp[k]*=3.0
                Ham1=deepcopy(bestHam)
                defcon=0

            #Awake the process
            if arate<10.0 and gama>20:
                metropa=metropa*100.0
                if metropa>metropas:
                    metropa=metropas
                    for k in stp.keys():
                        stp[k]=securestp[k]/2
                for k in stp.keys():
                    stp[k]*=2.0
                Ham1=deepcopy(bestHam)
                print(f"|---Process heated up---|")
            print(f"Step {gama} | Error: {ct1:.5e} | Best one: {besterror:.5e} | Rate: {arate:.1f}% | T: {metropa:.4e}")
            if Var.g!=0.0:
                print(f'gx={bestHam.g[0]} | gy={bestHam.g[1]} | gz={bestHam.g[2]}')
            if Var.A!=0.0:
                print(f'Ax={bestHam.A[0]} | Ay={bestHam.A[1]} | Az={bestHam.A[2]}')
            if Var.Q!=0.0:
                print(f'Qx={bestHam.Q[0]} | Qy={bestHam.Q[1]} | Qz={bestHam.Q[2]}')
            if Var.D!=0.0:
                print(f'D={bestHam.D[0]},E={bestHam.D[1]}')
            if np.any(Var.Hpp):
                print(f'Hppg={bestHam.Hpp[0]},Hppl={bestHam.Hpp[1]}')
    except KeyboardInterrupt:
        print("\n"+"="*50)
        print(f"Process stopped at iteration:{gama}")
        print("="*50)
        if Var.g!=0.0:
            print(f'gx={bestHam.g[0]} | gy={bestHam.g[1]} | gz={bestHam.g[2]}')
        if Var.A!=0.0:
            print(f'Ax={bestHam.A[0]} | Ay={bestHam.A[1]} | Az={bestHam.A[2]}')
        if Var.Q!=0.0:
            print(f'Qx={bestHam.Q[0]} | Qy={bestHam.Q[1]} | Qz={bestHam.Q[2]}')
        if Var.D!=0.0:
            print(f'D={bestHam.D[0]},E={bestHam.D[1]}')
        if np.any(Var.Hpp):
            print(f'Hppg={bestHam.Hpp[0]},Hppl={bestHam.Hpp[1]}')
        if funcname in ['Calpowder']:
            return Powder(bestHam,Exp,graph='False')[1]
        elif funcname in ['Eresonant']:
            return funtiona(bestHam,Exp,graph='False',table='False')[1]
    print("\n"+"="*50)
    print(f"Process stopped at iteration:{gama}")
    print("="*50)
    if Var.g!=0.0:
        print(f'gx={bestHam.g[0]} | gy={bestHam.g[1]} | gz={bestHam.g[2]}')
    if Var.A!=0.0:
        print(f'Ax={bestHam.A[0]} | Ay={bestHam.A[1]} | Az={bestHam.A[2]}')
    if Var.Q!=0.0:
        print(f'Qx={bestHam.Q[0]} | Qy={bestHam.Q[1]} | Qz={bestHam.Q[2]}')
    if Var.D!=0.0:
        print(f'D={bestHam.D[0]} | E={bestHam.D[1]}')
    if np.any(Var.Hpp):
        print(f'Hppg={bestHam.Hpp[0]} | Hppl={bestHam.Hpp[1]}')
    if funcname in ['Calpowder']:
        return Powder(bestHam,Exp,graph='False')[1]
    elif funcname in ['Eresonant']:
        return funtiona(bestHam,Exp,graph='False',table='False')[1]

def Metrostair2(Hamer,Exp,Var,date,stepsize,ocos,para,variable,aktsys,funcname,dtype='data'):
    Ham=deepcopy(Hamer)
    vma=Var.Mvary[aktsys]
    hma=Ham.Mulham[aktsys]
    if 'g' in variable:
        hma.g[0]+=np.random.normal(0,stepsize[f'gx_{aktsys}'])
        hma.g[1]+=np.random.normal(0,stepsize[f'gy_{aktsys}'])
        hma.g[2]+=np.random.normal(0,stepsize[f'gz_{aktsys}'])
        if not (vma.g[0]<=hma.g[0]<=vma.g[1] and vma.g[2]<=hma.g[1]<=vma.g[3] and vma.g[4]<=hma.g[2]<=vma.g[5]):
            return Hamer,ocos,False

    if 'A' in variable:
        hma.A[0]+=np.random.normal(0,stepsize[f'Ax_{aktsys}'])
        hma.A[1]+=np.random.normal(0,stepsize[f'Ay_{aktsys}'])
        hma.A[2]+=np.random.normal(0,stepsize[f'Az_{aktsys}'])
        if not (vma.A[0]<=hma.A[0]<=vma.A[1] and vma.A[2]<=hma.A[1]<=vma.A[3] and vma.A[4]<=hma.A[2]<=vma.A[5]):
            return Hamer,ocos,False

    if 'Q' in variable:
        hma.Q[0]+=np.random.normal(0,stepsize[f'Qx_{aktsys}'])
        hma.Q[1]+=np.random.normal(0,stepsize[f'Qy_{aktsys}'])
        hma.Q[2]+=np.random.normal(0,stepsize[f'Qz_{aktsys}'])
        if not (vma.Q[0]<=hma.Q[0]<=vma.Q[1] and vma.Q[2]<=hma.Q[1]<=vma.Q[3] and vma.Q[4]<=hma.Q[2]<=vma.Q[5]):
            return Hamer,ocos,False

    if 'D' in variable:
        hma.D[0]+=np.random.normal(0,stepsize[f'D_{aktsys}'])
        hma.D[1]+=np.random.normal(0,stepsize[f'E_{aktsys}'])
        if not (vma.D[0]<=hma.D[0]<=vma.D[1] and vma.D[2]<=hma.D[1]<=vma.D[3]):
            return Hamer,ocos,False

    if 'Hpp' in variable:
        hma.Hpp[0]+=np.random.normal(0,stepsize[f'Hpp1_{aktsys}'])
        hma.Hpp[1]+=np.random.normal(0,stepsize[f'Hpp2_{aktsys}'])
        if not (vma.Hpp[0]<=hma.Hpp[0]<=vma.Hpp[1] and vma.Hpp[2]<=hma.Hpp[1]<=vma.Hpp[3]):
            return Hamer,ocos,False
    if funcname in ['Mulpol']:
        fielda,intena=Mulpol(Ham,Exp,graph='False')
    elif funcname in ['Music']:
        fielda,intena=Music(Ham,Exp,graph='False',table='False')
    if dtype=='data':
        ncos=Costf(date,intena,metric='rmse')
    elif dtype=='integral':
        intenat=scii.cumulative_trapezoid(intena,fielda,initial=0)
        ncos=Costf(date,intenat,metric='rmse')
    else:
        ncos=1e6

    if np.isnan(ncos) or ncos>=1e6:
        return Hamer,ocos,False
    if ncos<ocos:
        return Ham,ncos,True
    else:
        Tsaf=max(para,1e-8)
        prob=np.exp(-(ncos-ocos)/Tsaf)
        if np.random.rand()<prob:
            return Ham,ncos,True
        else:
            return Hamer,ocos,False

def Metro2(Hamer,Exp,Var,dat,maximal,dtype='data',mode='p'):
    #global stopvar
    Ham1=deepcopy(Hamer)
    if mode=='p':
        funtiona=Mulpol#(Ham,Exp,graph='False')
    elif mode=='c':
        funtiona=Music#(Ham,Exp,graph='False',table='False')
    else:
        raise ValueError (f'Valid functions are powder (p) or cristal (c).')
    funcname=funtiona.__name__
    #First iteration
    if funcname in ['Mulpol']:
        fielda,intena=Mulpol(Ham,Exp,graph='False')
    elif funcname in ['Music']:
        fielda,intena=Music(Ham,Exp,graph='False',table='False')
    if dtype=='data':
        ct1=Costf(dat,intena)
    if dtype=='integral':
        dat=scii.cumulative_trapezoid(dat,fielda,initial=0)
        intena=scii.cumulative_trapezoid(intena,fielda,initial=0)
        ct1=Costf(dat,intena)
    pointx=[]
    stp={}
    lowfron=[]
    hifron=[]
    pg,pa,pq,pdr,php=20,50,50,50,50
    for ier in range(len(Ham.Mulham)):
        vma=Var.Mvary[ier]
        hma=Ham.Mulham[ier]
        if isinstance(vma.g,(list,np.ndarray)) or vma.g!=0.0:
            stp[f'gx_{ier}']=(vma.g[1]-vma.g[0])/pg
            stp[f'gy_{ier}']=(vma.g[3]-vma.g[2])/pg
            stp[f'gz_{ier}']=(vma.g[5]-vma.g[4])/pg
        if isinstance(vma.A,(list,np.ndarray)) or vma.A!=0.0:
            stp[f'Ax_{ier}']=(vma.A[1]-vma.A[0])/pa
            stp[f'Ay_{ier}']=(vma.A[3]-vma.A[2])/pa
            stp[f'Az_{ier}']=(vma.A[5]-vma.A[4])/pa
        if isinstance(vma.Q,(list, np.ndarray)) or vma.Q!=0.0:
            stp[f'Qx_{ier}']=(vma.Q[1]-vma.Q[0])/pq
            stp[f'Qy_{ier}']=(vma.Q[3]-vma.Q[2])/pq
            stp[f'Qz_{ier}']=(vma.Q[5]-vma.Q[4])/pq
        if isinstance(vma.D,(list,np.ndarray)) or vma.D!=0.0:
            stp[f'D_{ier}']=(vma.D[1]-vma.D[0])/pdr
            stp[f'E_{ier}']=(vma.D[3]-vma.D[2])/pdr
        if np.any(vma.Hpp):
            stp[f'Hpp1_{ier}']=(vma.Hpp[1]-vma.Hpp[0])/php
            stp[f'Hpp2_{ier}']=(vma.Hpp[3]-vma.Hpp[2])/php
    pointx=np.array(pointx,dtype=float)
    lowfron=np.array(lowfron,dtype=float)
    hifron=np.array(hifron,dtype=float)
    securestp=stp.copy()
    metropa=ct1*0.008 #Montecarlo Temperature
    metropas=metropa
    besterror=ct1
    laster=ct1
    defcon=0
    bestHam=deepcopy(Ham1)
    try:
        for gama in range(0,maximal):
            if stopvar:
                raise KeyboardInterrupt("Stopped by user")
            acepv=0
            tries=0
            for ira in range(len(Ham.Mulham)):
                if Var.Mvary[ira].g!=0.0:
                    hemetro=metropa
                    opa=0
                    while opa<pg:
                        Ham1,ct1,acep=Metrostair2(Ham1,Exp,Var,dat,stp,ct1,hemetro,['g'],ira,funcname,dtype)
                        if acep:
                            acepv+=1
                        tries+=1
                    hemetro=hemetro*0.99
                    opa+=1
                    if ct1<besterror:
                        besterror=ct1
                        bestHam=deepcopy(Ham1)
                    Ham1=deepcopy(bestHam)
                    ct1=besterror
                if Var.Mvary[ira].A!=0.0:
                    hemetro=metropa
                    opa=0
                    while opa<pa:
                        Ham1,ct1,acep=Metrostair2(Ham1,Exp,Var,dat,stp,ct1,hemetro,['A'],ira,funcname,dtype)
                        if acep:
                            acepv+=1
                        tries+=1
                    hemetro=hemetro*0.99
                    opa+=1
                    if ct1<besterror:
                        besterror=ct1
                        bestHam=deepcopy(Ham1)
                    Ham1=deepcopy(bestHam)
                    ct1=besterror
                if Var.Mvary[ira].Q!=0.0:
                    hemetro=metropa
                    opa=0
                    while opa<pq:
                        Ham1,ct1,acep=Metrostair2(Ham1,Exp,Var,dat,stp,ct1,hemetro,['Q'],ira,funcname,dtype)
                        if acep:
                            acepv+=1
                        tries+=1
                        hemetro=hemetro*0.99
                        opa+=1
                        if ct1<besterror:
                            besterror=ct1
                            bestHam=deepcopy(Ham1)
                    Ham1=deepcopy(bestHam)
                    ct1=besterror
                if Var.Mvary[ira].D!=0.0:
                    hemetro=metropa
                    opa=0
                    while opa<pdr:
                        Ham1,ct1,acep=Metrostair2(Ham1,Exp,Var,dat,stp,ct1,hemetro,['D'],ira,funcname,dtype)
                        if acep:
                            acepv+=1
                        tries+=1
                        hemetro=hemetro*0.99
                        opa+=1
                        if ct1<besterror:
                            besterror=ct1
                            bestHam=deepcopy(Ham1)
                    Ham1=deepcopy(bestHam)
                    ct1=besterror
                if np.any(Var.Mvary[ira].Hpp):
                    hemetro=metropa
                    opa=0
                    while opa<pdr:
                        Ham1,ct1,acep=Metrostair2(Ham1,Exp,Var,dat,stp,ct1,hemetro,['Hpp'],ira,funcname,dtype)
                        if acep:
                            acepv+=1
                        tries+=1
                        hemetro=hemetro*0.99
                        opa+=1
                        if ct1<besterror:
                            besterror=ct1
                            bestHam=deepcopy(Ham1)
                    Ham1=deepcopy(bestHam)
                    ct1=besterror

            #Last try with all
            hemetro=metropa*1.5
            opa=0
            while opa<30:
                for ira in range(len(Ham.Mulham)):
                    varact=[]
                    if Var.Mvary[ira].g!=0.0:
                        varact.append('g')
                    if Var.Mvary[ira].A!=0.0:
                        varact.append('A')
                    if Var.Mvary[ira].D!=0.0:
                        varact.append('D')
                    if np.any(Var.Mvary[ira].Hpp):
                        varact.append('Hpp')
                    if len(varact)>0:
                        Ham1,ct1,acep=Metrostair2(Ham1,Exp,Var,dat,stp,ct1,hemetro,varact,ira,funcname,dtype)
                        if acep:
                            acepv+=1
                        tries+=1
                hemetro=hemetro*0.99
                opa+=1
                if ct1<besterror:
                    besterror=ct1
                    bestHam=deepcopy(Ham1)

            metropa=metropa*0.95
            arate=(acepv/tries)*100 if tries> 0 else 0
            if arate>40.0:
                for k in stp.keys():
                    stp[k]*=1.05
            elif arate<20.0:
                for k in stp.keys():
                    stp[k]*=0.95
            #Shake values
            improv=laster-besterror
            if improv<1e-4:
                defcon+=1
            else:
                defcon=0
                laster=besterror
            if defcon>=10:
                print(f"|--- Varying values to avoid stagnation ---|")
                metropa=metropas*5.0
                for k in stp.keys():
                    stp[k]*=3.0
                Ham1=deepcopy(bestHam)
                defcon=0

            #Awake the process
            if arate<10.0 and gama>20:
                metropa=metropa*100.0
                if metropa>metropas:
                    metropa=metropas
                    for k in stp.keys():
                        stp[k]=securestp[k]/2
                for k in stp.keys():
                    stp[k]*=2.0
                Ham1=deepcopy(bestHam)
                print(f"|---Process heated up---|")
            print(f"Step {gama} | Error: {ct1:.5e} | Best one: {besterror:.5e} | Rate: {arate:.1f}% | T: {metropa:.4e}")
            for i in range(len(Ham.Mulham)):
                print(f"--- System {i+1} ---")
                if Var.Mvary[i].g!=0.0:
                    print(f'gx={bestHam.Mulham[i].g[0]:.4f} | gy={bestHam.Mulham[i].g[1]:.4f} | gz={bestHam.Mulham[i].g[2]:.4f}')
                if Var.Mvary[i].A!=0.0:
                    print(f'Ax={bestHam.Mulham[i].A[0]:.4f} | Ay={bestHam.Mulham[i].A[1]:.4f} | Az={bestHam.Mulham[i].A[2]:.4f}')
                if Var.Mvary[i].Q!=0.0:
                    print(f'Qx={bestHam.Mulham[i].Q[0]:.4f} | Qy={bestHam.Mulham[i].Q[1]:.4f} | Qz={bestHam.Mulham[i].Q[2]:.4f}')
                if Var.Mvary[i].D!=0.0:
                    print(f'D={bestHam.Mulham[i].D[0]:.4f} | E={bestHam.Mulham[i].D[1]:.4f}')
                if np.any(Var.Mvary[i].Hpp):
                    print(f'Hppg={Ham.Mulham[i].Hpp[0]:.4f} | Hppl={Ham.Mulham[i].Hpp[1]:.4f}')
    except KeyboardInterrupt:
        print("\n"+"="*50)
        print(f"Process stopped at iteration:{gama}")
        print("="*50)
        for i in range(len(Ham.Mulham)):
            print(f"--- System {i+1} ---")
            if Var.Mvary[i].g!=0.0:
                print(f'gx={bestHam.Mulham[i].g[0]:.4f} | gy={bestHam.Mulham[i].g[1]:.4f} | gz={bestHam.Mulham[i].g[2]:.4f}')
            if Var.Mvary[i].A!=0.0:
                print(f'Ax={bestHam.Mulham[i].A[0]:.4f} | Ay={bestHam.Mulham[i].A[1]:.4f} | Az={bestHam.Mulham[i].A[2]:.4f}')
            if Var.Mvary[i].Q!=0.0:
                print(f'Qx={bestHam.Mulham[i].Q[0]:.4f} | Qy={bestHam.Mulham[i].Q[1]:.4f} | Qz={bestHam.Mulham[i].Q[2]:.4f}')
            if Var.Mvary[i].D!=0.0:
                print(f'D={bestHam.Mulham[i].D[0]:.4f} | E={bestHam.Mulham[i].D[1]:.4f}')
            if np.any(Var.Mvary[i].Hpp):
                print(f'Hppg={Ham.Mulham[i].Hpp[0]:.4f} | Hppl={Ham.Mulham[i].Hpp[1]:.4f}')
        if funcname in ['Mulpol']:
            return funtiona(bestHam,Exp,graph='False')[1]
        elif funcname in ['Music']:
            return funtiona(bestHam,Exp,graph='False',table='False')[1]
    print("\n"+"="*50)
    print(f"Process stopped at iteration:{gama}")
    print("="*50)
    for i in range(len(Ham.Mulham)):
        print(f"--- System {i+1} ---")
        if Var.Mvary[i].g!=0.0:
            print(f'gx={bestHam.Mulham[i].g[0]:.4f} | gy={bestHam.Mulham[i].g[1]:.4f} | gz={bestHam.Mulham[i].g[2]:.4f}')
        if Var.Mvary[i].A!=0.0:
            print(f'Ax={bestHam.Mulham[i].A[0]:.4f} | Ay={bestHam.Mulham[i].A[1]:.4f} | Az={bestHam.Mulham[i].A[2]:.4f}')
        if Var.Mvary[i].Q!=0.0:
            print(f'Qx={bestHam.Mulham[i].Q[0]:.4f} | Qy={bestHam.Mulham[i].Q[1]:.4f} | Qz={bestHam.Mulham[i].Q[2]:.4f}')
        if Var.Mvary[i].D!=0.0:
            print(f'D={bestHam.Mulham[i].D[0]:.4f} | E={bestHam.Mulham[i].D[1]:.4f}')
        if np.any(Var.Mvary[i].Hpp):
            print(f'Hppg={bestHam.Mulham[i].Hpp[0]:.4f} | Hppl={bestHam.Mulham[i].Hpp[1]:.4f}')
    if funcname in ['Multa']:
        return funtiona(bestHam,Exp,graph='False')[1]
    elif funcname in ['Music']:
        return funtiona(bestHam,Exp,graph='False',table='False')[1]

def Metro(Hamer,Exp,Var,exper,maximal=2000,dtype='data',mode='p'):
    if type(Hamer)==Multham:
         if Exp.Mexp[0].Points!=len(exper):
            Exp.Mexp[0].Points=len(exper)
         scpe=Metro2(Hamer,Exp,Var,exper,maximal,dtype,mode)
    elif type(Hamer)==Hval:
         if Exp.Points!=len(exper):
            Exp.Points=len(exper)
         scpe=Metro1(Hamer,Exp,Var,exper,maximal,dtype,mode)
    return scpe

def Packtoscipy(Ham,Var):
    pointx=[]
    lowfron=[]
    hifron=[]
    if Var.g!=0.0:
        pointx.extend(Ham.g)
        lowfron.extend([Var.g[0],Var.g[2],Var.g[4]])
        hifron.extend([Var.g[1],Var.g[3],Var.g[5]])
    if Var.A!=0.0:
        pointx.extend(Ham.A)
        lowfron.extend([Var.A[0],Var.A[2],Var.A[4]])
        hifron.extend([Var.A[1],Var.A[3],Var.A[5]])
    if Var.Q!=0.0:
        pointx.extend(Ham.Q)
        lowfron.extend([Var.Q[0],Var.Q[2],Var.Q[4]])
        hifron.extend([Var.Q[1],Var.Q[3],Var.Q[5]])
    if Var.D!=0.0:
        pointx.extend(Ham.D)
        lowfron.extend([Var.D[0],Var.D[2]])
        hifron.extend([Var.D[1],Var.D[3]])
    if np.any(Var.Hpp):
        pointx.extend(Ham.Hpp)
        lowfron.extend([Var.Hpp[0],Var.Hpp[2]])
        if Var.Hpp[0]==Var.Hpp[1]:
            hifron.extend([Var.Hpp[1]+0.0001,Var.Hpp[3]])
        elif Var.Hpp[2]==Var.Hpp[3]:
            hifron.extend([Var.Hpp[1],Var.Hpp[3]+0.0001])
    pointx=np.array(pointx,dtype=float)
    lowfron=np.array(lowfron,dtype=float)
    hifron=np.array(hifron,dtype=float)
    return pointx,lowfron,hifron

def UnpackHam(point,Hamer,Var):
    Ham=deepcopy(Hamer)
    idx=0
    if Var.g!=0.0:
        Ham.g=point[idx:idx+3]
        idx+=3
    if Var.A!=0.0:
        Ham.A=point[idx:idx+3]
        idx+=3
    if Var.Q!=0.0:
        Ham.Q=point[idx:idx+3]
        idx+=3
    if Var.D!=0.0:
        Ham.D=point[idx:idx+2]
        idx+=2
    if np.any(Var.Hpp):
        Ham.Hpp=point[idx:idx+2]
    return Ham

def Residuals(point,Hame,Exp,exper,Vary,fname):
    #global stopvar
    if stopvar:
        raise KeyboardInterrupt("Stopped by user")
    Ham1=UnpackHam(point,Hame,Vary)
    if fname in ['Powder']:
        fielda,intena=Powder(Ham1,Exp,graph='False')
    elif fname in ['Eresonant']:
        fielda,intena=Eresonant(Ham1,Exp,graph='False',table='False')
    normi=np.linalg.norm(intena)
    if normi==0 or np.isnan(normi):
        return exper*1e6
    norme=np.linalg.norm(exper)
    enorm=exper/norme
    inorm=intena/normi
    return inorm-enorm

def LSquare1(Ham1,Expe,Vary,exper,maximal=1000,mode='p'):
    if mode=='p':
        funtiona=Powder#(Ham,Exp,graph='False')
    elif mode=='c':
        funtiona=Eresonant#(Ham,Exp,graph='False',table='False')
    else:
        raise ValueError (f'Valid functions are powder (p) or cristal (c).')
    funcname=funtiona.__name__
    pointx,lowfron,hifron=Packtoscipy(Ham1,Vary)
    frontier=lowfron,hifron
    eps=1e-6
    pointx=np.clip(pointx,lowfron+eps,hifron-eps)
    leasts
    result=leasts(fun=Residuals,args=(Ham1,Expe,exper,Vary,funcname),x0=pointx,bounds=frontier,method='trf',max_nfev=maximal,verbose=2,diff_step=1e-3,
                  xtol=1e-10,ftol=1e-10,gtol=1e-10)
    bestone=result.x
    bHam=UnpackHam(bestone,Ham1,Vary)

    print(f"Final cost: {result.cost:.5f}")
    print(f"Message: {result.message}")
    if Vary.g!=0.0:
        print(f'gx={bHam.g[0]} | gy={bHam.g[1]} | gz={bHam.g[2]}')
    if Vary.A!=0.0:
        print(f'Ax={bHam.A[0]} | Ay={bHam.A[1]} | Az={bHam.A[2]}')
    if Vary.Q!=0.0:
        print(f'Qx={bHam.Q[0]} | Qy={bHam.Q[1]} | Qz={bHam.Q[2]}')
    if Vary.D!=0.0:
        print(f'D={bHam.D[0]} | E={bHam.D[1]}')
    if np.any(Vary.Hpp):
        print(f'Hppg={bHam.Hpp[0]} | Hppl={bHam.Hpp[1]}')
    if funcname in ['Powder']:
        return Powder(bHam,Exp,graph='False')[1]
    elif funcname in ['Eresonant']:
        return Eresonant(bHam,Exp,graph='False',table='False')[1]

def Packtoscipy2(Ham,Var):
    pointx=[]
    lowfron=[]
    hifron=[]
    for jer in range(0,len(Ham.Mulham)):
        vmi=Var.Mvary[jer]
        hmi=Ham.Mulham[jer]
        if vmi.g!=0.0:
            pointx.extend(hmi.g)
            lowfron.extend([vmi.g[0],vmi.g[2],vmi.g[4]])
            hifron.extend([vmi.g[1],vmi.g[3],vmi.g[5]])
        if vmi.A!=0.0:
            pointx.extend(hmi.A)
            lowfron.extend([vmi.A[0],vmi.A[2],vmi.A[4]])
            hifron.extend([vmi.A[1],vmi.A[3],vmi.A[5]])
        if vmi.Q!=0.0:
            pointx.extend(hmi.Q)
            lowfron.extend([vmi.Q[0],vmi.Q[2],vmi.Q[4]])
            hifron.extend([vmi.Q[1],vmi.Q[3],vmi.Q[5]])
        if vmi.D!=0.0:
            pointx.extend(hmi.D)
            lowfron.extend([vmi.D[0],vmi.D[2]])
            hifron.extend([vmi.D[1],vmi.D[3]])
        if np.any(vmi.Hpp):
            pointx.extend(hmi.Hpp)
            lowfron.extend([vmi.Hpp[0],vmi.Hpp[2]])
            hifron.extend([vmi.Hpp[1],vmi.Hpp[3]])
    pointx=np.array(pointx,dtype=float)
    lowfron=np.array(lowfron,dtype=float)
    hifron=np.array(hifron,dtype=float)
    return pointx,lowfron,hifron

def UnpackHam2(point,Hamer,Var):
    Ham=deepcopy(Hamer)
    idx=0
    for jer in range(len(Ham.Mulham)):
        vmi=Var.Mvary[jer]
        hmi=Ham.Mulham[jer]
        if vmi.g!=0.0:
            hmi.g=point[idx:idx+3]
            idx+=3
        if vmi.A!=0.0:
            hmi.A=point[idx:idx+3]
            idx+=3
        if vmi.Q!=0.0:
            hmi.Q=point[idx:idx+3]
            idx+=3
        if vmi.D!=0.0:
            hmi.D=point[idx:idx+2]
            idx+=2
        if np.any(vmi.Hpp):
            hmi.Hpp=point[idx:idx+2]
            idx+=2
    return Ham

def Residuals2(point,Hame,Exp,exper,Vary,fname):
    #global stopvar
    if stopvar:
        raise KeyboardInterrupt("Stopped by user")
    Ham1=UnpackHam2(point,Hame,Vary)
    try:
        if fname in ['Mulpol']:
            fielda,intena=Mulpol(Ham1,Exp,graph='False')
        elif fname in ['Music']:
            fielda,intena=Music(Ham1,Exp,graph='False',table='False')
        normi=np.linalg.norm(intena)
        if normi==0 or np.isnan(normi):
            return exper*1e6
        norme=np.linalg.norm(exper)
        enorm=exper/norme
        inorm=intena/normi
        return inorm-enorm
    except Exception:
        return exper*1e6

def LSquare2(Ham1,Expe,Vary,exper,maximal=1000,mode='p'):
    if mode=='p':
        funtiona=Mulpol#(Ham,Exp,graph='False')
    elif mode=='c':
        funtiona=Music#(Ham,Exp,graph='False',table='False')
    else:
        raise ValueError (f'Valid functions are powder (p) or cristal (c).')
    funcname=funtiona.__name__
    pointx,lowfron,hifron=Packtoscipy2(Ham1,Vary)
    frontier=lowfron,hifron
    eps=1e-6
    pointx=np.clip(pointx,lowfron+eps,hifron-eps)
    leasts
    result=leasts(fun=Residuals2,args=(Ham1,Expe,exper,Vary,funcname),x0=pointx,bounds=frontier,method='trf',max_nfev=maximal,verbose=2,diff_step=1e-3,
                  xtol=1e-10,ftol=1e-10,gtol=1e-10)
    bestone=result.x
    bHam=UnpackHam2(bestone,Ham1,Vary)

    print(f"Final cost: {result.cost:.5f}")
    print(f"Message: {result.message}")
    for i in range(len(Ham.Mulham)):
        print(f"--- System {i+1} ---")
        if Vary.Mvary[i].g!=0.0:
            print(f'gx={bHam.Mulham[i].g[0]:.4f} | gy={bHam.Mulham[i].g[1]:.4f} | gz={bHam.Mulham[i].g[2]:.4f}')
        if Vary.Mvary[i].A!=0.0:
            print(f'Ax={bHam.Mulham[i].A[0]:.4f} | Ay={bHam.Mulham[i].A[1]:.4f} | Az={bHam.Mulham[i].A[2]:.4f}')
        if Vary.Mvary[i].Q!=0.0:
            print(f'Qx={bHam.Mulham[i].Q[0]:.4f} | Qy={bHam.Mulham[i].Q[1]:.4f} | Qz={bHam.Mulham[i].Q[2]:.4f}')
        if Vary.Mvary[i].D!=0.0:
            print(f'D={bHam.Mulham[i].D[0]:.4f} | E={bHam.Mulham[i].D[1]:.4f}')
        if np.any(Vary.Mvary[i].Hpp):
            print(f'Hppg={bHam.Mulham[i].Hpp[0]:.4f} | Hppl={bHam.Mulham[i].Hpp[1]:.4f}')
    if funcname in ['Mulpol']:
        return Mulpol(bHam,Exp,graph='False')[1]
    elif funcname in ['Music']:
        return Music(bHam,Exp,graph='False',table='False')[1]

def LSquare(Ham1,Expe,Vary,exper,maximal=1000,mode='p'):
    if type(Ham1)==Multham:
         if Expe.Mexp[0].Points!=len(exper):
            Expe.Mexp[0].Points=len(exper)
         scpe=LSquare2(Ham1,Expe,Vary,exper,maximal,mode)
    elif type(Ham1)==Hval:
         if Expe.Points!=len(exper):
            Expe.Points=len(exper)
         scpe=LSquare1(Ham1,Expe,Vary,exper,maximal,mode)
    return scpe

def Fitting(Hamer,Exper,Vara,datexp):
    global stopvar
    global result
    stopvar=False
    result={}
    lbl1=Label('Method:')
    wdg1=Dropdown(options=['Nelder-Mead','Genetic algorithm','Metropolis','Least squares'],index=2)
    frvar1=HBox([lbl1,wdg1])
    lbl2=Label('Data type:')
    wdg2=RadioButtons(options=['Spectrum','First Integral'],index=0)
    frvar2=HBox([lbl2,wdg2])
    lbl3=Label('Sample type:')
    wdg3=RadioButtons(options=['Powder','Cristal'],index=0)
    frvar3=HBox([lbl3,wdg3])
    lbl4=Label('Max. # of iterations:')
    wdg4=IntText(value=100,step=50,layout=Layout(width='200px'))
    frvar4=VBox([lbl4,wdg4])
    lbl5=Label('Max. error (10^x) [Input Exponent]:')
    wdg5=IntText(value=-10,step=1,layout=Layout(width='200px'))
    frvar5=VBox([lbl5,wdg5])
    frvar6=Button(description='Start',button_style='success',icon='play',layout=Layout(border='2px solid blue'))
    frvar7=Button(description='Stop (Interrupt)',button_style='danger',icon='stop',layout=Layout(border='2px solid red'))
    tapts=HBox([frvar6,frvar7])
    outside=Output()
    def Evalfunc(b):
        global stopvar
        stopvar=False
        with outside:
            outside.clear_output()
            Chmet=wdg1.value
            cdtype=wdg2.value
            csample=wdg3.value
            numtr=wdg4.value
            erroreps=10**float(wdg5.value)
            if cdtype=='Spectrum':
                cdtype='data'
            elif cdtype=='First Integral':
                cdtype='integral'
            if csample=='Powder':
                csample='p'
            elif csample=='Cristal':
                csample='c'
            print(f'Starting process: {Chmet}...')
            def functiontorun():
                try:
                    resultexper=None
                    if Chmet=='Nelder-Mead':
                        resultexper=Nelder(Hamer,Exper,Vara,datexp,eps=erroreps,maximal=numtr,dtype=cdtype,mode=csample)
                    elif Chmet=='Genetic algorithm':
                        resultexper=Genio(Hamer,Exper,Vara,datexp,eps=erroreps,maximal=numtr,dtype=cdtype,mode=csample)
                    elif Chmet=='Metropolis':
                        resultexper=Metro(Hamer,Exper,Vara,datexp,maximal=numtr,dtype=cdtype,mode=csample)
                    elif Chmet=='Least squares':
                        resultexper=LSquare(Hamer,Exper,Vara,datexp,maximal=numtr,mode=csample)
                    with outside:
                        #To show the graph
                        fig=Figure(figsize=(8,10))
                        canvas=FigureCanvasAgg(fig)
                        ax=fig.add_subplot(211)
                        if len(Exper)!=1:
                            Bla=np.linspace(Exper.Mexp[0].Frange[0],Exper.Mexp[0].Frange[1],Exper.Mexp[0].Points)
                        else:
                            Bla=np.linspace(Exper.Frange[0],Exper.Frange[1],Exper.Points)
                        ax.plot(Bla,datexp,color='blue',label='Data')
                        if resultexper is not None:
                            ax.plot(Bla,resultexper/np.max(resultexper)*np.max(datexp),color='red',label='Fit')
                        ax.legend()
                        ax.grid()
                        buf=io.BytesIO()
                        fig.savefig(buf,format='png',bbox_inches='tight')
                        buf.seek(0)
                        img=Image(data=buf.getvalue(),format='png')
                        display(img)
                        buf.close()
                        fig.clf()
                except Exception as e:
                    with outside:
                        print(f"\n[X] Error in function: {e}")

            partoeval=threading.Thread(target=functiontorun)
            partoeval.start()

    def Stopfunc(b):
        global stopvar
        stopvar=True
        with outside:
            print("\n[!] Stopping the process.")

    frvar6.on_click(Evalfunc)
    frvar7.on_click(Stopfunc)

    centerone=HBox([VBox([frvar1,frvar2,frvar3]),VBox([frvar4,frvar5])])
    display(VBox([centerone,tapts,outside]))
