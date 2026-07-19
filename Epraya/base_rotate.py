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
from itertools import product as iterproduct
from .base_cris import *
from .base_ham import *
from .base_powd import *


def Nrotate(Hamer,Expe,phi=0):
    Ham=deepcopy(Hamer)
    Exp=deepcopy(Expe)
    if Exp.Freq<=0:
        raise ValueError("Frequency can't be a negative or zero value")
    if Exp.Frange[0]<0:
        Exp.Frange[0]=0
        print("WARNING: A negative value has been used, changing it to zero.")
    if Exp.Frange[0]>=Exp.Frange[1]:
        raise ValueError("Field range can't be from higher to lower values.")
    Ham.D,Ham.A=np.asarray(Ham.D),np.asarray(Ham.A)
    Ham.Bk2,Ham.Bk4=np.asarray(Ham.Bk2),np.asarray(Ham.Bk4)
    Ham.Bk6,Ham.Q=np.asarray(Ham.Bk6),np.asarray(Ham.Q)
    slit,nlit,llit=PMsmi(Ham.I,Ham.S,Ham.L)
    Ham.A=Ham.A/1000
    Ham.D=Ham.D/1000
    Ham.Q=Ham.Q/1000
    Ham.Bk2=Ham.Bk2/1000
    Ham.Bk4=Ham.Bk4/1000
    Ham.Bk6=Ham.Bk6/1000
    dim=int(2*Ham.S+1)*int(2*Ham.I+1)*int(2*Ham.L+1)
    Ham=chaframe(Ham,Exp)
    sx,sy,sz=Pauli(Ham.S)
    ix,iy,iz=Pauli(Ham.I)
    isx=np.kron(sx,np.eye(int(2*Ham.I+1)))
    isx=np.kron(np.eye(int(2*Ham.L+1)),isx)
    isx=np.asarray(isx,dtype=np.complex128)
    isy=np.kron(sy,np.eye(int(2*Ham.I+1)))
    isy=np.kron(np.eye(int(2*Ham.L+1)),isy)
    isy=np.asarray(isy,dtype=np.complex128)
    isz=np.kron(sz,np.eye(int(2*Ham.I+1)))
    isz=np.kron(np.eye(int(2*Ham.L+1)),isz)
    isz=np.asarray(isz,dtype=np.complex128)
    E=Exp.Freq
    espac1=np.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
    beta=(scic.physical_constants["Bohr magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
    betan=(scic.physical_constants["nuclear magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
    h1=np.zeros((dim,dim),dtype='complex128')
    hzex=np.asarray(beta*Hze(sx,sy,sz,Ham.g,[1,0,0],dim),dtype=np.complex128)
    hzey=np.asarray(beta*Hze(sx,sy,sz,Ham.g,[0,1,0],dim),dtype=np.complex128)
    hzez=np.asarray(beta*Hze(sx,sy,sz,Ham.g,[0,0,1],dim),dtype=np.complex128)
    if Ham.S>=1:
        h1=h1+StevensO(sx,sy,sz,Ham.S,Ham,dim)
    if Ham.L!=0:
        h1=h1+Lorbit(sx,sy,sz,Ham.lc,dim,Ham.L)
    if Ham.I!=0:
        h1=h1+Hfi(sx,sy,sz,ix,iy,iz,Ham.A,dim)
        nhzex=np.asarray(betan*Nhze(Ham.I,ix,iy,iz,dim,Ham.Nucl,[1,0,0]),dtype=complex)
        nhzey=np.asarray(betan*Nhze(Ham.I,ix,iy,iz,dim,Ham.Nucl,[0,1,0]),dtype=complex)
        nhzez=np.asarray(betan*Nhze(Ham.I,ix,iy,iz,dim,Ham.Nucl,[0,0,1]),dtype=complex)
        hzex-=nhzex
        hzey-=nhzey
        hzez-=nhzez
    if np.any(Ham.Q):
        h1=h1+Qii(ix,iy,iz,Ham.Q,dim)
    h1=np.asarray(h1,dtype=np.complex128)
    #Rotations
    direct=[]
    theta=np.linspace(0,180,181)
    phi=np.deg2rad(phi)
    theta=np.deg2rad(theta)
    for alfa in theta:
        nx=np.sin(alfa)*np.cos(phi)
        ny=np.sin(alfa)*np.sin(phi)
        nz=np.cos(alfa)
        nd=np.array([nx,ny,nz])
        nd=nd/np.linalg.norm(nd)
        direct.append(nd)
    fpoints=1000
    cpoints=400
    Blist1=np.linspace(Exp.Frange[0],Exp.Frange[1],cpoints)
    Blist2=np.linspace(Exp.Frange[0],Exp.Frange[1],fpoints)
    anglexva=[]
    fieldy=[]
    intval=[]
    for ang,ne in enumerate(direct):
        nx,ny,nz=ne
        Elist1,Vlist1,h2=Padaptarray(Blist1,h1,hzex,hzey,hzez,nx,ny,nz)
        #Elist1,Vlis1t=Pretrack(Elist1,Vlist1)
        splines=cubichers(Blist1,Elist1,axis=0)
        Elist2=splines(Blist2)
        refil,intfil=Nresina(Blist2,Blist1,Elist2,Vlist1,dim,Exp.Freq,isx,isy,isz,nx,ny,nz,Exp.Temperature,h2)
        if len(refil)>0:
            anglexva.extend([theta[ang]]*len(refil))
            fieldy.extend(refil)
            intval.extend(intfil)

    plt.figure(figsize=(8,6))
    threshold=0.3
    anglex=np.array(anglexva)*180/np.pi
    fieldey=np.array(fieldy)
    intens=np.array(intval)
    imax=np.max(intens)
    if imax>0:
        normit=intens/imax
    else:
        normit=intens
    formask=normit<=threshold
    forx=anglex[formask]
    fory=fieldey[formask]
    plt.scatter(fory,forx,s=0.5,color='black',alpha=0.1,label='Forbidden')
    allmask=normit>threshold
    allx=anglex[allmask]
    ally=fieldey[allmask]
    plt.scatter(ally,allx,s=1.0,color='black',alpha=0.8,label='Allowed')
    plt.ylabel(r'Angle $\theta$ ($^{\circ}$)',fontsize=14)
    plt.xlabel('Line position (mT)',fontsize=14)
    plt.ylim(0,180)
    plt.yticks(np.arange(0,181,30))
    plt.grid(True)
    plt.title(f'Rotation Map: S={Ham.S}, I={Ham.I} and {E=:.2f} GHz',fontsize=16)
    plt.show()
    return anglex,fieldey,intens

def Pot(espac2,enegria,curvebasis,resonants,lab,espac1):
      slit,nlit,llit,transitions=Msmi(Ham.I,Ham.S,Ham.L)
      import plotly.graph_objects as pgo
      import plotly.colors as pc
      graphe=pgo.Figure()
      elk=0
      col=pc.sample_colorscale('Viridis',[k/(len(enegria[0])-1) for k in range(len(enegria[0]))])
      coel=pc.sample_colorscale('Jet',[k/(len(enegria[0])-1) for k in range(len(enegria[0]))])
      for elk in range(0,len(enegria[0])):
        basidx=curvebasis[elk]
        labelr=Getlabel(basidx,slit,nlit,llit,Ham.L,Ham.I)
        graphe.add_trace(pgo.Scatter(x=espac2,y=enegria[:,elk],mode='lines',line=dict(color=col[elk]),name=labelr,legendgroup="En",legendgrouptitle_text="Energies",legend="legend"))
      for r in resonants:
        fv=r['field']
        idi,idj=r['inx']
        eni=np.interp(fv,espac2,enegria[:,idi])
        enj=np.interp(fv,espac2,enegria[:,idj])
        if r['type']=='Allowed':
            graphe.add_trace(pgo.Scatter(x=[fv,fv],y=[eni,enj],mode='markers+lines',line=dict(color=coel[idi]),name=f"Field: {fv:.2f} mT",legendgroup="Fir",
        legendgrouptitle_text="R. Fields",legend="legend2"))
        else:
            graphe.add_trace(pgo.Scatter(x=[fv,fv],y=[eni,enj],mode='markers+lines',line=dict(color='gray'),name=f"Field: {fv:.2f} mT",legendgroup="Fi",
        legendgrouptitle_text="R. Fields",legend="legend2"))
      graphe.update_layout(
        title={'text':f'Energy VS Field: {lab} Orientation','xanchor':'center','yanchor':'auto','x':0.40, 'y':0.95,
               'font': dict(family='Georgia', size=24,color='black')
               },
        xaxis=dict(title='Field [mT]',showline=True,linecolor='black',mirror=True,linewidth=2,showgrid=True,gridcolor='black',range=[espac1[0],espac1[-1]+10])
        ,yaxis=dict(title='Energy [GHz]',showline=True,linecolor='black',mirror=True,linewidth=2,showgrid=True,gridcolor='black'),
        plot_bgcolor='white',
        width=1075,
        height=600,
        legend=dict(
            x=1.02,y=1,xanchor='left',yanchor='top',bgcolor='rgba(0,0,0,0)'
            ,groupclick="toggleitem"
        ),
        legend2=dict(
            x=1.25,y=1,xanchor='left',yanchor='top',bgcolor='rgba(0,0,0,0)'
            ,groupclick="toggleitem"
        ),
        margin=dict(l=50,r=250,b=50,t=70,pad=4),
        showlegend=True
      )
      upbutton=[
          dict(label="All", method="update", args=[{"visible": [True]*len(graphe.data)}]),
          dict(label="Fields", method="update", args=[{"visible": [trace.legendgroup!="Fi" for trace in graphe.data],}]),
          dict(label="Energy", method="update", args=[{"visible": [trace.legendgroup=='En' for trace in graphe.data]}])
      ]
      graphe.update_layout(

          updatemenus=[dict(
              type="buttons", direction="up",
              buttons=upbutton,
              x=0.01, xanchor="auto", y=1, yanchor="auto"
          )]
      )
      graphe.show()

def Ori(Hamer,Expe):
    Ham=deepcopy(Hamer)
    Exp=deepcopy(Expe)
    if Exp.Freq<=0:
        raise ValueError("Frequency can't be a negative or zero value")
    if Exp.Frange[0]<0:
        Exp.Frange[0]=0
        print("WARNING: A negative value has been used, changing it to zero.")
    if Exp.Frange[0]>=Exp.Frange[1]:
        raise ValueError("Field range can't be from higher to lower values.")
    Ham.D,Ham.A=np.asarray(Ham.D),np.asarray(Ham.A)
    Ham.Bk2,Ham.Bk4=np.asarray(Ham.Bk2),np.asarray(Ham.Bk4)
    Ham.Bk6,Ham.Q=np.asarray(Ham.Bk6),np.asarray(Ham.Q)
    slit,nlit,llit,transitions=Msmi(Ham.I,Ham.S,Ham.L)
    Ham.A=Ham.A/1000
    Ham.D=Ham.D/1000
    Ham.Q=Ham.Q/1000
    Ham.Bk2=Ham.Bk2/1000
    Ham.Bk4=Ham.Bk4/1000
    Ham.Bk6=Ham.Bk6/1000
    dim=int(2*Ham.S+1)*int(2*Ham.I+1)*int(2*Ham.L+1)
    Ham=chaframe(Ham,Exp)
    sx,sy,sz=Pauli(Ham.S)
    ix,iy,iz=Pauli(Ham.I)
    isx=np.kron(sx,np.eye(int(2*Ham.I+1)))
    isx=np.kron(np.eye(int(2*Ham.L+1)),isx)
    isx=np.asarray(isx,dtype=np.complex128)
    isy=np.kron(sy,np.eye(int(2*Ham.I+1)))
    isy=np.kron(np.eye(int(2*Ham.L+1)),isy)
    isy=np.asarray(isy,dtype=np.complex128)
    isz=np.kron(sz,np.eye(int(2*Ham.I+1)))
    isz=np.kron(np.eye(int(2*Ham.L+1)),isz)
    isz=np.asarray(isz,dtype=np.complex128)
    E=Exp.Freq
    espac1=np.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
    beta=(scic.physical_constants["Bohr magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
    betan=(scic.physical_constants["nuclear magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
    h1=np.zeros((dim,dim),dtype='complex128')
    hzex=np.asarray(beta*Hze(sx,sy,sz,Ham.g,[1,0,0],dim),dtype=np.complex128)
    hzey=np.asarray(beta*Hze(sx,sy,sz,Ham.g,[0,1,0],dim),dtype=np.complex128)
    hzez=np.asarray(beta*Hze(sx,sy,sz,Ham.g,[0,0,1],dim),dtype=np.complex128)
    if Ham.S>=1:
        h1=h1+StevensO(sx,sy,sz,Ham.S,Ham,dim)
    if Ham.L!=0:
        h1=h1+Lorbit(sx,sy,sz,Ham.lc,dim,Ham.L)
    if Ham.I!=0:
        h1=h1+Hfi(sx,sy,sz,ix,iy,iz,Ham.A,dim)
        nhzex=np.asarray(betan*Nhze(Ham.I,ix,iy,iz,dim,Ham.Nucl,[1,0,0]),dtype=complex)
        nhzey=np.asarray(betan*Nhze(Ham.I,ix,iy,iz,dim,Ham.Nucl,[0,1,0]),dtype=complex)
        nhzez=np.asarray(betan*Nhze(Ham.I,ix,iy,iz,dim,Ham.Nucl,[0,0,1]),dtype=complex)
        hzex-=nhzex
        hzey-=nhzey
        hzez-=nhzez
    if np.any(Ham.Q):
        h1=h1+Qii(ix,iy,iz,Ham.Q,dim)
    h1=np.asarray(h1,dtype=np.complex128)
    targettr=set()
    targettr.update(tuple(sorted(p)) for p in transitions["allowed"])
    targettr.update(tuple(sorted(p)) for p in transitions["for Dms2"])
    fpoints=500
    Blist=np.linspace(Exp.Frange[0],Exp.Frange[1],fpoints)
    rfield=[]
    alabel={'X':(1,0,0,hzex),'Y':(0,1,0,hzey),'Z':(0,0,1,hzez)}
    for lab,(nx,ny,nz,hop) in alabel.items():
        Elist,Vlist,h2=Padaptarray(Blist,h1,hzex,hzey,hzez,nx,ny,nz)
        Elist,Vlist=Pretrack(Elist,Vlist)
        maxvec=Vlist[-1]
        curvebasis=Assingstatestobasis(maxvec)
        resonants=[]
        for i in range(dim):
            for j in range(i+1,dim):
                basis1=curvebasis[i]
                basis2=curvebasis[j]
                pair=tuple(sorted((basis1,basis2)))
                diffv=np.abs(Elist[:,j]-Elist[:,i])-Exp.Freq
                for k in range(len(diffv)-1):
                    if (diffv[k]*diffv[k+1]<=0.0) and (diffv[k]!=diffv[k+1]):
                        dEk=diffv[k]
                        dEk1=diffv[k+1]
                        t=-dEk/(dEk1-dEk)
                        res=Blist[k]+(t*(Blist[k+1]-Blist[k]))
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
                        if res>Blist[0]+2:
                            state1=Getlabel(basis1,slit,nlit,llit,Ham.L,Ham.I)
                            state2=Getlabel(basis2,slit,nlit,llit,Ham.L,Ham.I)
                            resonants.append({'field': res,'inx': (i, j),'bainx': (basis1,basis2),'type': ttyp,'transition': f"{state1} <-> {state2}"})

        Pot(Blist,Elist,curvebasis,resonants,lab,espac1)
