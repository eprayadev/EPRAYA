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
from .base_ham import *
from .base_plot import *
from .base_rotate import *


@njit
def Lorentzp(field,Int,rfield,Hpp):
    espec=np.zeros(len(field),dtype=float)
    gamma=np.sqrt(3.0)*Hpp
    gamma2=gamma/2.0
    for wq in range(len(Int)):
        for we in range(len(field)):
            dif=field[we]-rfield[wq]
            espec[we]+=(-Int[wq]*dif*gamma/np.pi)/((dif**2+gamma2**2)**2)
    return espec

@njit
def Gaussp(field,Int,rfield,Hpp):
    espec=np.zeros(len(field),dtype=float)
    gamma=np.sqrt(np.log(2.0)/2.0)*Hpp
    ymax=np.sqrt(np.log(2.0)/np.pi)*(1/gamma)
    for wq in range(len(Int)):
        for we in range(len(field)):
            dif=field[we]-rfield[wq]
            espec[we]+=((-Int[wq]*ymax*2.0*np.log(2.0)*dif)/gamma**2)*np.exp(-np.log(2.0)*dif**2/gamma**2)
    return espec

@njit
def Voigtp(field,Int,rfield,Hpp,eta):
    if eta==0:
      return Gaussp(field,Int,rfield,Hpp[0])
    elif eta==1:
      return Lorentzp(field,Int,rfield,Hpp[1])
    elif Hpp[1]==0.0:
      return Gaussp(field,Int,rfield,Hpp[0])
    elif Hpp[0]==0.0:
      return Lorentzp(field,Int,rfield,Hpp[1])
    else:
      Hppl=Hpp[1]
      Hppg=Hpp[0]
      espec=np.zeros(len(field),dtype=float)
      gammal=np.sqrt(3.0)*Hppl
      gamma2l=gammal/2.0
      gammag=np.sqrt(np.log(2.0)/2.0)*Hppg
      ymaxg=np.sqrt(np.log(2.0)/np.pi)*(1/gammag)
      for wq in range(len(Int)):
          for we in range(len(field)):
              dif=field[we]-rfield[wq]
              lor=(-Int[wq]*dif*gammal/np.pi)/((dif**2+gamma2l**2)**2)
              gas=((-Int[wq]*ymaxg*2.0*np.log(2.0)*dif)/gammag**2)*np.exp(-np.log(2.0)*dif**2/gammag**2)
              espec[we]+=(lor*eta)+((1-eta)*gas)
      return espec

#Find energy values in function of field
def Padaptarray(espac,h1,hx,hy,hz,nx,ny,nz):
    h2=nx*hx+ny*hy+nz*hz
    h3=h1[np.newaxis,:,:]+h2[np.newaxis,:,:]*espac[:,np.newaxis,np.newaxis]
    Elist,Vlist=np.linalg.eigh(h3)
    return Elist,Vlist,h2

# Takes into account the possibility of crossing in the energies, then makes an approximation with
# the eigenvectors change, that will be "close" from each other, if the field difference is low
def Pretrack(Enegria,Vector):
    for i in range(Enegria.shape[0]-2,-1,-1):
        oncevals,oncevecs=Enegria[i+1],Vector[i+1]
        actvals,actvecs=Enegria[i],Vector[i]
        novovals,novovecs=Hungorder(oncevals,oncevecs,actvals,actvecs)
        Enegria[i],Vector[i]=novovals,novovecs
    return np.array(Enegria),np.array(Vector)

# Makes the approximation by the assigment problem solution
def Hungorder(onevals,onevecs,actvals,actvecs):
    supermatrix=np.abs(np.dot(onevecs.conj().T,actvecs))
    cost=1-supermatrix
    Edif=np.abs(onevals[:,None]-actvals[None,:])
    max=np.max(Edif)
    if max>0:
        coste=Edif/max
    else:
        coste=0
    cost=cost+0.1*coste
    #Minimaze cost to find the right configuration
    rowidx,colidx=sci.optimize.linear_sum_assignment(cost)
    novo=colidx[np.argsort(rowidx)]
    return actvals[novo],actvecs[:,novo]

def Formatfra(val):
    #Change format for fractions
    if np.isclose(val,0.0):
        return "0"
    if np.isclose(abs(val%1),0.5):
        num=int(2*val)
        return f"{num}/2"
    return f"{int(val)}" if val.is_integer() else f"{val:.1f}"

def Getlabel(basisidx,slit,nlit,llit,L,I):
    ms=Formatfra(slit[basisidx])
    mi=Formatfra(nlit[basisidx])
    ml=Formatfra(llit[basisidx])
    if L!=0:
        if I!=0:
            return f"|{ms},{mi},{ml}⟩"
        else:
            return f"|{ms},{ml}⟩"
    else:
        if I!=0:
            return f"|{ms},{mi}⟩"
        else:
            return f"|{ms}⟩"

@njit
def Boltfactor(Eghz,di,dj,Temp):
    if Temp<=0:
        return 1.0
    h=scc.h
    kb=scc.k
    conver=1e9*h # Provisional conversion value
    Ej=Eghz*conver
    beta=1.0/(kb*Temp)
    Emin=np.min(Ej)
    boltz=np.exp(-beta*(Ej-Emin))
    Z=np.sum(boltz)
    popui=boltz[di]/Z
    popuj=boltz[dj]/Z
    return np.abs(popui-popuj)

# Creates a plane where the triangle is created and then expanded to the surface of a
#1 radius sphere, with a correction parameter of weight (solid angle projection)
def Delaunay(Exp,M=35):
    vectors=[]
    weights=[]
    for i in range(M+1):
        for j in range(M+1-i):
            k=M-j-i
            R=np.sqrt(i**2+j**2+k**2)
            if R==0:
                continue
            vectors.append([i/R,j/R,k/R])
            weights.append(1/(R**3))
    vectors=np.asarray(vectors)
    weights=np.asarray(weights)
    #Symmetries definitions
    gframe=np.asarray(Exp.gframe)
    Dframe=np.asarray(Exp.Dframe)
    Aframe=np.asarray(Exp.Aframe)
    #Cubic, Axial and Rombic
    if np.allclose(gframe,Dframe) and np.allclose(gframe,Aframe):
        signs=[np.array([1,1,1])]
    #Monoclinic
    #Using the unique axis as Z for simplicity
    elif (np.allclose(gframe[0],Aframe[0]) and np.allclose(gframe[1],Aframe[1])) or (np.allclose(gframe[0],Dframe[0]) and np.allclose(gframe[1],Dframe[1])) or (np.allclose(Aframe[0],Dframe[0]) and np.allclose(Aframe[1],Dframe[1])):
        signs=[np.array([1,1,1]),np.array([-1,1,1])]
    #Triclinic
    else:
        signs=[np.array([1,1,1]),np.array([-1,1,1]),np.array([1,-1,1]),np.array([-1,-1,1])]
    avector=[]
    aweight=[]
    for s in signs:
        avector.append(vectors*s)
        aweight.append(weights)
    avector=np.vstack(avector)
    aweight=np.concatenate(aweight)
    #Eliminate the frontier values
    uvectors,idx=np.unique(np.round(avector,6),axis=0,return_index=True)
    uweight=aweight[idx]
    uweight=uweight/np.sum(uweight)
    #Convex hull creates the smaller convex polygon which contains the points
    hulk=ConvexHull(uvectors).simplices
    return uvectors[:,0],uvectors[:,1],uvectors[:,2],uweight,hulk

@njit
def Nresina(Blist,Blist2,Elist,Vlist,dim,Freq,isx,isy,isz,nx,ny,nz,Tem,h2):
    resfield=[]
    intensy=[]
    Bstart=Blist2[0]
    dB2=(Blist2[-1]-Blist2[0])/(len(Blist2)-1)
    for i in range(dim):
        for j in range(i+1,dim):
            diffv=np.abs(Elist[:,j]-Elist[:,i])-Freq
            for k in range(len(diffv)-1):
                if (diffv[k]*diffv[k+1]<=0.0) and (diffv[k]!=diffv[k+1]):
                    dEk=diffv[k]
                    dEk1=diffv[k+1]
                    t=-dEk/(dEk1-dEk)
                    res=Blist[k]+(t*(Blist[k+1]-Blist[k]))
                    if res>Blist[0]+2:
                        #Interpolate for intensities
                        ic=int((res-Bstart)/dB2)
                        if ic>=len(Blist2)-1:
                            ic=len(Blist2)-2
                        tc=(res-Blist2[ic])/dB2
                        vi0=np.ascontiguousarray(Vlist[ic,:,i])
                        vi1=np.ascontiguousarray(Vlist[ic+1,:,i])
                        vj0=np.ascontiguousarray(Vlist[ic,:,j])
                        vj1=np.ascontiguousarray(Vlist[ic+1,:,j])
                        vik=(1.0-tc)*vi0+tc*vi1
                        vjk=(1.0-tc)*vj0+tc*vj1
                        vik=vik/np.linalg.norm(vik)
                        vjk=vjk/np.linalg.norm(vjk)
                        #Probability definition and interpolation
                        Tx=np.dot(vjk.conj().T,np.dot(isx,vik))
                        Ty=np.dot(vjk.conj().T,np.dot(isy,vik))
                        Tz=np.dot(vjk.conj().T,np.dot(isz,vik))
                        M2=np.abs(Tx)**2+np.abs(Ty)**2+np.abs(Tz)**2
                        #Part that is parallel to B
                        Mn=nx*Tx+ny*Ty+nz*Tz
                        prob=M2-np.abs(Mn)**2
                        #Frecuency to field
                        dert=np.real(np.dot(np.conj(vik),np.dot(h2,vik)))
                        izrt=np.real(np.dot(np.conj(vjk),np.dot(h2,vjk)))
                        gma=np.abs(izrt-dert)
                        if gma<1e-4:
                            gma=1e-4
                        gema=1.0/gma
                        Energz=(1.0-t)*Elist[k]+t*Elist[k+1]
                        boltzman=Boltfactor(Energz,i,j,Tem)
                        resfield.append(res)
                        intensy.append(prob*boltzman*gema)
    return resfield,intensy


def Omegaparal(args):
    (nx,ny,nz,Blist1,Blist2,h1,hzex,hzey,hzez,dim,Freq,Temp,isx,isy,isz,espac1,eta)=args
    Elist1,Vlist1,h2=Padaptarray(Blist1,h1,hzex,hzey,hzez,nx,ny,nz)
    #Elist1,Vlist1=Pretrack(Elist1,Vlist1)
    splines=cubichers(Blist1,Elist1,axis=0)
    Elist2=splines(Blist2)
    resfield,intensy=Nresina(Blist2,Blist1,Elist2,Vlist1,dim,Freq,isx,isy,isz,nx,ny,nz,Temp,h2)
    if len(resfield)>0:
        allesint=np.array(intensy,dtype=np.float64)
        resfield=np.array(resfield,dtype=np.float64)
        return resfield,allesint
    else:
        return np.array([]),np.array([])


@njit
def Caltriangle(sketch,Bmin,dB,allres,allint,transi,hulk,weight):
    #Barycentral mesh of triangles
    numd=15  # Resolution
    tpoints=(numd*(numd+1))/2.0
    for tx in range(len(hulk)):
        i1=hulk[tx,0]
        i2=hulk[tx,1]
        i3=hulk[tx,2]
        n1=transi[i1]
        n2=transi[i2]
        n3=transi[i3]
        #Same number of transitions
        if n1==n2 and n2==n3 and n1>0:
            weig=(weight[i1]+weight[i2]+weight[i3])/3.0
            pointweg=weig/tpoints
            for eins in range(0,n1):
                B1,B2,B3=allres[i1,eins],allres[i2,eins],allres[i3,eins]
                I1,I2,I3=allint[i1,eins],allint[i2,eins],allint[i3,eins]
                for i in range(numd):
                    for j in range(numd-i):
                        k=numd-1-i-j
                        w1=i/(numd-1)
                        w2=j/(numd-1)
                        w3=k/(numd-1)
                        Bin=w1*B1+w2*B2+w3*B3
                        Iint=w1*I1+w2*I2+w3*I3
                        igdam=int((Bin-Bmin)/dB)
                        if 0<=igdam<len(sketch):
                            sketch[igdam]+=Iint*pointweg

def Powder(Hamer,Expe,graph='True'):  #Method ASG
    iwas,jwas,kwas,weight,hulk=Delaunay(Expe)
    Bfield,Intensity=Calpowder(Hamer,Expe,iwas,jwas,kwas,weight,hulk)
    if graph=='True':
        import plotly.graph_objects as pgo
        import plotly.colors as pc
        graphp=pgo.Figure(data=pgo.Scatter(x=Bfield,y=Intensity, mode='lines',name="Spectrum",line=dict(color='navy')))
        graphp.update_layout(
            title={'text':'EPR spectrum','xanchor':'center','yanchor':'auto','x':0.45, 'y':0.95,
                   'font': dict(family='Georgia', size=24,color='black')},
            xaxis=dict(title='Field [mT]',showline=True,linecolor='black',mirror=True,linewidth=2,showgrid=True,gridcolor='black',range=[Bfield[0],Bfield[-1]+10])
            ,yaxis=dict(title='Counts [U. A.]',showline=True,linecolor='black',mirror=True,linewidth=2,showgrid=True,gridcolor='black'),
            plot_bgcolor='white',
            width=1000,
            height=600,
            margin=dict(l=50,r=50,b=50,t=70,pad=4),
            showlegend=True)
        graphp.add_hline(y=0,line_color="black",line_width=1)
        graphp.add_vline(x=0,line_color="black",line_width=1)
        display(graphp)
        return Bfield,Intensity
    else:
        return Bfield,Intensity

def Calpowder(Hamer,Expe,iwas,jwas,kwas,weight,hulk):
    Ham=deepcopy(Hamer)
    Exp=deepcopy(Expe)
    #For powder samples using the method of the Delaunay triangles
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
    Ham.A=Ham.A/1000
    Ham.D=Ham.D/1000
    Ham.Q=Ham.Q/1000
    Ham.Bk2=Ham.Bk2/1000
    Ham.Bk4=Ham.Bk4/1000
    Ham.Bk6=Ham.Bk6/1000
    Ham.Hpp[0],Ham.Hpp[1]=Ham.Hpp[0]/1.0,Ham.Hpp[1]/1.0
    if Ham.Hpp[0]==0.0 and Ham.Hpp[1]!=0.0:
        eta=1.0
    elif Ham.Hpp[0]!=0.0 and Ham.Hpp[1]==0.0:
        eta=0.0
    else:
        eta=Ham.eta
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
    h1=np.zeros((dim,dim),dtype='complex')
    hzex=np.asarray(beta*Hze(sx,sy,sz,Ham.g,[1,0,0],dim),dtype=complex)
    hzey=np.asarray(beta*Hze(sx,sy,sz,Ham.g,[0,1,0],dim),dtype=complex)
    hzez=np.asarray(beta*Hze(sx,sy,sz,Ham.g,[0,0,1],dim),dtype=complex)
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

    h1=np.asarray(h1,dtype=complex)
    espectotal=np.zeros(Exp.Points)
    Blist1=np.linspace(Exp.Frange[0],Exp.Frange[1],500)
    Blist2=np.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
    ttdo=[]#Works to do in parallel
    for omega in range(0,len(weight)):
        if np.isclose(weight[omega],0):
            continue
        nx,ny,nz=iwas[omega],jwas[omega],kwas[omega]
        currenttdo=(nx,ny,nz,Blist1,Blist2,h1,hzex,hzey,hzez,dim,Exp.Freq,Exp.Temperature,isx,isy,isz,espac1,eta)
        ttdo.append(currenttdo)
    with threadpool_limits(limits=1,user_api='blas'):
        partresult=Parallel(n_jobs=-3,backend="loky")(delayed(Omegaparal)(t) for t in ttdo)
    #Calculate over the triangles
    nodes=len(partresult)
    mtrans=0
    if nodes>0:
        mtrans=max([len(res[0]) for res in partresult])
    allres=np.zeros((nodes,mtrans),dtype=np.float64)
    allint=np.zeros((nodes,mtrans),dtype=np.float64)
    ntrans=np.zeros(nodes,dtype=np.int32)
    for ide,(Bv,Iv) in enumerate(partresult):
        na=len(Bv)
        ntrans[ide]=na
        if na>0:
            allres[ide,:na]=Bv
            allint[ide,:na]=Iv
    sketch=np.zeros(Exp.Points)
    Bmin=Exp.Frange[0]
    dB=(Exp.Frange[1]-Exp.Frange[0])/(Exp.Points)
    Caltriangle(sketch,Bmin,dB,allres,allint,ntrans,hulk,weight)

    #Convolution of the function to create the derivated spectrum
    maxlenght=np.max(Ham.Hpp)*10
    kerpoint=int(maxlenght/dB)*2+1
    kaxis=np.arange(-kerpoint//2+1,kerpoint//2+1)*dB
    #Int=1, no resonant field
    kvoigt=Voigtp(kaxis,np.array([1.0]),np.array([0.0]),Ham.Hpp,eta)
    espectotal=scs.fftconvolve(sketch,kvoigt,mode='same')
    return espac1,espectotal

def Iee(ssx1,ssy1,ssz1,ssx2,ssy2,ssz2,X,dim):
    eet=(X[0,0]*np.kron(ssx1,ssx2))+(X[0,1]*np.kron(ssx1,ssy2))+(X[0,2]*np.kron(ssx1,ssz2))+(X[1,0]*np.kron(ssy1,ssx2))+(X[1,1]*np.kron(ssy1,ssy2))+    (X[1,2]*np.kron(ssy1,ssz2))+(X[2,0]*np.kron(ssz1,ssx2))+(X[2,1]*np.kron(ssz1,ssy2))+(X[2,2]*np.kron(ssz1,ssz2))
    eet=np.kron(eet,np.eye(int(dim/(eet).shape[1])))
    return eet

def Acodom(Xa):
    if Xa.ndim==1 and Xa.shape[0]==3:
        return Xa*np.eye
    elif Xa.ndim==2 and Xa.shape[0]==3 and Xa.shape[1]==3:
        return Xa
    else:
        raise ValueError(f"Values must be a 1-D 3 element array or a 3-D matrix.")

def Mulpol(Hamer,Expe,graph='True'):
    Ham=deepcopy(Hamer)
    iwas,jwas,kwas,weight,hulk=Delaunay(Expe.Mexp[0])
    numberes=len(Ham.Mulham)
    for inka in range(0,numberes):
        Expe.Mexp[inka].Freq=Expe.Mexp[0].Freq
        Expe.Mexp[inka].Points=Expe.Mexp[0].Points
        Expe.Mexp[inka].Temperature=Expe.Mexp[0].Temperature
        Expe.Mexp[inka].Fdirection=Expe.Mexp[0].Fdirection
        Expe.Mexp[inka].Frange=Expe.Mexp[0].Frange
    Exp=deepcopy(Expe)
    submixes=set()
    if hasattr(Ham,'Amix') and Ham.Amix:
        submixes.update(Ham.Amix.keys())
    if hasattr(Ham,'Xmix') and Ham.Xmix:
        submixes.update(Ham.Xmix.keys())
    relations={i: [] for i in range(numberes)}
    for (u,v) in submixes:
        relations[u].append(v)
        relations[v].append(u)
    numboint=[] #Number of systems that take place in the same process
    looked=set()
    for ipl in range(numberes):
        if ipl not in looked:
            aktuell=[]
            lastone=[ipl]
            while lastone:
                nowdat=lastone.pop(0)
                if nowdat not in looked:
                    looked.add(nowdat)
                    aktuell.append(nowdat)
                    lastone.extend(relations[nowdat])
            numboint.append(sorted(aktuell))
    sumespct=np.zeros(Exp.Mexp[0].Points)
    for elka in numboint:
        if len(elka)==1:
            Ham.Mulham[elka[0]].Hpp=Ham.Mulham[0].Hpp
            Ham.Mulham[elka[0]].eta=Ham.Mulham[0].eta
            fild1,fild2=Calpowder(Ham.Mulham[elka[0]],Exp.Mexp[elka[0]],iwas,jwas,kwas,weight,hulk)
            sumespct+=fild2
        else:
            if Exp.Mexp[0].Freq<=0:
                raise ValueError("Frequency can't be a negative or zero value")
            if Exp.Mexp[0].Frange[0]<0:
                Exp.Mexp[0].Frange[0]=0
                print("WARNING: A negative value has been used, changing it to zero.")
            if Exp.Mexp[0].Frange[0]>=Exp.Mexp[0].Frange[1]:
                raise ValueError("Field range can't be from higher to lower values.")
            Ham.Mulham[0].Hpp[0],Ham.Mulham[0].Hpp[1]=Ham.Mulham[0].Hpp[0]/1.0,Ham.Mulham[0].Hpp[1]/1.0
            if Ham.Mulham[0].Hpp[0]==0.0 and Ham.Mulham[0].Hpp[1]!=0.0:
                Ham.Mulham[0].eta=1.0
            elif Ham.Mulham[0].Hpp[0]!=0.0 and Ham.Mulham[0].Hpp[1]==0.0:
                Ham.Mulham[0].eta=0.0
            else:
                Ham.Mulham[0].eta=Ham.Mulham[0].eta
            Pmatrixs={}
            Pmatrixi={}
            E=Exp.Mexp[0].Freq
            espac1=np.linspace(Exp.Mexp[0].Frange[0],Exp.Mexp[0].Frange[1],Exp.Mexp[0].Points)
            beta=(scic.physical_constants["Bohr magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
            betan=(scic.physical_constants["nuclear magneton"][0]/scic.physical_constants["Planck constant"][0])/1e12
            dim=1
            dimerq=[]
            for orko in elka:
                dim*=int(2*Ham.Mulham[orko].S+1)*int(2*Ham.Mulham[orko].I+1)
                dimerq.append(int(2*Ham.Mulham[orko].S+1)*int(2*Ham.Mulham[orko].I+1))
            h1=np.zeros((dim,dim),dtype='complex')
            hzex=0
            hzey=0
            hzez=0
            def Kroexpand(HZE,lnumbe,dims):
                if lnumbe==0:
                    firstep=HZE
                else:
                    firstep=np.eye(dims[0],dtype=complex)
                for i in range(1,len(dims)):
                    if i==lnumbe:
                        firstep=np.kron(firstep,HZE)
                    else:
                        firstep=np.kron(firstep,np.eye(dims[i],dtype=complex))
                return firstep
            for orka in elka:
                Ham.Mulham[orka].D,Ham.Mulham[orka].A=np.asarray(Ham.Mulham[orka].D),np.asarray(Ham.Mulham[orka].A)
                Ham.Mulham[orka].Q=np.asarray(Ham.Mulham[orka].Q)
                Ham.Mulham[orka].A=Ham.Mulham[orka].A/1000
                Ham.Mulham[orka].D=Ham.Mulham[orka].D/1000
                Ham.Mulham[orka].Q=Ham.Mulham[orka].Q/1000
                Ham.Mulham[orka].Bk2,Ham.Mulham[orka].Bk4=np.asarray(Ham.Mulham[orka].Bk2),np.asarray(Ham.Mulham[orka].Bk4)
                Ham.Mulham[orka].Bk6=np.asarray(Ham.Mulham[orka].Bk6)
                Ham.Mulham[orka].Bk2=Ham.Mulham[orka].Bk2/1000
                Ham.Mulham[orka].Bk4=Ham.Mulham[orka].Bk4/1000
                Ham.Mulham[orka].Bk6=Ham.Mulham[orka].Bk6/1000
                Ham.Mulham[orka]=chaframe(Ham.Mulham[orka],Exp.Mexp[orka])
                Pmatrixs[orka]=Pauli(Ham.Mulham[orka].S)
                Pmatrixi[orka]=Pauli(Ham.Mulham[orka].I)
                hzexx=np.asarray(beta*Hze(Pmatrixs[orka][0],Pmatrixs[orka][1],Pmatrixs[orka][2],Ham.Mulham[orka].g,[1,0,0],dimerq[orka]),dtype=complex)
                hzex+=Kroexpand(hzexx,orka,dimerq)
                hzeyy=np.asarray(beta*Hze(Pmatrixs[orka][0],Pmatrixs[orka][1],Pmatrixs[orka][2],Ham.Mulham[orka].g,[0,1,0],dimerq[orka]),dtype=complex)
                hzey+=Kroexpand(hzeyy,orka,dimerq)
                hzezz=np.asarray(beta*Hze(Pmatrixs[orka][0],Pmatrixs[orka][1],Pmatrixs[orka][2],Ham.Mulham[orka].g,[0,0,1],dimerq[orka]),dtype=complex)
                hzez+=Kroexpand(hzezz,orka,dimerq)
                if Ham.Mulham[orka].S>=1:
                    h1=h1+StevensO(Pmatrixs[orka][0],Pmatrixs[orka][1],Pmatrixs[orka][2],Ham.Mulham[orka].S,Ham.Mulham[orka],dim)
                if Ham.Mulham[orka].I!=0:
                    h1=h1+Hfi(Pmatrixs[orka][0],Pmatrixs[orka][1],Pmatrixs[orka][2],Pmatrixi[orka][0],Pmatrixi[orka][1],Pmatrixi[orka][2],Ham.Mulham[orka].A,dim)
                    nhzexx=np.asarray(betan*Nhze(Ham.Mulham[orka].I,Pmatrixi[orka][0],Pmatrixi[orka][1],Pmatrixi[orka][2],dimerq[orka],Ham.Mulham[orka].Nucl,[1,0,0]),dtype=complex)
                    nhzex=Kroexpand(nhzexx,orka,dimerq)
                    nhzeyy=np.asarray(betan*Nhze(Ham.Mulham[orka].I,Pmatrixi[orka][0],Pmatrixi[orka][1],Pmatrixi[orka][2],dimerq[orka],Ham.Mulham[orka].Nucl,[0,1,0]),dtype=complex)
                    nhzey=Kroexpand(nhzeyy,orka,dimerq)
                    nhzezz=np.asarray(betan*Nhze(Ham.Mulham[orka].I,Pmatrixi[orka][0],Pmatrixi[orka][1],Pmatrixi[orka][2],dimerq[orka],Ham.Mulham[orka].Nucl,[0,0,1]),dtype=complex)
                    nhzez=Kroexpand(nhzezz,orka,dimerq)
                    hzex-=nhzex
                    hzey-=nhzey
                    hzez-=nhzez
                if np.any(Ham.Mulham[orka].Q):
                    h1=h1+Qii(Pmatrixi[orka][0],Pmatrixi[orka][1],Pmatrixi[orka][2],Ham.Mulham[orka].Q,dim)
            if isinstance(Ham,Multham):
                for ilar in elka:
                    for jlar in elka:
                        if ilar==jlar:
                            continue
                        Aref=Ham.Amix.get((ilar,jlar))
                        if Aref is not None and np.any(Aref):
                            Aref=np.asarray(Aref)/1000.0
                            Aref=Aref*np.eye(3)
                            h1+=Hfi(Pmatrixs[ilar][0],Pmatrixs[ilar][1],Pmatrixs[ilar][2],Pmatrixi[jlar][0],Pmatrixi[jlar][1],Pmatrixi[jlar][2],Aref,dim)
                        Xref=Ham.Xmix.get((ilar,jlar))
                        if Xref is not None and np.any(Xref):
                            Xref=np.asarray(Xref)/1000.0
                            Xref=Xref*np.eye(3)
                            h1+=Iee(Pmatrixs[ilar][0],Pmatrixs[ilar][1],Pmatrixs[ilar][2],Pmatrixs[jlar][0],Pmatrixs[jlar][1],Pmatrixs[jlar][2],Xref,dim)
            h1=np.asarray(h1,dtype=complex)
            #For the total magnetic moment of the system
            stodx=np.zeros((dim,dim),dtype='complex')
            stody=np.zeros((dim,dim),dtype='complex')
            stodz=np.zeros((dim,dim),dtype='complex')
            pla=0
            for pla in elka:
                if Ham.Mulham[pla].S>0:
                    sxe1,sye1,sze1=Pmatrixs[pla]
                    sxe2=np.array([[1.0]])
                    sye2=np.array([[1.0]])
                    sze2=np.array([[1.0]])
                    for plo in elka:
                        dims=int(2*Ham.Mulham[plo].S+1)
                        dimi=int(2*Ham.Mulham[plo].I+1)
                        dimt=dims*dimi
                        if pla==plo:
                            sx3=np.kron(sxe1,np.eye(dimi))
                            sy3=np.kron(sye1,np.eye(dimi))
                            sz3=np.kron(sze1,np.eye(dimi))
                        else:
                            sx3=np.eye(dimt)
                            sy3=np.eye(dimt)
                            sz3=np.eye(dimt)
                        sxe2=np.kron(sxe2,sx3)
                        sye2=np.kron(sye2,sy3)
                        sze2=np.kron(sze2,sz3)
                    stodx+=sxe2
                    stody+=sye2
                    stodz+=sze2
            espectotal=np.zeros(Exp.Mexp[0].Points)
            Blist1=np.linspace(Exp.Mexp[0].Frange[0],Exp.Mexp[0].Frange[1],50)
            Blist2=np.linspace(Exp.Mexp[0].Frange[0],Exp.Mexp[0].Frange[1],Exp.Mexp[0].Points)
            ttdo=[]#Works to do in parallel
            for omega in range(0,len(weight)):
                if np.isclose(weight[omega],0):
                    continue
                nx,ny,nz=iwas[omega],jwas[omega],kwas[omega]
                currenttdo=(nx,ny,nz,Blist1,Blist2,h1,hzex,hzey,hzez,dim,Exp.Mexp[0].Freq,Exp.Mexp[0].Temperature,stodx,stody,stodz,espac1,Ham.Mulham[0].eta)
                ttdo.append(currenttdo)
            with threadpool_limits(limits=1,user_api='blas'):
                partresult=Parallel(n_jobs=-3,backend="loky")(delayed(Betaparal)(t) for t in ttdo)
            #Calculate over the triangles
            nodes=len(partresult)
            mtrans=0
            if nodes>0:
                mtrans=max([len(res[0]) for res in partresult])
            allres=np.zeros((nodes,mtrans),dtype=np.float64)
            allint=np.zeros((nodes,mtrans),dtype=np.float64)
            ntrans=np.zeros(nodes,dtype=np.int32)
            for ide,(Bv,Iv) in enumerate(partresult):
                na=len(Bv)
                ntrans[ide]=na
                if na>0:
                    allres[ide,:na]=Bv
                    allint[ide,:na]=Iv
            sketch=np.zeros(Exp.Mexp[0].Points)
            Bmin=Exp.Mexp[0].Frange[0]
            dB=(Exp.Mexp[0].Frange[1]-Exp.Mexp[0].Frange[0])/(Exp.Mexp[0].Points)
            Caltriangle(sketch,Bmin,dB,allres,allint,ntrans,hulk,weight)

            #Convolution of the function to create the derivated spectrum
            maxlenght=np.max(Ham.Mulham[0].Hpp)*10
            kerpoint=int(maxlenght/dB)*2+1
            kaxis=np.arange(-kerpoint//2+1,kerpoint//2+1)*dB
            kvoigt=Voigtp(kaxis,np.array([1.0]),np.array([0.0]),Ham.Mulham[0].Hpp,Ham.Mulham[0].eta)
            espectotal=scs.fftconvolve(sketch,kvoigt,mode='same')
            sumespct+=espectotal
            fild1=espac1
    if graph=='True':
        import plotly.graph_objects as pgo
        import plotly.colors as pc
        graphp=pgo.Figure(data=pgo.Scatter(x=fild1,y=sumespct,mode='lines',name="Spectrum",line=dict(color='navy')))
        graphp.update_layout(
            title={'text':'EPR spectrum','xanchor':'center','yanchor':'auto','x':0.45, 'y':0.95,
                   'font': dict(family='Georgia', size=24,color='black')},
            xaxis=dict(title='Field [mT]',showline=True,linecolor='black',mirror=True,linewidth=2,showgrid=True,gridcolor='black',range=[fild1[0],fild1[-1]+10])
            ,yaxis=dict(title='Counts [U. A.]',showline=True,linecolor='black',mirror=True,linewidth=2,showgrid=True,gridcolor='black'),
            plot_bgcolor='white',
            width=1000,
            height=600,
            margin=dict(l=50,r=50,b=50,t=70,pad=4),
            showlegend=True)
        graphp.add_hline(y=0,line_color="black",line_width=1)
        graphp.add_vline(x=0,line_color="black",line_width=1)
        display(graphp)
    return fild1,sumespct

def Betaparal(args):
    (nx,ny,nz,Blist1,Blist2,h1,hzex,hzey,hzez,dim,Freq,Temp,isx,isy,isz,espac1,eta)=args
    nBlist=Blist1[:,np.newaxis,np.newaxis]
    h4=nx*hzex+ny*hzey+nz*hzez
    hzet=h4[np.newaxis,:,:]
    h3=h1[np.newaxis,:,:]+nBlist*hzet
    Elist1,Vlist1=np.linalg.eigh(h3)
    splines=cubichers(Blist1,Elist1,axis=0)
    Elist2=splines(Blist2)
    resfield,intensy=Nresina(Blist2,Blist1,Elist2,Vlist1,dim,Freq,isx,isy,isz,nx,ny,nz,Temp,h4)
    if len(resfield)>0:
        allesint=np.array(intensy,dtype=np.float64)
        resfield=np.array(resfield,dtype=np.float64)
        return resfield,allesint
    else:
        return np.array([]),np.array([])
