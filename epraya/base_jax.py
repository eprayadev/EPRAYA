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
from .base_powd import *
@jaxdatclass
class JHval:
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
    g: Union[list[float],float]=0.0
    A: Union[list[float],float]=0.0     # Hyperfine constant
    Q: Union[list[float],float]=0.0     # Quadrupole interaction constant
    D: Union[list[float],float]=0.0
    Hpp: List=dcfield(default_factory=lambda:jxn.array([0,0]))
    weight: float=0.0

def Jstart():
    global Exp,Vary,Ham
    Ham,Exp,Vary=JHval(),JEco(),JEva()

def JKronecker(a0,b0):
    return jxn.where(a0==b0,1.0,0.0)

def JPauli(s):
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
  if l!=0:
    lx,ly,lz=JPauli(l)
    orbe=lamda*(jxn.kron(lx,sx)+jxn.kron(ly,sy)+jxn.kron(lz,sz))
    orbe=jxn.kron(orbe,jxn.eye(int(dim/(orbe).shape[1])))
    return orbe
  else:
    return np.zeros(dim)

def JHfi(ssx,ssy,ssz,iix,iiy,iiz,at,dim):
    ta=(at[0,0]*jxn.kron(ssx,iix))+(at[0,1]*jxn.kron(ssx,iiy))+(at[0,2]*jxn.kron(ssx,iiz))+(at[1,0]*jxn.kron(ssy,iix))+(at[1,1]*jxn.kron(ssy,iiy))+(at[1,2]*jxn.kron(ssy,iiz))+(at[2,0]*jxn.kron(ssz,iix))+(at[2,1]*jxn.kron(ssz,iiy))+(at[2,2]*jxn.kron(ssz,iiz))
    ta=jxn.kron(ta,jxn.eye(int(dim/(ta).shape[1])))
    return ta

def JHze(ssx,ssy,ssz,g,biel,dim):
    hze=biel[0]*(g[0,0]*ssx+g[0,1]*ssy+g[0,2]*ssz)+biel[1]*(g[1,0]*ssx+g[1,1]*ssy+g[1,2]*ssz)+biel[2]*(g[2,0]*ssx+g[2,1]*ssy+g[2,2]*ssz)
    thz=jxn.kron(hze,jxn.eye(int(dim/(hze).shape[1])))
    return thz

def JIee(ssx1,ssy1,ssz1,ssx2,ssy2,ssz2,X,dim):
    eet=(X[0,0]*np.kron(ssx1,ssx2))+(X[0,1]*np.kron(ssx1,ssy2))+(X[0,2]*np.kron(ssx1,ssz2))+(X[1,0]*np.kron(ssy1,ssx2))+(X[1,1]*np.kron(ssy1,ssy2))+(X[1,2]*np.kron(ssy1,ssz2))+(X[2,0]*np.kron(ssz1,ssx2))+(X[2,1]*np.kron(ssz1,ssy2))+(X[2,2]*np.kron(ssz1,ssz2))
    eet=jxn.kron(eet,jxn.eye(int(dim/(eet).shape[1])))
    return eet

def JQii(iix,iiy,iiz,q,i,dim):
    hql=(q[0,0]*iix*iix)+(q[1,1]*iiy*iiy)+(q[2,2]*iiz*iiz)+(q[0,1]*(iix*iiy)-(iiy*iix))+(q[1,2]*(iiy*iiz)-(iiz*iiy))
    +(q[2,0]*(iiz*iix)-(iix*iiz))
    tql=jxn.kron(hql,jxn.eye(int(dim/(hql).shape[1])))
    return tql


def JNhze(I,iix,iiy,iiz,dim,gn,direction=[0,0,1]):
    direct=direction[0]*iix+direction[0]*iiy+direction[0]*iiz
    nhz=gn*direct
    nhz=jxn.kron(nhz,jxn.eye(int(dim/(nhz).shape[1])))
    return nhz

def gnfactor(Nucl='None'):
    if Nucl!='None':
        krle=read_csv('nucleardaat.txt',header=0,sep='\t')
        deq=krle[krle['Symbol']==Nucl]
        return float(deq['gN_factor'].values[0])
    return 0.0

def Jchaframe(Ham,Exp):
  Ham=JConvtarray(Ham)
  tA=jxn.eye(3)*Ham.A
  tQ=jxn.eye(3)*Ham.Q
  tg=jxn.eye(3)*Ham.g
  D2s=jxn.asarray([-Ham.D[0]/3+Ham.D[1],-Ham.D[0]/3-Ham.D[1],2*Ham.D[0]/3])
  tD=jxn.eye(3)*D2s
  A1=JRotmatrix(Exp.Aframe[0],Exp.Aframe[1],Exp.Aframe[2]).T@tA@(JRotmatrix(Exp.Aframe[0],Exp.Aframe[1],Exp.Aframe[2]))
  g1=(JRotmatrix(Exp.gframe[0],Exp.gframe[1],Exp.gframe[2])).T@tg@(JRotmatrix(Exp.gframe[0],Exp.gframe[1],Exp.gframe[2]))
  D2=(JRotmatrix(Exp.Dframe[0],Exp.Dframe[1],Exp.Dframe[2])).T@tD@(JRotmatrix(Exp.Dframe[0],Exp.Dframe[1],Exp.Dframe[2]))
  Q1=(JRotmatrix(Exp.Qframe[0],Exp.Qframe[1],Exp.Qframe[2])).T@tQ@(JRotmatrix(Exp.Qframe[0],Exp.Qframe[1],Exp.Qframe[2]))
  return Ham.replace(A=A1,g=g1,D=D2,Q=Q1)

def JRotationmat(Exp):
    RRmatrix=JRotmatrix(Exp.Sampleframe[0],Exp.Sampleframe[1],Exp.Sampleframe[2])@JRotmatrix(Exp.Molframe[0],Exp.Molframe[1],Exp.Molframe[2])
    return RRmatrix.T

def JRotmatrix(alfa,beta,gamma):
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
    if I<0 or S<0 or L<0:
        raise ValueError('Spin values cannot be negative')
    dim=int(2*S+1)*int(2*I+1)*int(2*L+1)
    poss=jxn.linspace(S,-S,int(2*S+1))
    posi=jxn.linspace(I,-I,int(2*I+1))
    posl=jxn.linspace(L,-L,int(2*L+1))
    ll,sl,il=jxn.meshgrid(posl,poss,posi,indexing='ij')
    return sl.flatten(),il.flatten(),ll.flatten()

@jx.jit
def JLorentzp(field,Int,rfield,Hpp):
    espec=jxn.zeros(len(field),dtype=float)
    gamma=jxn.sqrt(3.0)*Hpp
    gamma2=gamma/2.0
    dif=field[:,None]-rfield[None,:]
    ert=(-dif*gamma/jxn.pi)/((dif**2+gamma2**2)**2)
    espec=jxn.sum(Int*ert,axis=1)
    return espec
@jx.jit
def JGaussp(field,Int,rfield,Hpp):
    espec=jxn.zeros(len(field),dtype=float)
    gamma=jxn.sqrt(jxn.log(2.0)/2.0)*Hpp
    ymax=jxn.sqrt(jxn.log(2.0)/jxn.pi)*(1/gamma)
    dif=field[:,None]-rfield[None,:]
    ert=((-ymax*2.0*jxn.log(2.0)*dif)/gamma**2)*jxn.exp(-jxn.log(2.0)*dif**2/gamma**2)
    espec=jxn.sum(Int*ert,axis=1)
    return espec

@jx.jit
def JVoigtp(field,Int,rfield,Hpp,eta):
    hppg=jxn.where(Hpp[0]==0.0,1e-10,Hpp[0])
    hppl=jxn.where(Hpp[1]==0.0,1e-10,Hpp[1])
    gas=JGaussp(field,Int,rfield,hppg)
    lor=JLorentzp(field,Int,rfield,hppl)
    espec=(lor*eta)+(gas*(1.0-eta))
    return espec

#Find energy values in function of field
@jx.jit
def JPadaptarray(espac,h1,hx,hy,hz,nx,ny,nz):
    h2=nx*hx+ny*hy+nz*hz
    h3=h1[None,:,:]+h2[None,:,:]*espac[:,None,None]
    Elist,Vlist=jxn.linalg.eigh(h3)
    return Elist,Vlist,h2
# Makes the approximation by the assigment problem solution
def Hungarian(cost):
    rowidx,colidx=sci.optimize.linear_sum_assignment(np.array(cost))
    novo=colidx[np.argsort(rowidx)]
    return novo.astype(np.int32)

def Jungarian(cost):
    shake=jx.ShapeDtypeStruct((cost.shape[0],),jxn.int32)
    return jx.pure_callback(Hungarian,shake,cost,vmap_method='sequential')

@jx.jit
def JPretrack(Enegria, Vector):
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
    h=scc.h
    kb=scc.k
    conver=1e9*h # Provisional conversion value
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
def JNresina(Blist2,Blist1,Elist,Vlist,dim,Freq,isx,isy,isz,nx,ny,nz,Tem,h2):
    intercep=lambda yl: jxn.interp(Blist2,Blist1,yl)
    Elist2=jx.vmap(intercep,in_axes=1,out_axes=1)(Elist)
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
    interp=lambda yl: jxn.interp(Blist2,Blist1,yl)
    Int=jx.vmap(interp,in_axes=1,out_axes=1)(intensy)
    diffv=jxn.abs(Elist2[:,jidx]-Elist2[:,iidx])-Freq
    gdelta=(Blist2[1]-Blist2[0])*1.5
    wsti=Int*jxn.exp(-jxn.log(2.0)*(diffv**2)/(gdelta**2))
    intensy=jxn.sum(wsti,axis=1)
    return intensy


def JPowder(Hamer,Expe,Nucl='None',graph='True'):
    iwas,jwas,kwas,weight,_=Delaunay(Expe)
    iwas,jwas,kwas,weight=jxn.array(iwas),jxn.array(jwas),jxn.array(kwas),jxn.array(weight)
    Blist,epc=JCalpowder(Hamer,Expe,iwas,jwas,kwas,weight,Nucl)
    if graph=='True':
        plt.plot(Blist,epc,color='navy')
        plt.xlabel('Field [mT]')
        plt.ylabel('Counts [U. A.]')
        plt.xlim(Exp.Frange[0],Exp.Frange[1])
        plt.grid()
        plt.show(block=False)
    return Blist,epc

def JCalpowder(Hamer,Expe,iwas,jwas,kwas,weight,Nucl='None'):
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
    espac1=jxn.linspace(frange0,Exp.Frange[1],500)
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
        h1=h1+JQii(ix,iy,iz,Ham.Q,Ham.I,dim)
        gnk=gnfactor(Nucl)
        nhzex=jxn.asarray(betan*JNhze(Ham.I,ix,iy,iz,dim,gnk,[1,0,0]),dtype=complex)
        nhzey=jxn.asarray(betan*JNhze(Ham.I,ix,iy,iz,dim,gnk,[0,1,0]),dtype=complex)
        nhzez=jxn.asarray(betan*JNhze(Ham.I,ix,iy,iz,dim,gnk,[0,0,1]),dtype=complex)
        hzex-=nhzex
        hzey-=nhzey
        hzez-=nhzez
    h1=jxn.asarray(h1,dtype=complex)
    Blist1=jxn.linspace(Exp.Frange[0],Exp.Frange[1],400)
    Blist2=jxn.linspace(Exp.Frange[0],Exp.Frange[1],Exp.Points)
    gdelta=(Blist2[1]-Blist2[0])*1.5
    @jx.jit
    def Oneori(nx,ny,nz,w):
        Elist,Vlist,h2=JPadaptarray(Blist1,h1,hzex,hzey,hzez,nx,ny,nz)
        intensy=JNresina(Blist2,Blist1,Elist,Vlist,dim,Exp.Freq,isx,isy,isz,nx,ny,nz,Exp.Temperature,h2)
        allesint=intensy*w
        esp1=jxn.where(jxn.isclose(w,0.0),jxn.zeros_like(allesint),allesint)
        return (esp1)
    voneori=jx.vmap(Oneori,in_axes=(0,0,0,0))
    csize=50 #Divides the orientations blocks so the RAM doesn't explote
    tlen=len(weight)
    plen=(csize-(tlen%csize))%csize
    pdw=jxn.pad(weight,(0,plen))
    pdi=jxn.pad(iwas,(0,plen))
    pdj=jxn.pad(jwas,(0,plen))
    pdk=jxn.pad(kwas,(0,plen))
    nparts=len(pdw)//csize
    batw=pdw.reshape(nparts,csize)
    bati=pdi.reshape(nparts,csize)
    batj=pdj.reshape(nparts,csize)
    batk=pdk.reshape(nparts,csize)
    @jx.checkpoint
    def Processvmap(curspect,bat):
        bnx,bny,bnz,bw=bat
        batspect=voneori(bnx,bny,bnz,bw)
        nspect=curspect+jxn.sum(batspect,axis=0)
        return nspect,None
    espectotal,_=jx.lax.scan(Processvmap,jxn.zeros(Exp.Points),(bati,batj,batk,batw))
    maxlenght=jxn.max(jxn.array(Ham.Hpp))*10
    dB=(Exp.Frange[1]-Exp.Frange[0])/(Exp.Points-1)
    kpoints=501
    kaxis=jxn.arange(-kpoints//2+1,kpoints//2+1)*dB
    kvoigt=JVoigtp(kaxis,jxn.array([1.0]),jxn.array([0.0]),Ham.Hpp,etas)
    espectotal=jsig.fftconvolve(espectotal,kvoigt,mode='same')
    return Blist2,espectotal

def Jresonant(Hamer,Expe,graph='True',Nucl='None'):
    Blist,epc=Calresonant(Hamer,Expe,Nucl)
    if graph=='True':
        plt.plot(Blist,epc,color='navy')
        plt.xlabel('Field [mT]')
        plt.ylabel('Counts [U. A.]')
        plt.xlim(Exp.Frange[0],Exp.Frange[1])
        plt.grid()
        plt.show(block=False)
    return Blist,epc

def Calresonant(Hamer,Expe,Nucl='None'):
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
        h1=h1+JQii(ix,iy,iz,Ham.Q,Ham.I,dim)
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
    return Blist,spc
    
@jaxdatclass
class MJhval:
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
    global Exp,Vary,Ham
    Ham,Exp,Vary=MJhval(),JEmco(),JEmva()

def JMulpol(maham,Expe,Nucl1='None',Nucl2='None',graph='True'):
    Blist,epc=Jcalmulta(maham,Expe,Nucl1,Nucl2)
    if graph=='True':
        plt.plot(Blist,epc,color='navy')
        plt.xlabel('Field [mT]')
        plt.ylabel('Counts [U. A.]')
        plt.xlim(Exp.Frange[0],Exp.Frange[1])
        plt.grid()
        plt.show(block=False)
    return Blist,epc
def Jcalmulta(maham,Expe,Nucl1='None',Nucl2='None'):
    maham.X1_2=np.asarray(maham.X1_2)
    maham.A1_2=np.asarray(maham.A1_2)
    maham.A2_1=np.asarray(maham.A2_1)
    Ham1=JHval()
    Ham2=JHval()
    Exp1=JEco()
    Exp2=JEco()
    Ham1.S,Ham1.g,Ham1.I,Ham1.L,Ham1.A,Ham1.Q,Ham1.D,Ham1.Bk2,Ham1.Bk4,Ham1.Bk6,Ham1.lc,Ham1.Hpp,Ham1.eta=maham.S1,maham.g1,maham.I1,maham.L1,maham.A1,maham.Q1,maham.D1,maham.Bk2,maham.Bk4,maham.Bk6,maham.lc1,maham.Hpp,maham.eta
    Ham2.S,Ham2.g,Ham2.I,Ham1.L,Ham2.A,Ham2.Q,Ham2.D,Ham2.lc,Ham2.Hpp,Ham2.eta=maham.S2,maham.g2,maham.I2,maham.L2,maham.A2,maham.Q2,maham.D2,maham.lc2,maham.Hpp,maham.eta
    Exp1.Freq,Exp1.Points,Exp1.Temperature,Exp1.Fdirection,Exp1.Frange,Exp1.Sampleframe,Exp1.Molframe,Exp1.gframe,Exp1.Aframe,Exp1.Dframe,Exp1.Qframe=Expe.Freq,Expe.Points,Expe.Temperature,Expe.Fdirection,Expe.Frange,Expe.Sampleframe1,Expe.Molframe1,Expe.gframe1,Expe.Aframe1,Expe.Dframe1,Expe.Qframe1
    Exp2.Freq,Exp2.Points,Exp2.Temperature,Exp2.Fdirection,Exp2.Frange,Exp2.Sampleframe,Exp2.Molframe,Exp2.gframe,Exp2.Aframe,Exp2.Dframe,Exp2.Qframe=Expe.Freq,Expe.Points,Expe.Temperature,Expe.Fdirection,Expe.Frange,Expe.Sampleframe2,Expe.Molframe2,Expe.gframe2,Expe.Aframe2,Expe.Dframe2,Expe.Qframe2
    iwas,jwas,kwas,weight,_=Delaunay(Exp1)
    iwas,jwas,kwas,weight=jxn.array(iwas),jxn.array(jwas),jxn.array(kwas),jxn.array(weight)
    if np.allclose(maham.X1_2,0.0) and np.allclose(maham.A1_2,0.0) and np.allclose(maham.A2_1,0.0):
        fielde,specs1=JCalpowder(Ham1,Exp1,iwas,jwas,kwas,weight,Nucl1)
        _,specs2=JCalpowder(Ham2,Exp2,iwas,jwas,kwas,weight,Nucl2)
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
            h1=h1+JLorbit(sx1,sy1,s1z,Ham1.lc,dim,Ham1.L)
        if Ham2.L!=0:
            h1=h1+JLorbit(sx2,sy,sz2,Ham2.lc,dim,Ham2.L)
        if Ham1.I!=0:
            h1=h1+JHfi(sx1,sy1,sz1,ix1,iy1,iz1,Ham1.A,dim)
            h1=h1+JQii(ix1,iy1,iz1,Ham1.Q,Ham1.I,dim)
            gnk=gnfactor(Nucl1)
            nhzex=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[1,0,0]),dtype=complex)
            nhzey=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[0,1,0]),dtype=complex)
            nhzez=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[0,0,1]),dtype=complex)
            hzex-=jxn.kron(nhzex,jxn.eye(dim2,dtype=complex))
            hzey-=jxn.kron(nhzey,jxn.eye(dim2,dtype=complex))
            hzez-=jxn.kron(nhzez,jxn.eye(dim2,dtype=complex))
        if Ham2.I!=0:
            h1=h1+JHfi(sx2,sy2,sz2,ix2,iy2,iz2,Ham2.A,dim)
            h1=h1+JQii(ix2,iy2,iz2,Ham2.Q,Ham2.I,dim)
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
            Aref=jxn.asarray(Ham.A2_1)/1000.0
            Aref=Aref*jxn.eye(3)
            h1+=Hfi(sx2,sy2,sz2,ix1,iy1,iz1,Aref,dim)
        if jxn.any(maham.X1_2):
            Xref=jxn.asarray(Ham.X1_2)/1000.0
            Xref=Xref*jxn.eye(3)
            h1+=JIee(sx1,sy1,sz1,sx2,sy2,sz2,Xref,dim)
        h1=jxn.asarray(h1,dtype=complex)
        #For the total magnetic moment of the system
        stodx=np.zeros((dim,dim),dtype='complex')
        stody=np.zeros((dim,dim),dtype='complex')
        stodz=np.zeros((dim,dim),dtype='complex')
        dimwns1=int((2*Ham.I1+1)*(2*Ham.L1+1))
        dimwns2=int((2*Ham.I2+1)*(2*Ham.L2+1))
        if Ham.S1>0:
            sx1r=jxn.kron(sx1,jxn.eye(dimwns1,dtype=complex))
            sy1r=jxn.kron(sy1,jxn.eye(dimwns1,dtype=complex))
            sz1r=jxn.kron(sz1,jxn.eye(dimwns1,dtype=complex))
            stodx+=jxn.kron(sx1r,jxn.eye(dim2,dtype=complex))
            stody+=jxn.kron(sy1r,jxn.eye(dim2,dtype=complex))
            stodz+=jxn.kron(sz1r,jxn.eye(dim2,dtype=complex))
        if Ham.S2>0:
            sx2r=jxn.kron(sx2,jxn.eye(dimwns2,dtype=complex))
            sy2r=jxn.kron(sy2,jxn.eye(dimwns2,dtype=complex))
            sz2r=jxn.kron(sz2,jxn.eye(dimwns2,dtype=complex))
            stodx+=jxn.kron(jxn.eye(dim1,dtype=complex),sx2r)
            stody+=jxn.kron(jxn.eye(dim1,dtype=complex),sy2r)
            stodz+=jxn.kron(jxn.eye(dim1,dtype=complex),sz2r)
        Blist1=jxn.linspace(Exp1.Frange[0],Exp1.Frange[1],400)
        Blist2=jxn.linspace(Exp1.Frange[0],Exp1.Frange[1],Exp1.Points)
        gdelta=(Blist2[1]-Blist2[0])*1.5
        @jx.jit
        def Oneori(nx,ny,nz,w):
            Elist,Vlist,h2=JPadaptarray(Blist1,h1,hzex,hzey,hzez,nx,ny,nz)
            intensy=JNresina(Blist2,Blist1,Elist,Vlist,dim,Exp1.Freq,stodx,stody,stodz,nx,ny,nz,Exp1.Temperature,h2)
            allesint=intensy*w
            esp1=jxn.where(jxn.isclose(w,0.0),jxn.zeros_like(allesint),allesint)
            return (esp1)
        voneori=jx.vmap(Oneori,in_axes=(0,0,0,0))
        csize=5 #Divides the orientations blocks so the RAM doesn't explote
        tlen=len(weight)
        plen=(csize-(tlen%csize))%csize
        pdw=jxn.pad(weight,(0,plen))
        pdi=jxn.pad(iwas,(0,plen))
        pdj=jxn.pad(jwas,(0,plen))
        pdk=jxn.pad(kwas,(0,plen))
        nparts=len(pdw)//csize
        batw=pdw.reshape(nparts,csize)
        bati=pdi.reshape(nparts,csize)
        batj=pdj.reshape(nparts,csize)
        batk=pdk.reshape(nparts,csize)
        @jx.checkpoint
        def Processvmap(curspect,bat):
            bnx,bny,bnz,bw=bat
            batspect=voneori(bnx,bny,bnz,bw)
            nspect=curspect+jxn.sum(batspect,axis=0)
            return nspect,None
        espectotal,_=jx.lax.scan(Processvmap,jxn.zeros(Exp1.Points),(bati,batj,batk,batw))
        maxlenght=jxn.max(jxn.array(Ham.Hpp))*10
        dB=(Exp1.Frange[1]-Exp1.Frange[0])/(Exp1.Points-1)
        kpoints=501
        kaxis=jxn.arange(-kpoints//2+1,kpoints//2+1)*dB
        kvoigt=JVoigtp(kaxis,jxn.array([1.0]),jxn.array([0.0]),Ham.Hpp,etas)
        espectotal=jsig.fftconvolve(espectotal,kvoigt,mode='same')
        fielde,specs=Blist2,espectotal
    return fielde,specs

def JMusic(maham,Expe,Nucl1='None',Nucl2='None',graph='True'):
    Blist,epc=Jcalmusic(maham,Expe,Nucl1,Nucl2)
    if graph=='True':
        plt.plot(Blist,epc,color='navy')
        plt.xlabel('Field [mT]')
        plt.ylabel('Counts [U. A.]')
        plt.xlim(Exp.Frange[0],Exp.Frange[1])
        plt.grid()
        plt.show(block=False)
    return Blist,epc

def Jcalmusic(maham,Expe,Nucl1='None',Nucl2='None'):
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
            h1=h1+JLorbit(sx1,sy1,s1z,Ham1.lc,dim,Ham1.L)
        if Ham2.L!=0:
            h1=h1+JLorbit(sx2,sy,sz2,Ham2.lc,dim,Ham2.L)
        if Ham1.I!=0:
            h1=h1+JHfi(sx1,sy1,sz1,ix1,iy1,iz1,Ham1.A,dim)
            h1=h1+JQii(ix1,iy1,iz1,Ham1.Q,Ham1.I,dim)
            gnk=gnfactor(Nucl1)
            nhzex=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[1,0,0]),dtype=complex)
            nhzey=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[0,1,0]),dtype=complex)
            nhzez=jxn.asarray(betan*JNhze(Ham1.I,ix1,iy1,iz1,dim1,gnk,[0,0,1]),dtype=complex)
            hzex-=jxn.kron(nhzex,jxn.eye(dim2,dtype=complex))
            hzey-=jxn.kron(nhzey,jxn.eye(dim2,dtype=complex))
            hzez-=jxn.kron(nhzez,jxn.eye(dim2,dtype=complex))
        if Ham2.I!=0:
            h1=h1+JHfi(sx2,sy2,sz2,ix2,iy2,iz2,Ham2.A,dim)
            h1=h1+JQii(ix2,iy2,iz2,Ham2.Q,Ham2.I,dim)
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
            Aref=jxn.asarray(Ham.A2_1)/1000.0
            Aref=Aref*jxn.eye(3)
            h1+=Hfi(sx2,sy2,sz2,ix1,iy1,iz1,Aref,dim)
        if jxn.any(maham.X1_2):
            Xref=jxn.asarray(Ham.X1_2)/1000.0
            Xref=Xref*jxn.eye(3)
            h1+=JIee(sx1,sy1,sz1,sx2,sy2,sz2,Xref,dim)
        h1=jxn.asarray(h1,dtype=complex)
        #For the total magnetic moment of the system
        stodx=np.zeros((dim,dim),dtype='complex')
        stody=np.zeros((dim,dim),dtype='complex')
        stodz=np.zeros((dim,dim),dtype='complex')
        dimwns1=int((2*Ham.I1+1)*(2*Ham.L1+1))
        dimwns2=int((2*Ham.I2+1)*(2*Ham.L2+1))
        if Ham.S1>0:
            sx1r=jxn.kron(sx1,jxn.eye(dimwns1,dtype=complex))
            sy1r=jxn.kron(sy1,jxn.eye(dimwns1,dtype=complex))
            sz1r=jxn.kron(sz1,jxn.eye(dimwns1,dtype=complex))
            stodx+=jxn.kron(sx1r,jxn.eye(dim2,dtype=complex))
            stody+=jxn.kron(sy1r,jxn.eye(dim2,dtype=complex))
            stodz+=jxn.kron(sz1r,jxn.eye(dim2,dtype=complex))
        if Ham.S2>0:
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
        fielde=Blist
        specs=spc
    return fielde,specs
    
def Briggs(Hamer,Exp,Vary,expr,maximal=2000,eps=1e-11,mode='p'):
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
          iwas,jwas,kwas,weight=jxn.array(iwas),jxn.array(jwas),jxn.array(kwas),jxn.array(weight)
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
      optimus=optax.chain(optax.clip_by_global_norm(1.0),optax.zero_nans(),optax.adam(learning_rate=0.1))
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
              _,simul=JCalpowder(Hame,dExp,iwas,jwas,kwas,weight)
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
              Blis,espc=JCalpowder(Hat,dExp,iwas,jwas,kwas,weight)
              plt.figure(figsize=(8,6))
              plt.plot(Blis,expr,label='Data')
              plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
              plt.xlabel('Field [mT]')
              plt.ylabel('Counts [U. A.]')
              plt.grid()
              plt.title('EPR Spectrum')
              plt.show()
              return espc
          elif mode=='c':
              Blis,espc=Calresonant(Hat,dExp)
              plt.figure(figsize=(8,6))
              plt.plot(Blis,expr,label='Data')
              plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
              plt.xlabel('Field [mT]')
              plt.ylabel('Counts [U. A.]')
              plt.grid()
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
          Blis,espc=JCalpowder(Hat,dExp,iwas,jwas,kwas,weight)
          plt.figure(figsize=(8,6))
          plt.plot(Blis,expr,label='Data')
          plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
          plt.xlabel('Field [mT]')
          plt.ylabel('Counts [U. A.]')
          plt.grid()
          plt.title('EPR Spectrum')
          plt.show()
          return espc
      elif mode=='c':
          Blis,espc=Calresonant(Hat,dExp)
          plt.figure(figsize=(8,6))
          plt.plot(Blis,expr,label='Data')
          plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
          plt.xlabel('Field [mT]')
          plt.ylabel('Counts [U. A.]')
          plt.grid()
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
      optimus=optax.chain(optax.clip_by_global_norm(1.0),optax.zero_nans(),optax.adam(learning_rate=0.1))
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
              _,simul=JMulpol(Hame,dExp,graph='False')
          elif mode=='c':
              _,simul=JMusic(Hame,dExp,graph='False')
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
              Blis,espc=JMulpol(Hat,dExp,plot='False')
              plt.figure(figsize=(8,6))
              plt.plot(Blis,expr,label='Data')
              plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
              plt.xlabel('Field [mT]')
              plt.ylabel('Counts [U. A.]')
              plt.grid()
              plt.title('EPR Spectrum')
              plt.show()
              return espc
          elif mode=='c':
              Blis,espc=JMusic(Hat,dExp,plot='False')
              plt.figure(figsize=(8,6))
              plt.plot(Blis,expr,label='Data')
              plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
              plt.xlabel('Field [mT]')
              plt.ylabel('Counts [U. A.]')
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
          Blis,espc=JMulpol(Hat,dExp,plot='False')
          plt.figure(figsize=(8,6))
          plt.plot(Blis,expr,label='Data')
          plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
          plt.xlabel('Field [mT]')
          plt.ylabel('Counts [U. A.]')
          plt.grid()
          plt.title('EPR Spectrum')
          plt.show()
          return espc
      elif mode=='c':
          Blis,espc=JMusic(Hat,dExp,plot='False')
          plt.figure(figsize=(8,6))
          plt.plot(Blis,expr,label='Data')
          plt.plot(Blis,espc/np.max(espc)*np.max(expr),label='Fit')
          plt.xlabel('Field [mT]')
          plt.ylabel('Counts [U. A.]')
          plt.grid()
          plt.title('EPR Spectrum')
          plt.show()
          return espc

