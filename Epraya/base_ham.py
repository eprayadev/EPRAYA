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

@dataclass
class Hval:
    S: Union[float,int]=1/2   # Spin
    g: Union[List[float], float,int]=dcfield(default_factory=lambda:2.003)  # g value
    I: float=0.0   # Nuclear spin
    L: float=0.0   # Angular momentum
    A: Union[List[float], float]=dcfield(default_factory=lambda:np.array([0,0,0]))     # Hyperfine constant
    Q: Union[list[float], float]=dcfield(default_factory=lambda:np.array([0,0,0]))    # Quadrupole interaction constant
    D: Union[list[float], float]=dcfield(default_factory=lambda:np.array([0,0]))     # Zero field interaction D and E constants
    Bk2: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0])
    Bk4: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0,0,0,0,0])
    Bk6: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0,0,0,0,0,0,0,0,0])
    lc: float=0.0                          # Spin-orbit interaction constant
    Hpp: List=dcfield(default_factory=lambda:np.array([0,1]))
    eta: float=0.5
    weight: float=0.0
    Nucl: str='None'

@dataclass
class SHval:
    S: Union[float,int]=1/2   # Spin
    g: Union[List[float], float,int]=dcfield(default_factory=lambda:2.003)  # g value
    I: float=0.0   # Nuclear spin
    A: Union[List[float], float]=dcfield(default_factory=lambda:np.array([0,0,0]))     # Hyperfine constant
    Q: Union[list[float], float]=dcfield(default_factory=lambda:np.array([0,0,0]))    # Quadrupole interaction constant
    D: Union[list[float], float]=dcfield(default_factory=lambda:np.array([0,0]))     # Zero field interaction D and E constants
    Bk2: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0])
    Bk4: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0,0,0,0,0])
    Bk6: Union[list[float], float]=dcfield(default_factory=lambda:[0,0,0,0,0,0,0,0,0,0,0,0,0])
    weight: float=0.0
    Nucl: str='None'

@dataclass
class Multham:
    Mulham: List[Hval]=dcfield(default_factory=list)
    Amix: dict=dcfield(default_factory=dict)
    Xmix: dict=dcfield(default_factory=dict)

    def _parse_attr(self,name):
        match=re.match(r"^(S|g|I|L|A|Q|D|Bk2|Bk4|Bk6|lc|Hpp|eta|weight|Nucl)(\d+)$",name)
        if match:
            return "single",match.group(1),int(match.group(2))-1
        cross=re.match(r"^(A|X)(\d+)_(\d+)$",name)
        if cross:
            return "cross",cross.group(1),int(cross.group(2))-1,int(cross.group(3))-1
        return None,None,None

    def __getattr__(self,name):
        attrtype,*args=self._parse_attr(name)
        if attrtype=="single":
            attr,idx=args
            if 0<=idx<len(self.Mulham):
                return getattr(self.Mulham[idx],attr)
        elif attrtype=="cross":
            attr,idx1,idx2=args
            if idx1==idx2:
                raise ValueError(f"Auto-interactions of spins are not supported.")
            if 0<=idx1<len(self.Mulham) and 0<=idx2<len(self.Mulham) and idx1!=idx2:
                akey=(idx1,idx2)
                if attr=="A":
                    return self.Amix.get(akey,np.array([0.0,0.0,0.0]))
                if attr=="X":
                    return self.Xmix.get(akey,np.array([0.0,0.0,0.0]))
        raise AttributeError(f"'{type(self).__name__}' doesn't have '{name}'")

    def __setattr__(self,name,value):
        if name in ["Mulham","Amix","Xmix"]:
            super().__setattr__(name,value)
            return
        attrtype,*args=self._parse_attr(name)

        if attrtype=="single":
            attr,idx=args
            if 0<=idx<len(self.Mulham):
                setattr(self.Mulham[idx],attr,value)
                return

        elif attrtype=="cross":
            attr,idx1,idx2=args
            if idx1==idx2:
                raise ValueError(f"Auto-interactions of spins are not supported.")
            if 0<=idx1<len(self.Mulham) and 0<=idx2<len(self.Mulham) and idx1!=idx2:
                akey=(idx1,idx2)
                if attr=="A":
                    self.Amix[akey]=np.array(value)
                if attr=="X":
                    self.Xmix[akey]=np.array(value)
                return
        super().__setattr__(name,value)

@dataclass
class Eco:
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

@dataclass
class Eva:
    g: Union[list[float],float]=0.0
    A: Union[list[float],float]=0.0     # Hyperfine constant
    Q: Union[list[float],float]=0.0     # Quadrupole interaction constant
    D: Union[list[float],float]=0.0
    Hpp: List=dcfield(default_factory=lambda:np.array([0,0]))
    weight: float=0.0

@dataclass
class Muleva:
    Mvary: List[Eva]=dcfield(default_factory=list)
    Vamix: dict=dcfield(default_factory=dict)
    Vxmix: dict=dcfield(default_factory=dict)
    def _parse_attr(self,name):
        match=re.match(r"^(S|g|I|L|A|Q|D|Hpp|weight|)(\d+)$",name)
        if match:
            return "single",match.group(1),int(match.group(2))-1
        cross=re.match(r"^(A|X)(\d+)_(\d+)$",name)
        if cross:
            return "cross",cross.group(1),int(cross.group(2))-1,int(cross.group(3))-1
        return None,None,None

    def __getattr__(self,name):
        attrtype,*args=self._parse_attr(name)
        if attrtype=="single":
            attr,idx=args
            if 0<=idx<len(self.Mvary):
                return getattr(self.Mvary[idx],attr)
        elif attrtype=="cross":
            attr,idx1,idx2=args
            if idx1==idx2:
                raise ValueError(f"Auto-interactions of spins are not supported.")
            if 0<=idx1<len(self.Mvary) and 0<=idx2<len(self.Mvary) and idx1!=idx2:
                akey=(idx1,idx2)
                if attr=="A":
                    return self.Vamix.get(akey,np.array([0.0,0.0,0.0]))
                if attr=="X":
                    return self.Vxmix.get(akey,np.array([0.0,0.0,0.0]))
        raise AttributeError(f"'{type(self).__name__}' doesn't have '{name}'")

    def __setattr__(self,name,value):
        if name in ["Mvary","Vamix","Vxmix"]:
            super().__setattr__(name,value)
            return
        attrtype,*args=self._parse_attr(name)

        if attrtype=="single":
            attr,idx=args
            if 0<=idx<len(self.Mvary):
                setattr(self.Mvary[idx],attr,value)
                return
        elif attrtype=="cross":
            attr,idx1,idx2=args
            if idx1==idx2:
                raise ValueError(f"Auto-interactions of spins are not supported.")
            if 0<=idx1<len(self.Mvary) and 0<=idx2<len(self.Mvary) and idx1!=idx2:
                akey=(idx1,idx2)
                if attr=="A":
                    self.Vamix[akey]=np.array(value)
                if attr=="X":
                    self.Vxmix[akey]=np.array(value)
                return
        super().__setattr__(name,value)

@dataclass
class Mulexco:
    Mexp: List[Eco]=dcfield(default_factory=list)
    def _parse_attr(self,name):
        match=re.match(r"^(Freq|Points|Temperature|Fdirection|Frange|Sampleframe|Molframe|gframe|Aframe|Dframe|)(\d+)$",name)
        if match:
            return "single",match.group(1),int(match.group(2))-1
        return None,None,None

    def __getattr__(self,name):
        attrtype,*args=self._parse_attr(name)
        if attrtype=="single":
            attr,idx=args
            if 0<=idx<len(self.Mexp):
                return getattr(self.Mexp[idx],attr)
        raise AttributeError(f"'{type(self).__name__}' doesn't have '{name}'")

    def __setattr__(self,name,value):
        if name in ["Mexp"]:
            super().__setattr__(name,value)
            return
        attrtype,*args=self._parse_attr(name)

        if attrtype=="single":
            attr,idx=args
            if 0<=idx<len(self.Mexp):
                setattr(self.Mexp[idx],attr,value)
                return
        super().__setattr__(name,value)

def Start(num=1):
    global Exp,Vary,Ham
    ae=0
    if num==1:
        Ham=Hval()
        Vary=Eva()
        Exp=Eco()
    elif num<=0:
        raise ValueError(f"Number of spins must be at least 1, not {num}.")
    elif num>4:
        raise ValueError(f"Number of spins must be lower or equal to 4.")
    else:
        Ham=Multham()
        Vary=Muleva()
        Exp=Mulexco()
        for ae in range(0,num):
            Ham.Mulham.append(Hval())
            Vary.Mvary.append(Eva())
            Exp.Mexp.append(Eco())
            if ae>0:
                Ham.Mulham[ae].S=0
    return Ham,Exp,Vary
def Kronecker(a0,b0):
    return np.where(a0==b0,1.0,0.0)

def Pauli(s):
    #Defines the pauli matrix for all s values:
    ms=np.linspace(s,-s,int(2*s+1))
    z1=0.5j
    r,g=0,0
    sz=np.diag(ms)
    sx,sy=np.zeros([int(2*s+1),int(2*s+1)],dtype=complex),np.zeros([int(2*s+1),int(2*s+1)],dtype=complex)
    for r in range (0,int(2*s+1)):
        for g in range (0,int(2*s+1)):
            yn=Kronecker(r+1,g+1)
            ym=Kronecker(r+1,g+2)
            yl=Kronecker(r+2,g+1)
            sx[r,g]=(0.5*(ym+yl)*(np.sqrt((s+1)*(r+g+1)-((r+1)*(g+1)))))
            sy[r,g]=(z1*((ym-yl)*(np.sqrt((s+1)*(r+g+1)-((r+1)*(g+1))))))
    return sx,sy,sz

#Spin Orbit term
def Lorbit(sx,sy,sz,lamda,dim,l=0):
  if l !=0:
    lx,ly,lz=Pauli(l)
    orbe=lamda*(np.kron(lx,sx)+np.kron(ly,sy)+np.kron(lz,sz))
    orbe=np.kron(orbe,np.eye(int(dim/(orbe).shape[1])))
    return orbe
  else:
    return np.zeros(dim)

def Hfi(ssx,ssy,ssz,iix,iiy,iiz,at,dim):
    ta=(at[0,0]*np.kron(ssx,iix))+(at[0,1]*np.kron(ssx,iiy))+(at[0,2]*np.kron(ssx,iiz))+(at[1,0]*np.kron(ssy,iix))+(at[1,1]*np.kron(ssy,iiy))+(at[1,2]*np.kron(ssy,iiz))+(at[2,0]*np.kron(ssz,iix))+(at[2,1]*np.kron(ssz,iiy))+(at[2,2]*np.kron(ssz,iiz))
    ta=np.kron(ta,np.eye(int(dim/(ta).shape[1])))
    return ta

def Hze(ssx,ssy,ssz,g,biel,dim):
    hze=biel[0]*(g[0,0]*ssx+g[0,1]*ssy+g[0,2]*ssz)+biel[1]*(g[1,0]*ssx+g[1,1]*ssy+g[1,2]*ssz)+biel[2]*(g[2,0]*ssx+g[2,1]*ssy+g[2,2]*ssz)
    thz=np.kron(hze,np.eye(int(dim/(hze).shape[1])))
    return thz

def Qii(iix,iiy,iiz,q,dim):
    hql=(q[0,0]*iix*iix)+(q[1,1]*iiy*iiy)+(q[2,2]*iiz*iiz)+(q[0,1]*(iix*iiy)-(iiy*iix))+(q[1,2]*(iiy*iiz)-(iiz*iiy))
    +(q[2,0]*(iiz*iix)-(iix*iiz))
    tql=np.kron(hql,np.eye(int(dim/(hql).shape[1])))
    return tql

def Nhze(I,iix,iiy,iiz,dim,Nucl='None',direction=[0,0,1]):
    if Nucl!='None':
        krle=read_csv('nucleardaat.txt',header=0,sep='\t')
        deq=krle[krle['Symbol']==Nucl]
        gn=deq['gN_factor'].values[0]
    else:
        gn=0
    if direction==[0,0,1]:
        direct=iiz
    if direction==[0,1,0]:
        direct=iiy
    if direction==[1,0,0]:
        direct=iix
    else:
        direct=iiz
    nhz=gn*direct
    nhz=np.kron(nhz,np.eye(int(dim/(nhz).shape[1])))
    return nhz

def chaframe(Ham,Exp):
      Ham=Convtarray(Ham)
      D2s=np.asarray([-Ham.D[0]/3+Ham.D[1],-Ham.D[0]/3-Ham.D[1],2*Ham.D[0]/3])
      tA=np.eye(3)*Ham.A
      tQ=np.eye(3)*Ham.Q
      tg=np.eye(3)*Ham.g
      tD=np.eye(3)*D2s
      A1=Rotmatrix(Exp.Aframe[0],Exp.Aframe[1],Exp.Aframe[2]).T@tA@(Rotmatrix(Exp.Aframe[0],Exp.Aframe[1],Exp.Aframe[2]))
      g1=(Rotmatrix(Exp.gframe[0],Exp.gframe[1],Exp.gframe[2])).T@tg@(Rotmatrix(Exp.gframe[0],Exp.gframe[1],Exp.gframe[2]))
      D2=(Rotmatrix(Exp.Dframe[0],Exp.Dframe[1],Exp.Dframe[2])).T@tD@(Rotmatrix(Exp.Dframe[0],Exp.Dframe[1],Exp.Dframe[2]))
      Q1=(Rotmatrix(Exp.Qframe[0],Exp.Qframe[1],Exp.Qframe[2])).T@tQ@(Rotmatrix(Exp.Qframe[0],Exp.Qframe[1],Exp.Qframe[2]))
      Ham.A,Ham.g,Ham.D,Ham.Q=A1,g1,D2,Q1
      return Ham

def Rotationmat(Exp):
    RRmatrix=Rotmatrix(Exp.Sampleframe[0],Exp.Sampleframe[1],Exp.Sampleframe[2])@Rotmatrix(Exp.Molframe[0],Exp.Molframe[1],Exp.Molframe[2])
    return RRmatrix.T

def Rotmatrix(alfa,beta,gamma):
  alfa,beta,gamma=np.radians(alfa),np.radians(beta),np.radians(gamma)
  cosg,sing=np.cos(gamma),np.sin(gamma)
  cosa,sina=np.cos(alfa),np.sin(alfa)
  cosb,sinb=np.cos(beta),np.sin(beta)
  eps=10**-9
  cosg=np.where(np.abs(cosg)<eps,0.0,cosg)
  sing=np.where(np.abs(sing)<eps,0.0,sing)
  cosa=np.where(np.abs(cosa)<eps,0.0,cosa)
  sina=np.where(np.abs(sina)<eps,0.0,sina)
  cosb=np.where(np.abs(cosb)<eps,0.0,cosb)
  sinb=np.where(np.abs(sinb)<eps,0.0,sinb)
  Reuler=np.array([[(cosg*cosa*cosb)-(sing*sina),(cosg*cosa*sinb)+(sing*cosa),-cosg*sinb],
 [-(sing*cosb*cosa)-(cosg*sina),-(sing*cosb*sina)+(cosg*cosa),sing*sinb],
  [sinb*cosa,sina*sinb,cosb]])
  return Reuler

def Convtarray(Ham):
    def formatj(val,variable):
        iti=np.asarray(val,dtype=float)
        if iti.ndim==0:
            return iti*np.array([1.0,1.0,1.0])
        elif iti.ndim==1:
            if iti.shape[0]==3:
                return iti
            elif iti.shape[0]==2:
                return np.array([iti[0],iti[0],iti[1]])
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
    hD=np.asarray(Ham.D,dtype=float)
    if hD.ndim!=1:
        raise RuntimeError("Wrong data type/dimensions in Hval.D, must be 1D list or array.")
    if hD.shape[0]==2:
        hD=hD
    elif hD.shape[0]==3:
        hD=np.array([3.0*hD[2]/2.0,(hD[0]-hD[1])/2.0])
    else:
        raise RuntimeError(f"Wrong number of values in Hval.D, expected 2 or 3, got {hD.shape[0]}.")

    hQ=np.asarray(Ham.Q,dtype=float)
    if hQ.ndim!=1 or hQ.shape[0]!=3:
        raise RuntimeError(f"Wrong values in Hval.Q, expected exactly 3, got {hQ.shape[0] if hQ.ndim==1 else 'matrix'}.")
    hQ=hQ
    Ham.g,Ham.A,Ham.D,Ham.Q=hg,hA,hD,hQ
    return Ham


def Msmi(I,S,L=0):
    if I<0 or S<0 or L<0:
        raise ValueError('Spin values cannot be negative')
    dim=int(2*S+1)*int(2*I+1)*int(2*L+1)
    poss=np.linspace(S,-S,int(2*S+1))
    posi=np.linspace(I,-I,int(2*I+1))
    posl=np.linspace(L,-L,int(2*L+1))
    slit,nlit,llit=[],[],[]
    for ml in posl:
        for ms in poss:
            for mi in posi:
                llit.append(ml)
                slit.append(ms)
                nlit.append(mi)
    slit=np.array(slit)
    nlit=np.array(nlit)
    llit=np.array(llit)
    Jlit=slit+llit
    # Allowed and forbidden transitions
    transitions={"allowed": [],   #Delta Ms=+-1, Delta Mi=0
        "for Dms2": [], #Delta Ms=2 Half field
        "for nuclear": [],  #Delta Mi!=0 Ns flip
        "semi for L": []    #Changes in L
    }
    for idx1 in range(dim):
        for idx2 in range(idx1+1,dim):
            dms=np.abs(slit[idx1]-slit[idx2])
            dmi=np.abs(nlit[idx1]-nlit[idx2])
            dmj=np.abs(Jlit[idx1]-Jlit[idx2])
            sallowed=False
            if L==0:
                if np.isclose(dms,1.0) and np.isclose(dmi,0.0):
                    sallowed=True
            else:
                if np.isclose(dmj,1.0) and np.isclose(dmi,0.0):
                    sallowed=True
            if sallowed:
                transitions["allowed"].append([idx1,idx2])
                continue
            if np.isclose(dms,2.0) and np.isclose(dmi,0.0):
                transitions["for Dms2"].append([idx1,idx2])
                continue
            if np.isclose(dms,1.0) and dmi>0:
                transitions["for nuclear"].append([idx1,idx2])
                continue
    for key in transitions:
        transitions[key]=np.array(transitions[key])
    return slit,nlit,llit,transitions

# Relates the vectors to the basis of s, i with the last value at high field
def Assingstatestobasis(vectorhf):
    dima=vectorhf.shape[0]
    mapa={}
    cost=1-np.abs(vectorhf)**2
    row,col=sci.optimize.linear_sum_assignment(cost)
    for ka,ke in zip(row,col):
        mapa[ke]=ka
    return mapa


#Stevens Operators
#Rule: k<=2s
def StevensO(ssx,ssy,ssz,s,Ham,dim):
    k=int(2*s)
    B22,B21,B20,Bq21,Bq22=Ham.Bk2
    B20,B22=3*Ham.D[2,2]/2,(Ham.D[0,0]-Ham.D[1,1])/2
    B21=Ham.D[0,2]
    Bq21=Ham.D[1,2]
    Bq22=Ham.D[0,1]
    B44,B43,B42,B41,B40,Bq41,Bq42,Bq43,Bq44=Ham.Bk4
    B66,B65,B64,B63,B62,B61,B60,Bq61,Bq62,Bq63,Bq64,Bq65,Bq66=Ham.Bk6
    sour=(int(k-k%2))
    if sour==0:
        return np.kron(np.zeros((int(2*s+1),int(2*s+1))),np.eye(int(dim/(np.zeros((int(2*s+1),int(2*s+1)))).shape[1])))
    else:
        k=np.arange(0,sour+1,2)[1:]
        sxminus=ssx-(ssy*1.0j)
        sxsum=ssx+(ssy*1.0j)
        sxminus2=np.linalg.matrix_power(sxminus,2)
        sxsum2=np.linalg.matrix_power(sxsum,2)
        ssz2=np.linalg.matrix_power(ssz,2)
        xs=s*(s+1)
        eye=np.eye(int(2*s+1),dtype=complex)
        if k[-1]>=4:
            ssz3=np.linalg.matrix_power(ssz,3)
            ssz4=np.linalg.matrix_power(ssz,4)
            sxminus3=np.linalg.matrix_power(sxminus,3)
            sxsum3=np.linalg.matrix_power(sxsum,3)
            sxminus4=np.linalg.matrix_power(sxminus,4)
            sxsum4=np.linalg.matrix_power(sxsum,4)
        if k[-1]>=6:
            ssz6=np.linalg.matrix_power(ssz,6)
            ssz5=np.linalg.matrix_power(ssz,5)
            sxminus5=np.linalg.matrix_power(sxminus,5)
            sxsum5=np.linalg.matrix_power(sxsum,5)
            sxminus6=np.linalg.matrix_power(sxminus,6)
            sxsum6=np.linalg.matrix_power(sxsum,6)
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
        totales=np.kron(total,np.eye(int(dim/(total).shape[1])))
        return totales

def PMsmi(I,S,L=0):
    if I<0 or S<0 or L<0:
        raise ValueError('Spin values cannot be negative')
    dim=int(2*S+1)*int(2*I+1)*int(2*L+1)
    poss=np.linspace(S,-S,int(2*S+1))
    posi=np.linspace(I,-I,int(2*I+1))
    posl=np.linspace(L,-L,int(2*L+1))
    ll,sl,il=np.meshgrid(posl,poss,posi,indexing='ij')
    return sl.flatten(),il.flatten(),ll.flatten()
