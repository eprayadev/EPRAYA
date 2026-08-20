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
import threading
from itertools import product as iterproduct
from .base_ham import *
from .base_plot import *
from .base_rotate import *
from .base_powd import *


#Find energy values in function of field
def EAdaptarray(espac,h1,iser):
    '''
    Constructs the Zeeman hamiltonian, adds it to the complete one and finds the energy values and eigenvectors.
    
    Parameters
    ----------
    espac : np.array
        Array with the values of the magnetic field.
    h1 : np.array
        Hamiltonian matrix that contains all no Zeeman interactions.
    iser : np.array
        Hamiltonian matrix of the Zeeman interaction.
    Returns
    -------

    Elist : np.array
        Array of the energy values of the hamiltonian.

    Vlist : np.array
        Array of the eigenvectors of the hamiltonian.
    '''
    Blist=espac
    h2=iser
    Elist=np.zeros((len(Blist),h1.shape[0]),dtype=np.float64)
    Vlist=np.zeros((len(Blist),h1.shape[0],h1.shape[0]),dtype=np.complex128)
    for B in range(len(Blist)):
        h3=h1+h2*Blist[B]
        vals,vecs=np.linalg.eigh(h3)
        Elist[B]=vals
        Vlist[B]=vecs
    return Elist,Vlist
# Takes into account the possibility of crossing in the energies, then makes an approximation with
# the eigenvectors change, that will be "close" from each other, if the field difference is low
def ERetrack(field,energy,einvector):
    '''

    Organize the eigenvectors and energies to relate them with the quantum numbers of the system, taking as references the values at high field.
    
    Parameters
    ----------
    
    field : np.array
        Array with the values of the magnetic field.
        
    energy : np.array
        Array of the energy values of the hamiltonian.
        
    einvector : np.array
        Array of the eigenvectors of the hamiltonian.
    
    Returns
    -------
    
    Enegria : np.array
        Array of the sorted energy values of the hamiltonian.
    Vector : np.array
        Array of the sorted eigenvectors of the hamiltonian.

    '''
    Enegria=np.array(energy,copy=True)
    Vector=np.array(einvector,copy=True)
    for i in range(len(field)-2,-1,-1):
        oncevals,oncevecs=Enegria[i+1],Vector[i+1]
        actvals,actvecs=Enegria[i],Vector[i]
        novovals,novovecs=EHungorder(oncevals,oncevecs,actvals,actvecs)
        Enegria[i],Vector[i]=novovals,novovecs
    return Enegria,Vector

# Makes the approximation by the assigment problem solution
def EHungorder(onevals,onevecs,actvals,actvecs):
    '''
    Solves the assigment problem with the J-V method implemented in scipy, to organize the eigenvectors and energies of the hamiltonian and relate them with the quantum numbers.
    

    Parameters
    ----------
    
    onevlas : np.array 
        Array of the next point (i+1) energy values of the hamiltonian.
    onevecs : np.array
        Array of the next point (i+1) eigenvectors of the hamiltonian.

    actvals : np.array
        Array of the point (i) energy values of the hamiltonian.
    actvecs : np.array

        Array of the point (i) eigenvectors of the hamiltonian.
    Returns
    -------
    actvals[novo] : np.array
        Sorted energy value of the hamiltonian.
    actvecs[:,novo] : np.array
        Sorted eigenvectors of the hamiltonian.

    '''
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

@njit
def EBoltfactor(Eghz,di,dj,Temp):
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

    popui-popuj : np.array
        Temperature dependent Boltzmann distribution.
    
    '''
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

def Eresonant(Hamer,Exp,graph=True,table=True):  #Function for finding the resonant fields and energies
    '''
    Function for the simulation of the EPR spectrum for monocrystal samples, creating the energy diagrams and a table of the resonant fields.
    
    Parameters
    ----------
    
    Hamer : Class

        Container for the hamiltonian parameters of the system.
    
    Expe : Class

        Container for the experimental conditions.
        
    graph : Bool
        Plots the resulting spectrum and energy diagrams.
        
    table : Bool
        Generates the resonant fields and transitions table.
    
    Returns
    -------
    
    espac1 : np.array
        Array of the magnetic field.
    inten1 : np.array
        Array of the counts of the spectrum.
    
    Example
    -------
    .. code-block:: python

       import matplotlib.pyplot as plt
       import epraya as epr
       import numpy as np
       Ham,Exp,_=epr.Start()
       Ham.S=1
       Ham.I=1/2
       Ham.g=np.array([2.003, 2, 2])
       Ham.A=np.array([200, 200, 200])
       Ham.Hpp=[0, 20]
       Exp.Freq=9.4
       Exp.Points=4096
       Exp.Temperature=300
       Exp.Frange=[0,800]
       epr.Eresonant(Ham,Exp)
    .. image:: /_static/tabla.PNG
       :alt: Table of the resonant fields and transitions
       :align: center
    .. image:: /_static/cristal.PNG
       :alt: Plot of the spectrum of the Eresonant function
       :align: center
    .. image:: /_static/diagra.PNG
       :alt: Energy diagram produced with the Eresonant function
       :align: center

    '''
    Ham=deepcopy(Hamer)
    if Exp.Freq<=0:
        raise ValueError("Frequency can't be a negative or zero value")
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
    Ham.Hpp[0],Ham.Hpp[1]=Ham.Hpp[0]/1.0,Ham.Hpp[1]/1.0
    if Ham.Hpp[0]==0.0 and Ham.Hpp[1]!=0.0:
        eta=1.0
    elif Ham.Hpp[0]!=0.0 and Ham.Hpp[1]==0.0:
        eta=0.0
    else:
        eta=Ham.eta
    dim=int(2*Ham.S+1)*int(2*Ham.I+1)*int(2*Ham.L+1)
    dimi=int(2*Ham.I+1)
    diml=int(2*Ham.L+1)
    Ham=chaframe(Ham,Exp)
    rot=Rotationmat(Exp)
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
    hze=np.asarray(beta*Hze(sx,sy,sz,Ham.g,Exp.Fdirection,dim),dtype=complex)
    hmw=np.asarray(beta*Hze(sx,sy,sz,Ham.g,Exp.Mwdirection,dim),dtype=complex)
    if Ham.S>=1:
        h1=h1+StevensO(sx,sy,sz,Ham.S,Ham,dim)
    if Ham.L!=0:
        h1=h1+Lorbit(sx,sy,sz,Ham.lc,dim,Ham.L)
    if Ham.I!=0:
        h1=h1+Hfi(sx,sy,sz,ix,iy,iz,Ham.A,dim)
        hze-=np.asarray(betan*Nhze(Ham.I,ix,iy,iz,dim,Ham.Nucl,Exp.Fdirection),dtype=complex)
    if np.any(Ham.Q):
        h1=h1+Qii(ix,iy,iz,Ham.Q,dim)
    h1=np.asarray(h1,dtype=complex)
    Blist=np.linspace(Exp.Frange[0],Exp.Frange[1],100)
    Elist,Vlist=EAdaptarray(Blist,h1,hze)
    Elist,Vlist=ERetrack(Blist,Elist,Vlist)
    #Find allowed transitions
    targettr=set()
    targettr.update(tuple(sorted(p)) for p in transitions["allowed"])
    targettr.update(tuple(sorted(p)) for p in transitions["for Dms2"])
    maxvector=Vlist[-1]
    curvebasis=Assingstatestobasis(maxvector)
    #Cubic splines algorithm for the values
    splines=cubichers(Blist,Elist,axis=0)
    #Resonant fields determination
    resonants=[]
    resfield=[]
    intensy=[]
    for i in range(dim):
        for j in range(i+1,dim):
            basis1=curvebasis[i]
            basis2=curvebasis[j]
            pair=tuple(sorted((basis1,basis2)))
            if pair not in targettr:
                continue
            def deltaE(b):
                return np.real(np.abs(splines(b)[j]-splines(b)[i]))-Exp.Freq
            diffv=np.abs(Elist[:,j]-Elist[:,i])-Exp.Freq
            signch=np.where(np.diff(np.signbit(diffv)))[0]
            for k in signch:
                bstart,bend=Blist[k],Blist[k+1]
                try:
                    res=sci.optimize.root_scalar(deltaE,bracket=[bstart,bend],method='brentq')
                    if res.converged:
                        #Interpolate for intensities
                        t=(res.root-bstart)/(bend-bstart)
                        vik,vik1=Vlist[k,:,i],Vlist[k+1,:,i]
                        vjk,vjk1=Vlist[k,:,j],Vlist[k+1,:,j]
                        vecci=(1-t)*vik+t*vik1
                        veccj=(1-t)*vjk+t*vjk1
                        vecci/=np.linalg.norm(vecci)
                        veccj/=np.linalg.norm(veccj)
                        trament=veccj.conj().T@hmw@vecci
                        prob=np.abs(trament)**2
                        #Frecuency to field
                        dert=vecci.conj().T@hze@vecci
                        izrt=veccj.conj().T@hze@veccj
                        gma=np.abs(izrt-dert)
                        if gma<1e-6:
                            gma=1e-6
                        gema=1/gma
                        #Boltzmann distribution
                        Energz=splines(res.root)
                        boltzman=EBoltfactor(Energz,i,j,Exp.Temperature)
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
                        state1=Getlabel(basis1,slit,nlit,llit,Ham.L,Ham.I)
                        state2=Getlabel(basis2,slit,nlit,llit,Ham.L,Ham.I)
                        resonants.append({'field': res.root,'inx': (i, j),'bainx': (basis1,basis2),'type': ttyp,'transition': f"{state1} <-> {state2}"})
                        resfield.append(res.root)
                        intensy.append(prob*boltzman*gema)
                except ValueError:
                    pass
    inten1=Voigtp(espac1,intensy,resfield,Ham.Hpp,Ham.eta)
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
    if type(inten1)==int:
        print("No transition probability in range")
    else:
        if graph:
            Plotsim(espac1,inten1,resfield,Blist,Elist,curvebasis,splines,resonants)

    return espac1,inten1

def Plotsim(espac1,inten1,resfield,espac2,enegria,curvebasis,splines,resonants):
    '''
    Function to produce the graph of the spectrum and the energy diagram.
    
    Parameters
    ----------
    
    espac1 : np.array
        Array of the magnetic field values.
    inten1 : np.array
        Array of the intensities calculated in the resonant fields.
    resfield : np.array 
        Array of the resonant fields.
    espac2 : np.array
        Array of the magnetic field values to calculate the energy diagram.
    enegria : np.array
        Array of the energy values.
    curvebasis : np.array
        Base of the quantum state.
    splines : Scipy class
        Third order polinomium that is use to find the resonant fields.
    resonants : dictionary
        Contains transition type and resonant fields information.

    '''
    slit,nlit,llit,transitions=Msmi(Ham.I,Ham.S,Ham.L)
    if resfield!=[]:
      import plotly.graph_objects as pgo
      import plotly.colors as pc
      if len(resfield)>1:
        col=pc.sample_colorscale('Viridis',[k/(len(resfield)-1) for k in range(len(resfield))])
      else:
          col=['red']
      graphp=pgo.Figure(data=pgo.Scatter(x=espac1,y=inten1, mode='lines',name="Spectrum",line=dict(color='navy')))
      graphp.update_layout(
          title={'text':'EPR spectrum','xanchor':'center','yanchor':'auto','x':0.45, 'y':0.95,
                'font': dict(family='Georgia', size=24,color='black')
                },
          xaxis=dict(title='Field [mT]',showline=True,linecolor='black',mirror=True,linewidth=2,showgrid=True,gridcolor='black',range=[espac1[0],espac1[-1]+10])
          ,yaxis=dict(title='Counts [U. A.]',showline=True,linecolor='black',mirror=True,linewidth=2,showgrid=True,gridcolor='black'),
          plot_bgcolor='white',
          width=1000,
          height=600,
          margin=dict(l=50,r=50,b=50,t=70,pad=4),
          showlegend=True
      )
      i=0
      for r in resonants:
          fv=r['field']
          if r['type']=='Allowed':
              graphp.add_trace(pgo.Scatter(x=[fv, fv], y=[-np.mean(inten1)/2, np.mean(inten1)/2],mode='markers+lines',line=dict(color=col[i], dash='dot'),name=f"Field: {fv:.2f} mT"))
          else:
              graphp.add_trace(pgo.Scatter(x=[fv, fv], y=[-np.mean(inten1)/20, np.mean(inten1)/20],mode='markers+lines',line=dict(color='gray', dash='dot'),name=f"Field: {fv:.2f} mT"))
          i+=1
      upbutton = [
          dict(label="All", method="update", args=[{"visible": [True]*len(graphp.data)}]),
          dict(label="Fields", method="update", args=[{"visible": [trace.name.startswith('Field') for trace in graphp.data]}]),
          dict(label="Spectrum", method="update", args=[{"visible": [True if i == 0 else False for i, trace in enumerate(graphp.data)]}])
      ]
      graphp.update_layout(

          updatemenus=[dict(
              type="buttons", direction="down",
              buttons=upbutton,
              x=0.01,xanchor="auto",y=0.01,yanchor="auto"
          )]
      )
      graphp.add_hline(y=0,line_color="black",line_width=1)
      graphp.add_vline(x=0,line_color="black",line_width=1)
      display(graphp)

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
        eni=splines(fv)[idi]
        enj=splines(fv)[idj]
        if r['type']=='Allowed':
            graphe.add_trace(pgo.Scatter(x=[fv,fv],y=[eni,enj],mode='markers+lines',line=dict(color=coel[idi]),name=f"Field: {fv:.2f} mT",legendgroup="Fir",
        legendgrouptitle_text="R. Fields",legend="legend2"))
        else:
            graphe.add_trace(pgo.Scatter(x=[fv,fv],y=[eni,enj],mode='markers+lines',line=dict(color='gray'),name=f"Field: {fv:.2f} mT",legendgroup="Fi",
        legendgrouptitle_text="R. Fields",legend="legend2"))
      graphe.update_layout(
        title={'text':'Energy VS Field','xanchor':'center','yanchor':'auto','x':0.45, 'y':0.95,
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
      upbutton = [
          dict(label="All", method="update", args=[{"visible": [True] * len(graphe.data)}]),
          dict(label="Fields", method="update", args=[{"visible": [trace.legendgroup!='Fi' for trace in graphe.data]}]),
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


def DFormatfra(val):
    #Change format for fractions
    val=float(val)
    if np.isclose(val,0.0):
        return "0"
    if np.isclose(abs(val%1),0.5):
        num=int(2*val)
        return f"{num}/2"
    return f"{int(val)}" if val.is_integer() else f"{val:.1f}"

def Mulgetlabel(basisidx,slit,nlit):
    '''
    Creates the state ket for the energy level, using the quantum numbers.
    
    Parameters
    ----------
    
    basisidx : float
        Indicates the quantum number of the system.
    slit : np.array
        List of the spin quantum numbers.
    nlit : np.array
        List of the nuclear spin quantum numbers.
        
    Returns
    -------
    output : str
        Ket with the corresponding quantum numbers.
    
    '''
    ams=slit[basisidx]
    ami=nlit[basisidx]
    msf=[DFormatfra(s) for s in ams]
    msf=", ".join(msf)
    if len(ami)>0:
        msi=[DFormatfra(i) for i in ami]
        msi=", ".join(msi)
        return f"|Ms:{msf}, Mi:{msi}⟩"
    else:
        return f"|Ms:{msf}⟩"

def Cristalfm(Hamer,Exp):  #Function for finding the resonant fields and energies
    '''
    Function for the calculation of the spectrum of monocristals without producing the graphs.
    
    Parameters 
    ----------
    
    Hamer : Class
        Container for the hamiltonian parameters of the system.
    
    Exp : Class
        Container for the experimental conditions.
        
    Returns
    -------
    espac1 : np.array
        Array of the magnetic field.
    inten1 : np.array
        Array of the counts of the spectrum.
    Elist : np.array 
        Array of the energy values of the hamiltonian
    dfdis : pandas.Dataframe
        Table of the kets and transitions of the system.
    
    Example
    -------
    
    >>> import epraya as epr
    >>> Ham,Exp,_=epr.Start()
    >>> Ham.S=1
    >>> Ham.I=1/2
    >>> Ham.g=np.array([2.003, 2, 2])
    >>> Ham.A=np.array([200, 200, 200])
    >>> Ham.Hpp=[0, 20]
    >>> Exp.Freq=9.4
    >>> Exp.Points=4096
    >>> Exp.Temperature=300
    >>> Exp.Frange=[0,800]
    >>> print(epr.Cristalfm(Ham,Exp))
    (array([0.00000000e+00, 1.95360195e-01, 3.90720391e-01, ...,
    7.99609280e+02, 7.99804640e+02, 8.00000000e+02], shape=(4096,)),
    array([ 4.08051359e-12,  4.08762234e-12,  4.09474760e-12, ...,
    -1.55121819e-12, -1.54926300e-12, -1.54731109e-12], shape=(4096,)), 
    array([[-2.00000000e-01,  1.00000000e-01, -2.00000000e-01,
         1.00000000e-01,  1.00000000e-01,  1.00000000e-01],...,
       [-2.24948810e+01, -2.22939919e+01, -8.97066590e-04,
         8.89091169e-04,  2.22948889e+01,  2.24939919e+01]]), 
    Field (mT)             Transition     Type  Field (mT)  \
    0     332.234    |-1,1/2⟩<-> |0,1/2⟩  Allowed     164.254   
    1     339.302  |-1,-1/2⟩<-> |0,-1/2⟩  Allowed     171.399   
    2     332.154     |0,1/2⟩<-> |1,1/2⟩  Allowed     339.375   
    Transition           Type  
    0    |-1,1/2⟩<-> |1,1/2⟩  Forbidden (2) 
    1  |-1,-1/2⟩<-> |1,-1/2⟩  Forbidden (2) 
    2   |0,-1/2⟩<-> |1,-1/2⟩        Allowed  )

    '''
    Ham=deepcopy(Hamer)
    if Exp.Freq<=0:
        raise ValueError("Frequency can't be a negative or zero value")
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
    Ham.Hpp[0],Ham.Hpp[1]=Ham.Hpp[0]/1.0,Ham.Hpp[1]/1.0
    if Ham.Hpp[0]==0.0 and Ham.Hpp[1]!=0.0:
        eta=1.0
    elif Ham.Hpp[0]!=0.0 and Ham.Hpp[1]==0.0:
        eta=0.0
    else:
        eta=Ham.eta
    dim=int(2*Ham.S+1)*int(2*Ham.I+1)*int(2*Ham.L+1)
    dimi=int(2*Ham.I+1)
    diml=int(2*Ham.L+1)
    Ham=chaframe(Ham,Exp)
    rot=Rotationmat(Exp)
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
    hze=np.asarray(beta*Hze(sx,sy,sz,Ham.g,Exp.Fdirection,dim),dtype=complex)
    hmw=np.asarray(beta*Hze(sx,sy,sz,Ham.g,Exp.Mwdirection,dim),dtype=complex)
    if Ham.S>=1:
        h1=h1+StevensO(sx,sy,sz,Ham.S,Ham,dim)
    if Ham.L!=0:
        h1=h1+Lorbit(sx,sy,sz,Ham.lc,dim,Ham.L)
    if Ham.I!=0:
        h1=h1+Hfi(sx,sy,sz,ix,iy,iz,Ham.A,dim)
        hze-=np.asarray(betan*Nhze(Ham.I,ix,iy,iz,dim,Ham.Nucl,Exp.Fdirection),dtype=complex)
    if np.any(Ham.Q):
        h1=h1+Qii(ix,iy,iz,Ham.Q,dim)
    h1=np.asarray(h1,dtype=complex)
    Blist=np.linspace(Exp.Frange[0],Exp.Frange[1],100)
    Elist,Vlist=EAdaptarray(Blist,h1,hze)
    Elist,Vlist=ERetrack(Blist,Elist,Vlist)
    #Find allowed transitions
    targettr=set()
    targettr.update(tuple(sorted(p)) for p in transitions["allowed"])
    targettr.update(tuple(sorted(p)) for p in transitions["for Dms2"])
    maxvector=Vlist[-1]
    curvebasis=Assingstatestobasis(maxvector)
    #Cubic splines algorithm for the values
    splines=cubichers(Blist,Elist,axis=0)
    #Resonant fields determination
    resonants=[]
    resfield=[]
    intensy=[]
    for i in range(dim):
        for j in range(i+1,dim):
            basis1=curvebasis[i]
            basis2=curvebasis[j]
            pair=tuple(sorted((basis1,basis2)))
            if pair not in targettr:
                continue
            def deltaE(b):
                return np.real(np.abs(splines(b)[j]-splines(b)[i]))-Exp.Freq
            diffv=np.abs(Elist[:, j]-Elist[:, i])-Exp.Freq
            signch=np.where(np.diff(np.signbit(diffv)))[0]
            for k in signch:
                bstart,bend=Blist[k],Blist[k+1]
                try:
                    res=sci.optimize.root_scalar(deltaE,bracket=[bstart,bend],method='brentq')
                    if res.converged:
                        #Interpolate for intensities
                        t=(res.root-bstart)/(bend-bstart)
                        vik,vik1=Vlist[k,:,i],Vlist[k+1,:,i]
                        vjk,vjk1=Vlist[k,:,j],Vlist[k+1,:,j]
                        vecci=(1-t)*vik+t*vik1
                        veccj=(1-t)*vjk+t*vjk1
                        vecci/=np.linalg.norm(vecci)
                        veccj/=np.linalg.norm(veccj)
                        trament=veccj.conj().T@hmw@vecci
                        prob=np.abs(trament)**2
                        #Frecuency to field
                        dert=vecci.conj().T@hze@vecci
                        izrt=veccj.conj().T@hze@veccj
                        gma=np.abs(izrt-dert)
                        if gma<1e-6:
                            gma=1e-6
                        gema=1/gma
                        #Boltzmann distribution
                        Energz=splines(res.root)
                        boltzman=EBoltfactor(Energz,i,j,Exp.Temperature)
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
                        state1=Getlabel(basis1,slit,nlit,llit,0,Ham.I)
                        state2=Getlabel(basis2,slit,nlit,llit,0,Ham.I)
                        resonants.append({'field':res.root,'inx':(i, j),'bainx': (basis1,basis2),'type':ttyp,'transition':f"{state1}<-> {state2}"})
                        resfield.append(res.root)
                        intensy.append(prob*boltzman*gema)
                except ValueError:
                    pass
    inten1=Voigtp(espac1,intensy,resfield,Ham.Hpp,Ham.eta)
    if len(resfield)>0:
        df=DataFrame(data=resonants)
        dfdis=df[['field', 'transition', 'type']].copy()
        dfl=dfdis.iloc[::2].reset_index(drop=True)
        dfr=dfdis.iloc[1::2].reset_index(drop=True)
        dfdis=concat([dfl, dfr],axis=1)
        dfdis.columns=['Field (mT)','Transition','Type','Field (mT)','Transition','Type']
        dfdis['Field (mT)']=dfdis['Field (mT)'].round(3)
    else:
        dfdis=DataFrame()

    return espac1,inten1,Elist,dfdis

def Music(Hamer,Exper,graph=True,table=True):
    '''
    Wrap function that calculates the spectrum and table of transitions of multisystems. If there is an interaction between the systems (electron-eletron or hiperfine), solves the total hamiltonian. Otherwise, use the function Cristalfm and sums the contributions to the total spectrum.
    
    Parameters
    ----------
    
    Hamer : Class
        Container for the hamiltonian parameters of the system.
    
    Expe : Class
        Container for the experimental conditions.
        
    graph : Bool
        Plots the resulting spectrum and energy diagrams.
        
    table : Bool
        Generates the resonant fields and transitions table.
    
    Returns
    -------
    
    fild1 : np.array
        Array of the magnetic field.
    sumespct : np.array
        Array of the counts of the spectrum.
        
    Example
    -------
    
    .. code-block:: python
    
       import matplotlib.pyplot as plt
       import epraya as epr
       import numpy as np
       Ham,Exp,_=epr.Start(2)
       Ham.S2=1
       Ham.S1=1
       Ham.I1=1/2
       Ham.g1=np.array([2.003, 2, 2])
       Ham.A1=np.array([200, 200, 200])
       Ham.Hpp1=[0, 20]
       Ham.I2=1/2
       Ham.g2=np.array([3.003, 3, 3])
       Ham.A2=np.array([100, 100, 100])
       Exp.Freq1=9.4
       Exp.Points1=4096
       Exp.Temperature1=300
       Exp.Frange1=[0,800]
       B,spc=epr.Music(Ham,Exp)
    .. image:: /_static/tabla2.PNG
       :alt: Table of the Music function    
    .. image:: /_static/music.PNG
       :alt: Plot of the Music function
       :align: center
    '''
    Ham=deepcopy(Hamer)
    numberes=len(Ham.Mulham)
    for inka in range(0,numberes):
        Exper.Mexp[inka].Freq=Exper.Mexp[0].Freq
        Exper.Mexp[inka].Points=Exper.Mexp[0].Points
        Exper.Mexp[inka].Temperature=Exper.Mexp[0].Temperature
        Exper.Mexp[inka].Fdirection=Exper.Mexp[0].Fdirection
        Exper.Mexp[inka].Frange=Exper.Mexp[0].Frange
        Exper.Mexp[inka].Mwdirection=Exper.Mexp[0].Mwdirection
    Exp=deepcopy(Exper)
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
    energya=[]
    padata=[]
    for elka in numboint:
        if len(elka)==1:
            Ham.Mulham[elka[0]].Hpp=Ham.Mulham[0].Hpp
            Ham.Mulham[elka[0]].eta=Ham.Mulham[0].eta
            fild1,fild2,enegrias,datafr=Cristalfm(Ham.Mulham[elka[0]],Exp.Mexp[0])
            sumespct+=fild2
            energya.append(enegrias)
            padata.append(datafr)
        else:
            if Exp.Mexp[0].Freq<=0:
                raise ValueError("Frequency can't be a negative or zero value")
            if Exp.Mexp[0].Frange[0]<0:
                Exp.Mexp[0].Frange[0]=0
                print("WARNING: A negative value has been used, changing it to zero.")
            if Exp.Mexp[0].Frange[0]>=Exp.Mexp[0].Frange[1]:
                raise ValueError("Field range can't be from higher to lower values.")
            numberes=len(Ham.Mulham)
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
            hmw=0
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
                hmw+=Kroexpand(np.asarray(beta*Hze(Pmatrixs[orka][0],Pmatrixs[orka][1],Pmatrixs[orka][2],Ham.Mulham[orka].g,Exp.Mexp[0].Mwdirection,dimerq[orka]),dtype=complex),orka,dimerq)
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
            ndir=np.array(Exp.Mexp[0].Fdirection)
            ndir=ndir/np.linalg.norm(ndir)
            hze=ndir[0]*hzex+ndir[1]*hzey+ndir[2]*hzez
            Sval=[Ham.Mulham[se].S for se in elka]
            Ival=[Ham.Mulham[ie].I for ie in elka]
            slit,nlit,transitions=MMsmi(Sval,Ival)
            targettr=set()
            for key in ["allowed","for Dms2","for nuclear","cross spin"]:
                if key in transitions:
                    targettr.update(tuple(sorted(p)) for p in transitions[key])

            espectotal=np.zeros(Exp.Mexp[0].Points)
            Blist=np.linspace(Exp.Mexp[0].Frange[0],Exp.Mexp[0].Frange[1],500)
            Elist,Vlist=EAdaptarray(Blist,h1,hze)
            Elist,Vlist=ERetrack(Blist,Elist,Vlist)
            maxvector=Vlist[-1]
            curvebasis=Assingstatestobasis(maxvector)
            #Cubic splines algorithm for the values
            splines=cubichers(Blist,Elist,axis=0)
            #Resonant fields determination
            resonants=[]
            resfield=[]
            intensy=[]
            for i in range(dim):
                for j in range(i+1,dim):
                    basis1=curvebasis[i]
                    basis2=curvebasis[j]
                    pair=tuple(sorted((basis1,basis2)))
                    if pair not in targettr:
                        continue
                    def deltaE(b):
                        return np.real(np.abs(splines(b)[j]-splines(b)[i]))-Exp.Mexp[0].Freq
                    diffv=np.abs(Elist[:,j]-Elist[:,i])-Exp.Mexp[0].Freq
                    signch=np.where(np.diff(np.signbit(diffv)))[0]
                    for k in signch:
                        bstart,bend=Blist[k],Blist[k+1]
                        try:
                            res=sci.optimize.root_scalar(deltaE,bracket=[bstart,bend],method='brentq')
                            if res.converged:
                                t=(res.root-bstart)/(bend-bstart)
                                vik,vik1=Vlist[k,:,i],Vlist[k+1,:,i]
                                vjk,vjk1=Vlist[k,:,j],Vlist[k+1,:,j]
                                vecci=(1-t)*vik+t*vik1
                                veccj=(1-t)*vjk+t*vjk1
                                vecci/=np.linalg.norm(vecci)
                                veccj/=np.linalg.norm(veccj)
                                trament=veccj.conj().T@hmw@vecci
                                prob=np.abs(trament)**2
                                dert=np.real(vecci.conj().T@hze@vecci)
                                izrt=np.real(veccj.conj().T@hze@veccj)
                                gma=np.abs(izrt-dert)
                                if gma<1e-6:
                                    gma=1e-6
                                gema=1/gma
                                Energz=splines(res.root)
                                boltzman= EBoltfactor(Energz,i,j,Exp.Mexp[0].Temperature)
                                ttyp="Forbidden"
                                if pair in [tuple(sorted(p)) for p in transitions.get("allowed",[])]:
                                    ttyp="Allowed"
                                elif pair in [tuple(sorted(p)) for p in transitions.get("for Dms2",[])]:
                                    ttyp="Forbidden (2)"
                                elif pair in [tuple(sorted(p)) for p in transitions.get("for nuclear",[])]:
                                    ttyp="Forbidden (N)"
                                elif pair in [tuple(sorted(p)) for p in transitions.get("cross spin",[])]:
                                    ttyp="Cross-Spin"
                                state1=Mulgetlabel(basis1,slit,nlit)
                                state2=Mulgetlabel(basis2,slit,nlit)
                                resonants.append({'field':res.root,'inx':(i,j),'bainx':(basis1,basis2),'type': ttyp,'transition':f"{state1}<->{state2}"})
                                resfield.append(res.root)
                                intensy.append(prob*boltzman*gema)
                        except ValueError:
                            pass
            inten1=Voigtp(espac1,intensy,resfield,Ham.Mulham[0].Hpp,Ham.Mulham[0].eta)
            sumespct+=inten1
            fild1=espac1
            if len(resfield)>0:
                df=DataFrame(data=resonants)
                df=df[df['type']=='Allowed'].reset_index(drop=True)
                df['field']=df['field'].astype(float).round(3)
                df=df.drop_duplicates(subset=['field']).reset_index(drop=True)
                if not df.empty:
                    dfdis= df[['field','transition','type']].copy()
                    dfl=dfdis.iloc[::2].reset_index(drop=True)
                    dfr=dfdis.iloc[1::2].reset_index(drop=True)
                    dfdis=concat([dfl, dfr], axis=1)
                    if len(dfdis.columns)==6:
                        dfdis.columns=['Field (mT)','Transition','Type','Field (mT)','Transition','Type']
                    elif len(dfdis.columns)==3:
                        dfdis.columns=['Field (mT)','Transition','Type']
                else:
                    dfdis=DataFrame()
            energya.append(Elist)
            if 'dfdis' in locals() and not dfdis.empty:
                padata.append(dfdis)
    dfdis=concat(padata,ignore_index=True)
    if table:
        if is_notebook():
            from IPython.display import display
            display(dfdis)
        else:
            print(dfdis)
    if graph:
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
