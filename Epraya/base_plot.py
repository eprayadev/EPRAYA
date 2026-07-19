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

def Sload(dat,rows,cols): #Loads counts and field data
    dt=np.loadtxt(dat,usecols=(cols))
    sh=np.shape(dt)
    if rows<0 or rows>sh[0]:
        raise ValueError(f"The file just has {sh[0]} rows, can't read {rows} rows.")
    if sh[1]!=2:
        raise ValueError("Columns number must be 2")
    if rows==0:
        rows=sh[0]
    dt=dt[:rows]
    dt=dt[~np.isnan(dt).any(axis=1)] #ELiminates Nan values
    field=dt[:,0]
    rcount=dt[:,1]
    return field,rcount

def Splot(field,count):
    plt.figure(figsize=(20,7))
    plt.xlabel('Magnetic Field [mT]')
    plt.ylabel('Counts [U. A.]')
    plt.title('EPR spectrum')
    plt.plot(field, count)#, color='green')
    plt.show()

def Sfilter(field,count,startl=0,endl=-1,lt=51,pol=3):
    #Applies a Savitzky-Golay to the count data, finds resonant fields and the peak-to-peak width
    spc=scs.savgol_filter(count,window_length=lt,polyorder=pol)
    # Base line determination
    mx=(spc[endl]-spc[startl])/(field[endl]-field[startl])
    bx=spc[startl]-mx*field[startl]
    basel=bx+(mx*field)
    return spc-basel

def Overseer(field,counts,lt=51,pol=3,i=0,startl=0,endli=-1, epsilon=(5*10**-6),plot="True"): # Shows tools for spectrum analysis and creates variable Sdata, with Hpp, resonant fields, filtered spectrum and integral of the spectrum.
  #Sliders conditions
  hystoria=[]
  maind=len(counts)-1
  sslider=IntSlider(value=0,min=0,max=maind-1,description='First index:',continuous_update=False,style={'description_width': 'initial'})
  eslider=IntSlider(value=maind,min=1,max=maind,description='Last index:',continuous_update=False,style={'description_width': 'initial'})
  ltslider=widgets.IntSlider(value=51,min=5,max=101,step=2,description='Window Length:',style={'description_width': 'initial'})
  polslider=widgets.IntSlider(value=3,min=1,max=10,step=1,description='Polyorder:',style={'description_width': 'initial'})
  #Normalized sliders
  def fschange(chg):
    eslider.min=chg.new+1
    if eslider.value<eslider.min:
      eslider.value=eslider.min
  def lschange(chg):
    sslider.max=chg.new-1
    if sslider.value>sslider.max:
      sslider.value=sslider.max
  def Wrapper(field,count,startl,endli,lt,pol):
    devul=Sfilter(field,counts,startl=startl,endli=endli,lt=lt,pol=pol)
    if devul is not None:
      hystoria.append(devul)
  sslider.observe(fschange,names='value')
  eslider.observe(lschange,names='value')
  savev=widgets.Button(description="Save configuration", button_style='success')
  ouarea=widgets.Output()
  def on_save_clicked(b):
    global Sdata
    with ouarea:
      ouarea.clear_output(wait=True)
      if hystoria:
        # Save results
        Sdata=hystoria[-1]
        print(f"Data saved")
  savev.on_click(on_save_clicked)
  out=interactive_output(Wrapper,{ 'field': fixed(field),'count': fixed(counts),'startl': sslider,'endli': eslider,'lt': ltslider,'pol': polslider})
  controls=VBox([sslider, eslider,ltslider,polslider,savev,ouarea])
  #controls.layout=widgets.Layout(width='500px',border='solid 1px #cccccc',padding='10px',margin='20px 0px 0 700px' )
  app_layout=widgets.HBox([controls, out])
  display(controls,out)

#For tkinter 
def Sload1(dat,rows,cols): #Loads counts and field data
    dt=np.loadtxt(dat,usecols=(cols))
    sh=np.shape(dt)
    if rows<0 or rows>sh[0]:
        raise ValueError("Rows out of index")
    if sh[1]!=2:
        raise ValueError("Columns number must be 2")
    if rows==0:
        rows=sh[0]
    dt=dt[:rows]
    dt=dt[~np.isnan(dt).any(axis=1)] #ELiminates Nan values
    field=dt[:,0]
    rcount=dt[:,1]
    return field,rcount


def Spmanipulation(fig,axes,field,count,lt=51,pol=3,startl=0,endli=-1,einmal=0,aufmal=-1,epsilon=(5*10**-6),plot="True"):
    for ax in axes.flat:
        ax.clear()
    if endli<0: 
        endli=int(len(count)+endli)
    if startl>=endli:
        axes[0,0].set_title("Invalid range",color='red')
        return 0
    dfr=DataFrame()
    spc=scs.savgol_filter(count,window_length=lt,polyorder=pol)
    # Base line determination
    mx=(spc[endli]-spc[startl])/(field[endli]-field[startl])
    bx=spc[startl]-mx*field[startl]
    basel=bx+(mx*field)
    #Determination of the absorption curve
    integ=scii.cumulative_trapezoid(spc-basel,field,initial=0)
    mxi=(integ[aufmal]-integ[einmal])/(field[aufmal]-field[einmal])
    bxi=integ[aufmal]-mxi*field[aufmal]
    baselini=bxi+(mxi*field)
    integ=integ-baselini
    peaks,_=scs.find_peaks(integ)
    prominences=scs.peak_prominences(integ,peaks)[0]
    tempa,val=[],[] #tempa= values of the field in the peaks, val= Values of the peaks that follow the epsilon condition
    for ir in range (0,len(prominences)):
        if prominences[ir]>epsilon:
            val.append(peaks[ir])
            tempa.append(field[peaks[ir]])
    val=np.asarray(val)
    tempa=np.asarray(tempa)
    #Makes another try to find the field by normalizing the spectrum
    if len(tempa)==0:
        maxinteg=np.max((integ))
        if maxinteg!=0:
            peaks,_=scs.find_peaks(integ/np.max(integ),1)
        prominences=scs.peak_prominences(integ/np.max(integ),peaks)[0]
        tempa,val=[],[] #tempa= values of the field in the peaks, val= Values of the peaks that follow the epsilon condition
        for ir in range (0,len(prominences)):
            if prominences[ir]>epsilon:
                val.append(peaks[ir])
                tempa.append(field[peaks[ir]])
        val=np.asarray(val)
        tempa=np.asarray(tempa)
    
    #Finding the Hpp
    Hpp,hp,itsmin,itsmax=np.nan,np.nan,np.nan,np.nan
    if len(tempa)==1:
        tearma=spc[np.where(field<tempa[0])]
        tearmi=spc[np.where(field>tempa[0])]
        if tearma.size>0 and tearmi.size>0:
            itmaxif=np.where(spc==np.nanmax(tearma))[0]
            itminif=np.where(spc==np.nanmin(tearmi))[0]
            if itmaxif.size>0 and itminif.size>0:
                itmaxidx=np.take(itmaxif,itmaxif.size//2)
                itminidx=np.take(itminif,itminif.size//2)
                itmax=field[itmaxidx] #max
                itmin=field[itminidx] #min
                itsmax=itmaxidx 
                itsmin=itminidx 
                hp=itmin-itmax
                Hpp=hp
    elif len(tempa)!=0:
        hp=np.full(len(tempa),np.nan)
        itsmin=np.full(len(tempa),np.nan)
        itsmax=np.full(len(tempa),np.nan)
        for sw in range (0,len(tempa)):
            tearma,tearmi=np.array([]),np.array([])
            if sw==0:
              tearma=spc[np.where(field<tempa[sw])]
              tearmi=spc[np.where((field>tempa[sw])&(field <tempa[sw+1]))]
            elif sw==(len(tempa)-1):
              tearma=spc[np.where((field<tempa[sw])&(field >tempa[sw-1]))]
              tearmi=spc[np.where(field>tempa[sw])]
            else:
              tearma=spc[np.where((field<tempa[sw])&(field >tempa[sw-1]))]
              tearmi=spc[np.where((field>tempa[sw])&(field <tempa[sw+1]))]
            if tearma.size>0 and tearmi.size>0:
                    itmaxif=np.where(spc==np.nanmax(tearma))[0]
                    itminif=np.where(spc==np.nanmin(tearmi))[0]
                    if itmaxif.size>0 and itminif.size>0:
                        itsmaxidx=np.take(itmaxif,itmaxif.size//2)
                        itminidx=np.take(itminif,itminif.size//2)
                        itsmax[sw]=itsmaxidx
                        itsmin[sw]=itminidx
                        itmaxval=field[itsmaxidx]
                        itminval=field[itminidx]
                        hp[sw]=itminval-itmaxval
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category=RuntimeWarning)
            Hpp=np.nanmean(hp)

    # Find the second integral of the spectrum
    integto=scii.cumulative_trapezoid(integ,field,initial=0)
    if len(tempa)==1:
        drr={'Hpp distance [mT]':Hpp,'Resonant fields [mT]':tempa}
        dfr=DataFrame(data=drr)
    elif len(tempa)!=1 and len(tempa)!=0:
        drr={'Hpp distance [mT]':hp,'Resonant fields [mT]':tempa}
        dfr=DataFrame(data=drr)
    ax1=axes[0,0]
    ax1.plot(field,spc)
    ax1.plot(field,basel,"--",color='green',label='Baseline')
    ax1.plot(field[startl],spc[startl],"o",color='blue',label='Line start')
    ax1.plot(field[endli],spc[endli],"o",color='blue',label='Line end')
    ax1.set_title('EPR spectrum')
    ax1.set_xlabel('Magnetic Field [mT]')
    ax1.set_ylabel('Counts [U. A.]')
    ax1.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    if len(tempa)!=0:
        if len(tempa)==1:
            if not np.isnan(itsmin) and not np.isnan(itsmax):
                ax1.axvline(field[int(itsmin)],color='blue',linestyle='--',label='Min. point')
                ax1.axvline(field[int(itsmax)],color='red',linestyle='--',label='Max. point')
        else:
            for kls in range(0,len(tempa)):
                if not np.isnan(itsmin[kls]) and not np.isnan(itsmax[kls]):
                    if kls==0:
                        ax1.axvline(field[int(itsmin[kls])],color='blue',linestyle='--',label='Min. point')
                        ax1.axvline(field[int(itsmax[kls])],color='red',linestyle='--',label='Max. point')
                    else:
                        ax1.axvline(field[int(itsmin[kls])],color='blue',linestyle='--')
                        ax1.axvline(field[int(itsmax[kls])],color='red',linestyle='--')
    ax1.grid(True)
    ax1.legend(fontsize='small')
    
    ax2=axes[0,1]; 
    ax2.plot(field,(spc-basel))
    ax2.set_title('EPR spectrum')
    ax2.set_xlabel('Magnetic Field [mT]')
    ax2.set_ylabel('Counts [U. A.]')
    if len(tempa)!=0:
        ax2.plot(tempa,(spc-basel)[val],"*",color='red',label='Resonant field')
        ax2.legend()
    ax2.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax2.grid(True)
    
    ax3=axes[1,0]
    ax3.plot(field,integ)
    principal=[integ[einmal],integ[aufmal]]
    principal1=[field[einmal],field[aufmal]]
    ax3.plot(principal1,principal,"--",color='green',label='Baseline')
    ax3.plot(field[einmal],integ[einmal],"o",color='blue',label='Line start')
    ax3.plot(field[aufmal],integ[aufmal],"o",color='blue',label='Line end')
    ax3.set_title('Absorption curve')
    ax3.set_xlabel('Magnetic Field [mT]')
    ax3.set_ylabel('Counts [U. A.]')
    if len(tempa)!=0:
        ax3.plot(tempa,integ[val],"*",color='red',label='Resonant field')
        ax3.legend()
    ax3.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax3.grid(True)
    
    ax4=axes[1,1]
    ax4.plot(field,integto)
    if len(tempa)!=0:
        ax4.plot(tempa,integto[val],"*",color='red',label='Resonant field')
        ax4.legend()
    ax4.set_title('EPR Intensity')
    ax4.set_xlabel('Magnetic Field [mT]')
    ax4.set_ylabel('Counts [U. A.]')
    ax4.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    ax4.grid(True)
    
    fig.tight_layout()
    
    if len(tempa)==0:
        return (dfr,field,spc-basel,integ)
    else:
        return (dfr,field,spc-basel,Hpp,tempa,integ)

class TkinterApp2:
    def __init__(self,root,field,counts):
        self.root=root
        self.root.title("Spectrum analysis")
        style=ttk.Style()
        style.configure('.',font=('Helvetica',11))
        style.configure('TNotebook.Tab',font=('Helvetica',10),padding=[2,2])
        self.field=field
        self.counts=counts
        self.Sdat=None
        maxval=len(self.counts)-1
        self.ffield=field 
        self.fcounts=counts
        self.unit="mT"
        #Baseline change caused by data change
        self.lastdstart=0
        self.lastdend=maxval
        # Values on display
        self.dsvar=tk.IntVar(value=0)
        self.denvar=tk.IntVar(value=maxval)
        self.svar=tk.IntVar(value=0)
        self.envar=tk.IntVar(value=maxval)
        self.ltvar=tk.IntVar(value=51)
        self.polvar=tk.IntVar(value=3)
        self.epvar=tk.StringVar(value="5")
        self.rowsvar=tk.StringVar(value="0")
        self.colsvar=tk.StringVar(value="2, 3") 
        #For integral line
        self.sinvar=tk.IntVar(value=0) 
        self.finvar=tk.IntVar(value=maxval) 
        # Measure and zoom
        self.meamode=False
        self.meapoints=[]
        self.distancevar=tk.StringVar(value="Dist. X: --")
        self.ldistancet="Dist. X: --"
        self.tempmealines=[]
        self.expandedax= None
        self.originalpositions={}
        # Frames
        mainfra=ttk.Frame(self.root,padding="10")
        leftfra=ttk.Frame(mainfra) 
        rightfra=ttk.Frame(mainfra)
        controlfra=ttk.LabelFrame(leftfra,text="Controls",padding="10")
        plotfra=ttk.Frame(mainfra)
        tablefra=ttk.LabelFrame(leftfra,text="Resonant fields",padding="10")
        # Tabs for the controls
        self.notebook=ttk.Notebook(controlfra)
        self.notebook.pack(fill='x',pady=(0,10))
        self.tabadata=ttk.Frame(self.notebook)
        self.tabaseline=ttk.Frame(self.notebook)
        self.notebook.add(self.tabadata,text='Data')
        self.notebook.add(self.tabaseline,text='Baseline')
        #Frames for controls
        daframe=ttk.LabelFrame(self.tabadata,text="Data range",padding=10)
        daframe.pack(fill='x',expand=True,padx=2,pady=2)
        dsframe=ttk.Frame(daframe)
        dsframe.pack(fill='x',pady=(0,2))
        ttk.Label(dsframe,text="Start").pack(side=tk.LEFT)
        ttk.Label(dsframe,textvariable=self.dsvar,width=5).pack(side=tk.RIGHT)
        #Sliders for data 
        self.dsslider=ttk.Scale(dsframe,from_=0,to=maxval-1,orient=tk.HORIZONTAL,variable=self.dsvar,command=self.updall)
        self.dsslider.pack(fill='x')
        deframe=ttk.Frame(daframe)
        deframe.pack(fill='x',pady=(2,0))
        ttk.Label(deframe,text="End").pack(side=tk.LEFT)
        ttk.Label(deframe,textvariable=self.denvar,width=5).pack(side=tk.RIGHT)
        self.deslider=ttk.Scale(deframe,from_=1,to=maxval,orient=tk.HORIZONTAL,variable=self.denvar,command=self.updall)
        self.deslider.pack(fill='x')
        #Sliders for baseline
        grpspec=ttk.LabelFrame(self.tabaseline,text="Spectrum Baseline",padding=5)
        grpspec.pack(fill='x',padx=5,pady=(5,5))
        sasframe=ttk.Frame(grpspec)
        sasframe.pack(fill='x')
        ttk.Label(sasframe,text="Start").pack(side=tk.LEFT)
        ttk.Label(sasframe,textvariable=self.svar,width=5).pack(side=tk.RIGHT)
        self.sslider=ttk.Scale(sasframe,from_=0,to=self.ltvar.get()-1,orient=tk.HORIZONTAL,variable=self.svar,command=self.updall)
        self.sslider.pack(fill='x')
        easframe=ttk.Frame(grpspec)
        easframe.pack(fill='x')
        ttk.Label(easframe,text="End").pack(side=tk.LEFT)
        ttk.Label(easframe,textvariable=self.envar,width=5).pack(side=tk.RIGHT)
        self.eslider=ttk.Scale(easframe,from_=self.ltvar.get(),to=maxval,orient=tk.HORIZONTAL,variable=self.envar,command=self.updall)
        self.eslider.pack(fill='x')

        # Integral baseline
        grpint=ttk.LabelFrame(self.tabaseline,text="Integral Baseline",padding=5)
        grpint.pack(fill='x',padx=5,pady=(5,5))
        siframe=ttk.Frame(grpint)
        siframe.pack(fill='x')
        ttk.Label(siframe,text="Start").pack(side=tk.LEFT)
        ttk.Label(siframe, textvariable=self.sinvar,width=5).pack(side=tk.RIGHT)
        self.sinslider=ttk.Scale(siframe,from_=0,to=self.ltvar.get()-1,orient=tk.HORIZONTAL,variable=self.sinvar,command=self.updall)
        self.sinslider.pack(fill='x')
        fisframe=ttk.Frame(grpint)
        fisframe.pack(fill='x')
        ttk.Label(fisframe,text="End").pack(side=tk.LEFT)
        ttk.Label(fisframe,textvariable=self.finvar,width=5).pack(side=tk.RIGHT)
        self.finslider=ttk.Scale(fisframe,from_=self.ltvar.get(),to=maxval,orient=tk.HORIZONTAL,variable=self.finvar,command=self.updall)
        self.finslider.pack(fill='x')

        # Sliders and frames
        self.ltframe=ttk.Frame(controlfra)
        self.ltframe.pack(fill='x',pady=(0,0))
        ttk.Label(self.ltframe,text="Window length").pack(side=tk.LEFT)
        ttk.Label(self.ltframe,textvariable=self.ltvar,width=5).pack(side=tk.RIGHT)
        self.ltslider=ttk.Scale(self.ltframe,from_=5,to=101,orient=tk.HORIZONTAL,variable=self.ltvar,command=self.updall)
        self.ltslider.pack(fill='x')
        self.polframe=ttk.Frame(controlfra)
        self.polframe.pack(fill='x',pady=(2,5))
        ttk.Label(self.polframe,text="Polyorder").pack(side=tk.LEFT)
        ttk.Label(self.polframe,textvariable=self.polvar,width=5).pack(side=tk.RIGHT)
        self.polslider=ttk.Scale(self.polframe,from_=1,to=10,orient=tk.HORIZONTAL,variable=self.polvar,command=self.updall)
        self.polslider.pack(fill='x')
        promframe=ttk.Frame(controlfra)
        promframe.pack(fill='x',pady=2)
        ttk.Label(promframe,text='Prominence (E-6):').pack(side=tk.LEFT)
        self.epentry=ttk.Entry(promframe,textvariable=self.epvar,width=15,font=('Helvetica',11))
        self.epentry.pack(side=tk.RIGHT)
        self.epvar.trace_add("write",self.updall)
        self.rowse=ttk.Entry(controlfra,textvariable=self.rowsvar,width=10,font=('Helvetica',11))
        self.colse=ttk.Entry(controlfra,textvariable=self.colsvar,width=10,font=('Helvetica',11))
        
        #Buttons
        self.loadbutton=ttk.Button(controlfra,text="Load data",command=self.load)
        self.meabutton=ttk.Button(controlfra,text="Measure",command=self.togglemeamode)
        self.savebutton=ttk.Button(controlfra,text="Save and close",command=self.saveaquit)
        #Tables and plots
        self.table=ttk.Treeview(tablefra,columns=('hpp','field'),show='headings',height=5)
        self.distancelabel=ttk.Label(controlfra,textvariable=self.distancevar,relief=tk.SUNKEN,anchor=tk.CENTER,font=('Helvetica',12,),padding=5)
        
        self.fig=Figure(figsize=(10,8))
        self.axes=self.fig.subplots(2,2)
        self.canvas=FigureCanvasTkAgg(self.fig,master=plotfra)
        self.toolbar=NavigationToolbar2Tk(self.canvas,plotfra,pack_toolbar=False)
        self.canvas.mpl_connect('button_press_event',self.onclick) 
        self.table.heading('hpp',text='Hpp [mT]')
        self.table.heading('field',text='R fields [mT]')
        self.table.column('hpp',width=150)
        self.table.column('field',width=150)
        
        #Display configuration
        mainfra.pack(fill=tk.BOTH,expand=True)
        leftfra.pack(side=tk.LEFT,fill=tk.Y,padx=5,pady=5)
        rightfra.pack(side=tk.LEFT,fill=tk.Y,padx=2,pady=2)
        controlfra.pack(fill=tk.X,anchor='n')
        plotfra.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True,padx=5,pady=5)
        
        loadframe=ttk.Frame(controlfra)
        loadframe.pack(fill=tk.X,pady=5)
        rowsframe=ttk.Frame(loadframe)
        rowsframe.pack(side=tk.LEFT,fill=tk.X,expand=True,padx=(0,2))
        ttk.Label(rowsframe, text="Rows (0 = all):").pack(side=tk.LEFT)
        self.rowse=ttk.Entry(rowsframe,textvariable=self.rowsvar,width=3,font=('Helvetica',11))
        self.rowse.pack(side=tk.RIGHT,fill=tk.X,expand=True)
        colsframe=ttk.Frame(loadframe)
        colsframe.pack(side=tk.LEFT,fill=tk.X,expand=True,padx=(2,0))
        ttk.Label(colsframe,text="[B, spc]: ").pack(side=tk.LEFT)
        self.colse=ttk.Entry(colsframe,textvariable=self.colsvar,width=6,font=('Helvetica',11))
        self.colse.pack(side=tk.RIGHT,fill=tk.X,expand=True)
        self.loadbutton.pack(fill=tk.X,pady=(5,0))
        self.meabutton.pack(fill=tk.X,pady=(2,0))
        self.savebutton.pack(fill=tk.X)
        self.distancelabel.pack(fill=tk.X,pady=(2,0))
        tablefra.pack(fill=tk.BOTH,expand=True,anchor='n',pady=5)
        self.table.pack(fill=tk.BOTH,expand=True)
        
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP,fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.updall()
        
    def load(self):
        fileh=filedialog.askopenfilename(title="Select data file",filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("Dat files", "*.dat"),("All files", "*.*")])
        if not fileh: 
            return
        try:
            rowsval=int(self.rowsvar.get())
            colsval=self.colsvar.get()
            colsval=tuple(int(c.strip()) for c in colsval.split(','))
            fields,countss=Sload1(fileh,rows=rowsval,cols=colsval)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load file: {e}\n\n Check the parameters.")
            return
            
        dialog=tk.Toplevel(self.root)
        dialog.title("Field units")
        dialog.grab_set()
        self.tfields=fields
        self.tunit="mT"
        def conversion(factor,unit):
            self.tfields=fields*factor
            self.tunit=unit
            dialog.destroy()
        ttk.Label(dialog,text="Select field units change:", font=('Helvetica',11,'bold')).pack(padx=20,pady=10)
        ttk.Button(dialog,text="Keep data in mT",command=lambda:conversion(1,"mT")).pack(fill='x',padx=20,pady=5)
        ttk.Button(dialog,text="Data in Gauss -> Change to mT (/10)",command=lambda:conversion(0.1,"mT")).pack(fill='x',padx=20,pady=5)
        
        self.root.wait_window(dialog) 
        fields=self.tfields
        self.unit=self.tunit
        self.table.heading('hpp', text=f'Hpp [{self.unit}]')
        self.table.heading('field', text=f'R fields [{self.unit}]')
            
        self.ffield=fields
        self.fcounts=countss
        nmaxval=len(self.fcounts)-1
        self.dsslider.config(to=nmaxval-1)
        self.deslider.config(to=nmaxval)
        self.lastdstart=-1
        self.lastdend=-1
        self.dsvar.set(0)
        self.denvar.set(nmaxval)
        self.clearmeafeedback() 
        self.updall()

    def togglemeamode(self):
        cancele=self.meamode and len(self.meapoints)==0
        self.meamode=not self.meamode
        if self.meamode:
            self.meabutton.config(text="Cancel measure")
            self.distancevar.set("Click first point")
            self.clearmeafeedback() 
            self.meapoints=[]
        else:
            self.meabutton.config(text="Measure")
            if len(self.meapoints)==1 or cancele: 
                self.clearmeafeedback() 
                self.meapoints=[]
                self.distancevar.set(self.ldistancet)
            
    def onclick(self,event):
        if not event.inaxes or self.toolbar.mode: 
            return
        ax=event.inaxes
        if event.dblclick and event.button==1:
            if self.meamode:
                self.togglemeamode()
            elif self.expandedax is None:
                self.expandplot(ax) 
            elif self.expandedax==ax:
                self.restoreplots()
            else: 
                self.restoreplots(drawc=False)
                self.expandplot(ax)
            return

        # measure
        if self.meamode and event.button==1:
            x,y=event.xdata,event.ydata
            line=ax.axvline(x,color='black',linestyle=':',linewidth=3)
            self.tempmealines.append(line)
            self.canvas.draw_idle()
            if len(self.meapoints)==0:
                self.meapoints.append((x,y))
                self.distancevar.set("Click second point...")
            else:
                self.meapoints.append((x,y))
                x1,y1=self.meapoints[0]
                x2,y2=self.meapoints[1]
                dist_x=abs(x2-x1)
                vardistancia=f"Distance: {dist_x:.4f} [mT]"
                self.distancevar.set(vardistancia) 
                self.ldistancet=vardistancia
                print(f"Measured distance X: {dist_x:.4f} [mT]")
                self.meapoints=[]
                self.togglemeamode() 
    #Cleaning
    def clearmeafeedback(self):
        for line in self.tempmealines:
            try:
                line.remove()
            except ValueError:
                pass 
        self.tempmealines=[]
        self.canvas.draw_idle()

    def updatable(self,df):
        for item in self.table.get_children(): 
            self.table.delete(item)
        if df is not None and not df.empty:
            for index, row in df.iterrows():
                hppval=f"{row['Hpp distance [mT]']:.4f}"
                fielval=f"{row['Resonant fields [mT]']:.4f}"
                self.table.insert('', tk.END, values=(hppval, fielval))

    def updall(self,*args):
        currstart=round(self.dsvar.get())
        currend=round(self.denvar.get())
        self.dsvar.set(currstart)
        self.denvar.set(currend)
        datachanged=(currstart!=self.lastdstart) or (currend!=self.lastdend)
        self.field=self.ffield[currstart:currend]
        self.counts=self.fcounts[currstart:currend]
        curlen=len(self.counts)
        maxidx=curlen-1 if curlen>0 else 0
        self.sslider.config(to=maxidx)
        self.eslider.config(to=maxidx)
        self.sinslider.config(to=maxidx)
        self.finslider.config(to=maxidx)
        ltval=self.ltvar.get()
        if ltval%2==0:
            ltval+=1
        if ltval>curlen and curlen>3:
            ltval=curlen if curlen%2!=0 else curlen-1
            self.root.after(10,lambda: self.ltvar.set(ltval))
        if datachanged:
            sval=0
            enval=maxidx
            sinval=0
            finval=maxidx
            self.svar.set(sval)
            self.envar.set(enval)
            self.sinvar.set(sinval)
            self.finvar.set(finval)
            self.lastdstart=currstart
            self.lastdend=currend
        else:
            sval=round(self.svar.get())
            enval=round(self.envar.get())
            sinval=round(self.sinvar.get())
            finval=round(self.finvar.get())
            sval=max(0,min(sval,maxidx))
            enval=max(0,min(enval,maxidx))
            sinval=max(0,min(sinval,maxidx))
            finval=max(0,min(finval,maxidx))
            if sval>=enval:
                sval=max(0,enval-1)
            if sinval>=finval:
                sinval=max(0,finval-1)
            self.svar.set(sval)
            self.envar.set(enval)
            self.sinvar.set(sinval)
            self.finvar.set(finval)

        self.ltvar.set(round(self.ltvar.get()))
        self.polvar.set(round(self.polvar.get()))
        
        polval=self.polvar.get()
        if polval>=ltval:
            polval=ltval-1
            self.root.after(10, lambda: self.polvar.set(polval))
        
        try: 
            epval=float(self.epvar.get())*1e-6
        except ValueError:epval=5e-6

        results=Spmanipulation(self.fig,self.axes,self.field,self.counts,startl=sval,endli=enval,einmal=sinval,aufmal=finval,lt=ltval,pol=polval,epsilon=epval)        
        if results: 
            self.updatable(results[0])
        self.clearmeafeedback()
        self.canvas.draw()
        self.toolbar.update()

    def saveaquit(self):
        self.updall() 
        sval=self.svar.get()
        enval=self.envar.get()
        sinval=self.sinvar.get()
        finval=self.finvar.get()
        ltval=self.ltvar.get()
        polval=self.polvar.get()
        try: 
            epval=float(self.epvar.get())*1e-6
        except: 
            epval=5e-6
        self.Sdat=Spmanipulation(self.fig,self.axes,self.field,self.counts,startl=sval,endli=enval,einmal=sinval,aufmal=finval,lt=ltval,pol=polval,epsilon=epval)
        self.root.quit()
        self.root.destroy()
        
    def expandplot(self,axexpand):
        if not self.originalpositions:
            for i,ax in enumerate(self.axes.flat):
                 self.originalpositions[ax]=ax.get_subplotspec()
        for ax in self.axes.flat:
            if ax!=axexpand: 
                ax.set_visible(False)
            else: 
                ax.set_subplotspec(self.fig.add_gridspec(1,1)[0,0])
                ax.set_visible(True)
        self.expandedax=axexpand
        self.fig.tight_layout()
        self.canvas.draw()
        self.toolbar.update()

    def restoreplots(self,drawc=True):
        if not self.originalpositions: 
            return
        for ax in self.axes.flat:
            if ax in self.originalpositions:
                ax.set_subplotspec(self.originalpositions[ax])
            ax.set_visible(True)
        self.expandedax=None
        self.originalpositions={}
        if drawc:
            self.fig.tight_layout()
            self.canvas.draw()
            self.toolbar.update()
def Seek(field=None,counts=None):
    if field is None or counts is None:
        findpath=resources.files(__package__).joinpath('STRONG PITCH.dat')
        dfield,dcounts=Sload(findpath,2046,[2,3])
        if field is None:
            field=dfield/10
        if counts is None:
            counts=decounts
    root=tk.Tk()
    app=TkinterApp2(root,field,counts)
    root.mainloop()   
    if app.Sdat is not None:
        return app.Sdat[1:3]
    else:
        print('Exit application without saving')

class BaselineTuner: #For data tuning
    def __init__(self,root,field,counts):
        self.root=root
        self.top=tk.Toplevel(root)
        self.top.title("Tuning Baselines & Parameters")
        self.top.grab_set() 
        self.field=field
        self.counts=counts
        self.ffield=field
        self.fcounts=counts
        self.fdata=None
        maxidx=len(field)-1
        self.expandedax=None
        self.originalpositions={}
        self.ltvar=tk.IntVar(value=51)
        self.polvar=tk.IntVar(value=3)
        self.svar=tk.IntVar(value=0)
        self.envar=tk.IntVar(value=maxidx)
        self.ssvar=tk.IntVar(value=0)
        self.esnvar=tk.IntVar(value=maxidx)
        self.sinvar=tk.IntVar(value=0)
        self.finvar=tk.IntVar(value=maxidx)
        self.lastdstart=0
        self.lastdend=maxidx
        self.epvar=tk.StringVar(value="5")
        
        #Frames
        mainfra=ttk.Frame(self.top,padding="10")
        mainfra.pack(fill=tk.BOTH,expand=True)
        controlfra=ttk.LabelFrame(mainfra,text="Parameters",padding="10")
        controlfra.pack(side=tk.LEFT,fill=tk.Y,padx=5)
        plotfra=ttk.Frame(mainfra)
        plotfra.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True)

        #Sliders
        self.ltframe=ttk.Frame(controlfra)
        self.ltframe.pack(fill='x',pady=(5,2))
        ttk.Label(self.ltframe,text="Window length").pack(side=tk.LEFT)
        ttk.Label(self.ltframe,textvariable=self.ltvar,width=5).pack(side=tk.RIGHT)
        self.ltslider=ttk.Scale(self.ltframe,from_=5,to=101,orient=tk.HORIZONTAL,variable=self.ltvar,command=self.updatep)
        self.ltslider.pack(fill='x')
        self.polframe=ttk.Frame(controlfra)
        self.polframe.pack(fill='x',pady=(2,5))
        ttk.Label(self.polframe,text="Polyorder").pack(side=tk.LEFT)
        ttk.Label(self.polframe,textvariable=self.polvar,width=5).pack(side=tk.RIGHT)
        self.polslider=ttk.Scale(self.polframe,from_=1,to=10,orient=tk.HORIZONTAL,variable=self.polvar,command=self.updatep)
        self.polslider.pack(fill='x')
        
        self.notebook=ttk.Notebook(controlfra)
        self.notebook.pack(fill='x',pady=(0,10))
        self.tabadata=ttk.Frame(self.notebook)
        self.tabaseline=ttk.Frame(self.notebook)
        self.notebook.add(self.tabadata,text='Data')
        self.notebook.add(self.tabaseline,text='Baseline')
        #Frames for controls
        daframe=ttk.LabelFrame(self.tabadata,text="Data range",padding=10)
        daframe.pack(fill='x',expand=True,padx=2,pady=2)
        dsframe=ttk.Frame(daframe)
        dsframe.pack(fill='x',pady=(0,2))
        ttk.Label(dsframe,text="Start").pack(side=tk.LEFT)
        ttk.Label(dsframe,textvariable=self.svar,width=5).pack(side=tk.RIGHT)
        
        #Sliders for data 
        self.dsslider=ttk.Scale(dsframe,from_=0,to=maxidx-1,orient=tk.HORIZONTAL,variable=self.ssvar,command=self.updatep)
        self.dsslider.pack(fill='x')
        deframe=ttk.Frame(daframe)
        deframe.pack(fill='x',pady=(2,0))
        ttk.Label(deframe,text="End").pack(side=tk.LEFT)
        ttk.Label(deframe,textvariable=self.envar,width=5).pack(side=tk.RIGHT)
        self.deslider=ttk.Scale(deframe,from_=1,to=maxidx,orient=tk.HORIZONTAL,variable=self.esnvar,command=self.updatep)
        self.deslider.pack(fill='x')
        #Sliders for baseline
        grpspec=ttk.LabelFrame(self.tabaseline,text="Spectrum Baseline",padding=5)
        grpspec.pack(fill='x',padx=5,pady=(5,5))
        sasframe=ttk.Frame(grpspec)
        sasframe.pack(fill='x')
        ttk.Label(sasframe,text="Start").pack(side=tk.LEFT)
        ttk.Label(sasframe,textvariable=self.svar,width=5).pack(side=tk.RIGHT)
        self.sslider=ttk.Scale(sasframe,from_=0,to=maxidx-1,orient=tk.HORIZONTAL,variable=self.svar,command=self.updatep)
        self.sslider.pack(fill='x')
        easframe=ttk.Frame(grpspec)
        easframe.pack(fill='x')
        ttk.Label(easframe,text="End").pack(side=tk.LEFT)
        ttk.Label(easframe,textvariable=self.envar,width=5).pack(side=tk.RIGHT)
        self.eslider=ttk.Scale(easframe,from_=1,to=maxidx,orient=tk.HORIZONTAL,variable=self.envar,command=self.updatep)
        self.eslider.pack(fill='x')

        # Integral baseline
        grpint=ttk.LabelFrame(self.tabaseline,text="Integral Baseline",padding=5)
        grpint.pack(fill='x',padx=5,pady=(5,5))
        siframe=ttk.Frame(grpint)
        siframe.pack(fill='x')
        ttk.Label(siframe,text="Start").pack(side=tk.LEFT)
        ttk.Label(siframe, textvariable=self.sinvar,width=5).pack(side=tk.RIGHT)
        self.sinslider=ttk.Scale(siframe,from_=0,to=maxidx-1,orient=tk.HORIZONTAL,variable=self.sinvar,command=self.updatep)
        self.sinslider.pack(fill='x')
        fisframe=ttk.Frame(grpint)
        fisframe.pack(fill='x')
        ttk.Label(fisframe,text="End").pack(side=tk.LEFT)
        ttk.Label(fisframe,textvariable=self.finvar,width=5).pack(side=tk.RIGHT)
        self.finslider=ttk.Scale(fisframe,from_=1,to=maxidx,orient=tk.HORIZONTAL,variable=self.finvar,command=self.updatep)
        self.finslider.pack(fill='x')
        promframe=ttk.Frame(controlfra)
        promframe.pack(fill='x',pady=2)
        ttk.Label(promframe,text='Prominence (E-6):').pack(side=tk.LEFT)
        self.epentry=ttk.Entry(promframe,textvariable=self.epvar,width=15,font=('Helvetica',11))
        self.epentry.pack(side=tk.RIGHT)
        self.epvar.trace_add("write",self.updatep)
        #Button
        ttk.Button(controlfra,text="Save and Return",command=self.saveaclo).pack(fill='x',pady=10)
        ttk.Button(controlfra,text="Cancel",command=self.cance).pack(fill='x')
        self.fig=Figure(figsize=(10,6))
        self.ax=self.fig.subplots(1,2)
        self.fig.tight_layout(pad=2.0)
        self.canvas=FigureCanvasTkAgg(self.fig,master=plotfra)
        self.toolbar=NavigationToolbar2Tk(self.canvas,plotfra,pack_toolbar=False)
        self.canvas.mpl_connect('button_press_event',self.onclick) 
        self.toolbar.pack(side=tk.TOP,fill=tk.X)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
        self.updatep()

    def updatep(self,*args):
        currstart=round(self.ssvar.get())
        currend=round(self.esnvar.get())
        if currstart>=currend:
            currstart=max(0,currend-1)
        self.ssvar.set(currstart)
        self.esnvar.set(currend)

        self.field=self.ffield[currstart:currend]
        self.counts=self.fcounts[currstart:currend]
        curlen=len(self.counts)
        maxidx=curlen-1 if curlen>0 else 0
        self.sslider.config(to=maxidx)
        self.eslider.config(to=maxidx)
        self.sinslider.config(to=maxidx)
        self.finslider.config(to=maxidx)
        ltval=self.ltvar.get()
        if ltval%2==0:
            ltval+=1
        if ltval>curlen:
            ltval=curlen if curlen%2!=0 else curlen-1
            if ltval<1:
                ltval=1
            self.root.after(10,lambda: self.ltvar.set(ltval))
        datachanged=(currstart!=self.lastdstart) or (currend!=self.lastdend)
        if datachanged:
            sind=0
            eind=maxidx
            siind=0
            eiind=maxidx
            self.lastdstart=currstart
            self.lastdend=currend
        else:
            sind=max(0,min(round(self.svar.get()),maxidx))
            eind=max(0,min(round(self.envar.get()),maxidx))
            siind=max(0,min(round(self.sinvar.get()),maxidx))
            eiind=max(0,min(round(self.finvar.get()),maxidx))
        if sind>=eind: 
            sind=max(0,eind-1)
        if siind>=eiind: 
            siind=max(0,eiind-1)
        self.svar.set(sind)
        self.envar.set(eind)
        self.sinvar.set(siind)
        self.finvar.set(eiind)
        
        lt=ltval
        polval=self.polvar.get()
        if polval>=lt:
            polval=t-1
            if polval<0:
                polval=0 
            self.root.after(10,lambda: self.polvar.set(polval))
        pol=polval
        if curlen>=lt and lt>pol and curlen>0:
            spc=scs.savgol_filter(self.counts,window_length=lt,polyorder=pol)
        else:
            spc=np.copy(self.counts)
        mx=(spc[eind]-spc[sind])/(self.field[eind]-self.field[sind])
        bx=spc[sind]-mx*self.field[sind]
        basel=bx+(mx*self.field)
        self.spcc=spc-basel

        # Integral Baseline
        integ=scii.cumulative_trapezoid(self.spcc,self.field,initial=0)
        mxi=(integ[eiind]-integ[siind])/(self.field[eiind]-self.field[siind])
        bxi=integ[siind]-mxi*self.field[siind]
        baselini=bxi+(mxi*self.field)
        self.integc=integ-baselini
        peaks,_=scs.find_peaks(self.integc)
        try:
            epval=float(self.epvar.get())
        except ValueError:
            epval=5

        epsilon=epval*10**-6
        prominences=scs.peak_prominences(self.integc,peaks)[0]
        tempa,val=[],[] #tempa= values of the field in the peaks, val= Values of the peaks that follow the epsilon condition
        for ir in range (0,len(prominences)):
            if prominences[ir]>epsilon:
                val.append(peaks[ir])
                tempa.append(self.field[peaks[ir]])
        val=np.asarray(val)
        tempa=np.asarray(tempa)
        #Makes another try to find the field by normalizing the spectrum
        if len(tempa)==0:
            maxinteg=np.max(self.integc)
            if maxinteg!=0:
                peaks,_=scs.find_peaks(self.integc/np.max(self.integc),1)
                prominences=scs.peak_prominences(self.integc/np.max(self.integc),peaks)[0]
            tempa,val=[],[] #tempa= values of the field in the peaks, val= Values of the peaks that follow the epsilon condition
            for ir in range (0,len(prominences)):
                if prominences[ir]>epsilon:
                    val.append(peaks[ir])
                    tempa.append(self.field[peaks[ir]])
        self.val=np.asarray(val)
        self.tempa=np.asarray(tempa)
        #Finding the Hpp
        Hpp,hp,itsmin,itsmax=np.nan,np.nan,np.nan,np.nan
        if len(tempa)==1:
            tearma=spc[np.where(self.field<tempa[0])]
            tearmi=spc[np.where(self.field>tempa[0])]
            if tearma.size>0 and tearmi.size>0:
                itmaxif=np.where(spc==np.nanmax(tearma))[0]
                itminif=np.where(spc==np.nanmin(tearmi))[0]
                if itmaxif.size>0 and itminif.size>0:
                    itmaxidx=np.take(itmaxif,itmaxif.size//2)
                    itminidx=np.take(itminif,itminif.size//2)
                    itmax=self.field[itmaxidx] #max
                    itmin=self.field[itminidx] #min
                    itsmax=itmaxidx 
                    itsmin=itminidx 
                    hp=itmin-itmax
                    Hpp=hp
        elif len(tempa)!=0:
            hp=np.full(len(tempa),np.nan)
            itsmin=np.full(len(tempa),np.nan)
            itsmax=np.full(len(tempa),np.nan)
            for sw in range (0,len(tempa)):
                tearma,tearmi=np.array([]),np.array([])
                if sw==0:
                  tearma=spc[np.where(self.field<tempa[sw])]
                  tearmi=spc[np.where((self.field>tempa[sw])&(self.field<tempa[sw+1]))]
                elif sw==(len(tempa)-1):
                  tearma=spc[np.where((self.field<tempa[sw])&(self.field>tempa[sw-1]))]
                  tearmi=spc[np.where(self.field>tempa[sw])]
                else:
                  tearma=spc[np.where((self.field<tempa[sw])&(self.field>tempa[sw-1]))]
                  tearmi=spc[np.where((self.field>tempa[sw])&(self.field<tempa[sw+1]))]
                if tearma.size>0 and tearmi.size>0:
                        itmaxif=np.where(spc==np.nanmax(tearma))[0]
                        itminif=np.where(spc==np.nanmin(tearmi))[0]
                        if itmaxif.size>0 and itminif.size>0:
                            itsmaxidx=np.take(itmaxif,itmaxif.size//2)
                            itminidx=np.take(itminif,itminif.size//2)
                            itsmax[sw]=itsmaxidx
                            itsmin[sw]=itminidx
                            itmaxval=self.field[itsmaxidx]
                            itminval=self.field[itminidx]
                            hp[sw]=itminval-itmaxval
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",category=RuntimeWarning)
                Hpp=np.nanmean(hp)

        self.integto=scii.cumulative_trapezoid(self.integc,self.field,initial=0)
        self.Hpp=Hpp
        self.hp=hp
        axspc=self.ax[0]
        axint=self.ax[1]

        axspc.clear()
        principal=[self.spcc[sind],self.spcc[eind]]
        principal1=[self.field[sind],self.field[eind]]
        axspc.plot(principal1,principal,"--",color='green')
        axspc.plot(self.field,self.spcc,color='navy')
        axspc.plot(self.field[sind],self.spcc[sind],"o",color='blue')
        axspc.plot(self.field[eind],self.spcc[eind],"o",color='blue')
        if len(tempa)!=0:
            axspc.plot(tempa,(spc-basel)[val],"*",color='red')
        axspc.set_title("Spectrum")
        axspc.set_xlabel('Magnetic Field [mT]')
        axspc.set_ylabel('Counts [a.u.]')
        axspc.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
        axspc.grid(True)
        axint.clear()
        principal=[self.integc[siind],self.integc[eiind]]
        principal1=[self.field[siind],self.field[eiind]]
        axint.plot(principal1,principal,"--",color='green')
        axint.plot(self.field,self.integc,color='green')
        axint.plot(self.field[siind],self.integc[siind],"o",color='blue')
        axint.plot(self.field[eiind],self.integc[eiind],"o",color='blue')
        if len(tempa)!=0:
            axint.plot(tempa,(self.integc)[val],"*",color='red')
        axint.set_title("Integral")
        axint.set_xlabel('Magnetic Field [mT]')
        axint.set_ylabel('Counts [a.u.]')
        axint.grid(True)
        self.canvas.draw_idle()
        
    def saveaclo(self):
        self.fdata=(self.field,self.spcc,self.integc,self.tempa,self.val,self.Hpp,self.hp,self.integto)
        self.top.destroy()
    def cance(self):
        self.top.destroy()
    def onclick(self,event):
        if not event.inaxes or self.toolbar.mode: 
            return
        ax=event.inaxes
        if event.dblclick and event.button == 1:
            if self.expandedax is None:
                self.expandplot(ax) 
            elif self.expandedax==ax:
                self.restoreplots()
            else: 
                self.restoreplots(drawc=False)
                self.expandplot(ax)

    def expandplot(self, axexpand):
        if not self.originalpositions:
            for ax in self.ax.flat:
                 self.originalpositions[ax]=ax.get_subplotspec()
        
        gs=axexpand.get_subplotspec().get_gridspec()
        
        for ax in self.ax.flat:
            if ax!=axexpand: 
                ax.set_visible(False)
            else: 
                ax.set_subplotspec(gs[:,:])
                ax.set_visible(True)
                
        self.expandedax=axexpand
        self.canvas.draw()
        

    def restoreplots(self,drawc=True):
        if not self.originalpositions: 
            return
        for ax in self.ax.flat:
            if ax in self.originalpositions:
                ax.set_subplotspec(self.originalpositions[ax])
            ax.set_visible(True)     
        self.expandedax=None
        if drawc:
            self.canvas.draw()

class TkinterApp3:
    def __init__(self,root):
        self.root=root
        self.root.title("Spectrum Analysis")
        style=ttk.Style()
        style.configure('.',font=('Helvetica',11))
        
        self.datasets={} 
        self.Sdat={}
        self.rowsvar=tk.StringVar(value="0")
        self.colsvar=tk.StringVar(value="2, 3")
        self.temvar=tk.StringVar(value="293")
        self.expandedax=None
        self.originalpositions={}
        #For units
        self.uchosen=False
        self.fieltor=1.0
        self.unit="mT"
        #Frames
        mainfra=ttk.Frame(self.root,padding="10")
        mainfra.pack(fill=tk.BOTH,expand=True)
        leftfra=ttk.Frame(mainfra) 
        plotfra=ttk.Frame(mainfra)
        leftfra.pack(side=tk.LEFT,fill=tk.Y,padx=5,pady=5)
        plotfra.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True,padx=5,pady=5)
        controlfra=ttk.LabelFrame(leftfra,text="Data Loader",padding="10")
        controlfra.pack(fill=tk.X,anchor='n')
        #Entries
        loadframe=ttk.Frame(controlfra)
        loadframe.pack(fill=tk.X,pady=5)
        
        rowsframe=ttk.Frame(loadframe)
        rowsframe.pack(side=tk.LEFT,fill=tk.X,expand=True,padx=2)
        ttk.Label(rowsframe,text="Rows (0 = all):").pack(side=tk.TOP,anchor='w')
        self.rowse=ttk.Entry(rowsframe,textvariable=self.rowsvar,width=5,font=('Helvetica',11))
        self.rowse.pack(side=tk.TOP,fill=tk.X)
        colsframe=ttk.Frame(loadframe)
        colsframe.pack(side=tk.LEFT,fill=tk.X,expand=True,padx=2)
        ttk.Label(colsframe,text="Cols [B, spc]:").pack(side=tk.TOP,anchor='w')
        self.colse=ttk.Entry(colsframe,textvariable=self.colsvar,width=5,font=('Helvetica',11))
        self.colse.pack(side=tk.TOP,fill=tk.X)
        temframe=ttk.Frame(loadframe)
        temframe.pack(side=tk.LEFT,fill=tk.X,expand=True,padx=2)
        ttk.Label(temframe,text="Temperature [K]:").pack(side=tk.TOP,anchor='w')
        self.tem=ttk.Entry(temframe,textvariable=self.temvar,width=5,font=('Helvetica',11))
        self.tem.pack(side=tk.TOP,fill=tk.X)
        #Button
        ttk.Button(controlfra,text="Load Data",command=self.loade).pack(fill='x',pady=10)
        delframe = ttk.Frame(controlfra)
        delframe.pack(fill='x',pady=5)
        
        self.erasee=ttk.Combobox(delframe,state="readonly",width=10,font=('Helvetica',11))
        self.erasee.pack(side=tk.LEFT,fill='x',expand=True,padx=(0,2))
        
        ttk.Button(delframe,text="Delete",command=self.delet).pack(side=tk.RIGHT)
        #Figures 
        self.ascreen=1
        self.contf=ttk.Frame(plotfra)
        self.contf.pack(fill=tk.BOTH, expand=True)
        self.frame1=ttk.Frame(self.contf)
        self.fig1=Figure(figsize=(10,8))
        self.fig1.subplots_adjust(left=0.08,right=0.97,top=0.94,bottom=0.10,wspace=0.25,hspace=0.35)
        self.ax1=self.fig1.subplots(2,2) 
        self.canvas1=FigureCanvasTkAgg(self.fig1,master=self.frame1)
        self.toolbar1=NavigationToolbar2Tk(self.canvas1,self.frame1,pack_toolbar=False)
        self.canvas1.mpl_connect('button_press_event',self.onclick) 
        self.toolbar1.pack(side=tk.TOP,fill=tk.X)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH,expand=True)

        self.frame2=ttk.Frame(self.contf)
        self.fig2=Figure(figsize=(10,8))
        self.fig2.subplots_adjust(left=0.08,right=0.97,top=0.94,bottom=0.10,wspace=0.25,hspace=0.35)
        self.ax2=self.fig2.subplots(2,2) 
        self.canvas2=FigureCanvasTkAgg(self.fig2,master=self.frame2)
        self.toolbar2=NavigationToolbar2Tk(self.canvas2,self.frame2,pack_toolbar=False)
        self.canvas2.mpl_connect('button_press_event',self.onclick) 
        self.toolbar2.pack(side=tk.TOP,fill=tk.X)
        self.frame1.pack(fill=tk.BOTH, expand=True)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        ttk.Button(controlfra,text="Change graphics",command=self.cview).pack(fill='x',pady=5)
        #For temperature analisys
        ttk.Button(controlfra,text=r"Linear regression (1/I)",command=self.regression).pack(fill='x',pady=5)
        self.curie=ttk.Label(controlfra,text="Curie-Weiss Temperature (\u03B8): --- K",font=('Helvetica',11,'bold'),foreground="blue")
        self.curie.pack(fill='x',pady=5)
        self.ferrotype=ttk.Label(controlfra,text="-----",font=('Helvetica',11,'bold'),foreground="blue")
        self.ferrotype.pack(fill='x',pady=5)
        self.adparam=ttk.Label(controlfra,text="Adjustment parameters:",font=('Helvetica',11,'bold'),foreground="blue")
        self.adparam.pack(fill='x',pady=5)
        
        self.daframe=ttk.Frame(controlfra)
        self.daframe.pack(fill='x',pady=5)
        self.seda=ttk.Treeview(self.daframe,columns=('Select','Temp','1/I'),show='headings',height=12)
        self.seda.heading('Select',text='Select data')
        self.seda.heading('Temp',text='T [K]')
        self.seda.heading('1/I',text='1/Max')
        self.seda.column('Select',width=40,anchor='center')
        self.seda.column('Temp',width=60,anchor='center')
        self.seda.column('1/I',width=80,anchor='center')
        self.seda.bind('<ButtonRelease-1>',self.tocheck)
        scrollbar=ttk.Scrollbar(self.daframe,orient=tk.VERTICAL,command=self.seda.yview)
        self.seda.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT,fill=tk.Y)
        self.seda.pack(side=tk.LEFT,fill=tk.X,expand=True)
        
    def loade(self):
        asval=self.temvar.get().strip()
        if not asval:
            messagebox.showwarning("Warning", "Please enter a temperature value before loading.")
            return
        latext=f"{asval} K"
        
        if asval in self.datasets:
            anw=messagebox.askyesno("Value exists", f"The temperature {asval} K was already used.\nOverwrite it?")
            if not anw:
                return 
        fileh=filedialog.askopenfilename(title="Select data file",filetypes=[("Text files", "*.txt"),
                                                                               ("CSV files", "*.csv"),
                                                                               ("Dat files", "*.dat"),
                                                                               ("All files", "*.*")])           
        if not fileh: 
            return
        try:
            rowsval=int(self.rowsvar.get())
            colsval=tuple(int(c.strip()) for c in self.colsvar.get().split(','))
            fields,countss=Sload1(fileh,rows=rowsval,cols=colsval)
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load file: {e}")
            return
        #Units
        if not self.uchosen:
            dialog=tk.Toplevel(self.root)
            dialog.title("Field units")
            dialog.grab_set()
            self.tfactor=1.0
            self.tunit="mT"
            def conversion(factor,unit):
                self.tfactor=factor
                self.tunit=unit
                dialog.destroy()
            ttk.Label(dialog,text="Select field units change:",font=('Helvetica',11,'bold')).pack(padx=20,pady=10)
            ttk.Button(dialog,text="Keep data in mT",command=lambda: conversion(1.0,"mT")).pack(fill='x',padx=20,pady=5)
            ttk.Button(dialog,text="Data in Gauss -> Change to mT (/10)",command=lambda:conversion(0.1,"mT")).pack(fill='x',padx=20,pady=5)
            self.root.wait_window(dialog)
            self.fieltor=self.tfactor
            self.unit=self.tunit
            self.uchosen=True
        fields=fields*self.fieltor
        #Data configuration
        tuner=BaselineTuner(self.root,fields,countss)
        self.root.wait_window(tuner.top)
    
        if tuner.fdata is None:
            return
        self.datasets[asval]=(fields,countss)
        self.Sdat[asval]=(fields,countss)
        fieldss,spcc,integc,tempa,val,phpp,hpp,dintegc=tuner.fdata
        self.datasets[asval]=(fieldss,spcc,integc,dintegc)
        reten=np.full(len(tempa),float(asval))
        axspc=self.ax1[0,0]
        axint=self.ax1[0,1]
        axres=self.ax1[1,0]
        axhpp=self.ax1[1,1]
        aximax=self.ax2[0,0] 
        axi2s=self.ax2[0,1] 
        axiint=self.ax2[1,0] 
        precedent=False
        vax=np.max(dintegc)
        temm=float(asval)
        for line in axspc.get_lines():
            if line.get_label()==latext:
                precedent=True
                break
        if precedent:
            axdat=[(axspc,fieldss,spcc),(axint,fieldss,integc),(axres,np.atleast_1d(reten),np.atleast_1d(tempa)),
                   (aximax,np.atleast_1d(temm),np.atleast_1d(vax)),(axi2s,fieldss,dintegc)]
            for ax,xt,yt in axdat:
                for line in ax.get_lines():
                    if line.get_label()==latext:
                        line.set_data(xt,yt)
                        ax.relim()            
                        ax.autoscale_view() 
                        break 
            if vax!=0:
                for line in axiint.get_lines():
                    if line.get_label()==latext:
                        line.set_data(np.atleast_1d(temm),np.atleast_1d(1/vax))
                        axiint.relim()
                        axiint.autoscale_view()
                        break
            linepr=None
            phlin=None
            for line in axhpp.get_lines():
                if line.get_label()==latext:
                    linepr=line
                elif line.get_label()==f"_{latext}_phpp":
                    phlin=line
            if linepr:
                linepr.set_data(np.atleast_1d(reten),np.atleast_1d(hpp))
                if not isinstance(hpp,float):
                    if phlin:
                        phlin.set_data(np.atleast_1d(reten[0]),np.atleast_1d(phpp))
                    else:
                        axhpp.plot(reten[0],phpp,'s',label=f"_{latext}_phpp")
                else:
                    if phlin:
                        phlin.remove()

            axhpp.relim()
            axhpp.autoscale_view()
            self.canvas1.draw()
            self.canvas2.draw()
        if not precedent:
            axspc.plot(fieldss,spcc,label=latext)
            axint.plot(fieldss,integc,label=latext)
            axspc.set_title('Spectrum')
            axspc.set_xlabel('Magnetic Field [mT]')
            axspc.set_ylabel('Counts [a.u.]')
            axspc.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
            axspc.grid(True)
            axspc.legend()
            axspc.relim()
            axspc.autoscale_view()
            axint.set_title('Absorption curve')
            axint.set_xlabel('Magnetic Field [mT]')
            axint.set_ylabel('Integral [a.u.]')
            axint.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
            axint.grid(True)
            axint.legend()
            axint.relim()
            axint.autoscale_view()
            axres.set_title('Resonant Fields')
            axres.set_ylabel('Magnetic Field [mT]')
            axres.set_xlabel('Temperature [K]')
            axres.plot(reten,tempa,'o',label=latext)
            axres.grid(True)
            axres.legend()
            axres.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axres.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axres.relim()
            axres.autoscale_view()
            
            axhpp.set_title('Hpp distance')
            axhpp.set_ylabel('Magnetic Field [mT]')
            axhpp.set_xlabel('Temperature [K]')
            axhpp.plot(reten,hpp,'o',label=latext)
            if not isinstance(hpp,float):
                axhpp.plot(reten[0],phpp,'s',label=f"_{latext}_phpp")
            axhpp.grid(True)
            axhpp.legend()
            axhpp.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axhpp.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            axhpp.relim()
            axhpp.autoscale_view()

            aximax.plot(temm,vax,'o',label=latext)
            aximax.set_title('Intensity Maximums')
            aximax.set_xlabel('Temperature [K]')
            aximax.set_ylabel('Max Counts [A. U.]')
            aximax.grid(True)
            aximax.relim()
            aximax.legend()
            aximax.autoscale_view()
    
            axi2s.plot(fieldss,dintegc,label=latext)
            axi2s.set_title('EPR Intensity')
            axi2s.set_xlabel('Magnetic Field [mT]')
            axi2s.set_ylabel('Counts [U. A.]')
            axi2s.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
            axi2s.grid(True)
            axi2s.relim()
            axi2s.legend()
            axi2s.autoscale_view()
            self.canvas1.draw()
            self.canvas2.draw()
            self.erasee['values']=list(self.datasets.keys())
            self.erasee.set(asval)
            self.actable()
            if vax!=0:
                axiint.plot(temm,1/vax,'o',label=latext)
                axiint.set_title('Inverse of Intensity')
                axiint.set_xlabel('Temperature [K]')
                axiint.set_ylabel('Max Counts [A. U.]')
                axiint.grid(True)
                axiint.relim()
                axiint.autoscale_view()
                axiint.legend()
                self.canvas1.draw()
                self.canvas2.draw()
                self.erasee['values']=list(self.datasets.keys())
                self.erasee.set(asval)
        
    def onclick(self,event):
        toolbar=self.toolbar1 if self.ascreen==1 else self.toolbar2
        if not event.inaxes or toolbar.mode: 
            return
        ax=event.inaxes
        if event.dblclick and event.button == 1:
            if self.expandedax is None:
                self.expandplot(ax) 
            elif self.expandedax==ax:
                self.restoreplots()
            else: 
                self.restoreplots(drawc=False)
                self.expandplot(ax)
                
    def tocheck(self, event):
        region=self.seda.identify_region(event.x,event.y)
        if region=='cell':
            place=self.seda.identify_column(event.x)
            if place=='#1':
                row=self.seda.identify_row(event.y)
                actual=self.seda.set(row,'Select')
                nstate="[   ]"if actual=="[ X ]" else "[ X ]"
                self.seda.set(row,'Select',nstate)
                
    def expandplot(self,axexpand):
        axes=self.ax1 if self.ascreen==1 else self.ax2
        canvas=self.canvas1 if self.ascreen==1 else self.canvas2
        if not self.originalpositions:
            for ax in axes.flat:
                 self.originalpositions[ax]=ax.get_subplotspec()
        gs=axexpand.get_subplotspec().get_gridspec()
        for ax in axes.flat:
            if ax!=axexpand: 
                ax.set_visible(False)
            else: 
                ax.set_subplotspec(gs[:,:])
                ax.set_visible(True)
        self.expandedax=axexpand
        canvas.draw()

    def restoreplots(self,drawc=True):
        if not self.originalpositions: 
            return
        axes=self.ax1 if self.ascreen==1 else self.ax2
        canvas=self.canvas1 if self.ascreen==1 else self.canvas2
        for ax in axes.flat:
            if ax in self.originalpositions:
                ax.set_subplotspec(self.originalpositions[ax])
            ax.set_visible(True)     
        self.expandedax=None
        self.originalpositions.clear()
        
        if drawc:
            canvas.draw()

    def delet(self):
            tar=self.erasee.get()
            if not tar:
                messagebox.showwarning("Warning!","No dataset selected")
                return
            confim=messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete the data for {tar} K?")
            if not confim:
                return
            if tar in self.datasets:
                del self.datasets[tar]
            if tar in self.Sdat:
                del self.Sdat[tar]
            tarl=f"{tar} K"
            tarlphpp=f"_{tarl}_phpp"
            lastk=list(self.datasets.keys())
            self.erasee['values']=lastk
            self.erasee.set(lastk[0] if lastk else '')
            tarl=f"{tar} K"
            screen=[(self.ax1,self.canvas1),(self.ax2,self.canvas2)]
            
            for axes,canvas in screen:
                for ax in axes.flat:
                    for line in ax.lines[:]: 
                        if line.get_label() in [tarl,tarlphpp]:
                            line.remove()
                    for col in ax.collections[:]:
                        if col.get_label() in [tarl,tarlphpp]:
                            col.remove()
                    ax.relim()
                    ax.autoscale_view()
                    hand,lab=ax.get_legend_handles_labels()
                    if hand: 
                        ax.legend()
                    else:
                        ll=ax.get_legend()
                        if ll is not None:
                            ll.remove()
                canvas.draw()
                self.actable()

    def cview(self):
        if self.expandedax is not None:
            self.restoreplots()
        self.originalpositions.clear()
        if self.ascreen==1:
            self.frame1.pack_forget()
            self.frame2.pack(fill=tk.BOTH,expand=True)
            self.ascreen=2
        else:
            self.frame2.pack_forget()
            self.frame1.pack(fill=tk.BOTH,expand=True)
            self.ascreen=1
            
    def actable(self):
        for item in self.seda.get_children():
            self.seda.delete(item)
        for tname in sorted(self.datasets.keys(),key=float):
            dates=self.datasets[tname]
            vax=np.max(dates[3]) 
            if vax!=0:
                invax=1.0/vax
                self.seda.insert('','end',iid=tname,values=("[ X ]",f"{float(tname):.1f}",f"{invax:.2e}"))
    
    def regression(self):
        xva=[]
        yva=[]
        for tname in self.seda.get_children():
            if self.seda.set(tname,'Select')=="[ X ]":
                tval=float(tname)
                dates=self.datasets[tname]
                vax=np.max(dates[3])
                if vax!=0:
                    xva.append(tval)
                    yva.append(1.0/vax)
        if len(xva)<2:
            messagebox.showwarning("Warning","At least two points are required for the linear regression.")
            return
        xva=np.array(xva)
        yva=np.array(yva)
        res=linregress(xva,yva)
        m=res.slope
        b=res.intercept
        em=res.stderr
        eb=res.intercept_stderr
        coef=res.rvalue**2
        if m!=0:
            tcurie=-b/m
            self.curie.config(text=f"Curie-Weiss temperature (\u03B8): {tcurie:.2f} K")
            if tcurie<0:
                self.ferrotype.config(text=f"Antiferromagnetic behavior.")
            else:
                self.ferrotype.config(text=f"Ferromagnetic behavior.")
            self.adparam.config(text=f'Adjustment parameters:\n m = ({m:.4f} \u00B1 {em:.4f})\n b = ({b:.4f} \u00B1 {eb:.4f})\n R\u00B2 = {coef:.4f}' )
        else:
            self.curie.config(text="Curie-Weiss temperature (\u03B8): Undefined.")

        xf=np.array([np.min(xva),np.max(xva)])
        yf=m*xf+b
        axitem=self.ax2[1,1]
        for line in list(axitem.get_lines()):
            if line.get_label().startswith('y=') or line.get_label()=='Data':
                line.remove()
        axitem.plot(xva,yva,'o',color='blue',label='Data')
        axitem.set_title('Inverse of Intensity')
        axitem.set_xlabel('Temperature [K]')
        axitem.set_ylabel('1/I [A. U.]')
        axitem.grid(True)
        if b<0:
            kl=np.abs(b)
            elabel=f'y={m:.4f}x - {kl:.4f}' 
        else:
            elabel=f'y={m:.4f}x + {b:.4f}'
        axitem.plot(xf,yf,'--',color='red',label=elabel)
        axitem.relim()
        axitem.autoscale_view()
        axitem.legend()
        axitem.grid(True)
        self.canvas2.draw()
            
def Termal():
    root=tk.Tk()
    app=TkinterApp3(root)
    root.option_add('*TCombobox*Listbox.font',('Helvetica',11))
    root.mainloop()   
    if app.Sdat:
        return app.Sdat
    else:
        print('Exit application without saving')
