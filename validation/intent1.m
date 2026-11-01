clear; clc;

% Conversion factor 
cm2MHz = 29979.25;

% 1.System (Fe^3+, S = 5/2)
Sys.S = 5/2;
Sys.g = 2.1215; 
% D and E in cm^-1
D_cm = 0.020;                        % cm^-1
E_cm = 0.0065;                       % cm^-1
Sys.D = [D_cm, E_cm] * cm2MHz;       % [D, E] en MHz
Sys.lwpp = [0, 18]; 

% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [200, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 295.15;
[B_mT, spec] = pepper(Sys, Exp);

data = [B_mT(:),spec(:)];
writematrix(data,'espectro_Bi25FeO40.txt','Delimiter','tab');



clear; clc;
Opt.GridSize = 40;
% Conversion factor 
cm2MHz = 29979.25;

% 1.System (Fe^3+, S = 5/2)
Sys.S = 5/2;
Sys.g = 2.1215; 
% D and E in cm^-1
D_cm = 0.020;                        % cm^-1
E_cm = 0.0065;                       % cm^-1
Sys.D = [D_cm, E_cm] * cm2MHz;       % [D, E] en MHz
Sys.lwpp = [0, 18]; 

% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [200, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 295.15;
[B_mT, spec] = pepper(Sys, Exp,Opt);

data = [B_mT(:),spec(:)];
writematrix(data,'espectro_Bi25FeO40_40not.txt','Delimiter','tab');

clear; clc;
% 2.System (CaCO3, S = 5/2, I = 5/2)
Sys.S = 5/2;
Sys.I= 5/2;
Sys.g = [2.00123 2.00123 2.00131];
Sys.A = [-264.51 -264.51 -262.62];
Sys.B4 = [0 3.36 0 0 -0.141 0 0 0 0];
Sys.Q = [0.566 -0.214 -0.352];
Sys.D = [-227.84 0];
Sys.lwpp = [1, 8]; 
Sys.Nucs = 'Mn';

% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [250, 400];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);

data = [B_mT(:),spec(:)];
writematrix(data,'espectro_CaCO3.txt','Delimiter','tab');

clear; clc;
Opt.GridSize = 40;
% 2.System (CaCO3 with Mn, S = 5/2, I = 5/2)
Sys.S = 5/2;
Sys.I= 5/2;
Sys.g = [2.00123 2.00123 2.00131];
Sys.A = [-264.51 -264.51 -262.62];
Sys.B4 = [0 3.36 0 0 -0.141 0 0 0 0];
Sys.Q = [0.566 -0.214 -0.352];
Sys.D = [-227.84 0];
Sys.lwpp = [1, 8]; 
Sys.Nucs = 'Mn';
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [250, 400];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);

data = [B_mT(:),spec(:)];
writematrix(data,'espectro_CaCO3_40not.txt','Delimiter','tab');

clear; clc;
%DPPH 1
Sys.S = 1/2;
Sys.g = 2.004; 
Sys.lwpp = [0, 2]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [250, 400];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_DPPH.txt','Delimiter','tab');

clear; clc;
Opt.GridSize = 40;
%DPPH 1
Sys.S = 1/2;
Sys.g = 2.004; 
Sys.lwpp = [0, 2]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [250, 400];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_DPPH_40not.txt','Delimiter','tab');

clear; clc;
%Hidroxia
Sys.S = 1/2;
Sys.g = [2.0026 2.0026 1.996]; 
Sys.lwpp = [0, 0.5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [250, 400];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_Hidroxia.txt','Delimiter','tab');

clear; clc;
%Hidroxia
Opt.GridSize = 40;
Sys.S = 1/2;
Sys.g = [2.0026 2.0026 1.996]; 
Sys.lwpp = [0, 0.5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [250, 400];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_Hidroxia_40not.txt','Delimiter','tab');

clear; clc;
%A isotropic
Sys.S = 1/2;
Sys.I = 1/2;
Sys.g = [2.0026 2.0026 2.0026]; 
Sys.Nucs = '13C';
Sys.A = [200 200 200];
Sys.lwpp = [0, 5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_Aiso.txt','Delimiter','tab');

clear; clc;
%A isotropic
Opt.GridSize = 40;
Sys.S = 1/2;
Sys.I = 1/2;
Sys.g = [2.0026 2.0026 2.0026]; 
Sys.Nucs = '13C';
Sys.A = [200 200 200];
Sys.lwpp = [0, 5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_Aiso_40not.txt','Delimiter','tab');

clear; clc;
%A anisotropic
Sys.S = 1/2;
Sys.I = 1/2;
Sys.g = [2.0026 2.0026 2.0026]; 
Sys.Nucs = '13C';
Sys.A = [200 500 300];
Sys.lwpp = [0, 5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_Aniso.txt','Delimiter','tab');

clear; clc;
%A anisotropic
Opt.GridSize = 40;
Sys.S = 1/2;
Sys.I = 1/2;
Sys.g = [2.0026 2.0026 2.0026]; 
Sys.Nucs = '13C';
Sys.A = [200 500 300];
Sys.lwpp = [0, 5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_Aniso_40not.txt','Delimiter','tab');

clear; clc;
%D and E=0
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026]; 
Sys.D = [50 0];
Sys.lwpp = [0, 5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_DE0.txt','Delimiter','tab');

clear; clc;
%D and E=0
Opt.GridSize = 40;
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026]; 
Sys.D = [50 0];
Sys.lwpp = [0, 5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_DE0_40not.txt','Delimiter','tab');

clear; clc;
%D and E
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026]; 
Sys.D = [50 50*3];
Sys.lwpp = [5,0]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_DE.txt','Delimiter','tab');

clear; clc;
%D and E
Opt.GridSize = 40;
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026]; 
Sys.D = [50 50*3];
Sys.lwpp = [5,0]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_DE_40not.txt','Delimiter','tab');

clear; clc;
%Q and A
Sys.S = 1;
Sys.I = 1;
Sys.g = [2.0026 2.0026 2.0026]; 
Sys.A = [200 200 200];
Sys.Q = [50 20 50];
Sys.Nucs = '14N';
Sys.lwpp = [10,0]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_QA.txt','Delimiter','tab');

clear; clc;
%Q and A
Opt.GridSize = 40;
Sys.S = 1;
Sys.I = 1;
Sys.g = [2.0026 2.0026 2.0026]; 
Sys.A = [200 200 200];
Sys.Q = [50 20 50];
Sys.Nucs = '14N';
Sys.lwpp = [10,0]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_QA_40not.txt','Delimiter','tab');

clear; clc;
%D, g and A
Sys.S = 1;
Sys.I= 1;
Sys.g = [2.0026 1.806 2.001]; 
Sys.D = [50 50*3];
Sys.A = [200 200 200];
Sys.Nucs = '14N';
Sys.lwpp = [0, 5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_Dga.txt','Delimiter','tab');

clear; clc;
%D, g and A
Opt.GridSize = 40;
Sys.S = 1;
Sys.I= 1;
Sys.g = [2.0026 1.806 2.001]; 
Sys.D = [50 50*3];
Sys.A = [200 200 200];
Sys.Nucs = '14N';
Sys.lwpp = [0, 5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.43;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_Dga_40not.txt','Delimiter','tab');

clear; clc;
%S3/2D
Sys.S = 3/2;
Sys.g = [2.01 2.01 2.01]; 
Sys.D = [-1780 -90];
Sys.lwpp = [0.5,0]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.439;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_S32D.txt','Delimiter','tab');

clear; clc;
%S3/2D
Opt.GridSize = 40;
Sys.S = 3/2;
Sys.g = [2.01 2.01 2.01]; 
Sys.D = [-1780 -90];
Sys.lwpp = [0.5,0]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.439;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_S32D_40not.txt','Delimiter','tab');

clear; clc;
%ZnCrO
Sys.S = 3/2;
Sys.g = [1.96 1.96 1.97]; 
Sys.D = [500 1500];
Sys.lwpp = [0, 0.5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.439;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_ZnCro.txt','Delimiter','tab');

clear; clc;
%ZnCrO
Opt.GridSize = 40;
Sys.S = 3/2;
Sys.g = [1.96 1.96 1.97]; 
Sys.D = [500 1500];
Sys.lwpp = [0, 0.5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.439;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_ZnCro_40not.txt','Delimiter','tab');


clear; clc;
%ZnCoO
Sys.S = 3/2;
Sys.g = [4.43 4.43 2.2]; 
Sys.D = [700 0];
Sys.lwpp = [0, 1.5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.439;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_ZnCo.txt','Delimiter','tab');

clear; clc;
%ZnCoO
Opt.GridSize = 40;
Sys.S = 3/2;
Sys.g = [4.43 4.43 2.2]; 
Sys.D = [700 0];
Sys.lwpp = [0, 1.5]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.439;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_ZnCo_40not.txt','Delimiter','tab');

clear; clc;
%ZnCrO2
Sys.S = 3/2;
Sys.g = [2.17 1.25 1.97]; 
Sys.D = [1839.85 959.99];
Sys.lwpp = [0, 15]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.439;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 90;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_ZnCr2.txt','Delimiter','tab');

clear; clc;
%ZnCrO2
Opt.GridSize = 40;
Sys.S = 3/2;
Sys.g = [2.17 1.25 1.97]; 
Sys.D = [1839.85 959.99];
Sys.lwpp = [0, 15]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.439;                 
Exp.Range = [100, 500];             
Exp.nPoints = 2048;                
Exp.Temperature = 90;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_ZnCr2_40not.txt','Delimiter','tab');

clear; clc;
%Y itrium
% Conversion factor 
cm2MHz = 29979.25;
Sys.S = 5/2;
Sys.g = [2.0062 2.0062 2.0062]; 
Sys.D = [-1779.87 0];
acm  = 39e-4;        % cm^-1
Fcm  = 4e-4;         % cm^-1
a = acm*cm2MHz;   % ~ 116.92 MHz
F = Fcm*cm2MHz;   % ~ 11.99 MHz
B40 = (F - a) / 180;
B43 = (a * sqrt(2)) / 60;
Sys.B4 = [0, B43, 0, 0, B40, 0, 0, 0, 0];
Sys.lwpp = [15,0]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.76;          
Exp.Range  = [0, 600];  
Exp.nPoints = 2048;            
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_itrium.txt','Delimiter','tab');

clear; clc;
%Y itrium
% Conversion factor 
cm2MHz = 29979.25;
Opt.GridSize = 40;
Sys.S = 5/2;
Sys.g = [2.0062 2.0062 2.0062]; 
Sys.D = [-1779.87 0];
acm  = 39e-4;        % cm^-1
Fcm  = 4e-4;         % cm^-1
a = acm*cm2MHz;   % ~ 116.92 MHz
F = Fcm*cm2MHz;   % ~ 11.99 MHz
B40 = (F - a) / 180;
B43 = (a * sqrt(2)) / 60;
Sys.B4 = [0, B43, 0, 0, B40, 0, 0, 0, 0];
Sys.lwpp = [15, 0]; 
% Experimental conditions (X band)
Exp.mwFreq = 9.76;          
Exp.Range  = [0, 600];  
Exp.nPoints = 2048;            
Exp.Temperature = 300;
[B_mT, spec] = pepper(Sys, Exp,Opt);
data = [B_mT(:),spec(:)];
writematrix(data,'espectro_itrium_40not.txt','Delimiter','tab');

