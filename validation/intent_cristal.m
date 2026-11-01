clear; clc;
clear Sys Exp;
% cal_g | orientation x
Sys.S = 0.5;
Sys.g = [1.9 2.0 2.1];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.43;
Exp.Range = [300.0 370.0];
Exp.nPoints = 4096;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_cal_g_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_cal_g_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% cal_g | orientation y
Sys.S = 0.5;
Sys.g = [1.9 2.0 2.1];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.43;
Exp.Range = [300.0 370.0];
Exp.nPoints = 4096;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_cal_g_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_cal_g_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% cal_g | orientation z
Sys.S = 0.5;
Sys.g = [1.9 2.0 2.1];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.43;
Exp.Range = [300.0 370.0];
Exp.nPoints = 4096;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_cal_g_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_cal_g_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% cal_g | orientation o1
Sys.S = 0.5;
Sys.g = [1.9 2.0 2.1];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.43;
Exp.Range = [300.0 370.0];
Exp.nPoints = 4096;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 40 25]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_cal_g_o1.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_cal_g_o1.txt', 'Delimiter', 'tab');

clear Sys Exp;
% cal_g | orientation o1chi
Sys.S = 0.5;
Sys.g = [1.9 2.0 2.1];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.43;
Exp.Range = [300.0 370.0];
Exp.nPoints = 4096;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[90 40 25]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_cal_g_o1chi.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_cal_g_o1chi.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Fe3_S52_ZFS | orientation z
Sys.S = 2.5;
Sys.g = [2.1215 2.1215 2.1215];
Sys.D = [599.584916 194.8650977];
Sys.lwpp = [0.0 18.0];
Exp.mwFreq = 9.43;
Exp.Range = [200.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 295.15;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Fe3_S52_ZFS_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Fe3_S52_ZFS_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Fe3_S52_ZFS | orientation x
Sys.S = 2.5;
Sys.g = [2.1215 2.1215 2.1215];
Sys.D = [599.584916 194.8650977];
Sys.lwpp = [0.0 18.0];
Exp.mwFreq = 9.43;
Exp.Range = [200.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 295.15;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Fe3_S52_ZFS_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Fe3_S52_ZFS_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Fe3_S52_ZFS | orientation y
Sys.S = 2.5;
Sys.g = [2.1215 2.1215 2.1215];
Sys.D = [599.584916 194.8650977];
Sys.lwpp = [0.0 18.0];
Exp.mwFreq = 9.43;
Exp.Range = [200.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 295.15;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Fe3_S52_ZFS_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Fe3_S52_ZFS_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Fe3_S52_ZFS | orientation gen
Sys.S = 2.5;
Sys.g = [2.1215 2.1215 2.1215];
Sys.D = [599.584916 194.8650977];
Sys.lwpp = [0.0 18.0];
Exp.mwFreq = 9.43;
Exp.Range = [200.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 295.15;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Fe3_S52_ZFS_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Fe3_S52_ZFS_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DPPH 1 | orientation z
Sys.S = 0.5;
Sys.g = [2.004 2.004 2.004];
Sys.lwpp = [0.0 2.0];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DPPH_1_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DPPH_1_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DPPH 1 | orientation x
Sys.S = 0.5;
Sys.g = [2.004 2.004 2.004];
Sys.lwpp = [0.0 2.0];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DPPH_1_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DPPH_1_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DPPH 1 | orientation y
Sys.S = 0.5;
Sys.g = [2.004 2.004 2.004];
Sys.lwpp = [0.0 2.0];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DPPH_1_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DPPH_1_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DPPH 1 | orientation gen
Sys.S = 0.5;
Sys.g = [2.004 2.004 2.004];
Sys.lwpp = [0.0 2.0];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DPPH_1_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DPPH_1_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Hidroxia | orientation z
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 1.996];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Hidroxia_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Hidroxia_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Hidroxia | orientation x
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 1.996];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Hidroxia_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Hidroxia_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Hidroxia | orientation y
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 1.996];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Hidroxia_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Hidroxia_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Hidroxia | orientation gen
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 1.996];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Hidroxia_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Hidroxia_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Aisotro | orientation z
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '13C';
Sys.A = [200.0 200.0 200.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Aisotro_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Aisotro_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Aisotro | orientation x
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '13C';
Sys.A = [200.0 200.0 200.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Aisotro_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Aisotro_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Aisotro | orientation y
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '13C';
Sys.A = [200.0 200.0 200.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Aisotro_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Aisotro_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Aisotro | orientation gen
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '13C';
Sys.A = [200.0 200.0 200.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Aisotro_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Aisotro_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Anisotro | orientation z
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '13C';
Sys.A = [200.0 500.0 300.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Anisotro_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Anisotro_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Anisotro | orientation x
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '13C';
Sys.A = [200.0 500.0 300.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Anisotro_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Anisotro_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Anisotro | orientation y
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '13C';
Sys.A = [200.0 500.0 300.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Anisotro_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Anisotro_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% Anisotro | orientation gen
Sys.S = 0.5;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '13C';
Sys.A = [200.0 500.0 300.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_Anisotro_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_Anisotro_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DandE0 | orientation z
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.D = [50.0 0.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DandE0_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DandE0_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DandE0 | orientation x
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.D = [50.0 0.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DandE0_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DandE0_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DandE0 | orientation y
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.D = [50.0 0.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DandE0_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DandE0_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DandE0 | orientation gen
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.D = [50.0 0.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DandE0_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DandE0_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DandE | orientation z
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.D = [50.0 150.0];
Sys.lwpp = [5.0 0.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DandE_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DandE_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DandE | orientation x
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.D = [50.0 150.0];
Sys.lwpp = [5.0 0.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DandE_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DandE_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DandE | orientation y
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.D = [50.0 150.0];
Sys.lwpp = [5.0 0.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DandE_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DandE_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DandE | orientation gen
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.D = [50.0 150.0];
Sys.lwpp = [5.0 0.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DandE_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DandE_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% QandA | orientation z
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '14N';
Sys.A = [200.0 200.0 200.0];
Sys.Q = [50.0 20.0 50.0];
Sys.lwpp = [10.0 0.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_QandA_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_QandA_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% QandA | orientation x
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '14N';
Sys.A = [200.0 200.0 200.0];
Sys.Q = [50.0 20.0 50.0];
Sys.lwpp = [10.0 0.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_QandA_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_QandA_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% QandA | orientation y
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '14N';
Sys.A = [200.0 200.0 200.0];
Sys.Q = [50.0 20.0 50.0];
Sys.lwpp = [10.0 0.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_QandA_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_QandA_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% QandA | orientation gen
Sys.S = 1;
Sys.g = [2.0026 2.0026 2.0026];
Sys.Nucs = '14N';
Sys.A = [200.0 200.0 200.0];
Sys.Q = [50.0 20.0 50.0];
Sys.lwpp = [10.0 0.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_QandA_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_QandA_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DgandA | orientation z
Sys.S = 1;
Sys.g = [2.0026 1.806 2.001];
Sys.Nucs = '14N';
Sys.A = [200.0 200.0 200.0];
Sys.D = [50.0 150.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DgandA_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DgandA_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DgandA | orientation x
Sys.S = 1;
Sys.g = [2.0026 1.806 2.001];
Sys.Nucs = '14N';
Sys.A = [200.0 200.0 200.0];
Sys.D = [50.0 150.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DgandA_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DgandA_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DgandA | orientation y
Sys.S = 1;
Sys.g = [2.0026 1.806 2.001];
Sys.Nucs = '14N';
Sys.A = [200.0 200.0 200.0];
Sys.D = [50.0 150.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DgandA_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DgandA_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% DgandA | orientation gen
Sys.S = 1;
Sys.g = [2.0026 1.806 2.001];
Sys.Nucs = '14N';
Sys.A = [200.0 200.0 200.0];
Sys.D = [50.0 150.0];
Sys.lwpp = [0.0 5.0];
Exp.mwFreq = 9.43;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_DgandA_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_DgandA_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnFeO | orientation z
Sys.S = 1.5;
Sys.g = [2.01 2.01 2.01];
Sys.D = [-1780.0 -90.0];
Sys.lwpp = [0.5 0.0];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnFeO_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnFeO_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnFeO | orientation x
Sys.S = 1.5;
Sys.g = [2.01 2.01 2.01];
Sys.D = [-1780.0 -90.0];
Sys.lwpp = [0.5 0.0];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnFeO_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnFeO_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnFeO | orientation y
Sys.S = 1.5;
Sys.g = [2.01 2.01 2.01];
Sys.D = [-1780.0 -90.0];
Sys.lwpp = [0.5 0.0];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnFeO_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnFeO_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnFeO | orientation gen
Sys.S = 1.5;
Sys.g = [2.01 2.01 2.01];
Sys.D = [-1780.0 -90.0];
Sys.lwpp = [0.5 0.0];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnFeO_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnFeO_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCrO | orientation z
Sys.S = 1.5;
Sys.g = [1.96 1.96 1.97];
Sys.D = [500.0 1500.0];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCrO_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCrO_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCrO | orientation x
Sys.S = 1.5;
Sys.g = [1.96 1.96 1.97];
Sys.D = [500.0 1500.0];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCrO_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCrO_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCrO | orientation y
Sys.S = 1.5;
Sys.g = [1.96 1.96 1.97];
Sys.D = [500.0 1500.0];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCrO_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCrO_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCrO | orientation gen
Sys.S = 1.5;
Sys.g = [1.96 1.96 1.97];
Sys.D = [500.0 1500.0];
Sys.lwpp = [0.0 0.5];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCrO_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCrO_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCoO | orientation z
Sys.S = 1.5;
Sys.g = [4.43 4.43 2.2];
Sys.D = [700.0 0.0];
Sys.lwpp = [0.0 1.5];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCoO_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCoO_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCoO | orientation x
Sys.S = 1.5;
Sys.g = [4.43 4.43 2.2];
Sys.D = [700.0 0.0];
Sys.lwpp = [0.0 1.5];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCoO_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCoO_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCoO | orientation y
Sys.S = 1.5;
Sys.g = [4.43 4.43 2.2];
Sys.D = [700.0 0.0];
Sys.lwpp = [0.0 1.5];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCoO_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCoO_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCoO | orientation gen
Sys.S = 1.5;
Sys.g = [4.43 4.43 2.2];
Sys.D = [700.0 0.0];
Sys.lwpp = [0.0 1.5];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCoO_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCoO_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCrO2 | orientation z
Sys.S = 1.5;
Sys.g = [2.17 1.25 1.97];
Sys.D = [1839.85 959.99];
Sys.lwpp = [0.0 15.0];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 90;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCrO2_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCrO2_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCrO2 | orientation x
Sys.S = 1.5;
Sys.g = [2.17 1.25 1.97];
Sys.D = [1839.85 959.99];
Sys.lwpp = [0.0 15.0];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 90;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCrO2_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCrO2_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCrO2 | orientation y
Sys.S = 1.5;
Sys.g = [2.17 1.25 1.97];
Sys.D = [1839.85 959.99];
Sys.lwpp = [0.0 15.0];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 90;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCrO2_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCrO2_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% ZnCrO2 | orientation gen
Sys.S = 1.5;
Sys.g = [2.17 1.25 1.97];
Sys.D = [1839.85 959.99];
Sys.lwpp = [0.0 15.0];
Exp.mwFreq = 9.439;
Exp.Range = [100.0 500.0];
Exp.nPoints = 2048;
Exp.Temperature = 90;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_ZnCrO2_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_ZnCrO2_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% itrium | orientation z
Sys.S = 2.5;
Sys.g = [2.0062 2.0062 2.0062];
Sys.D = [-1779.87 0.0];
Sys.B4 = [0.0 2.755809026 0.0 0.0 -0.5829298611 0.0 0.0 0.0 0.0];
Sys.lwpp = [15.0 0.0];
Exp.mwFreq = 9.76;
Exp.Range = [0.0 600.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_itrium_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_itrium_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% itrium | orientation x
Sys.S = 2.5;
Sys.g = [2.0062 2.0062 2.0062];
Sys.D = [-1779.87 0.0];
Sys.B4 = [0.0 2.755809026 0.0 0.0 -0.5829298611 0.0 0.0 0.0 0.0];
Sys.lwpp = [15.0 0.0];
Exp.mwFreq = 9.76;
Exp.Range = [0.0 600.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_itrium_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_itrium_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% itrium | orientation y
Sys.S = 2.5;
Sys.g = [2.0062 2.0062 2.0062];
Sys.D = [-1779.87 0.0];
Sys.B4 = [0.0 2.755809026 0.0 0.0 -0.5829298611 0.0 0.0 0.0 0.0];
Sys.lwpp = [15.0 0.0];
Exp.mwFreq = 9.76;
Exp.Range = [0.0 600.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_itrium_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_itrium_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% itrium | orientation gen
Sys.S = 2.5;
Sys.g = [2.0062 2.0062 2.0062];
Sys.D = [-1779.87 0.0];
Sys.B4 = [0.0 2.755809026 0.0 0.0 -0.5829298611 0.0 0.0 0.0 0.0];
Sys.lwpp = [15.0 0.0];
Exp.mwFreq = 9.76;
Exp.Range = [0.0 600.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_itrium_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_itrium_gen.txt', 'Delimiter', 'tab');

clear Sys Exp;
% CaCO_3 | orientation z
Sys.S = 2.5;
Sys.g = [2.00123 2.00123 2.00131];
Sys.Nucs = '55Mn';
Sys.A = [-264.51 -264.51 -262.62];
Sys.Q = [0.566 -0.214 -0.352];
Sys.D = [-227.84 0.0];
Sys.B4 = [0.0 3.36 0.0 0.0 -0.141 0.0 0.0 0.0 0.0];
Sys.lwpp = [1.0 8.0];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 0 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_CaCO_3_z.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_CaCO_3_z.txt', 'Delimiter', 'tab');

clear Sys Exp;
% CaCO_3 | orientation x
Sys.S = 2.5;
Sys.g = [2.00123 2.00123 2.00131];
Sys.Nucs = '55Mn';
Sys.A = [-264.51 -264.51 -262.62];
Sys.Q = [0.566 -0.214 -0.352];
Sys.D = [-227.84 0.0];
Sys.B4 = [0.0 3.36 0.0 0.0 -0.141 0.0 0.0 0.0 0.0];
Sys.lwpp = [1.0 8.0];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 0]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_CaCO_3_x.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_CaCO_3_x.txt', 'Delimiter', 'tab');

clear Sys Exp;
% CaCO_3 | orientation y
Sys.S = 2.5;
Sys.g = [2.00123 2.00123 2.00131];
Sys.Nucs = '55Mn';
Sys.A = [-264.51 -264.51 -262.62];
Sys.Q = [0.566 -0.214 -0.352];
Sys.D = [-227.84 0.0];
Sys.B4 = [0.0 3.36 0.0 0.0 -0.141 0.0 0.0 0.0 0.0];
Sys.lwpp = [1.0 8.0];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 90 90]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_CaCO_3_y.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_CaCO_3_y.txt', 'Delimiter', 'tab');

clear Sys Exp;
% CaCO_3 | orientation gen
Sys.S = 2.5;
Sys.g = [2.00123 2.00123 2.00131];
Sys.Nucs = '55Mn';
Sys.A = [-264.51 -264.51 -262.62];
Sys.Q = [0.566 -0.214 -0.352];
Sys.D = [-227.84 0.0];
Sys.B4 = [0.0 3.36 0.0 0.0 -0.141 0.0 0.0 0.0 0.0];
Sys.lwpp = [1.0 8.0];
Exp.mwFreq = 9.43;
Exp.Range = [250.0 400.0];
Exp.nPoints = 2048;
Exp.Temperature = 300;
Exp.CrystalSymmetry = 'P1';
Exp.SampleFrame = -[0 50 30]*pi/180;   % [phi theta chi]
[B, spec] = pepper(Sys, Exp);
writematrix([B(:), spec(:)], 'cr_CaCO_3_gen.txt', 'Delimiter', 'tab');
[Pos, Amp] = resfields(Sys, Exp);
writematrix([Pos(:), Amp(:)], 'rf_CaCO_3_gen.txt', 'Delimiter', 'tab');
