# analysis two-dimensional dynamical systems in neuroscience
# Mike Wendels
# last update: 2022-12-11

# ===== MODULES ===== #
import os
import numpy as np
import matplotlib.pyplot as plt
import ode2d_models as model
import ode2d_analysis as analysis

# ===== ANALYSIS IZHIKEVICH MODEL A ===== #

# Izh_tmesh_TSA = 200.,0.01 
# Izh_tmesh_PPA = 200.,0.01
# Izh_tmesh_GIF = 100.,1
# Izh_eqtol = 1e-3

case = "IZH-B" #"IZH-D"

if case == "IZH-A":

    xmesh = -100.0, 50.0, 30, 500, "$v$"
    ymesh = -50.0, 200.0, 30, 500, "$u$"
    tmesh = 500.,0.01 #200.,0.1
    X0 = [np.array([-45,-20])] #[np.array([-60,0]),np.array([-45,-20])]
    eqtol = 1e-3
    model_lab = "IZH"

    C = 100
    k = 0.7
    vr = -60
    vt = -40
    vpeak = 35
    a = 0.03
    b = -2
    c = -50
    d = 100

    Iext = [0,20,40,60,80,100,200,300]

    savepath_main = "./IZH_model_A/"
    if not os.path.exists(savepath_main):
        os.mkdir(savepath_main)

elif case == "IZH-B":

    xmesh = -100.0, 50.0, 30, 500, "$v$"
    ymesh = -50.0, 250.0, 30, 500, "$u$"
    xmesh = -60.0, -40.0, 30, 500, "$v$"
    ymesh = 40.0, 100.0, 30, 500, "$u$"
    xmesh = -100.0, 50.0, 30, 500, "$v$"
    ymesh = -50.0, 300.0, 30, 500, "$u$"
    tmesh = 500.,0.01 #200.,0.1
    X0 = [np.array([-60,100])] #[np.array([-40,50])] #[np.array([-60,100])] # compared to (-60,0) in model A
    eqtol = 1e-3
    model_lab = "IZH"

    C = 100
    k = 0.7
    vr = -60
    vt = -40
    vpeak = 35
    a = 0.03
    b = 5 # compared to -2 in model A
    c = -50
    d = 100

    Iext = [200,300,400,500] #[120,124.5,125,126.5,127,127.5]

    savepath_main = "./IZH_model_B/"
    if not os.path.exists(savepath_main):
        os.mkdir(savepath_main)

elif case == "IZH-C":

    xmesh = -100.0, 50.0, 30, 500, "$v$"
    ymesh = -50.0, 450.0, 30, 500, "$u$"
    tmesh = 500.,0.01 #200.,0.1
    X0 = [np.array([-80,0])]
    eqtol = 1e-3
    model_lab = "IZH"

    C = 150
    k = 1.2
    vr = -75
    vt = -45
    vpeak = 50
    a = 0.01
    b = 5
    c = -56
    d = 130

    Iext = [0,300,370,500,550]

    savepath_main = "./IZH_model_C/"
    if not os.path.exists(savepath_main):
        os.mkdir(savepath_main)

elif case == "IZH-D":

    xmesh = -100.0, 50.0, 30, 500, "$v$"
    ymesh = 0,20000,30,500,"$u$" 
    ymesh = -50.0, 250.0, 30, 500, "$u$"
    tmesh = 500.,0.01
    X0 = [np.array([30,19000])] 
    X0 = [np.array([-50,50])] #[np.array([-70,200])] #[np.array([-60,0])]
    eqtol = 1e-3
    model_lab = "IZH"

    C = 50
    k = 0.5
    vr = -60
    vt = -45
    vpeak = 40
    a = 0.02
    b = 0.5
    c = -35 #fig 8.35
    d = 60

    Iext = [0,10,30,50,70]

    savepath_main = "./IZH_model_D/"
    if not os.path.exists(savepath_main):
        os.mkdir(savepath_main)

if case == "IZH-E":

    xmesh = -100.0, 50.0, 30, 500, "$v$"
    ymesh = -50.0, 200.0, 30, 500, "$u$"
    tmesh = 500.,0.01 #200.,0.1
    X0 = [np.array([-40,150])] #[np.array([-60,0]),np.array([-45,-20])]
    eqtol = 1e-3
    model_lab = "IZH"

    C = 100
    k = 0.7
    vr = -60
    vt = -40
    vpeak = 35
    a = 0.5
    b = 0.2
    c = -65
    d = 2

    Iext = [0,20,40,60,80,100,200,300]

    savepath_main = "./IZH_model_E/"
    if not os.path.exists(savepath_main):
        os.mkdir(savepath_main)

elif case == "FHN-A":

    xmesh = -0.5, 1.0, 30, 50, "$V$"
    ymesh = -0.05, 0.2, 30, 50, "$w$"
    tmesh = 500.,0.01
    X0 = [np.array([0,0.15])]
    eqtol = 1e-3
    model_lab = "FHN"

    a = 0.1
    b = 0.01
    c = 0.02

    Iext = [0,10,30,50,70]

    savepath_main = "./FHN_model_A/"
    if not os.path.exists(savepath_main):
        os.mkdir(savepath_main)

elif case == "FHN-B":

    xmesh = -0.5, 1.0, 30, 50, "$V$"
    ymesh = -0.05, 0.2, 30, 50, "$w$"
    tmesh = 500.,0.01
    X0 = [np.array([0,0.15])]
    eqtol = 1e-3
    model_lab = "FHN"

    a = -0.1
    b = 0.01
    c = 0.02

    Iext = [0,10,30,50,70]

    savepath_main = "./FHN_model_B/"
    if not os.path.exists(savepath_main):
        os.mkdir(savepath_main)

elif case == "FHN-C":

    xmesh = -0.5, 1.0, 30, 50, "$V$"
    ymesh = -0.05, 0.2, 30, 50, "$w$"
    tmesh = 500.,0.01
    X0 = [np.array([0,0.5])]
    eqtol = 1e-3
    model_lab = "FHN"

    a = 0.1
    b = 0.01
    c = 0.1

    Iext = [0,10,30,50,70]

    savepath_main = "./FHN_model_C/"
    if not os.path.exists(savepath_main):
        os.mkdir(savepath_main)

elif case == "PNK-A":

    xmesh = -90, 30.0, 30, 50, "$V$"
    ymesh = 0.0, 1.0, 30, 50, "$n$"
    tmesh = 500.,0.01
    X0 = [np.array([0,0.5])]
    eqtol = 1e-1
    model_lab = "PNK"

    f_minf = lambda V,Vhalf,k : 1/(1+np.exp((Vhalf-V)/k)) # np.power(1+np.exp((Vhalf-V)/k),-1)
    df_minf = lambda V,Vhalf,k : np.exp((Vhalf-V)/k)/(1+np.exp((Vhalf-V)/k))/k

    C = 1
    gL = 8
    gNa = 20
    gK = 10
    EL = -80
    ENa = 60
    EK = -90
    minf = lambda V : f_minf(V,-20,15)
    ninf = lambda V : f_minf(V,-25,5)
    tau = lambda V : 1
    dminf = lambda V : df_minf(V,-20,15)
    dninf = lambda V : df_minf(V,-25,5)
    dtau = lambda V : 0

    Iext = [0,10,30,50,70]

    savepath_main = "./PNK_model_A/"
    if not os.path.exists(savepath_main):
        os.mkdir(savepath_main)

elif case == "PNK-B":

    xmesh = -90, 30.0, 30, 50, "$V$"
    ymesh = 0.0, 1.0, 30, 50, "$n$"
    tmesh = 500.,0.01
    X0 = [np.array([0,0.5])]
    eqtol = 1e-1
    model_lab = "PNK"

    f_minf = lambda V,Vhalf,k : 1/(1+np.exp((Vhalf-V)/k)) # np.power(1+np.exp((Vhalf-V)/k),-1)
    df_minf = lambda V,Vhalf,k : np.exp((Vhalf-V)/k)/(1+np.exp((Vhalf-V)/k))/k

    C = 1
    gL = 8
    gNa = 20
    gK = 10
    EL = -80
    ENa = 60
    EK = -90
    minf = lambda V : f_minf(V,-20,15)
    ninf = lambda V : f_minf(V,-45,5)
    tau = lambda V : 1
    dminf = lambda V : df_minf(V,-20,15)
    dninf = lambda V : df_minf(V,-45,5)
    dtau = lambda V : 0

    Iext = [0,10,30,50,70]

    savepath_main = "./PNK_model_B/"
    if not os.path.exists(savepath_main):
        os.mkdir(savepath_main)


if model_lab == "IZH":
    fmodel = lambda f : model.Izh(fIext=f,C=C,k=k,vr=vr,vt=vt,vpeak=vpeak,a=a,b=b,c=c,d=d)
elif model_lab == "FHN":
    fmodel = lambda f : model.FHN(fIext=f,a=a,b=b,c=c)
elif model_lab == "PNK":
    fmodel = lambda f : model.pNK(fIext=f,C=C,gL=gL,gNa=gNa,gK=gK,EL=EL,ENa=ENa,EK=EK,minf=minf,ninf=ninf,tau=tau,dminf=dminf,dninf=dninf,dtau=dtau)
else:
    print("ERROR: specify model")

# plotting cases
tmesh = 1000.,0.01 
tthres = 50
Tf6_1 = [40,60]
Tf6_2 = [50,250]
Tf6 = [Tf6_1,Tf6_2]
Tf7 = [[[180,200],[580,600]],[[380,400],[780,800]]]
# Tf7_1 = [[0,10],[20,30],[40,50],[60,70],[80,90],[100,110]]
# Tf7_2 = [[10,20],[30,40],[50,60],[70,80],[90,100],[110,120]]
# Tf7_3 = [[20,40],[60,80],[100,120]]
# Tf7 = [Tf7_1,Tf7_2,Tf7_3]
Tplot = [0,tmesh[0]]

analysis_f1 = 0
analysis_f2 = 0
analysis_f3 = 0
analysis_f4 = 0
analysis_f5 = 0
analysis_f6 = 0
analysis_f7 = 1

opt_plot = 2

for tplot in Tplot:

    #########
    if analysis_f1:

        savedir = f"f1_tend={tmesh[0]}_tplot{tplot}/"
        savepath = savepath_main+savedir
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        # izhikevich fig 8.12
        # Iext = [0,20,40,50,60,80,100]
        # tmesh = 500.,0.01 
        for iext in Iext:
            f_i = lambda t : model.fI01(t,iext)
            savename_i = f"I{iext}"
            case_i = "" #f"parameter set A, $I(t) = f_1(t,{iext})$"
            plotI_i = [f_i,"I",[0,iext+10]]
            analysis.phasePlaneAnalysis(fmodel(f_i),savename_i,xmesh,ymesh,tmesh,X0=X0,t=tplot,plot_traj=2,eqtol=1e-3,opt_plot=opt_plot,legend=False,title=case_i,plotI=plotI_i,savepath=savepath)

    #########
    if analysis_f2:

        savedir = f"f2_tend={tmesh[0]}_tthres={tthres}_tplot{tplot}/"
        savepath = savepath_main+savedir
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        # Iext = [0,50,100,200,300]
        # tmesh = 500.,0.01 
        # tthres = 50
        for iext in Iext:
            f_i = lambda t : model.fI02(t,iext,tthres)
            savename_i = f"I{iext}"
            case_i = ""
            plotI_i = [f_i,"I",[0,iext+10]]
            analysis.phasePlaneAnalysis(fmodel(f_i),savename_i,xmesh,ymesh,tmesh,X0=X0,t=tplot,plot_traj=2,eqtol=1e-3,opt_plot=opt_plot,legend=False,title=case_i,plotI=plotI_i,savepath=savepath)

    #########
    if analysis_f3:

        savedir = f"f3_tend={tmesh[0]}_tthres={tthres}_tplot{tplot}/"
        savepath = savepath_main+savedir
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        Iext = [0,50,100,200,300]
        tmesh = 500.,0.01 
        tthres = 50
        for iext in Iext:
            f_i = lambda t : model.fI03(t,iext,tthres)
            savename_i = f"I{iext}"
            case_i = ""
            plotI_i = [f_i,"I",[0,iext+10]]
            analysis.phasePlaneAnalysis(fmodel(f_i),savename_i,xmesh,ymesh,tmesh,X0=X0,t=tplot,plot_traj=2,eqtol=1e-3,opt_plot=opt_plot,legend=False,title=case_i,plotI=plotI_i,savepath=savepath)

    #########
    if analysis_f4:

        savedir = f"f4_tend={tmesh[0]}_tthres={tthres}_tplot{tplot}/"
        savepath = savepath_main+savedir
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        Iext = [0,50,100,200,300]
        tmesh = 500.,0.01 
        tthres = 50
        for iext in Iext:
            f_i = lambda t : model.fI04(t,iext,tthres,tmesh[0]) 
            savename_i = f"I{iext}"
            case_i = ""
            plotI_i = [f_i,"I",[0,iext+10]]
            analysis.phasePlaneAnalysis(fmodel(f_i),savename_i,xmesh,ymesh,tmesh,X0=X0,t=tplot,plot_traj=2,eqtol=1e-3,opt_plot=opt_plot,legend=False,title=case_i,plotI=plotI_i,savepath=savepath)

    #########
    if analysis_f5:

        savedir = f"f5_tend={tmesh[0]}_tthres={tthres}_tplot{tplot}/"
        savepath = savepath_main+savedir
        if not os.path.exists(savepath):
            os.mkdir(savepath)

        Iext = [0,20,40,50,60,80,100,200,300]
        tmesh = 500.,0.01 
        tthres = 50
        for iext in Iext:
            f_i = lambda t : model.fI05(t,iext,tthres) 
            savename_i = f"I{iext}"
            case_i = ""
            plotI_i = [f_i,"I",[0,iext+10]]
            analysis.phasePlaneAnalysis(fmodel(f_i),savename_i,xmesh,ymesh,tmesh,X0=X0,t=tplot,plot_traj=2,eqtol=1e-3,opt_plot=opt_plot,legend=False,title=case_i,plotI=plotI_i,savepath=savepath)

    #########
    if analysis_f6:

        for T in Tf6:

            savedir = f"f6_tend={tmesh[0]}_T={T}_excitatory_tplot{tplot}/"
            savepath = savepath_main+savedir
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            for iext in Iext:
                f_i = lambda t : model.fI06(t,[iext,0],T)
                savename_i = f"I{iext}"
                case_i = ""
                plotI_i = [f_i,"I",[0,iext+10]]
                analysis.phasePlaneAnalysis(fmodel(f_i),savename_i,xmesh,ymesh,tmesh,X0=X0,t=tplot,plot_traj=2,eqtol=1e-3,opt_plot=opt_plot,legend=False,title=case_i,plotI=plotI_i,savepath=savepath)

        for T in Tf6:

            savedir = f"f6_tend={tmesh[0]}_T={T}_inhibitory_tplot{tplot}/"
            savepath = savepath_main+savedir
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            for iext in Iext:
                f_i = lambda t : model.fI06(t,[0,iext],T)
                savename_i = f"I{iext}"
                case_i = ""
                plotI_i = [f_i,"I",[0,iext+10]]
                analysis.phasePlaneAnalysis(fmodel(f_i),savename_i,xmesh,ymesh,tmesh,X0=X0,t=tplot,plot_traj=2,eqtol=1e-3,opt_plot=opt_plot,legend=False,title=case_i,plotI=plotI_i,savepath=savepath)

    #########
    if analysis_f7:

        for T in Tf7:

            savedir = f"f7_tend={tmesh[0]}_T={T}_Iref0_tplot{tplot}/"
            savepath = savepath_main+savedir
            if not os.path.exists(savepath):
                os.mkdir(savepath)

            for iext in Iext:
                T_i = [[T[0][0],T[0][1],iext]]
                for j in range(1,len(T)):
                    T_i.extend([[T[j][0],T[j][1],iext]])
                print(f"T_i = {T_i}")
                # T_i = [[10,20,iext],[30,40,iext],[50,60,iext],[70,80,iext],[90,100,iext],[110,120,iext]]
                f_i = lambda t : model.fI07(t,124.5,T_i)
                savename_i = f"I{iext}"
                case_i = ""
                plotI_i = [f_i,"I",[0,iext+10]]
                analysis.phasePlaneAnalysis(fmodel(f_i),savename_i,xmesh,ymesh,tmesh,X0=X0,t=tplot,plot_traj=2,eqtol=1e-3,opt_plot=opt_plot,legend=False,title=case_i,plotI=plotI_i,savepath=savepath)

    #########

