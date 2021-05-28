Etimating Best Position of a Third Detector [ET or CE] Using Combined SNR[¶](#Etimating-Best-Position-of-a-Third-Detector-[ET-or-CE]-Using-Combined-SNR) {#Etimating-Best-Position-of-a-Third-Detector-[ET-or-CE]-Using-Combined-SNR}
========================================================================================================================================================

In [1]:

    import numpy as np
    from tqdm import trange
    from astropy.cosmology import FlatLambdaCDM
    from Ang_transform import SkyDet, Psi_transform
    from Waveform import Waveform_PN, Pattern_Func60, Pattern_Func90
    cosmo = FlatLambdaCDM(67.66, 0.3111)

Functions (scalar-product and SNR[ET,CE])

In [2]:

    def sc_prod(V1,V2):
        V1 = np.array(V1)
        V2 = np.array(V2)
        return sum(V1*V2)

    def Snr_CE(hp,hx,alpha,beta,psi):
        Fp, Fx = Pattern_Func90(alpha,beta,psi) # F = (Fplus,Fcross)
        h = Fp*hp+Fx*hx
        mod_h = np.real(h*np.conj(h))
        integ = np.sum(mod_h*df_Sn_CE)
        rho = 2*np.sqrt(integ)
        return rho

    def Snr_ET(hp,hx,alpha,beta,psi):
        Fp1, Fx1 = Pattern_Func60(alpha,beta,psi) # F = (Fplus,Fcross)
        Fp2, Fx2 = Pattern_Func60(alpha+2*np.pi/3,beta,psi)
        Fp3, Fx3 = Pattern_Func60(alpha+4*np.pi/3,beta,psi)
        h1 = Fp1*hp+Fx1*hx
        h2 = Fp2*hp+Fx2*hx
        h3 = Fp3*hp+Fx3*hx
        mod_h = np.real(h1*np.conj(h1) + h2*np.conj(h2) + h3*np.conj(h3))
        integ = np.sum(mod_h*df_Sn_ET)
        rho = 2*np.sqrt(integ)
        return rho

Open Sensitivity-Curves Data [CE and ET]

In [3]:

    data = np.loadtxt('Sn_CE.txt') # CE sensitivity
    freq_CE = data[:,0]
    Sn = data[:,3]**2
    df = np.zeros(len(freq_CE))
    for i in range(1,len(freq_CE)): df[i] =freq_CE[i]-freq_CE[i-1]
    df_Sn_CE = df/Sn
        
    data = np.loadtxt('Sn_ET.txt') # ET sensitivity
    freq_ET = data[:,0]
    Sn = data[:,1]**2
    df = np.zeros(len(freq_ET))
    for i in range(1,len(freq_ET)): df[i] =freq_ET[i]-freq_ET[i-1]
    df_Sn_ET = df/Sn

Location of both detectors [CE and ET]

In [11]:

    Lat1, Lon1 = 47.75, -120.74 # Location of CE
    phi1 = Lon1*np.pi/180 ; theta1 = np.pi-Lat1*np.pi/180

    Lat2, Lon2 = 40.078072, 9.283447 # Location of ET
    phi2 = Lon2*np.pi/180 ; theta2 = np.pi-Lat2*np.pi/180

    P1 = ( np.sin(theta1)*np.cos(phi1), np.sin(theta1)*np.sin(phi1), np.cos(theta1) )
    P2 = ( np.sin(theta2)*np.cos(phi2), np.sin(theta2)*np.sin(phi2), np.cos(theta2) )
    Sep12 = np.arccos(sc_prod(P1,P2))*180/np.pi
    print('Separation between ET and CE: Sep = %.1f deg' % Sep12)

    Separation between ET and CE: Sep = 78.0 deg

In [25]:

    z = 0.5
    dL = cosmo.luminosity_distance(z).value
    m1 = m2 = 1.5*(1+z)

    #Theta3 = np.array([0,30,60,90,120,150,180])*np.pi/180
    #Phi3 = np.array([0,45,90,135,180,225,270,315,360])*np.pi/180

    Theta3 = np.arange(0,181,5)*np.pi/180
    Phi3 = np.arange(0,361,5)*np.pi/180

    Beta = Theta3
    Alpha = Phi3

    n_lat = len(Theta3)
    n_lon = len(Phi3)

    iota = 0
    psi = 0

    N = n_lat*n_lon
    print('n_lat = ', n_lat)
    print('n_lon = ', n_lon)
    print('N Tot = ', N)

    col_theta = np.zeros(N)
    col_phi = np.zeros(N)
    col_snr_ce = np.zeros(N)
    col_snr_et = np.zeros(N)
    col_sep1 = np.zeros(N)
    col_sep2 = np.zeros(N)
    Cont = 0

    n_lat =  37
    n_lon =  73
    N Tot =  2701

In [39]:

    cond = True
    if(cond):
        for i in range(n_lat):
            print('[%d of %d]' % (i+1,n_lat))        
            theta3 = Theta3[i]
            for j in trange(n_lon):
                phi3 = Phi3[j]
                P3 =  ( np.sin(theta3)*np.cos(phi3), np.sin(theta3)*np.sin(phi3), np.cos(theta3) )
                col_sep1[Cont] = np.arccos(sc_prod(P1,P3)) 
                col_sep2[Cont] = np.arccos(sc_prod(P2,P3))
                SNR_3rd_ET = np.zeros(N)
                SNR_3rd_CE = np.zeros(N)
                cont = 0
                for k in range(50):    
                    alpha = np.random.uniform(0,2*np.pi)
                    beta = np.random.uniform(0,np.pi)
                    
                    psi1 = Psi_transform(alpha,beta,0,theta1,phi1)
                    psi2 = Psi_transform(alpha,beta,0,theta2,phi2)
                    psi3 = Psi_transform(alpha,beta,0,theta3,phi3)
                    
                    alpha1, beta1 = SkyDet(alpha,beta,theta1,phi1)
                    alpha2, beta2 = SkyDet(alpha,beta,theta2,phi2)
                    alpha3, beta3 = SkyDet(alpha,beta,theta3,phi3)
                    
                    hp1, hx1 = Waveform_PN(iota,m1,m2,dL,freq_CE)   # Cosmic Explorer
                    hp2, hx2 = Waveform_PN(iota,m1,m2,dL,freq_ET)   # Einstein Teslescope
                    
                    SNR_CE = Snr_CE(hp1,hx1,alpha1,beta1,psi1)
                    SNR_ET = Snr_ET(hp2,hx2,alpha2,beta2,psi2)
                    
                    SNR_3rd_CE[cont] += SNR_CE**2 + SNR_ET**2 + Snr_CE(hp1,hx1,alpha3,beta3,psi3)**2
                    SNR_3rd_ET[cont] += SNR_CE**2 + SNR_ET**2 + Snr_ET(hp2,hx2,alpha3,beta3,psi3)**2
                    cont+=1
                col_theta[Cont] = theta3
                col_phi[Cont] = phi3
                col_snr_ce[Cont] = np.sqrt(sum(SNR_3rd_CE))
                col_snr_et[Cont] = np.sqrt(sum(SNR_3rd_ET))
                Cont+=1
        print("Concluded!!!")

      0%|          | 0/73 [00:00<?, ?it/s]

    [1 of 37]

    100%|██████████| 73/73 [02:59<00:00,  2.46s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [2 of 37]

    100%|██████████| 73/73 [02:54<00:00,  2.40s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [3 of 37]

    100%|██████████| 73/73 [02:50<00:00,  2.34s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [4 of 37]

    100%|██████████| 73/73 [02:57<00:00,  2.43s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [5 of 37]

    100%|██████████| 73/73 [03:22<00:00,  2.77s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [6 of 37]

    100%|██████████| 73/73 [03:08<00:00,  2.59s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [7 of 37]

    100%|██████████| 73/73 [03:17<00:00,  2.70s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [8 of 37]

    100%|██████████| 73/73 [03:27<00:00,  2.84s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [9 of 37]

    100%|██████████| 73/73 [03:16<00:00,  2.70s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [10 of 37]

    100%|██████████| 73/73 [03:09<00:00,  2.59s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [11 of 37]

    100%|██████████| 73/73 [03:15<00:00,  2.68s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [12 of 37]

    100%|██████████| 73/73 [03:01<00:00,  2.49s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [13 of 37]

    100%|██████████| 73/73 [03:14<00:00,  2.66s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [14 of 37]

    100%|██████████| 73/73 [03:10<00:00,  2.60s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [15 of 37]

    100%|██████████| 73/73 [03:05<00:00,  2.54s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [16 of 37]

    100%|██████████| 73/73 [03:03<00:00,  2.52s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [17 of 37]

    100%|██████████| 73/73 [03:04<00:00,  2.53s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [18 of 37]

    100%|██████████| 73/73 [03:07<00:00,  2.57s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [19 of 37]

    100%|██████████| 73/73 [03:09<00:00,  2.60s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [20 of 37]

    100%|██████████| 73/73 [03:05<00:00,  2.55s/it]
      0%|          | 0/73 [00:00<?, ?it/s]

    [21 of 37]

     89%|████████▉ | 65/73 [02:54<00:21,  2.68s/it]

    ---------------------------------------------------------------------------
    KeyboardInterrupt                         Traceback (most recent call last)
    <ipython-input-39-b0bb928c8e56> in <module>
         24                                 alpha3, beta3 = SkyDet(alpha,beta,theta3,phi3)
         25 
    ---> 26                                 hp1, hx1 = Waveform_PN(iota,m1,m2,dL,freq_CE)   # Cosmic Explorer
         27                                 hp2, hx2 = Waveform_PN(iota,m1,m2,dL,freq_ET)   # Einstein Teslescope
         28 

    ~/Área de Trabalho/Cosmic Explorer Project/Jupyter_programs/Waveform.py in Waveform_PN(iota, m1, m2, DL, freq)
         56 
         57         cp, cx = (1.+np.cos(iota)**2)/2, np.cos(iota)
    ---> 58         Hplus, Hcross = h0*cp*np.exp(1.j*phase), 1.j*h0*cx*np.exp(1.j*phase)
         59         return Hplus, Hcross
         60 

    KeyboardInterrupt: 

In [16]:

    header = "Theta3[rad] \t Phi3[rad] \t SNR_CE \t SNR_ET \t Sep1[rad] \t Sep2[rad]"
    Columns = [col_theta,col_phi,col_snr_ce,col_snr_et,col_sep1,col_sep2]
    np.savetxt('3rd_SNR.txt',np.transpose(Columns),fmt='%f',delimiter='\t',header=header)

Plotting the results:

In [24]:

    import matplotlib.pyplot as plt
    sep1 = col_sep1*180/np.pi
    sep2 = col_sep2*180/np.pi
    snr_et = col_snr_et
    snr_ce = col_snr_ce

    fig = plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title('Eintein Telescope')

    c = plt.tricontourf(sep1,sep2,snr_et,levels=20,cmap='jet')
    plt.xlabel('Separation From CE [deg]')
    plt.ylabel('Separation From ET [deg]')
    plt.grid(linestyle='--',alpha=0.5)
    plt.xlim(0,180) ; plt.ylim(0,180)
    plt.xticks(np.arange(0,181,30))
    plt.yticks(np.arange(0,181,30))
    plt.colorbar(c,orientation='horizontal')

    plt.subplot(1,2,2)
    plt.title('Cosmic Explorer')

    c = plt.tricontourf(sep1,sep2,snr_ce,levels=20,cmap='jet')
    plt.xlabel('Separation From CE [deg]')
    plt.ylabel('Separation From ET [deg]')
    plt.grid(linestyle='--',alpha=0.5)
    plt.xlim(0,180) ; plt.ylim(0,180)
    plt.xticks(np.arange(0,181,30))
    plt.yticks(np.arange(0,181,30))
    plt.colorbar(c,orientation='horizontal')

    plt.show()

![](data:image/png;base64,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%0A)

In [ ]:

     
