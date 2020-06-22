#!/anaconda3/envs/fenicsproject/bin/python

## #!/home/icsrsss/anaconda3/envs/blockdata/bin/python
## #!/anaconda3/envs/fenicsproject/bin/python

from fenics import *
import math, os, sys, getopt, time, random, subprocess, logging
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from zipfile import ZipFile
  

"""
Solve 3D linear quasistatic viscoelasticity problem in time to generate virtual
sensor data on the surface of parallelapiped block with a localised internal vibrating
source disturbance.

Copyright (c) 2020, Simon Shaw (simon.shaw89@alumni.imperial.ac.uk)
The moral right of the author has been asserted.

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License
as published by the Free Software Foundation; either version 2
of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You can obtain a copy of the GNU General Public License from
https://www.gnu.org/licenses/gpl-3.0.en.html

Alternatively, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
Also add information on how to contact you by electronic and paper mail.

This code is based on demo_elastodynamics.py as downloaded on 10 Dec 2019 from
# https://fenicsproject.org/docs/dolfin/latest/python/demos/elastodynamics/

And also on ft06_elasticity.py as downloaded on 10 Dec 2019 from
# https://github.com/hplgit/fenics-tutorial/blob/master/pub/python/vol1/ft06_elasticity.py

Remarks:
- For line styles, see https://matplotlib.org/gallery/lines_bars_and_markers/linestyles.html?highlight=linestyles
- For the property cycler see https://matplotlib.org/tutorials/intermediate/color_cycle.html
- this isn't needed with the plot cyler:
  markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
- marker style:
  - https://matplotlib.org/api/markers_api.html#matplotlib.markers.MarkerStyle
  - https://matplotlib.org/gallery/lines_bars_and_markers/marker_fillstyle_reference.html
  - https://matplotlib.org/3.1.1/tutorials/toolkits/mplot3d.html

"""

# these are chosen in order when plotting - for five plots or less they're unique
default_cycler                                                                       \
  = (cycler(color=['r', 'g', 'b', 'm', 'c', 'k'])                                    \
  +  cycler(linestyle=[(0,(3,1,1,1,1,1)), '--', ':', '-.', (0,(3,5,1,5,1,5)), '-' ]) \
    )

# Form compiler options
#parameters["form_compiler"]["cpp_optimize"] = True
#parameters["form_compiler"]["optimize"] = True

class blockdata:
  '''
  Linear quasistatic viscoelasticity solver on discrete [0,T] and
  3D rectangular prism [xL, xH] x [yL, yH] x [zL, zH]
  '''
  
  # immutable solver parameters:
  Ey      =  50000.0              # Young's modulus
  nu      =  0.48                 # Poisson's ratio
  xL, xH  =  0.0, 0.3             # Domain's extent on x axis 
  yL, yH  = -0.05, 0.05           # Domain's extent on y axis 
  zL, zH  =  0.0, 0.05            # Domain's extent on z axis 
  lmbda   = Ey*nu/(1+nu)/(1-2*nu) # Lame parameter
  mu      = Ey/2.0/(1+nu)         # Lame parameter, shear modulus

  # accelerometer positions in rows: (x1,y1,zH),  (x2,y2,zH),  ... (x1,y1,zH)
  accl = np.array([[0.12, 0.01, zH],                 [0.18, 0.01, zH],
                                    [0.15, 0.00, zH],
                   [0.12,-0.01, zH],                 [0.18,-0.01, zH],
                  ])
  # microphone positions in rows: (x1,y1,zL),  (x2,y2,zL),  ... (x1,y1,zL)
  mics = np.array([                 [0.15, 0.01, zH],
                   [0.12, 0.00, zH],                 [0.18, 0.00, zH],
                   
                                    [0.15,-0.01, zH],
                  ])
  
  # set defaults - all mutable
  T       = 0.15    # the final time...
  freqm   = 1       # source modulation frequency
  freqc   = 2       # source carrier frequency
  Fo      = 1000    # amplitude of source pulse 
  eps     = 0.001   # width of source pulse
  const   = 0       # which mode to use for the load. See help() 
  Nx      = 30      # default value for the mesh density in x direction
  Ny      = 10      # default value for the mesh density in y direction
  Nz      = 5       # default value for the mesh density in z direction
  Nt      = 5       # default value for the number of time steps
  rdeg    = 2       # order of CG elements
  Rdeg    = 5       # order of polynomials for exact solutions
  dt      = T/Nt    # time step
  #hx     = 1.0/Nx  # mesh width in x direction
  #hy     = 1.0/Ny  # mesh width in y direction
  #hz     = 1.0/Nz  # mesh width in z direction
  
  # variables for output file handles
  tp      = None    # collect up discrete times in one file (avoid duplicates)
  u1_a    = None    # columns of u1 at each accelerometer through time
  u2_a    = None    # columns of u2 at each accelerometer through time
  u3_a    = None    # columns of u3 at each accelerometer through time
  u1t_a   = None    # columns of u1_t at each accelerometer through time
  u2t_a   = None    # columns of u2_t at each accelerometer through time
  u3t_a   = None    # columns of u3_t at each accelerometer through time
  u1tt_a  = None    # columns of u1_tt at each accelerometer through time
  u2tt_a  = None    # columns of u2_tt at each accelerometer through time
  u3tt_a  = None    # columns of u3_tt at each accelerometer through time
  
  u1x_a   = None    # columns of u1_x at each accelerometer through time
  u2x_a   = None    # columns of u2_x at each accelerometer through time
  u3x_a   = None    # columns of u3_x at each accelerometer through time
  u1xt_a  = None    # columns of u1_xt at each accelerometer through time
  u2xt_a  = None    # columns of u2_xt at each accelerometer through time
  u3xt_a  = None    # columns of u3_xt at each accelerometer through time
  u1xtt_a = None    # columns of u1_xtt at each accelerometer through time
  u2xtt_a = None    # columns of u2_xtt at each accelerometer through time
  u3xtt_a = None    # columns of u3_xtt at each accelerometer through time
  
  u1y_a   = None    # columns of u1_y at each accelerometer through time
  u2y_a   = None    # columns of u2_y at each accelerometer through time
  u3y_a   = None    # columns of u3_y at each accelerometer through time
  u1yt_a  = None    # columns of u1_yt at each accelerometer through time
  u2yt_a  = None    # columns of u2_yt at each accelerometer through time
  u3yt_a  = None    # columns of u3_yt at each accelerometer through time
  u1ytt_a = None    # columns of u1_ytt at each accelerometer through time
  u2ytt_a = None    # columns of u2_ytt at each accelerometer through time
  u3ytt_a = None    # columns of u3_ytt at each accelerometer through time

  u1z_a   = None    # columns of u1_z at each accelerometer through time
  u2z_a   = None    # columns of u2_z at each accelerometer through time
  u3z_a   = None    # columns of u3_z at each accelerometer through time
  u1zt_a  = None    # columns of u1_zt at each accelerometer through time
  u2zt_a  = None    # columns of u2_zt at each accelerometer through time
  u3zt_a  = None    # columns of u3_zt at each accelerometer through time
  u1ztt_a = None    # columns of u1_ztt at each accelerometer through time
  u2ztt_a = None    # columns of u2_ztt at each accelerometer through time
  u3ztt_a = None    # columns of u3_ztt at each accelerometer through time

  # variables for output file handles
  u1_m    = None    # columns of u1 at each microphone through time
  u2_m    = None    # columns of u2 at each microphone through time
  u3_m    = None    # columns of u3 at each microphone through time
  u1t_m   = None    # columns of u1_t at each microphone through time
  u2t_m   = None    # columns of u2_t at each microphone through time
  u3t_m   = None    # columns of u3_t at each microphone through time
  u1tt_m  = None    # columns of u1_tt at each microphone through time
  u2tt_m  = None    # columns of u2_tt at each microphone through time
  u3tt_m  = None    # columns of u3_tt at each microphone through time
  
  u1x_m   = None    # columns of u1_x at each microphone through time
  u2x_m   = None    # columns of u2_x at each microphone through time
  u3x_m   = None    # columns of u3_x at each microphone through time
  u1xt_m  = None    # columns of u1_xt at each microphone through time
  u2xt_m  = None    # columns of u2_xt at each microphone through time
  u3xt_m  = None    # columns of u3_xt at each microphone through time
  u1xtt_m = None    # columns of u1_xtt at each microphone through time
  u2xtt_m = None    # columns of u2_xtt at each microphone through time
  u3xtt_m = None    # columns of u3_xtt at each microphone through time
  
  u1y_m   = None    # columns of u1_y at each microphone through time
  u2y_m   = None    # columns of u2_y at each microphone through time
  u3y_m   = None    # columns of u3_y at each microphone through time
  u1yt_m  = None    # columns of u1_yt at each microphone through time
  u2yt_m  = None    # columns of u2_yt at each microphone through time
  u3yt_m  = None    # columns of u3_yt at each microphone through time
  u1ytt_m = None    # columns of u1_ytt at each microphone through time
  u2ytt_m = None    # columns of u2_ytt at each microphone through time
  u3ytt_m = None    # columns of u3_ytt at each microphone through time

  u1z_m   = None    # columns of u1_z at each microphone through time
  u2z_m   = None    # columns of u2_z at each microphone through time
  u3z_m   = None    # columns of u3_z at each microphone through time
  u1zt_m  = None    # columns of u1_zt at each microphone through time
  u2zt_m  = None    # columns of u2_zt at each microphone through time
  u3zt_m  = None    # columns of u3_zt at each microphone through time
  u1ztt_m = None    # columns of u1_ztt at each microphone through time
  u2ztt_m = None    # columns of u2_ztt at each microphone through time
  u3ztt_m = None    # columns of u3_ztt at each microphone through time

  # arrays for sensor data. For accelerometer c ...
  accl_u  = None    # accl[c,n,v] has v = u1,u1_t,u1_tt, ..., u3_tt at tn-dt
  accl_ux = None    # accl[c,n,v] has v = u1_x,u1_xt,u1_xtt, ..., u3_xtt at tn-dt
  accl_uy = None    # accl[c,n,v] has v = u1_x,u1_xt,u1_xtt, ..., u3_xtt at tn-dt
  accl_uz = None    # accl[c,n,v] has v = u1_x,u1_xt,u1_xtt, ..., u3_xtt at tn-dt
  # ... and now for microphone c ...
  mics_u  = None    # mics[c,n,v] has v = u1,u1_t,u1_tt, ..., u3_tt at tn-dt
  mics_ux = None    # mics[c,n,v] has v = u1_x,u1_xt,u1_xtt, ..., u3_xtt at tn-dt
  mics_uy = None    # mics[c,n,v] has v = u1_x,u1_xt,u1_xtt, ..., u3_xtt at tn-dt
  mics_uz = None    # mics[c,n,v] has v = u1_x,u1_xt,u1_xtt, ..., u3_xtt at tn-dt

  # runtime config variables
  gfx     = 0       # if 0, then do not create sensor graphics for the run
  
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   

  def __init__(self, T, eps, Nt, Nx, Ny, Nz):
    self.T = T; self.eps = eps; self.f.eps = self.eps
    self.Nt = Nt; self.dt = self.T/self.Nt
    self.Nx = Nx;
    self.Ny = Ny;
    self.Nz = Nz;
    # if these ever get used then update the command line overrides
    # self.hx = (self.xH-self.xL)/self.Nx
    # self.hy = (self.yH-self.yL)/self.Ny
    # self.hz = (self.zH-self.zL)/self.Nz

    logging.basicConfig(level=logging.ERROR)   # to turn off no transparency eps message
    #logging.error('This will get logged')
    
    set_log_level(30)
    """
    where level is either WARNING or ERROR. You can experiment with
    different levels to get different amounts of diagnostic messages:

    CRITICAL  = 50, // errors that may lead to data corruption and suchlike
    ERROR     = 40, // things that go boom
    WARNING   = 30, // things that may go boom later
    INFO      = 20, // information of general interest
    PROGRESS  = 16, // what's happening (broadly)
    TRACE     = 13, // what's happening (in detail)
    DBG       = 10  // sundry
    To turn of logging completely, use set_log_active(False)
    """
    
  # vector form for RHS body force -  uniform in all directions - modulated by f_mod(x,y,z,t)
  f = Expression(("F*exp(-( (x[0]-xc)*(x[0]-xc)+(x[1]-yc)*(x[1]-yc)+(x[2]-zc)*(x[2]-zc))/eps)",
                  "F*exp(-( (x[0]-xc)*(x[0]-xc)+(x[1]-yc)*(x[1]-yc)+(x[2]-zc)*(x[2]-zc))/eps)",
                  "F*exp(-( (x[0]-xc)*(x[0]-xc)+(x[1]-yc)*(x[1]-yc)+(x[2]-zc)*(x[2]-zc))/eps)"
                 ), degree=2, F=0, xc=0, yc=0, zc=0, eps=0.001)

  # temporal modulation of RHS body force f 
  def f_mod(self,t):
    return self.Fo*sin(2*pi*self.freqc*t)*(sin(pi*self.freqm*t))*(sin(pi*self.freqm*t))

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   

  # plot sensor readings
  def plot_sensor_trace(self, savedir, fn, u_label, sens_type, v, qn):
      # set up the plot
      nfs = 16          # normal font size
      nlw = 3; tlw=1    # normal, thin line width
      nms = 10          # normal marker size
      plt.rc('axes', prop_cycle=default_cycler)   # move this to be first command! (see postpro.sty)
      plt.rc('text', usetex=True)
      plt.rc('font', family='serif')
      plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]  # for \boldsymbol etc
      plt.rcParams.update({'font.size': nfs})
      plt.tick_params(labelsize = nfs)
      plt.gcf().subplots_adjust(left=0.20)
#      plt.rc('axes', prop_cycle=default_cycler)   # move this to be first command! (see postpro.sty)
      plt.rc('lines', linewidth=nlw)

      times = np.linspace(self.dt,self.T,num=self.Nt-1, endpoint=False)
      fig = plt.figure(); ax = fig.add_subplot(111)
      ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
      if sens_type == 'accelerometers':
        label = u_label+r'(\mathrm{accl\ }'
      elif sens_type == 'microphones':
        label = u_label+r'(\mathrm{mic\ }'
      else:
        print('Error in plotting sensor traces. Exiting...')
        exit(0)
      plt.xlabel(r'$t$ (seconds)',fontsize = nfs)
      plt.ylabel(u_label+'$ at the '+sens_type,fontsize = nfs)
#      print('u_label = '+u_label+'$'); cont = input()
      for c in range(0, v.shape[0]):
        plt.plot(times, v[c,:,qn], label=label+ str(c)+')$')
#        if u_label == '$u_{1y}':
#          print(sens_type+' ('+str(v.shape[0])+'): u_label = '+u_label+'$ for c = '+str(c)+', and qn = '+str(qn))
#          print(times, v[c,:,qn])
      plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
      plt.legend(loc="lower left",fontsize=nfs)
      plt.savefig(savedir+'/png/'+fn+'.png', format='png', dpi=750)
      plt.savefig(savedir+'/eps/'+fn+'.eps', format='eps', dpi=1000)
      plt.grid(True)
      plt.clf()
      plt.close()

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   

# is n being used correctly in the below? And for mics? and times array in sensor plot routine?

  # obtain and plot the accelerometer readings (retarded by one time step in files)
  def postpro_accl_readings(self, tn, dt, n, u, up, upp, finished, rr, Pv, savedir):
 
    if up == None:
      self.accl_u  = np.zeros((self.accl.shape[0],self.Nt-1, 9))
      self.accl_ux = np.zeros((self.accl.shape[0],self.Nt-1, 9))
      self.accl_uy = np.zeros((self.accl.shape[0],self.Nt-1, 9))
      self.accl_uz = np.zeros((self.accl.shape[0],self.Nt-1, 9))
    else:
      # move this away - it isn't symmeterised in the mics routine      !!!!!!!!!!!!!!!!!!!!!!!!
      self.tp.write("{0:10.3e}\n".format(tn-dt) )

    for c in range(0, self.accl.shape[0]):
      x,y,z = self.accl[c]
      if up == None:
         if c == 0:
           rr.write("Output accl data is retarded by dt; dt, 2*dt,...,T-dt data are recorded\n")
      else:
        u1,   u2,   u3      = u.split();
        up1,  up2,  up3     = up.split();
        upp1, upp2, upp3    = upp.split()
        self.accl_u[c,n,0]  = up1(x,y,z)
        self.accl_u[c,n,1]  = (u1(x,y,z)-upp1(x,y,z))/(2*dt)
        self.accl_u[c,n,2]  = (2*up1(x,y,z)-u1(x,y,z)-upp1(x,y,z))/(dt*dt)
        self.accl_u[c,n,3]  = up2(x,y,z)
        self.accl_u[c,n,4]  = (u2(x,y,z)-upp2(x,y,z))/(2*dt)
        self.accl_u[c,n,5]  = (2*up2(x,y,z)-u2(x,y,z)-upp2(x,y,z))/(dt*dt)
        self.accl_u[c,n,6]  = up3(x,y,z)
        self.accl_u[c,n,7]  = (u3(x,y,z)-upp3(x,y,z))/(2*dt)
        self.accl_u[c,n,8]  = (2*up3(x,y,z)-u3(x,y,z)-upp3(x,y,z))/(dt*dt)
        # x derivatives
        ux   = Function(Pv); ux   = project(  u.dx(0), Pv)
        upx  = Function(Pv); upx  = project( up.dx(0), Pv)
        uppx = Function(Pv); uppx = project(upp.dx(0), Pv)
        u1x,   u2x,   u3x   = ux.split();
        up1x,  up2x,  up3x  = upx.split();
        upp1x, upp2x, upp3x = uppx.split();
        self.accl_ux[c,n,0] = up1x(x,y,z)
        self.accl_ux[c,n,1] = (u1x(x,y,z)-upp1x(x,y,z))/(2*dt)
        self.accl_ux[c,n,2] = (2*up1x(x,y,z)-u1x(x,y,z)-upp1x(x,y,z))/(dt*dt)
        self.accl_ux[c,n,3] = up2x(x,y,z)
        self.accl_ux[c,n,4] = (u2x(x,y,z)-upp2x(x,y,z))/(2*dt)
        self.accl_ux[c,n,5] = (2*up2x(x,y,z)-u2x(x,y,z)-upp2x(x,y,z))/(dt*dt)
        self.accl_ux[c,n,6] = up3x(x,y,z)
        self.accl_ux[c,n,7] = (u3x(x,y,z)-upp3x(x,y,z))/(2*dt)
        self.accl_ux[c,n,8] = (2*up3x(x,y,z)-u3x(x,y,z)-upp3x(x,y,z))/(dt*dt)
        # y derivatives
        uy   = Function(Pv); uy   = project(  u.dx(1), Pv)
        upy  = Function(Pv); upy  = project( up.dx(1), Pv)
        uppy = Function(Pv); uppy = project(upp.dx(1), Pv)
        u1y,   u2y,   u3y   = uy.split();
        up1y,  up2y,  up3y  = upy.split();
        upp1y, upp2y, upp3y = uppy.split();
        self.accl_uy[c,n,0] = up1y(x,y,z)
        self.accl_uy[c,n,1] = (u1y(x,y,z)-upp1y(x,y,z))/(2*dt)
        self.accl_uy[c,n,2] = (2*up1y(x,y,z)-u1y(x,y,z)-upp1y(x,y,z))/(dt*dt)
        self.accl_uy[c,n,3] = up2y(x,y,z)
        self.accl_uy[c,n,4] = (u2y(x,y,z)-upp2y(x,y,z))/(2*dt)
        self.accl_uy[c,n,5] = (2*up2y(x,y,z)-u2y(x,y,z)-upp2y(x,y,z))/(dt*dt)
        self.accl_uy[c,n,6] = up3y(x,y,z)
        self.accl_uy[c,n,7] = (u3y(x,y,z)-upp3y(x,y,z))/(2*dt)
        self.accl_uy[c,n,8] = (2*up3y(x,y,z)-u3y(x,y,z)-upp3y(x,y,z))/(dt*dt)
        # z derivatives
        uz   = Function(Pv); uz   = project(  u.dx(2), Pv)
        upz  = Function(Pv); upz  = project( up.dx(2), Pv)
        uppz = Function(Pv); uppz = project(upp.dx(2), Pv)
        u1z,   u2z,   u3z   = uz.split();
        up1z,  up2z,  up3z  = upz.split();
        upp1z, upp2z, upp3z = uppz.split();
        self.accl_uz[c,n,0] = up1z(x,y,z)
        self.accl_uz[c,n,1] = (u1z(x,y,z)-upp1z(x,y,z))/(2*dt)
        self.accl_uz[c,n,2] = (2*up1z(x,y,z)-u1z(x,y,z)-upp1z(x,y,z))/(dt*dt)
        self.accl_uz[c,n,3] = up2z(x,y,z)
        self.accl_uz[c,n,4] = (u2z(x,y,z)-upp2z(x,y,z))/(2*dt)
        self.accl_uz[c,n,5] = (2*up2z(x,y,z)-u2z(x,y,z)-upp2z(x,y,z))/(dt*dt)
        self.accl_uz[c,n,6] = up3z(x,y,z)
        self.accl_uz[c,n,7] = (u3z(x,y,z)-upp3z(x,y,z))/(2*dt)
        self.accl_uz[c,n,8] = (2*up3z(x,y,z)-u3z(x,y,z)-upp3z(x,y,z))/(dt*dt)

        # write files when data is complete
        if c == self.accl.shape[0]-1:
          for col in range(0, self.accl.shape[0]):
            # displacements components
            bd.u1_a.write   ("{0:10.3e} ".format( self.accl_u[col,n,0]))
            bd.u1t_a.write  ("{0:10.3e} ".format( self.accl_u[col,n,1]))
            bd.u1tt_a.write ("{0:10.3e} ".format( self.accl_u[col,n,2]))
            bd.u2_a.write   ("{0:10.3e} ".format( self.accl_u[col,n,3]))
            bd.u2t_a.write  ("{0:10.3e} ".format( self.accl_u[col,n,4]))
            bd.u2tt_a.write ("{0:10.3e} ".format( self.accl_u[col,n,5]))
            bd.u3_a.write   ("{0:10.3e} ".format( self.accl_u[col,n,6]))
            bd.u3t_a.write  ("{0:10.3e} ".format( self.accl_u[col,n,7]))
            bd.u3tt_a.write ("{0:10.3e} ".format( self.accl_u[col,n,8]))
            # x-derivative components
            bd.u1x_a.write  ("{0:10.3e} ".format(self.accl_ux[col,n,0]))
            bd.u1xt_a.write ("{0:10.3e} ".format(self.accl_ux[col,n,1]))
            bd.u1xtt_a.write("{0:10.3e} ".format(self.accl_ux[col,n,2]))
            bd.u2x_a.write  ("{0:10.3e} ".format(self.accl_ux[col,n,3]))
            bd.u2xt_a.write ("{0:10.3e} ".format(self.accl_ux[col,n,4]))
            bd.u2xtt_a.write("{0:10.3e} ".format(self.accl_ux[col,n,5]))
            bd.u3x_a.write  ("{0:10.3e} ".format(self.accl_ux[col,n,6]))
            bd.u3xt_a.write ("{0:10.3e} ".format(self.accl_ux[col,n,7]))
            bd.u3xtt_a.write("{0:10.3e} ".format(self.accl_ux[col,n,8]))
            # y-derivative components
            bd.u1y_a.write  ("{0:10.3e} ".format(self.accl_uy[col,n,0]))
            bd.u1yt_a.write ("{0:10.3e} ".format(self.accl_uy[col,n,1]))
            bd.u1ytt_a.write("{0:10.3e} ".format(self.accl_uy[col,n,2]))
            bd.u2y_a.write  ("{0:10.3e} ".format(self.accl_uy[col,n,3]))
            bd.u2yt_a.write ("{0:10.3e} ".format(self.accl_uy[col,n,4]))
            bd.u2ytt_a.write("{0:10.3e} ".format(self.accl_uy[col,n,5]))
            bd.u3y_a.write  ("{0:10.3e} ".format(self.accl_uy[col,n,6]))
            bd.u3yt_a.write ("{0:10.3e} ".format(self.accl_uy[col,n,7]))
            bd.u3ytt_a.write("{0:10.3e} ".format(self.accl_uy[col,n,8]))
            # z-derivative components
            bd.u1z_a.write  ("{0:10.3e} ".format(self.accl_uz[col,n,0]))
            bd.u1zt_a.write ("{0:10.3e} ".format(self.accl_uz[col,n,1]))
            bd.u1ztt_a.write("{0:10.3e} ".format(self.accl_uz[col,n,2]))
            bd.u2z_a.write  ("{0:10.3e} ".format(self.accl_uz[col,n,3]))
            bd.u2zt_a.write ("{0:10.3e} ".format(self.accl_uz[col,n,4]))
            bd.u2ztt_a.write("{0:10.3e} ".format(self.accl_uz[col,n,5]))
            bd.u3z_a.write  ("{0:10.3e} ".format(self.accl_uz[col,n,6]))
            bd.u3zt_a.write ("{0:10.3e} ".format(self.accl_uz[col,n,7]))
            bd.u3ztt_a.write("{0:10.3e} ".format(self.accl_uz[col,n,8]))

          bd.u1_a.write("\n");  bd.u1t_a.write("\n");  bd.u1tt_a.write("\n")
          bd.u2_a.write("\n");  bd.u2t_a.write("\n");  bd.u2tt_a.write("\n")
          bd.u3_a.write("\n");  bd.u3t_a.write("\n");  bd.u3tt_a.write("\n")
          bd.u1x_a.write("\n"); bd.u1xt_a.write("\n"); bd.u1xtt_a.write("\n")
          bd.u2x_a.write("\n"); bd.u2xt_a.write("\n"); bd.u2xtt_a.write("\n")
          bd.u3x_a.write("\n"); bd.u3xt_a.write("\n"); bd.u3xtt_a.write("\n")
          bd.u1y_a.write("\n"); bd.u1yt_a.write("\n"); bd.u1ytt_a.write("\n")
          bd.u2y_a.write("\n"); bd.u2yt_a.write("\n"); bd.u2ytt_a.write("\n")
          bd.u3y_a.write("\n"); bd.u3yt_a.write("\n"); bd.u3ytt_a.write("\n")
          bd.u1z_a.write("\n"); bd.u1zt_a.write("\n"); bd.u1ztt_a.write("\n")
          bd.u2z_a.write("\n"); bd.u2zt_a.write("\n"); bd.u2ztt_a.write("\n")
          bd.u3z_a.write("\n"); bd.u3zt_a.write("\n"); bd.u3ztt_a.write("\n")
  
    if finished and self.gfx:
      # first for u etc
      self.plot_sensor_trace(savedir,'u1_accl',  '$u_{1}',     'accelerometers',self.accl_u,qn=0)
      self.plot_sensor_trace(savedir,'u1t_accl', '$u_{1t}',    'accelerometers',self.accl_u,qn=1)
      self.plot_sensor_trace(savedir,'u1tt_accl','$u_{1tt}',   'accelerometers',self.accl_u,qn=2)
      self.plot_sensor_trace(savedir,'u2_accl',  '$u_{2}',     'accelerometers',self.accl_u,qn=3)
      self.plot_sensor_trace(savedir,'u2t_accl', '$u_{2t}',    'accelerometers',self.accl_u,qn=4)
      self.plot_sensor_trace(savedir,'u2tt_accl','$u_{2tt}',   'accelerometers',self.accl_u,qn=5)
      self.plot_sensor_trace(savedir,'u3_accl',  '$u_{3}',     'accelerometers',self.accl_u,qn=6)
      self.plot_sensor_trace(savedir,'u3t_accl', '$u_{3t}',    'accelerometers',self.accl_u,qn=7)
      self.plot_sensor_trace(savedir,'u3tt_accl','$u_{3tt}',   'accelerometers',self.accl_u,qn=8)
      # for u_x etc..
      self.plot_sensor_trace(savedir,'u1x_accl',  '$u_{1x}',   'accelerometers',self.accl_ux,qn=0)
      self.plot_sensor_trace(savedir,'u1xt_accl', '$u_{1xt}',  'accelerometers',self.accl_ux,qn=1)
      self.plot_sensor_trace(savedir,'u1xtt_accl','$u_{1xtt}', 'accelerometers',self.accl_ux,qn=2)
      self.plot_sensor_trace(savedir,'u2x_accl',  '$u_{2x}',   'accelerometers',self.accl_ux,qn=3)
      self.plot_sensor_trace(savedir,'u2xt_accl', '$u_{2xt}',  'accelerometers',self.accl_ux,qn=4)
      self.plot_sensor_trace(savedir,'u2xtt_accl','$u_{2xtt}', 'accelerometers',self.accl_ux,qn=5)
      self.plot_sensor_trace(savedir,'u3x_accl',  '$u_{3x}',   'accelerometers',self.accl_ux,qn=6)
      self.plot_sensor_trace(savedir,'u3xt_accl', '$u_{3xt}',  'accelerometers',self.accl_ux,qn=7)
      self.plot_sensor_trace(savedir,'u3xtt_accl','$u_{3xtt}', 'accelerometers',self.accl_ux,qn=8)
      # for u_y etc..
      self.plot_sensor_trace(savedir,'u1y_accl',  '$u_{1y}',   'accelerometers',self.accl_uy,qn=0)
      self.plot_sensor_trace(savedir,'u1yt_accl', '$u_{1yt}',  'accelerometers',self.accl_uy,qn=1)
      self.plot_sensor_trace(savedir,'u1ytt_accl','$u_{1ytt}', 'accelerometers',self.accl_uy,qn=2)
      self.plot_sensor_trace(savedir,'u2y_accl',  '$u_{2y}',   'accelerometers',self.accl_uy,qn=3)
      self.plot_sensor_trace(savedir,'u2yt_accl', '$u_{2yt}',  'accelerometers',self.accl_uy,qn=4)
      self.plot_sensor_trace(savedir,'u2ytt_accl','$u_{2ytt}', 'accelerometers',self.accl_uy,qn=5)
      self.plot_sensor_trace(savedir,'u3y_accl',  '$u_{3y}',   'accelerometers',self.accl_uy,qn=6)
      self.plot_sensor_trace(savedir,'u3yt_accl', '$u_{3yt}',  'accelerometers',self.accl_uy,qn=7)
      self.plot_sensor_trace(savedir,'u3ytt_accl','$u_{3ytt}', 'accelerometers',self.accl_uy,qn=8)
      # for u_z etc..
      self.plot_sensor_trace(savedir,'u1z_accl',  '$u_{1z}',   'accelerometers',self.accl_uz,qn=0)
      self.plot_sensor_trace(savedir,'u1zt_accl', '$u_{1zt}',  'accelerometers',self.accl_uz,qn=1)
      self.plot_sensor_trace(savedir,'u1ztt_accl','$u_{1ztt}', 'accelerometers',self.accl_uz,qn=2)
      self.plot_sensor_trace(savedir,'u2z_accl',  '$u_{2z}',   'accelerometers',self.accl_uz,qn=3)
      self.plot_sensor_trace(savedir,'u2zt_accl', '$u_{2zt}',  'accelerometers',self.accl_uz,qn=4)
      self.plot_sensor_trace(savedir,'u2ztt_accl','$u_{2ztt}', 'accelerometers',self.accl_uz,qn=5)
      self.plot_sensor_trace(savedir,'u3z_accl',  '$u_{3z}',   'accelerometers',self.accl_uz,qn=6)
      self.plot_sensor_trace(savedir,'u3zt_accl', '$u_{3zt}',  'accelerometers',self.accl_uz,qn=7)
      self.plot_sensor_trace(savedir,'u3ztt_accl','$u_{3ztt}', 'accelerometers',self.accl_uz,qn=8)

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   

  # obtain and plot the microphone readings (retarded by one time step in files)
  def postpro_mics_readings(self, tn, dt, n, u, up, upp, finished, rr, Pv, savedir):
    print('GOT HERE 1')
    if up == None:
      self.mics_u  = np.zeros((self.mics.shape[0],self.Nt-1, 9))
      self.mics_ux = np.zeros((self.mics.shape[0],self.Nt-1, 9))
      self.mics_uy = np.zeros((self.mics.shape[0],self.Nt-1, 9))
      self.mics_uz = np.zeros((self.mics.shape[0],self.Nt-1, 9))
    # This not needed if the accl routine has done it. !!!!!!!!!!!
    # BUT we should check that it is getting done somewhere. !!!!!!!!!!!
    #else:
      #self.tp.write("{0:10.3e}\n".format(tn-dt) )

    # loop over each sensor, get its location and read off the traces
    print('GOT HERE 2')
    for c in range(0, self.mics.shape[0]):
      x,y,z = self.mics[c]
      if up == None:
         if c == 0:
           rr.write("Output mics data is retarded by dt; dt, 2*dt,...,T-dt data are recorded\n")
         print('GOT HERE 3')
      else:
        print('GOT HERE 4')
        u1,   u2,   u3      = u.split();
        up1,  up2,  up3     = up.split();
        upp1, upp2, upp3    = upp.split()
        # component values and t derivatives
        self.mics_u[c,n,0]  = up1(x,y,z)
        self.mics_u[c,n,1]  = (u1(x,y,z)-upp1(x,y,z))/(2*dt)
        self.mics_u[c,n,2]  = (2*up1(x,y,z)-u1(x,y,z)-upp1(x,y,z))/(dt*dt)
        self.mics_u[c,n,3]  = up2(x,y,z)
        self.mics_u[c,n,4]  = (u2(x,y,z)-upp2(x,y,z))/(2*dt)
        self.mics_u[c,n,5]  = (2*up2(x,y,z)-u2(x,y,z)-upp2(x,y,z))/(dt*dt)
        self.mics_u[c,n,6]  = up3(x,y,z)
        self.mics_u[c,n,7]  = (u3(x,y,z)-upp3(x,y,z))/(2*dt)
        self.mics_u[c,n,8]  = (2*up3(x,y,z)-u3(x,y,z)-upp3(x,y,z))/(dt*dt)
        # x derivatives: use dx(0)
        ux   = Function(Pv); ux   = project(  u.dx(0), Pv)
        upx  = Function(Pv); upx  = project( up.dx(0), Pv)
        uppx = Function(Pv); uppx = project(upp.dx(0), Pv)
        u1x,   u2x,   u3x   = ux.split();
        up1x,  up2x,  up3x  = upx.split();
        upp1x, upp2x, upp3x = uppx.split();
        self.mics_ux[c,n,0] = up1x(x,y,z)
        self.mics_ux[c,n,1] = (u1x(x,y,z)-upp1x(x,y,z))/(2*dt)
        self.mics_ux[c,n,2] = (2*up1x(x,y,z)-u1x(x,y,z)-upp1x(x,y,z))/(dt*dt)
        self.mics_ux[c,n,3] = up2x(x,y,z)
        self.mics_ux[c,n,4] = (u2x(x,y,z)-upp2x(x,y,z))/(2*dt)
        self.mics_ux[c,n,5] = (2*up2x(x,y,z)-u2x(x,y,z)-upp2x(x,y,z))/(dt*dt)
        self.mics_ux[c,n,6] = up3x(x,y,z)
        self.mics_ux[c,n,7] = (u3x(x,y,z)-upp3x(x,y,z))/(2*dt)
        self.mics_ux[c,n,8] = (2*up3x(x,y,z)-u3x(x,y,z)-upp3x(x,y,z))/(dt*dt)
        # y derivatives: use dx(1)
        uy   = Function(Pv); uy   = project(  u.dx(1), Pv)
        upy  = Function(Pv); upy  = project( up.dx(1), Pv)
        uppy = Function(Pv); uppy = project(upp.dx(1), Pv)
        u1y,   u2y,   u3y   = uy.split();
        up1y,  up2y,  up3y  = upy.split();
        upp1y, upp2y, upp3y = uppy.split();
        self.mics_uy[c,n,0] = up1y(x,y,z)
        self.mics_uy[c,n,1] = (u1y(x,y,z)-upp1y(x,y,z))/(2*dt)
        self.mics_uy[c,n,2] = (2*up1y(x,y,z)-u1y(x,y,z)-upp1y(x,y,z))/(dt*dt)
        self.mics_uy[c,n,3] = up2y(x,y,z)
        self.mics_uy[c,n,4] = (u2y(x,y,z)-upp2y(x,y,z))/(2*dt)
        self.mics_uy[c,n,5] = (2*up2y(x,y,z)-u2y(x,y,z)-upp2y(x,y,z))/(dt*dt)
        self.mics_uy[c,n,6] = up3y(x,y,z)
        self.mics_uy[c,n,7] = (u3y(x,y,z)-upp3y(x,y,z))/(2*dt)
        self.mics_uy[c,n,8] = (2*up3y(x,y,z)-u3y(x,y,z)-upp3y(x,y,z))/(dt*dt)
        # z derivatives: use dx(2)
        uz   = Function(Pv); uz   = project(  u.dx(2), Pv)
        upz  = Function(Pv); upz  = project( up.dx(2), Pv)
        uppz = Function(Pv); uppz = project(upp.dx(2), Pv)
        u1z,   u2z,   u3z   = uz.split();
        up1z,  up2z,  up3z  = upz.split();
        upp1z, upp2z, upp3z = uppz.split();
        self.mics_uz[c,n,0] = up1z(x,y,z)
        self.mics_uz[c,n,1] = (u1z(x,y,z)-upp1z(x,y,z))/(2*dt)
        self.mics_uz[c,n,2] = (2*up1z(x,y,z)-u1z(x,y,z)-upp1z(x,y,z))/(dt*dt)
        self.mics_uz[c,n,3] = up2z(x,y,z)
        self.mics_uz[c,n,4] = (u2z(x,y,z)-upp2z(x,y,z))/(2*dt)
        self.mics_uz[c,n,5] = (2*up2z(x,y,z)-u2z(x,y,z)-upp2z(x,y,z))/(dt*dt)
        self.mics_uz[c,n,6] = up3z(x,y,z)
        self.mics_uz[c,n,7] = (u3z(x,y,z)-upp3z(x,y,z))/(2*dt)
        self.mics_uz[c,n,8] = (2*up3z(x,y,z)-u3z(x,y,z)-upp3z(x,y,z))/(dt*dt)

        # write files when data is complete
        if c == self.mics.shape[0]-1:
          for col in range(0, self.mics.shape[0]):
            # displacements components
            bd.u1_m.write  ("{0:10.3e} ".format(self.mics_u[col,n,0]))
            bd.u1t_m.write ("{0:10.3e} ".format(self.mics_u[col,n,1]))
            bd.u1tt_m.write("{0:10.3e} ".format(self.mics_u[col,n,2]))
            bd.u2_m.write  ("{0:10.3e} ".format(self.mics_u[col,n,3]))
            bd.u2t_m.write ("{0:10.3e} ".format(self.mics_u[col,n,4]))
            bd.u2tt_m.write("{0:10.3e} ".format(self.mics_u[col,n,5]))
            bd.u3_m.write  ("{0:10.3e} ".format(self.mics_u[col,n,6]))
            bd.u3t_m.write ("{0:10.3e} ".format(self.mics_u[col,n,7]))
            bd.u3tt_m.write("{0:10.3e} ".format(self.mics_u[col,n,8]))
            # x-derivative components
            bd.u1x_m.write  ("{0:10.3e} ".format(self.mics_ux[col,n,0]))
            bd.u1xt_m.write ("{0:10.3e} ".format(self.mics_ux[col,n,1]))
            bd.u1xtt_m.write("{0:10.3e} ".format(self.mics_ux[col,n,2]))
            bd.u2x_m.write  ("{0:10.3e} ".format(self.mics_ux[col,n,3]))
            bd.u2xt_m.write ("{0:10.3e} ".format(self.mics_ux[col,n,4]))
            bd.u2xtt_m.write("{0:10.3e} ".format(self.mics_ux[col,n,5]))
            bd.u3x_m.write  ("{0:10.3e} ".format(self.mics_ux[col,n,6]))
            bd.u3xt_m.write ("{0:10.3e} ".format(self.mics_ux[col,n,7]))
            bd.u3xtt_m.write("{0:10.3e} ".format(self.mics_ux[col,n,8]))
            # y-derivative components
            bd.u1y_m.write  ("{0:10.3e} ".format(self.mics_uy[col,n,0]))
            bd.u1yt_m.write ("{0:10.3e} ".format(self.mics_uy[col,n,1]))
            bd.u1ytt_m.write("{0:10.3e} ".format(self.mics_uy[col,n,2]))
            bd.u2y_m.write  ("{0:10.3e} ".format(self.mics_uy[col,n,3]))
            bd.u2yt_m.write ("{0:10.3e} ".format(self.mics_uy[col,n,4]))
            bd.u2ytt_m.write("{0:10.3e} ".format(self.mics_uy[col,n,5]))
            bd.u3y_m.write  ("{0:10.3e} ".format(self.mics_uy[col,n,6]))
            bd.u3yt_m.write ("{0:10.3e} ".format(self.mics_uy[col,n,7]))
            bd.u3ytt_m.write("{0:10.3e} ".format(self.mics_uy[col,n,8]))
            # z-derivative components
            bd.u1z_m.write  ("{0:10.3e} ".format(self.mics_uz[col,n,0]))
            bd.u1zt_m.write ("{0:10.3e} ".format(self.mics_uz[col,n,1]))
            bd.u1ztt_m.write("{0:10.3e} ".format(self.mics_uz[col,n,2]))
            bd.u2z_m.write  ("{0:10.3e} ".format(self.mics_uz[col,n,3]))
            bd.u2zt_m.write ("{0:10.3e} ".format(self.mics_uz[col,n,4]))
            bd.u2ztt_m.write("{0:10.3e} ".format(self.mics_uz[col,n,5]))
            bd.u3z_m.write  ("{0:10.3e} ".format(self.mics_uz[col,n,6]))
            bd.u3zt_m.write ("{0:10.3e} ".format(self.mics_uz[col,n,7]))
            bd.u3ztt_m.write("{0:10.3e} ".format(self.mics_uz[col,n,8]))
          bd.u1_m.write("\n");  bd.u1t_m.write("\n");  bd.u1tt_m.write("\n")
          bd.u2_m.write("\n");  bd.u2t_m.write("\n");  bd.u2tt_m.write("\n")
          bd.u3_m.write("\n");  bd.u3t_m.write("\n");  bd.u3tt_m.write("\n")
          bd.u1x_m.write("\n"); bd.u1xt_m.write("\n"); bd.u1xtt_m.write("\n")
          bd.u2x_m.write("\n"); bd.u2xt_m.write("\n"); bd.u2xtt_m.write("\n")
          bd.u3x_m.write("\n"); bd.u3xt_m.write("\n"); bd.u3xtt_m.write("\n")
          bd.u1y_m.write("\n"); bd.u1yt_m.write("\n"); bd.u1ytt_m.write("\n")
          bd.u2y_m.write("\n"); bd.u2yt_m.write("\n"); bd.u2ytt_m.write("\n")
          bd.u3y_m.write("\n"); bd.u3yt_m.write("\n"); bd.u3ytt_m.write("\n")
          bd.u1z_m.write("\n"); bd.u1zt_m.write("\n"); bd.u1ztt_m.write("\n")
          bd.u2z_m.write("\n"); bd.u2zt_m.write("\n"); bd.u2ztt_m.write("\n")
          bd.u3z_m.write("\n"); bd.u3zt_m.write("\n"); bd.u3ztt_m.write("\n")

    if finished and self.gfx:
      # first for u etc
      self.plot_sensor_trace(savedir,'u1_mics',  '$u_{1}',     'microphones',self.mics_u,qn=0)
      self.plot_sensor_trace(savedir,'u1t_mics', '$u_{1t}',    'microphones',self.mics_u,qn=1)
      self.plot_sensor_trace(savedir,'u1tt_mics','$u_{1tt}',   'microphones',self.mics_u,qn=2)
      self.plot_sensor_trace(savedir,'u2_mics',  '$u_{2}',     'microphones',self.mics_u,qn=3)
      self.plot_sensor_trace(savedir,'u2t_mics', '$u_{2t}',    'microphones',self.mics_u,qn=4)
      self.plot_sensor_trace(savedir,'u2tt_mics','$u_{2tt}',   'microphones',self.mics_u,qn=5)
      self.plot_sensor_trace(savedir,'u3_mics',  '$u_{3}',     'microphones',self.mics_u,qn=6)
      self.plot_sensor_trace(savedir,'u3t_mics', '$u_{3t}',    'microphones',self.mics_u,qn=7)
      self.plot_sensor_trace(savedir,'u3tt_mics','$u_{3tt}',   'microphones',self.mics_u,qn=8)
      # for u_x etc..
      self.plot_sensor_trace(savedir,'u1x_mics',  '$u_{1x}',   'microphones',self.mics_ux,qn=0)
      self.plot_sensor_trace(savedir,'u1xt_mics', '$u_{1xt}',  'microphones',self.mics_ux,qn=1)
      self.plot_sensor_trace(savedir,'u1xtt_mics','$u_{1xtt}', 'microphones',self.mics_ux,qn=2)
      self.plot_sensor_trace(savedir,'u2x_mics',  '$u_{2x}',   'microphones',self.mics_ux,qn=3)
      self.plot_sensor_trace(savedir,'u2xt_mics', '$u_{2xt}',  'microphones',self.mics_ux,qn=4)
      self.plot_sensor_trace(savedir,'u2xtt_mics','$u_{2xtt}', 'microphones',self.mics_ux,qn=5)
      self.plot_sensor_trace(savedir,'u3x_mics',  '$u_{3x}',   'microphones',self.mics_ux,qn=6)
      self.plot_sensor_trace(savedir,'u3xt_mics', '$u_{3xt}',  'microphones',self.mics_ux,qn=7)
      self.plot_sensor_trace(savedir,'u3xtt_mics','$u_{3xtt}', 'microphones',self.mics_ux,qn=8)
      # for u_y etc..
      self.plot_sensor_trace(savedir,'u1y_mics',  '$u_{1y}',   'microphones',self.mics_uy,qn=0)
      self.plot_sensor_trace(savedir,'u1yt_mics', '$u_{1yt}',  'microphones',self.mics_uy,qn=1)
      self.plot_sensor_trace(savedir,'u1ytt_mics','$u_{1ytt}', 'microphones',self.mics_uy,qn=2)
      self.plot_sensor_trace(savedir,'u2y_mics',  '$u_{2y}',   'microphones',self.mics_uy,qn=3)
      self.plot_sensor_trace(savedir,'u2yt_mics', '$u_{2yt}',  'microphones',self.mics_uy,qn=4)
      self.plot_sensor_trace(savedir,'u2ytt_mics','$u_{2ytt}', 'microphones',self.mics_uy,qn=5)
      self.plot_sensor_trace(savedir,'u3y_mics',  '$u_{3y}',   'microphones',self.mics_uy,qn=6)
      self.plot_sensor_trace(savedir,'u3yt_mics', '$u_{3yt}',  'microphones',self.mics_uy,qn=7)
      self.plot_sensor_trace(savedir,'u3ytt_mics','$u_{3ytt}', 'microphones',self.mics_uy,qn=8)
      # for u_z etc..
      self.plot_sensor_trace(savedir,'u1z_mics',  '$u_{1z}',   'microphones',self.mics_uz,qn=0)
      self.plot_sensor_trace(savedir,'u1zt_mics', '$u_{1zt}',  'microphones',self.mics_uz,qn=1)
      self.plot_sensor_trace(savedir,'u1ztt_mics','$u_{1ztt}', 'microphones',self.mics_uz,qn=2)
      self.plot_sensor_trace(savedir,'u2z_mics',  '$u_{2z}',   'microphones',self.mics_uz,qn=3)
      self.plot_sensor_trace(savedir,'u2zt_mics', '$u_{2zt}',  'microphones',self.mics_uz,qn=4)
      self.plot_sensor_trace(savedir,'u2ztt_mics','$u_{2ztt}', 'microphones',self.mics_uz,qn=5)
      self.plot_sensor_trace(savedir,'u3z_mics',  '$u_{3z}',   'microphones',self.mics_uz,qn=6)
      self.plot_sensor_trace(savedir,'u3zt_mics', '$u_{3zt}',  'microphones',self.mics_uz,qn=7)
      self.plot_sensor_trace(savedir,'u3ztt_mics','$u_{3ztt}', 'microphones',self.mics_uz,qn=8)

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   

  # manage the solution
  def solver(self,xc,yc,zc,rr,savedir):
    # set up the load with the current 'location' and 'width'
    self.f.xc = xc; self.f.yc = yc; self.f.zc = zc; 
    # acquire the mesh and set up the essential boundary condition
    mesh = BoxMesh(Point(self.xL, self.yL, self.zL),
                   Point(self.xH, self.yH, self.zH), self.Nx, self.Ny, self.Nz)
    # Define finite element space for displacement, velocity and acceleration; and a gradient space
    V = VectorFunctionSpace(mesh, "CG", self.rdeg)
    W = VectorFunctionSpace(mesh, "DG", self.rdeg-1)  # for gradients
    xdim = 3;
    TV = TensorFunctionSpace(mesh, "DG", self.Rdeg-1, shape=(xdim, xdim))

    # Define strain and stress
    def epsilon(u):
      return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    def sigma(u):
      return self.lmbda*div(u)*Identity(xdim) + 2*self.mu*epsilon(u)
    def bottom(x, on_boundary):
      return on_boundary and near(x[2],self.zL)
    consistency_checking = 0
    if consistency_checking > 0:
      from ihcc_linear_elastostatic_exact_data import lin_lin_3D as pdedata
      #from ihcc_linear_elastostatic_exact_data import non_lin_3D as pdedata
      coeffs = {"a":1,"b":1,"c":1}
      ux     = pdedata.u(   coeffs, degree=self.Rdeg, t=0,                              element=V.ufl_element())
      epsx   = pdedata.epsx(coeffs, degree=self.Rdeg, t=0,                              element=TV.ufl_element())
      sigx   = pdedata.sigx(coeffs, degree=self.Rdeg, t=0, lmbda=self.lmbda, G=self.mu, element=TV.ufl_element())
      self.f = pdedata.f(   coeffs, degree=self.Rdeg, t=0, lmbda=self.lmbda, G=self.mu, element=V.ufl_element())
#      T    = pdedata.g(   coeffs, degree=self.Rdeg, t=0, lmbda=self.lmbda, G=self.mu, element=V.ufl_element())
      bc_value = ux
      T = dot(sigx, FacetNormal(mesh))

    else:
      bc_value = Constant((0.0, 0.0, 0.0))
      T = Constant((0, 0, 0))
    
    # Set up boundary condition on the bottom face
    bc = DirichletBC(V, bc_value, bottom)
    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    d = u.geometric_dimension()  # space dimension
    a = inner(sigma(u), epsilon(v))*dx
    L = dot(self.f, v)*dx + dot(T, v)*ds

    # Compute solution through time
    u   = Function(V)
    up  = Function(V)
    upp = Function(V)
    for n in range(0,1+self.Nt):
      tn = n*self.dt
      if consistency_checking > 0:
        ux.t = tn; epsx.t = tn; sigx.t = tn; self.f.t = tn;
        bc = DirichletBC(V, ux, bottom)
        L = dot(self.f, v)*dx + dot(dot(sigx, FacetNormal(mesh)), v)*ds
      else:
        self.f.F = self.Fo if self.const>0 else self.f_mod(tn)

      solve(a == L, u, bc)
      
      if consistency_checking > 0:
        # compute errors
        # Define finite element space(s) for 'exact' solution and gradient
        VV = VectorFunctionSpace(mesh, "CG", self.Rdeg)    # dim of mesh is implied
        u_err_L2  = errornorm(ux, u, norm_type = "L2")
        u_err_H1  = errornorm(ux, u, norm_type = "H1")
        #epsh = epsilon(u); sigh = sigma(u)
        eps_err_L2 = sqrt( assemble(inner(epsilon(u)-epsx,epsilon(u)-epsx)*dx ) )
        sig_err_L2 = sqrt( assemble(inner(sigma(u)-sigx,sigma(u)-sigx)*dx ) )
        # elastic-energy norm
        uex = interpolate(ux, VV)
        #help(errornorm) says this might be unstable
        u_err_En = sqrt( assemble( inner(sigma(uex-u),epsilon(uex-u) )*dx()) )
        print((  '||u-uh||_0 = {0:10.3e}  '
                +'||u-uh||_1 = {1:10.3e}  '
                +'||u-uh||_V = {2:10.3e}  '
                +'||e-eh||_0 = {3:10.3e}  '
                +'||s-sh||_0 = {4:10.3e}\n').format(
                     u_err_L2, u_err_H1, u_err_En, eps_err_L2, sig_err_L2))
      else:
        if self.const == 1:
          print('GOT HERE C')
          # first call to allocate storage and intialise
          self.postpro_mics_readings(tn, self.dt, n, u, None, None, 0, rr, W, savedir)
          self.postpro_accl_readings(tn, self.dt, n, u, None, None, 0, rr, W, savedir)
          # second call to save static readings - the time differences will be zero and useless here
          self.postpro_mics_readings(tn, self.dt, n, u, u, u, 1, rr, W, savedir)
          self.postpro_accl_readings(tn, self.dt, n, u, u, u, 1, rr, W, savedir)
          break;
        else:
          if n == 0:
            print('GOT HERE A')
            self.postpro_mics_readings(tn, self.dt, n, u, None, None, 0, rr, W, savedir)
            self.postpro_accl_readings(tn, self.dt, n, u, None, None, 0, rr, W, savedir)
          # skip n=1 and n=Nt to get central difference of velocities: finish at self.Nt-1 
          elif n > 1 and n < 1+self.Nt:
            print('GOT HERE B')
            self.postpro_mics_readings(tn, self.dt, n-2, u, up, upp, 1 if n == self.Nt-0 else 0, rr, W, savedir)
            self.postpro_accl_readings(tn, self.dt, n-2, u, up, upp, 1 if n == self.Nt-0 else 0, rr, W, savedir)
        
      # prepare for next time
      upp.assign(up)
      up.assign(u)
      
      # check if a file called quitnow exists - if it does remove it and quit.
      if os.path.exists('quitnow') and os.path.isfile('quitnow'):
        print('\nA file called "quitnow" exists. Removing it and quitting')
        print('at stepping point n/Nt = {0:d}/{1:d}\n'.format(n,self.Nt))
        rr.write('\nA file called "quitnow" exists. Removing it and quitting\n')
        rr.write('at stepping point n/Nt = {0:d}/{1:d}\n\n'.format(n,self.Nt))
        os.remove('quitnow')
        exit(0)

    # Plot solution/mesh
    #plot(mesh, title='mesh', color="red", edgecolor="black"); plt.show()

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   

# plot the random source locations in the volume
# based on https://matplotlib.org/3.1.1/gallery/mplot3d/scatter3d.html
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
def save_xcyczc(xcyczc, bd, plotName, results_dir):
  # for runs on the server we wont be able to generate gfx
  if bd.gfx:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
   
    ax.auto_scale_xyz([0, 0.3], [-0.02, 0.02], [0.0, 0.04])
    xc,yc,zc = xcyczc[:,0], xcyczc[:,1], xcyczc[:,2]
    ax.scatter(xc, yc, zc, marker='o', s=2, c='b')
   
    # draw the accelerometers
    xa,ya,za = bd.accl[:,0], bd.accl[:,1], bd.accl[:,2]
    ax.scatter(xa, ya, za, marker='x', s=6, c='k', depthshade=False)
    # draw the microphones
    xm,ym,zm = bd.mics[:,0], bd.mics[:,1], bd.mics[:,2]
    ax.scatter(xm, ym, zm, marker='+', s=6, c='k', depthshade=False)
    # draw the block domain - x lines first
    ax.plot3D([bd.xL,bd.xH], [bd.yL,bd.yL], [bd.zL,bd.zL], 'gray', linewidth=1, linestyle='-')
    ax.plot3D([bd.xL,bd.xH], [bd.yH,bd.yH], [bd.zL,bd.zL], 'gray', linewidth=1, linestyle='-')
    ax.plot3D([bd.xL,bd.xH], [bd.yL,bd.yL], [bd.zH,bd.zH], 'gray', linewidth=1, linestyle='-')
    ax.plot3D([bd.xL,bd.xH], [bd.yH,bd.yH], [bd.zH,bd.zH], 'gray', linewidth=1, linestyle='-')
    # draw the block - y lines
    ax.plot3D([bd.xL,bd.xL], [bd.yL,bd.yH], [bd.zL,bd.zL], 'gray', linewidth=1, linestyle='-')
    ax.plot3D([bd.xL,bd.xL], [bd.yL,bd.yH], [bd.zH,bd.zH], 'gray', linewidth=1, linestyle='-')
    ax.plot3D([bd.xH,bd.xH], [bd.yL,bd.yH], [bd.zL,bd.zL], 'gray', linewidth=1, linestyle='-')
    ax.plot3D([bd.xH,bd.xH], [bd.yL,bd.yH], [bd.zH,bd.zH], 'gray', linewidth=1, linestyle='-')
    # draw the block - y lines
    ax.plot3D([bd.xL,bd.xL], [bd.yL,bd.yL], [bd.zL,bd.zH], 'gray', linewidth=1, linestyle='-')
    ax.plot3D([bd.xL,bd.xL], [bd.yH,bd.yH], [bd.zL,bd.zH], 'gray', linewidth=1, linestyle='-')
    ax.plot3D([bd.xH,bd.xH], [bd.yL,bd.yL], [bd.zL,bd.zH], 'gray', linewidth=1, linestyle='-')
    ax.plot3D([bd.xH,bd.xH], [bd.yH,bd.yH], [bd.zL,bd.zH], 'gray', linewidth=1, linestyle='-')
    
    # This was taken from the following in an attempt to get the right 'Bounding Box'
    # https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    #max_range = np.array([xc.max()-xc.min(), yc.max()-yc.min(), zc.max()-zc.min()]).max() / 2.0
    max_range = (bd.xH-bd.xL) / 2
    mid_x = (xc.max()+xc.min()) * 0.5
    mid_y = (yc.max()+yc.min()) * 0.5
    mid_z = (zc.max()+zc.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
   
    ax.view_init(20, 310)
    ax.set_axis_off()
    plt.savefig(results_dir+'/'+plotName+'.png', format='png', dpi=750)
    plt.savefig(results_dir+'/'+plotName+'.eps', format='eps', dpi=1000)
    plt.close()
  else:
    fwarn = open(results_dir+'/'+'xcyczc_warning.txt', 'w')
    fwarn.write('\nThe xcyczc plot could not be made - the gfx option was not given.\n')
    fwarn.write('The plot job needs to be offloaded to post processing using\n')
    fwarn.write('the position data in the xcyczc.txt file. Plot also the acclerometer\n')
    fwarn.write('and microphone positions from the accs.txt and mics.txt files.\n\n')
    fwarn.close()      
  
  # now write the file of random source points
  np.savetxt(results_dir+'/'+"xcyczc.txt", xcyczc, delimiter=' ', newline='\n')

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   

# help function
def usage():
  print("Solver methods available through coding used to be available with...")
  print("list_lu_solver_methods(); list_krylov_solver_methods(); list_krylov_solver_preconditioners()")
  print("\nCommand line options with n an integer and f a float:\n")
  print("-h       or --help")
  print("-v n     or --garrulous n   for verbosity")
  print("-b n     or --first n       integer label for the first of the batch runs (default 1, inclusive)")
  print("-B n     or --last n        integer label for the last of the batch runs (default 2, inclusive)")
  print("-C n     or --const n       run in constant load mode n (default 0, not constant)")
  print("                              n=0: use vibrating pulse, step through time")
  print("                              n=1: use linear elasticity, constant load, solve for t=0 only")
  print("-x       or --xc x          specify xc in source location N O T NORMALIZED (xc,yc,zc) (default, random)")
  print("-y       or --yc y          specify yc in source location N O T NORMALIZED (xc,yc,zc) (default, random)")
  print("-z       or --zc z          specify zc in source location N O T NORMALIZED (xc,yc,zc) (default, random)")
  print("-e e     or --eps e         specify source 'width' denominator epsilon (default, eps=0.001)")
  print("-F a     or --Fo a          specify source amplitude Fo (default, Fo=1000)")
  print("-g n     or --gfx  n        to ask for graphics every n'th batch run, or none if not given")
  print("-r n     or --rdeg n        to specify FE polynomial degree")
  print("-R n     or --Rdeg n        to specify 'exact solution' polynomial degree")
  print("-X n     or --example n     to specify the example to run")
  print("-N n     or --Nxyz n        to specify Nx = Ny = Nz")
  print("-n n     or --Nt n          to specify Nt")
  print("-T f     or --Tfinal f      to specify final time, T")
  print("-K       or --backup        make timestamped backup of this and postpro.py to ../backup/ and quit")
  print("-F       or --pdf           make enscript PDF of this and postpro.py")
  print(" ")
  print("            --Nx n          to specify Nx")
  print("            --Ny n          to specify Ny")
  print("            --Nz n          to specify Nz")
#  os.system('date +%Y_%m_%d_%H-%M-%S')
#  print(time.strftime("%d/%m/%Y at %H:%M:%S"))
  print('\n\nIf a file called "quitnow" is found, it is removed and termination is immediate\n')
  print('\nTypical run line (after a chmod u+x ./blockdata.py ./postpro.py):')
  print(' python ./blockdata.py -v 20 --Nt 20 --Nx 30 --Ny 10 --Nz 5 -b 300 -B 310 -g 5 | tee out.txt\n')

# controlling routine
if __name__ == '__main__':

  # initialise for a batch of runs, we can alter class defaults here,
  bd = blockdata(T=2.0, eps=0.001, Nt=10, Nx=15, Ny=5, Nz=3)
  # ... then set up the batch defaults,
  beingloud = 0
  gfx_step = 0
  # where we give these runs labels (end_run doesn't get executed due to range() )
  start_run=200; end_run=202
  xc=0; yc=0; zc=0; xcyczc_is_random = 1;
  
  # ... and now parse the command line to alter these defaults again if need be
  try:
    opts, args = getopt.getopt(sys.argv[1:], "hv:b:B:C:x:y:z:e:F:g:r:R:N:n:T:X:KF",
                   [
                    "help"         ,  # obvious
                    "garrulous="   ,  # level of verbosity
                    "first="       ,  # integer for first batch run label (default 1, inclusive)
                    "last="        ,  # integer for last batch run label (default 2, inclusive)
                    "const="       ,  # run in constant load mode n (default 0, not constant)
                    "xc="          ,  # specify xc in source location (xc,yc,zc) (default, random)
                    "yc="          ,  # specify yc in source location (xc,yc,zc) (default, random)
                    "zc="          ,  # specify zc in source location (xc,yc,zc) (default, random)
                    "eps="         ,  # specify source 'width' denominator epsilon (default, eps=0.001)
                    "Fo="          ,  # specify source amplitude Fo (default, Fo=1000)
                    "gfx="         ,  # if non-zero d generate gfx every d-th batch run 
                    "rdeg="        ,  # FE polynomial degree
                    "Rdeg="        ,  # solution polynomial degree
                    "Nxyz="        ,  # set Nx = Ny = Nz to the given argument 
                    "Nx="          ,  # set Nx to the given argument 
                    "Ny="          ,  # set Ny to the given argument 
                    "Nz="          ,  # set Nz to the given argument 
                    "Nt="          ,  # set Nt to the given argument
                    "Tfinal="      ,  # final time, T
                    "example="     ,  # integer for example number
                    "backup"       ,  # only make timestamped backup of this and postpro.py to ../backup/
                    "pdf"             # make enscript PDF of this and postpro.py
                    ])

  except getopt.GetoptError as err:
    # print help information and exit:
    print(err) # will print something like "option -a not recognized"
    usage()
    sys.exit(2)

  # get the command line modifications and ensure consistency of class variables 
  for o, a in opts:
    if o in ("-v","--garrulous"):
      beingloud = int(a)
      if beingloud > 19: print('Command Line: using: ')
      if beingloud > 19: print('loud level %d;' % beingloud),
    elif o in ("-h", "--help"):
      usage()
      sys.exit()
    elif o in ("-b", "--first"):
      start_run = int(a)
      if beingloud > 19: print('start_run = %d  (inclusive);' % start_run),
    elif o in ("-B", "--last"):
      end_run = 1+int(a)
      if beingloud > 19: print('end_run = %d (not inclusive);' % end_run),
    elif o in ("-C", "--const"):
      bd.const = int(a)
      if beingloud > 19: print('constant mode = %d;' % bd.const),
    elif o in ("-x", "--xc"):
      xc = float(a); xcyczc_is_random = 0
      if beingloud > 19: print('xc = %f;' % xc),
    elif o in ("-y", "--yc"):
      yc = float(a); xcyczc_is_random = 0
      if beingloud > 19: print('yc = %f;' % yc),
    elif o in ("-z", "--zc"):
      zc = float(a); xcyczc_is_random = 0
      if beingloud > 19: print('zc = %f;' % zc),
    elif o in ("-e", "--eps"):
      bd.eps = float(a)
      if beingloud > 19: print('eps = %f;' % bd.eps),
    elif o in ("-F", "--Fo"):
      bd.Fo = float(a)
      if beingloud > 19: print('Fo = %f;' % bd.Fo),
    elif o in ("-g", "--gfx"):
      gfx_step = int(a)
      if beingloud > 19: print('gfx_step = %d;' % gfx_step),
    elif o in ("-r", "--rdeg"):
      bd.rdeg = int(a)
      if beingloud > 19: print('rdeg = %d;' % bd.rdeg),
    elif o in ("-R", "--Rdeg"):
      bd.Rdeg = int(a)
      if beingloud > 19: print('Rdeg = %d;' % bd.Rdeg),
    elif o in ("-N", "--Nxyz"):
      bd.Nx = bd.Ny = bd.Nz = int(a)
      if beingloud > 19: print('Nx, Ny, Nz = %d, %d, %d;' % (bd.Nx,bd.Ny,bd.Nz) ),
    elif o in ("--Nx"):
      bd.Nx = int(a)
      if beingloud > 19: print('Nx = %d' % bd.Nx),
    elif o in ("--Ny"):
      bd.Ny = int(a)
      if beingloud > 19: print('Ny = %d' % bd.Ny),
    elif o in ("--Nz"):
      bd.Nz = int(a)
      if beingloud > 19: print('Nz = %d' % bd.Nz),
    elif o in ("-n", "--Nt"):
      bd.Nt = int(a); bd.dt = bd.T/bd.Nt
      if beingloud > 19: print('Nt = %d;' % bd.Nt),
    elif o in ("-T", "--Tfinal"):
      bd.T = float(a); bd.dt = bd.T/bd.Nt
      if beingloud > 19: print('T = %f;' % bd.T),
    elif o in ("-X", "--example"):
      example = int(a)
      if beingloud > 19: print('example %d;' % example),
    elif o in ("-K", "--backup"):
      os.system('DATE=`date +%Y_%m_%d_%H-%M-%S`; cp blockdata.py ../backup/$DATE-blockdata.py')
      os.system('DATE=`date +%Y_%m_%d_%H-%M-%S`; cp postpro.py ../backup/$DATE-postpro.py')
      if beingloud > 19: print('backing up this and postpro.py to ../backup/;'),
      exit(0)
    elif o in ("-F", "--pdf"):
      enscrstr = 'enscript --color=1 --margins 10c::: -C -f Courier8 -E -M A4 --landscape -p '
      os.system(enscrstr + './blockdata.ps blockdata.py')
      os.system('ps2pdf ./blockdata.ps; rm blockdata.ps')
      os.system(enscrstr + './postpro.ps postpro.py')
      os.system('ps2pdf ./postpro.ps; rm postpro.ps')
      if beingloud > 19: print('enscript PDF creation of this and postpro.py;'),
      exit(0)
    else:
      assert False, "unhandled option"

  print(""); print('Command line parsing complete...')

  # create a 'results_start_end' directory, if it doesn't already exist
  results_dir = "./results_"+str(start_run)+"_"+str(end_run-1)
  if xcyczc_is_random:
    xcyczc = np.random.rand(end_run - start_run,3)
  else:
    # these 
    xcyczc = np.stack( (xc*np.ones(end_run - start_run).T,
                        yc*np.ones(end_run - start_run).T,
                        zc*np.ones(end_run - start_run).T ), axis=-1)
  #print('not yet normalized to domain'); print(xcyczc)
  print('Running for ', xcyczc.shape[0], ' random source locations')
  if not os.path.exists(results_dir):
    os.mkdir(results_dir)
    print(str(time.strftime("%d/%m/%Y at %H:%M:%S"))+": directory " , results_dir ,  " created ")
  else:
    os.system("rm -rf "+results_dir+"/* ; ls "+results_dir) 
    print(str(time.strftime("%d/%m/%Y at %H:%M:%S"))+": directory " , results_dir ,  " already exists, now sterilized for new data")

  for run in range(start_run, end_run):
    xc,yc,zc = xcyczc[run-start_run]
    xc = bd.xL + xc*(bd.xH-bd.xL)
    yc = bd.yL + yc*(bd.yH-bd.yL)
    zc = bd.zL + zc*(bd.zH-bd.zL)
    xcyczc[run-start_run] = xc,yc,zc
    # determine whether to create sensor traces or not
    bd.gfx = 1 if (gfx_step and not (run-start_run) % gfx_step) else 0;
    # get a string for this run's output directory
    output_dir = results_dir+"/"+str(run)
    # either it doesn't exist or it is dirty with old runs; deal with both 
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
      print(str(time.strftime("%d/%m/%Y at %H:%M:%S"))+": directory " , output_dir , " created ")
    else:
      os.system("rm -rf "+output_dir+"/* ; ls "+output_dir) 
      print(str(time.strftime("%d/%m/%Y at %H:%M:%S"))+": directory " , output_dir , " already exists, now sterilized for new data")
    # in either case, create the data subdirectories
    if bd.gfx:
      os.mkdir(output_dir+"/eps")
      os.mkdir(output_dir+"/png")
    os.mkdir(output_dir+"/txt")

    # open files for sensor traces: for timesteps, u, du/dt, ... at acclerometers
    bd.tp     = open(output_dir+"/txt/times.txt","w")
    bd.u1_a   = open(output_dir+"/txt/u1_a.txt","w");     bd.u2_a    = open(output_dir+"/txt/u2_a.txt","w");    bd.u3_a   = open(output_dir+"/txt/u3_a.txt","w"); 
    bd.u1t_a  = open(output_dir+"/txt/u1t_a.txt","w");    bd.u2t_a   = open(output_dir+"/txt/u2t_a.txt","w");   bd.u3t_a  = open(output_dir+"/txt/u3t_a.txt","w"); 
    bd.u1tt_a = open(output_dir+"/txt/u1tt_a.txt","w");   bd.u2tt_a  = open(output_dir+"/txt/u2tt_a.txt","w");  bd.u3tt_a = open(output_dir+"/txt/u3tt_a.txt","w"); 

    bd.u1x_a   = open(output_dir+"/txt/u1x_a.txt","w");   bd.u2x_a   = open(output_dir+"/txt/u2x_a.txt","w");   bd.u3x_a   = open(output_dir+"/txt/u3x_a.txt","w"); 
    bd.u1xt_a  = open(output_dir+"/txt/u1xt_a.txt","w");  bd.u2xt_a  = open(output_dir+"/txt/u2xt_a.txt","w");  bd.u3xt_a  = open(output_dir+"/txt/u3xt_a.txt","w"); 
    bd.u1xtt_a = open(output_dir+"/txt/u1xtt_a.txt","w"); bd.u2xtt_a = open(output_dir+"/txt/u2xtt_a.txt","w"); bd.u3xtt_a = open(output_dir+"/txt/u3xtt_a.txt","w"); 

    bd.u1y_a   = open(output_dir+"/txt/u1y_a.txt","w");   bd.u2y_a   = open(output_dir+"/txt/u2y_a.txt","w");   bd.u3y_a   = open(output_dir+"/txt/u3y_a.txt","w"); 
    bd.u1yt_a  = open(output_dir+"/txt/u1yt_a.txt","w");  bd.u2yt_a  = open(output_dir+"/txt/u2yt_a.txt","w");  bd.u3yt_a  = open(output_dir+"/txt/u3yt_a.txt","w"); 
    bd.u1ytt_a = open(output_dir+"/txt/u1ytt_a.txt","w"); bd.u2ytt_a = open(output_dir+"/txt/u2ytt_a.txt","w"); bd.u3ytt_a = open(output_dir+"/txt/u3ytt_a.txt","w"); 

    bd.u1z_a   = open(output_dir+"/txt/u1z_a.txt","w");   bd.u2z_a   = open(output_dir+"/txt/u2z_a.txt","w");   bd.u3z_a   = open(output_dir+"/txt/u3z_a.txt","w"); 
    bd.u1zt_a  = open(output_dir+"/txt/u1zt_a.txt","w");  bd.u2zt_a  = open(output_dir+"/txt/u2zt_a.txt","w");  bd.u3zt_a  = open(output_dir+"/txt/u3zt_a.txt","w"); 
    bd.u1ztt_a = open(output_dir+"/txt/u1ztt_a.txt","w"); bd.u2ztt_a = open(output_dir+"/txt/u2ztt_a.txt","w"); bd.u3ztt_a = open(output_dir+"/txt/u3ztt_a.txt","w"); 

    # ... and then at microphones
    bd.u1_m   = open(output_dir+"/txt/u1_m.txt","w");     bd.u2_m    = open(output_dir+"/txt/u2_m.txt","w");    bd.u3_m    = open(output_dir+"/txt/u3_m.txt","w"); 
    bd.u1t_m  = open(output_dir+"/txt/u1t_m.txt","w");    bd.u2t_m   = open(output_dir+"/txt/u2t_m.txt","w");   bd.u3t_m   = open(output_dir+"/txt/u3t_m.txt","w"); 
    bd.u1tt_m = open(output_dir+"/txt/u1tt_m.txt","w");   bd.u2tt_m  = open(output_dir+"/txt/u2tt_m.txt","w");  bd.u3tt_m  = open(output_dir+"/txt/u3tt_m.txt","w"); 

    bd.u1x_m   = open(output_dir+"/txt/u1x_m.txt","w");   bd.u2x_m   = open(output_dir+"/txt/u2x_m.txt","w");   bd.u3x_m   = open(output_dir+"/txt/u3x_m.txt","w"); 
    bd.u1xt_m  = open(output_dir+"/txt/u1xt_m.txt","w");  bd.u2xt_m  = open(output_dir+"/txt/u2xt_m.txt","w");  bd.u3xt_m  = open(output_dir+"/txt/u3xt_m.txt","w"); 
    bd.u1xtt_m = open(output_dir+"/txt/u1xtt_m.txt","w"); bd.u2xtt_m = open(output_dir+"/txt/u2xtt_m.txt","w"); bd.u3xtt_m = open(output_dir+"/txt/u3xtt_m.txt","w"); 

    bd.u1y_m   = open(output_dir+"/txt/u1y_m.txt","w");   bd.u2y_m   = open(output_dir+"/txt/u2y_m.txt","w");   bd.u3y_m   = open(output_dir+"/txt/u3y_m.txt","w"); 
    bd.u1yt_m  = open(output_dir+"/txt/u1yt_m.txt","w");  bd.u2yt_m  = open(output_dir+"/txt/u2yt_m.txt","w");  bd.u3yt_m  = open(output_dir+"/txt/u3yt_m.txt","w"); 
    bd.u1ytt_m = open(output_dir+"/txt/u1ytt_m.txt","w"); bd.u2ytt_m = open(output_dir+"/txt/u2ytt_m.txt","w"); bd.u3ytt_m = open(output_dir+"/txt/u3ytt_m.txt","w"); 

    bd.u1z_m   = open(output_dir+"/txt/u1z_m.txt","w");   bd.u2z_m   = open(output_dir+"/txt/u2z_m.txt","w");   bd.u3z_m   = open(output_dir+"/txt/u3z_m.txt","w"); 
    bd.u1zt_m  = open(output_dir+"/txt/u1zt_m.txt","w");  bd.u2zt_m  = open(output_dir+"/txt/u2zt_m.txt","w");  bd.u3zt_m  = open(output_dir+"/txt/u3zt_m.txt","w"); 
    bd.u1ztt_m = open(output_dir+"/txt/u1ztt_m.txt","w"); bd.u2ztt_m = open(output_dir+"/txt/u2ztt_m.txt","w"); bd.u3ztt_m = open(output_dir+"/txt/u3ztt_m.txt","w"); 

    # open files to store x,y,z in columns of source, and each accelerometer and microphone, in rows
    s_f = open(output_dir+"/txt/srce.txt","w")
    # open files to store x,y,z (in columns) of accelerometer and microphone positions (in rows)
    if not os.path.exists(results_dir+"/accs.txt"):
      a_f = open(results_dir+"/accs.txt","w")
      for c in range(0, bd.accl.shape[0]):
        x,y,z = bd.accl[c]
        a_f.write("%e %e %e\n" % (x,y,z) )
      a_f.close()
    if not os.path.exists(results_dir+"/mics.txt"):
      m_f = open(results_dir+"/mics.txt","w")
      for c in range(0, bd.mics.shape[0]):
        x,y,z = bd.mics[c]
        m_f.write("%e %e %e\n" % (x,y,z) )
      m_f.close()
    # open a file to keep a record of this run
    run_report = open(output_dir+"/txt/run_report.txt","w")
    run_report.write("Begin...\n")
    run_report.write(str(time.strftime("%d/%m/%Y at %H:%M:%S"))); run_report.write("\n")
    run_report.write(str( ('python version ',sys.version) ));     run_report.write("\n")
    dolver = subprocess.check_output("dolfin-version").decode("utf-8")
    run_report.write("dolfin version %s\n" % dolver)
    if bd.gfx:
      run_report.write("Sensor trace graphics are being generated for this run\n")
    else:
      run_report.write("Sensor trace graphics are not being generated for this run\n")
    run_report.write("E = %e; nu = %e; lambda = %e; mu = %e\n" % (bd.Ey,bd.nu,bd.lmbda,bd.mu) )
    run_report.write("T = %e; freqm = %e; freqc = %e; eps = %e\n" % (bd.T,bd.freqm,bd.freqc,bd.eps) )
    run_report.write("(xL,xH)x(yL,yH)x(zL,zH) = (%e, %e)x(%e, %e)x(%e, %e)\n" % (bd.xL, bd.xH, bd.yL, bd.yH, bd.zL, bd.zH) )
    run_report.write("Nt = %d, Nx = %d, Ny = %d, Nz = %d\n" % (bd.Nt,bd.Nx,bd.Ny,bd.Nz) )
    run_report.write("rdeg = %d, Rdeg = %d\n" % (bd.rdeg,bd.Rdeg) )
    run_report.write("xc,yc,zc = (in e then f format)\n  %e %e %e\n  %f %f %f\n" % (xc,yc,zc,xc,yc,zc))
    run_report.write("\n")
    # output individual data sets
    s_f.write("%f %f %f\n" % (xc,yc,zc))
    for c in range(0, bd.accl.shape[0]):
      x,y,z = bd.accl[c]
      run_report.write("accelerometer %d at (x,y,z) = (%e,%e,%e)\n" % (c,x,y,z) )
    run_report.write("\n")
    for c in range(0, bd.mics.shape[0]):
      x,y,z = bd.mics[c]
      run_report.write("microphone  %d  at  (x,y,z) = (%e,%e,%e)\n" % (c,x,y,z) )
    # now solve
    bd.solver(xc,yc,zc,run_report,output_dir)
    run_report.write("\n")
    # close the files
    bd.tp.close()
    bd.u1_a.close();      bd.u2_a.close();      bd.u3_a.close(); 
    bd.u1t_a.close();     bd.u2t_a.close();     bd.u3t_a.close(); 
    bd.u1tt_a.close();    bd.u2tt_a.close();    bd.u3tt_a.close(); 

    bd.u1x_a.close();     bd.u2x_a.close();     bd.u3x_a.close(); 
    bd.u1xt_a.close();    bd.u2xt_a.close();    bd.u3xt_a.close(); 
    bd.u1xtt_a.close();   bd.u2xtt_a.close();   bd.u3xtt_a.close(); 

    bd.u1y_a.close();     bd.u2y_a.close();     bd.u3y_a.close(); 
    bd.u1yt_a.close();    bd.u2yt_a.close();    bd.u3yt_a.close(); 
    bd.u1ytt_a.close();   bd.u2ytt_a.close();   bd.u3ytt_a.close(); 

    bd.u1z_a.close();     bd.u2z_a.close();     bd.u3z_a.close(); 
    bd.u1zt_a.close();    bd.u2zt_a.close();    bd.u3zt_a.close(); 
    bd.u1ztt_a.close();   bd.u2ztt_a.close();   bd.u3ztt_a.close(); 

    bd.u1_m.close();      bd.u2_m.close();      bd.u3_m.close(); 
    bd.u1t_m.close();     bd.u2t_m.close();     bd.u3t_m.close(); 
    bd.u1tt_m.close();    bd.u2tt_m.close();    bd.u3tt_m.close(); 

    bd.u1x_m.close();     bd.u2x_m.close();     bd.u3x_m.close(); 
    bd.u1xt_m.close();    bd.u2xt_m.close();    bd.u3xt_m.close(); 
    bd.u1xtt_m.close();   bd.u2xtt_m.close();   bd.u3xtt_m.close(); 

    bd.u1y_m.close();     bd.u2y_m.close();     bd.u3y_m.close(); 
    bd.u1yt_m.close();    bd.u2yt_m.close();    bd.u3yt_m.close(); 
    bd.u1ytt_m.close();   bd.u2ytt_m.close();   bd.u3ytt_m.close(); 

    bd.u1z_m.close();     bd.u2z_m.close();     bd.u3z_m.close(); 
    bd.u1zt_m.close();    bd.u2zt_m.close();    bd.u3zt_m.close(); 
    bd.u1ztt_m.close();   bd.u2ztt_m.close();   bd.u3ztt_m.close(); 

    run_report.write(str(time.strftime("%d/%m/%Y at %H:%M:%S"))); run_report.write("\n")
    run_report.write("...End\n")
    run_report.close(); s_f.close()

  # finish by saving a plot (if gfx allowed) and list of the random source points
  plotName = "scatter_"+str(start_run)+"_"+str(end_run-1)
  save_xcyczc(xcyczc, bd, plotName, results_dir)
    
  
  # create a zip file of the results directory. From (on 27 Feb 2020)
  # https://thispointer.com/python-how-to-create-a-zip-archive-from-multiple-files-or-directory/
  # create a ZipFile object'
  os.system('rm -rf '+results_dir+'.zip') 
  with ZipFile(results_dir+'.zip', 'w') as zipObj:
    # Iterate over all the files in directory
    for folderName, subfolders, filenames in os.walk(results_dir):
      for filename in filenames:
        #create complete filepath of file in directory
        filePath = os.path.join(folderName, filename)
        # Add file to zip
        zipObj.write(filePath)

  exit(0)

