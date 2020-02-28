#!/anaconda3/envs/fenicsproject/bin/python

'''
MAC
git clone https://github.com/martinsbruveris/waves2d.git
conda env create -f environment.yml

in waves2d...
660  conda activate waves2d
  661  ls
  662  mkdir data
  663  cd data
  664  cp ../../../forward/samples_z.npz .
  665  ls
  666  mv samples_z.npz samples_3d_500.npz 
  668  cd ..
  669  ls
  670  echo $PYTHONPATH; export PYTHONPATH=./src:$PYTHONPATH
  671  echo $PYTHONPATH
  672  python scripts/2020_02_08_test_v3.py 


ADA
conda env remove --name waves2d
conda env create -f environment.yml

From (8 Feb 2020): https://stackoverflow.com/questions/19371860/python-open-file-from-zip-without-temporary-extracting-it
import zipfile
archive = zipfile.ZipFile('images.zip', 'r')
imgdata = archive.read('img_01.png')

'''

import os, zipfile, numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from cycler import cycler

# LaTeX trace names
fnames   = [r'u_{1}'  ,r'u_{1x}'  ,r'u_{1y}'  ,r'u_{1z}',
            r'u_{2}'  ,r'u_{2x}'  ,r'u_{2y}'  ,r'u_{2z}',
            r'u_{3}'  ,r'u_{3x}'  ,r'u_{3y}'  ,r'u_{3z}',
            r'u_{1t}' ,r'u_{1xt}' ,r'u_{1yt}' ,r'u_{1zt}',
            r'u_{2t}' ,r'u_{2xt}' ,r'u_{2yt}' ,r'u_{2zt}',
            r'u_{3t}' ,r'u_{3xt}' ,r'u_{3yt}' ,r'u_{3zt}',
            r'u_{1tt}',r'u_{1xtt}',r'u_{1ytt}',r'u_{1ztt}',
            r'u_{2tt}',r'u_{2xtt}',r'u_{2ytt}',r'u_{2ztt}',
            r'u_{3tt}',r'u_{3xtt}',r'u_{3ytt}',r'u_{3ztt}']

# accelerometers: loop over all samples, open and read files
fnames_a = ["u1_a.txt"  ,"u1x_a.txt"  ,"u1y_a.txt"  ,"u1z_a.txt",
            "u2_a.txt"  ,"u2x_a.txt"  ,"u2y_a.txt"  ,"u2z_a.txt",
            "u3_a.txt"  ,"u3x_a.txt"  ,"u3y_a.txt"  ,"u3z_a.txt",
            "u1t_a.txt" ,"u1xt_a.txt" ,"u1yt_a.txt" ,"u1zt_a.txt",
            "u2t_a.txt" ,"u2xt_a.txt" ,"u2yt_a.txt" ,"u2zt_a.txt",
            "u3t_a.txt" ,"u3xt_a.txt" ,"u3yt_a.txt" ,"u3zt_a.txt",
            "u1tt_a.txt","u1xtt_a.txt","u1ytt_a.txt","u1ztt_a.txt",
            "u2tt_a.txt","u2xtt_a.txt","u2ytt_a.txt","u2ztt_a.txt",
            "u3tt_a.txt","u3xtt_a.txt","u3ytt_a.txt","u3ztt_a.txt"]

# microphones: loop over all samples, open and read files
fnames_m = ["u1_m.txt"  ,"u1x_m.txt"  ,"u1y_m.txt"  ,"u1z_m.txt",
            "u2_m.txt"  ,"u2x_m.txt"  ,"u2y_m.txt"  ,"u2z_m.txt",
            "u3_m.txt"  ,"u3x_m.txt"  ,"u3y_m.txt"  ,"u3z_m.txt",
            "u1t_m.txt" ,"u1xt_m.txt" ,"u1yt_m.txt" ,"u1zt_m.txt",
            "u2t_m.txt" ,"u2xt_m.txt" ,"u2yt_m.txt" ,"u2zt_m.txt",
            "u3t_m.txt" ,"u3xt_m.txt" ,"u3yt_m.txt" ,"u3zt_m.txt",
            "u1tt_m.txt","u1xtt_m.txt","u1ytt_m.txt","u1ztt_m.txt",
            "u2tt_m.txt","u2xtt_m.txt","u2ytt_m.txt","u2ztt_m.txt",
            "u3tt_m.txt","u3xtt_m.txt","u3ytt_m.txt","u3ztt_m.txt"]

gfx = 1  # if true then make pictures

# these eventually can be parameterised - but best to re-configure the forward solver
Nsamples   = 2 # 500
Nt         = 24 # 99
Nsignals_a = 1+len(fnames)
Nsignals_m = 1+len(fnames)
Naccls     = 5
Nmics      = 4
start_sample = 200 # 1  # the directory number of the first in the consecutively numbered sample set
#in_path    = './results_1_500/'
in_zip    = './results_200_201.zip' #'../../../../data/results_1_500.zip' #'./results_1_500.zip'
in_path    = 'results_200_201/' # 'results_1_500/'
#in_path    = '/Users/simon/offline/MartinsML/MachineLearning/phase3/data/results_1_500' # 'results_1_500/'

acc_samples = np.zeros((Nsamples, Nt, Naccls, Nsignals_a))
mic_samples = np.zeros((Nsamples, Nt, Nmics , Nsignals_m))
src_xcyczc  = np.zeros((Nsamples,3))

## obtain the source positions
#src_f = open(in_path + "xcyczc.txt", "r")
#srclines = src_f.readlines() 

archive = zipfile.ZipFile(in_zip, 'r')
src_f = archive.open(in_path + "xcyczc.txt")
srclines = src_f.readlines() 

#print(srclines)

count = 0
for line in srclines: 
  vals = np.array(line.split()); vals = np.asfarray(vals,np.float64)
  #print('count = ', count)
  #print('line  = ', line)
  #print('vals  = ', vals)
  #print('array = ', np.array(line))
  for i in range(0,3):
    src_xcyczc[count,:] = vals[:]
  #print('src_..[', count, '] = ', np.array(line))
  count = count+1
  
#print('\n',src_xcyczc)
src_f.close()

# get the times - trust that the first sample results represents them all.
src_f = archive.open(in_path+str(start_sample)+'/txt/times.txt')
times = np.loadtxt(src_f, dtype=np.float64)

#print(times)


'''
for sample in range(0, Nsamples):
######  name_count = 0
  for fname in fnames_a:
    name_count = 0
    src_f = archive.open(in_path+str(start_sample+sample)+'/txt/'+fname)
    #src_f = open(in_path+str(1+sample)+'/txt/'+fname, "r")
    srclines = src_f.readlines() 
#    print(srclines); input()
    time_count = 0
#    print(fname)
    for line in srclines:
#      print(line); input()
      acc_samples[sample, time_count, :, 0] = times[time_count]
      vals = np.array(line.split()); vals = np.asfarray(vals,np.float64)
      for i in range(0,Naccls):
#        print(vals); input()
        acc_samples[sample, time_count, :, 1+name_count] = vals[:]
      time_count = time_count+1
    src_f.close()
  name_count = name_count + 1
'''

for sample in range(0, Nsamples):
  name_count = 0
  for fname in fnames_a:
    src_f = archive.open(in_path+str(start_sample+sample)+'/txt/'+fname)
    srclines = src_f.readlines() 
#    print(srclines); input()
    time_count = 0
#    print(fname)
    for line in srclines:
#      print(line); input()
      acc_samples[sample, time_count, :, 0] = times[time_count]
      vals = np.array(line.split()); vals = np.asfarray(vals,np.float64)
#      print(vals); input()
      for i in range(0,Naccls):
#        acc_samples[sample, time_count, :, 1+name_count] = vals[:]
        acc_samples[sample, time_count, i, 1+name_count] = vals[i]
      time_count = time_count+1
    src_f.close()
    name_count = name_count + 1

#print(acc_samples[0,:,0,0]); input(); print(times)
#print(acc_samples[0,:,0,1]); input()



'''
for sample in range(0, Nsamples):
  name_count = 0
  for fname in fnames_m:
#    print('Opening '+in_path+str(sample)+'/txt/'+fname)
    src_f = archive.open(in_path+str(start_sample+sample)+'/txt/'+fname)
    #src_f = open(in_path+str(1+sample)+'/txt/'+fname, "r")
    srclines = src_f.readlines() 
    time_count = 0
    for line in srclines:
      mic_samples[sample, time_count, :, 0] = times[time_count]
      vals = np.array(line.split()); vals = np.asfarray(vals,float)
      for i in range(0,Naccls): WRONG!
        mic_samples[sample, time_count, :, 1+name_count] = vals[:]
      time_count = time_count+1
    src_f.close()
  name_count = name_count + 1
'''

for sample in range(0, Nsamples):
  name_count = 0
  for fname in fnames_m:
    src_f = archive.open(in_path+str(start_sample+sample)+'/txt/'+fname)
    srclines = src_f.readlines() 
#    print(srclines); input()
    time_count = 0
#    print(fname)
    for line in srclines:
#      print(line); input()
      mic_samples[sample, time_count, :, 0] = times[time_count]
      vals = np.array(line.split()); vals = np.asfarray(vals,np.float64)
#      print(vals); input()
      for i in range(0,Nmics):
#        mic_samples[sample, time_count, :, 1+name_count] = vals[:]
        mic_samples[sample, time_count, i, 1+name_count] = vals[i]
      time_count = time_count+1
    src_f.close()
    name_count = name_count + 1


#np.savez('samples.npz', src_xcyczc  = src,
np.savez_compressed('samples_z.npz',
                        src = src_xcyczc,
                        acc = acc_samples,
                        mic = mic_samples)

#npzfile = np.load('samples.npz')
npzfile = np.load('samples_z.npz')
print('Loading samples from ', 'samples_z.npz', 'with ',sorted(npzfile.files))
src   = npzfile['src']
acc1  = npzfile['acc']
mic1  = npzfile['mic']

#print(acc1[0,:,0,1]); input()

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   

# these are chosen in order when plotting - for six plots or less they're unique
default_cycler                                                                       \
  = (cycler(color=['r', 'g', 'b', 'm', 'c', 'k'])                                    \
  +  cycler(linestyle=[(0,(3,1,1,1,1,1)), '--', ':', '-.', (0,(3,5,1,5,1,5)), '-' ]) \
    )

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   

# plot sensor readings
def plot_sensor_trace(savedir, fn, u_label, sens_type, v, sample, signal):
    # set up the plot
    nfs = 16          # normal font size
    nlw = 3; tlw=1    # normal, thin line width
    nms = 10          # normal marker size
    plt.rc('axes', prop_cycle=default_cycler) # must come first for plt.*** to take effect on first plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]  # for \boldsymbol etc
    plt.rcParams.update({'font.size': nfs})
    plt.tick_params(labelsize = nfs)
    plt.gcf().subplots_adjust(left=0.20)
    plt.rc('lines', linewidth=nlw)

#    times = np.linspace(self.dt,self.T,num=self.Nt-1, endpoint=False)
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
#    _, ax_f = plt.subplots(); ax_f.set_ylim(auto=True)
    for c in range(0, v.shape[2]):
      plt.plot(v[0,:,0,0], v[sample,:,c,signal], label=label+ str(c)+')$')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.legend(loc="lower left",fontsize=nfs)
    plt.savefig(savedir+'/png/'+fn+'.png', format='png', dpi=750)
    plt.savefig(savedir+'/eps/'+fn+'.eps', format='eps', dpi=1000)
    #plt.savefig(savedir+'/'+fn+'.png', format='png', dpi=750)
    #plt.savefig(savedir+'/'+fn+'.eps', format='eps', dpi=1000)
    plt.grid(True)
    plt.clf()
    plt.close()

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   

if gfx:
  # Create/clean the save directories for gfx
  savedir = r'./gfx'
  if not os.path.exists(savedir):
    os.makedirs(savedir+'/png/')
    os.makedirs(savedir+'/eps/')
  else:
    if not os.path.exists(savedir+'/png/'):
      os.makedirs(savedir+'/png/')
    else:
      os.system('rm -rf '+savedir+'/png/*')
    if not os.path.exists(savedir+'/eps/'):
      os.makedirs(savedir+'/eps/')
    else:
      os.system('rm -rf '+savedir+'/eps/*')

#  for sample in range(0, Nsamples):
  for sample in range(0, 2):
    name_count = 0
    for fname, lnm in zip(fnames_m, fnames):
      nm,_ = fname.split('_')
#      lnm = fnames[name_count]
      tmp = nm+'_'+str(start_sample+sample)
#      plot_sensor_trace(savedir, tmp+'_a0', '$'+lnm, 'accelerometers', acc_samples, sample, 1+name_count)
#      plot_sensor_trace(savedir, tmp+'_m0', '$'+lnm, 'microphones', mic_samples, sample, 1+name_count)
      plot_sensor_trace(savedir, tmp+'_a', '$'+lnm, 'accelerometers', acc1, sample, 1+name_count)
      plot_sensor_trace(savedir, tmp+'_m', '$'+lnm, 'microphones', mic1, sample, 1+name_count)
      name_count = name_count+1
  
'''
#  for sample in range(0, Nsamples):
  for sample in range(0, 2):
    name_count = 0
    for fname in fnames_m:
      nm,_ = fname.split('_')
      lnm = fnames[name_count]
      plot_sensor_trace(savedir,nm+'_'+str(1+sample)+'_a', '$'+lnm, 'accelerometers', acc_samples, 1+sample, name_count)
      plot_sensor_trace(savedir,nm+'_'+str(1+sample)+'_m', '$'+lnm, 'microphones', mic_samples, 1+sample, name_count)
      plot_sensor_trace(savedir,nm+'_'+str(1+sample)+'_a1', '$'+lnm, 'accelerometers', acc1, 1+sample, name_count)
      plot_sensor_trace(savedir,nm+'_'+str(1+sample)+'_m1', '$'+lnm, 'microphones', mic1, 1+sample, name_count)
      name_count = name_count+1
  
>>> for f1, f2 in zip(fnames, fnames_m):
...   print(f1+'\t\t'+f2)
... 
u_1		u1_m.txt
u_{1x}		u1x_m.txt
u_{1y}		u1y_m.txt


>>> s='file.txt.m.a._a1'
>>> s.split()
['file.txt.m.a._a1']
>>> s.split(.)
  File "<stdin>", line 1
    s.split(.)
            ^
SyntaxError: invalid syntax
>>> s.split('.')
['file', 'txt', 'm', 'a', '_a1']
>>> ,,c,, = s.split('.')
  File "<stdin>", line 1
    ,,c,, = s.split('.')
    ^
SyntaxError: invalid syntax
>>> _,,c,, = s.split('.')
  File "<stdin>", line 1
    _,,c,, = s.split('.')
      ^
SyntaxError: invalid syntax
>>> _,_,c,_,_ = s.split('.')
>>> c
'm'
>>> exit()
'''

  
print('\nThings To Do Now')
print('================')
print(' - Check that output graphics agree with those from the primitive forward data')
print(' - Archive the primitive forward data to the offline directory')
print(' - Archive the working data to the data (or shared Dropbox) directory')
print(' - Need these graphics to have all run time creation data, plus (xc,yc,zc)')
print(' - ')
print('\n')
