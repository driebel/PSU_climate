import numpy as np
import os.path
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from numba import jit
from mpl_toolkits.mplot3d import axes3d
from matplotlib.mlab import griddata
import math
#from matplotlib import pylab

start = time.time()            

boot_num = 100000

@jit
def jetstream_model(p,d,y):
    len = d.shape[0]
    a = 2*np.pi*d/len
    a2 = 2*a
    zmod = (p[0] +
            p[1]*y +
            p[2]*y**2 +
            p[3]*np.sin(a) +
            p[4]*np.cos(a) +
            p[5]*np.sin(a2) +
            p[6]*np.cos(a2) +
            p[7]*y*np.sin(a) +
            p[8]*y*np.cos(a) +
            p[9]*y*np.sin(a2) +
            p[10]*y*np.cos(a2) +
            p[11]*y**2*np.sin(a) +
            p[12]*y**2*np.cos(a) +
            p[13]*y**2*np.sin(a2) +
            p[14]*y**2*np.cos(a2))
    return zmod
                         
    
home_dir = '/home/driebel/Dropbox/psu_climate/'
var_dir = home_dir+'variability_data/'
waterfall_dir = home_dir+'waterfall_data/'
fit_dir = home_dir + 'quadratic_fits/'
model_type = 'era'
scenario=''

#for model_type in ['gfdl','mri']:
#    for scenario in ['hist','reanal','rcp45','rcp60','rcp85']:
base = model_type
print 'working on ' + base
waterfall_file = waterfall_dir + base + '_waterfall.dat'
var_file = var_dir + base + '.dat'
fit_file = fit_dir + base + '/fitting_coefficients.dat'
if os.path.exists(waterfall_file):
    data = np.genfromtxt(waterfall_file,delimiter=',',dtype='float')
    day = np.array(range(1,data.shape[0]+1))
    year = np.array(range(1,data.shape[1]+1))
    data_list = data.ravel()
    d, y = np.meshgrid(day,year,indexing='ij')
    d_list = d.ravel()
    y_list = y.ravel()
    a_list = 2.*np.pi*d_list/(day.shape[0])
    a2_list = 2.*a_list
    x = np.vstack([np.ones(y_list.size),
                   y_list,
                   y_list**2,
                   np.sin(a_list),
                   np.cos(a_list),
                   np.sin(a2_list),
                   np.cos(a2_list),
                   y_list*np.sin(a_list),
                   y_list*np.cos(a_list),
                   y_list*np.sin(a2_list),
                   y_list*np.cos(a2_list),
                   y_list**2*np.sin(a_list),
                   y_list**2*np.cos(a_list),
                   y_list**2*np.sin(a2_list),
                   y_list**2*np.cos(a2_list)]).T
    best_coeffs = np.linalg.lstsq(x,data_list)[0]
    model = jetstream_model(best_coeffs,d,y)
    resid_norm = (data - model)/model
    resid_norm_list = resid_norm.ravel()
    model_list = model.ravel()
    thresh_list = np.linspace(1,3.5,26)
    best_extreme_event_slope = np.zeros_like(thresh_list)
    year_x = np.vstack([year, np.ones(len(year))]).T
    for thresh_index,thresh in enumerate(thresh_list):
        bad_day = np.where(resid_norm_list >= thresh)
        best_extreme_event_count = np.zeros_like(year)
        for k in range(0,year.size):
            best_extreme_event_count[k] = len(
                np.where(y_list[bad_day] == year[k])[0])

        best_extreme_event_slope[thresh_index],junk = (
            np.linalg.lstsq(year_x,best_extreme_event_count)[0])


    all_coeffs = np.zeros([best_coeffs.size,boot_num])
    num_extremes = np.zeros_like(year)
    boot_extreme_event_slopes = np.zeros([thresh_list.size,boot_num])
    for bindex in range(0,boot_num):
        sample = np.random.randint(data_list.size,size=data_list.size)
        boot_data_list = data_list[sample]
        boot_year_list = y_list[sample]
        boot_day_list = d_list[sample]
        boot_resid = resid_norm_list[sample]
        boot_model_list = model_list[sample]
        all_coeffs[:,bindex] = np.linalg.lstsq(x[sample,:],boot_data_list)[0]
        boot_resid = (boot_data_list - boot_model_list)/boot_model_list
        for thresh_index,thresh in enumerate(thresh_list):
            bad_day = np.where(boot_resid >= thresh)
            boot_extreme_event_count = np.zeros_like(year)
            for k in range(0,year.size):
                boot_extreme_event_count[k] = len(
                    np.where(boot_year_list[bad_day] == year[k])[0])

            boot_extreme_event_slopes[thresh_index,bindex],junk = (
                np.linalg.lstsq(year_x,boot_extreme_event_count)[0])
            #End Bootstrap loop

    max_coeffs = np.zeros_like(best_coeffs)
    min_coeffs = np.zeros_like(best_coeffs)
    for k in range(0,best_coeffs.size):
        coeff_range = all_coeffs[k,:]
        bottom = int(coeff_range.size*0.025)
        top = coeff_range.size - bottom - 1
        min_coeffs[k] = np.sort(coeff_range)[bottom]
        max_coeffs[k] = np.sort(coeff_range)[top]

    max_slopes = np.zeros_like(thresh_list)
    min_slopes = np.zeros_like(thresh_list)
    for k in range(0,thresh_list.size):
        slope_range = boot_extreme_event_slopes[k,:]
        bottom = int(slope_range.size*0.025)
        top = slope_range.size - bottom - 1
        min_slopes[k] = np.sort(slope_range)[bottom]
        max_slopes[k] = np.sort(slope_range)[top]

    outdir = waterfall_dir+'python_output/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fit_file = outdir+base+'_fit_info.dat'
    labels = ['const','y','y^2','sin(a)','cos(a)','sin(2a)',
              'cos(2a)','ysin(a)','ycos(a)','ysin(2a)','ycos(2a)',
              'y2sin(a)','y2cos(a)','y2sin(2a)','y2cos(2a)']
    with open(fit_file,'w') as f:
        f.write('{0:11s}{1:^13s}{2:^13s}{3:^13s}\n'.format('','Best','Min','Max'))
        for k in range(len(labels)):
            f.write('{0:11}{1:13.5E}{2:13.5E}{3:13.5E}\n'.format(
                labels[k],best_coeffs[k],min_coeffs[k],max_coeffs[k]))

    thresh_file = outdir+base+'_threshold_slopes.dat'
    with open(thresh_file,'w') as f:
        f.write('{0:^5s}{1:^13s}{2:^13s}{3:^13s}\n'.format('T','Best','Min','Max'))
        for k in range(len(thresh_list)):
            f.write('{0:<5.1f}{1:13.4E}{2:13.4E}{3:13.4E}\n'.format(
                thresh_list[k],best_extreme_event_slope[k],min_slopes[k],max_slopes[k]))

    model_file = outdir+base+'_model.dat'
    np.savetxt(model_file,model,delimiter=',',fmt='%12.2f')

    fig = plt.figure()
    plt.ylabel("Day")
    plt.xlabel("Year")
    cs = plt.contourf(y, d, model, cmap=cm.jet)
    cbar = plt.colorbar()
    plt.savefig(outdir+base+'_model.png', bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    plt.xlabel('Threshold Value')
    plt.ylabel('Yearly Trend')
    lines = plt.plot(thresh_list,best_extreme_event_slope,color='k')
    lines = plt.plot(thresh_list,max_slopes,color='r')
    lines = plt.plot(thresh_list,min_slopes,color='b')
    lines = plt.plot(thresh_list,np.zeros_like(thresh_list),color='k')
    plt.ylim = [(max(max_slopes)*1.1,min(min_slopes)*1.1)]
    plt.savefig(outdir+base+'_threshold_trend.png', bbox_inches='tight')
    plt.close()

print time.time() - start      
        
