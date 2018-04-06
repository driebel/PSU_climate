import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import os
from numba import jit
import time
import argparse
import easygui

print 'Started at '+time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()
parser.add_argument("-boot_num",help="Enter the number of bootstrap loops",
                    type=int, default = 100)
parser.add_argument('-manual',help = 'Enter 1 to manually choose parameters',
                    type=int,default=0)
args = parser.parse_args()
boot_num = args.boot_num
manual = args.manual


#Here is the section you must edit to describe the data structure

if manual:
    nc_file = easygui.fileopenbox(msg='Choose the netCDF File to process',
                                  title = 'File Selection')
    title = 'Enter the First and Last years of the dataset'
    msg = 'Enter the First and Last years of the Dataset'
    fieldnames = ['First Year:','Last Year:']
    limits = easygui.multenterbox(msg,title,fieldnames)
    for i in range(len(limits)):
        limits[i] = int(limits[i])
    years = np.array(range(limits[0],limits[1])) - (limits[0]-1)
    daysperyear = int(easygui.enterbox(msg='Enter the number of time steps in a year'))
    msg = 'Enter the First and Last PCs to use to reconstruc the index'
    fieldnames = ['First Index:','Last Index:']
    title='Enter PCs'
    indicies = easygui.multenterbox(msg,title,fieldnames)
    first_index = int(indicies[0])-1
    last_index = int(indicies[1])-1
    msg = 'Enter the number of bootstrap reps:'
    title = 'Bootstrap Selection'
    boot_num = easugui.integerbox(msg,title,boot_num)
else:
    nc_file = '/home/driebel/Dropbox/psu_climate/gfdl_1979/reanal_1979_withleap.nc'
    years = np.array(range(1979,2004)) - 1978  #now just 1 - 25
    daysperyear = 1461
    first_index = 0
    last_index = 16
    boot_num = 100000

climdir = os.path.dirname(nc_file)+'/' # the climate trend is saved in the same directory
base = os.path.basename(nc_file)[:-3]
waterfall_file = '/home/driebel/Dropbox/psu_climate/waterfall_data/'+base+'.dat'
num_years = len(years)

month = 30*(daysperyear/365) #length of one month plus one obs, for smoothing function
if month % 2 == 0:
    month += 1
#month must be odd.


retain_explained = 90 
# Retain enough PCs to explain this percent of the variance
used_index = 16 
# How many PCs to use in the Jetstream Index

if not os.path.exists(climdir):
    os.mkdir(climdir)

if os.path.exists(nc_file):
    nc = nc.Dataset(nc_file,'r')

vars = [var for var in nc.variables]

# These values for the NCIP reanalysis
lons = nc.variables['lon'][:] # 144
lats = nc.variables['lat'][:] # 37
hgt = nc.variables['hgt']  #(36500, 1, 37, 144)
days = nc.variables['time'][:] #36500
hgt = np.swapaxes(np.squeeze(hgt),0,2) #reorder axes to be in MATLAB order (lon,lat,time)
#hgt now (144,37,36500)


'''This section for the ERA model
lons = nc.variables['longitude'][:] #240
lats = nc.variables['latitude'][:] # 121
gp = nc.variables['gp']  #(12775, 1, 121, 240)
days = nc.variables['time'][:] #12775
years = np.array(range(1979,2014)) - 1978
num_years = len(years)
daysperyear = 365
hgt = np.squeeze(gp)/9.8 # (12775,121,240) & divide by g to convert to units of actual height
hgt = np.swapaxes(hgt,0,2) #reorder axes to be in MATLAB order (lon,lat,time)
#hgt now (240,121,12775)
hgt = hgt[::2,lats>=0,:] #eliminate sounthern hemishere and cut lon resolution
hgt = hgt[:,::2,:]
lons = lons[::2]
lats = lats[lats>=0]
lats = lats[::2]
'''

#This file covers 25 years.  25 years*1460 obs/year = 36500 obs.

#We need to remove the quadratic trend from the raw data before performing PCA.
#First we need to "wrap" the data so that instead of a huge list of 36500 days, it is a list of 25 years, each with 1460 days

hgt4d = np.reshape(hgt,(lons.size,lats.size,daysperyear,num_years),order='F')

#we're going to fit a quadratic to every single point, one evry single day, over all 25 years.  So a fit to one location on all 25 Jan 15ths, a fit to one location on every Feb 10...

climate_trend_file = climdir+'smooth_climate_trend.npy'
if os.path.exists(climate_trend_file):
    print 'Trend file exists!  Woo-hoo!'
    smoothclim = np.load(climate_trend_file)
    new_num_days = len(days) - 2*daysperyear
else:
    t1=time.time()
    @jit
    def find_clim(heights):
        clim = np.zeros_like(heights)
        tmpyr = np.arange(heights.shape[3]) + 1
        x = np.vstack([tmpyr**2,tmpyr,np.ones(len(tmpyr))]).T
        for date in range(heights.shape[2]):
            for lat in range(heights.shape[1]):
                for lon in range(heights.shape[0]):
                    y = np.squeeze(heights[lon,lat,date,:])
                    bhat = np.linalg.lstsq(x,y)[0]
                    clim[lon,lat,date,:] = np.dot(x,bhat)
        return clim

    comp_input = np.zeros([10,10,10,10])
    junk = find_clim(comp_input)
    print 'Finding Climate Trend'
    clim = find_clim(hgt4d)
    #now we need a 3-D version of clim
    clim1 = np.reshape(clim,(lons.size,lats.size,days.size),order='F')
    new_num_days = len(days) - 2*daysperyear
    # We will be cutting off the first and last year after smoothing.
    # This counts the number of days in that new 33 year period: 12,045
    # We now run a 31 day moving average smoother on the entire 35 year dataset.
    # Since the edges are not properly smoothed, cut the first and last years,
    # leaving us with 33 good years.
    window_len = month 
    smoothclim = np.zeros_like(clim1)
    @jit
    def smooth(x,window_len=11):
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        w=np.ones(window_len,'d')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]

    for lon in range(len(lons)):
        for lat in range(len(lats)):
            smoothclim[lon,lat,:] = smooth(np.squeeze(clim1[lon,lat,:]),window_len)
    #now remove first and last years:
    smoothclim = smoothclim[:,:,daysperyear:(daysperyear+new_num_days)]
    np.save(climate_trend_file,smoothclim)
    print 'Finding climate trend took: '+ str(time.time()-t1)

hgt = hgt[:,:,daysperyear:(daysperyear+new_num_days)] - smoothclim
#hgt is now the middle 23 years of the original height file - the smoothed climate model
#Clear up some RAM:
clim1 = 0
smoothclim = 0
clim = 0

#calculate PCAs
y = np.reshape(hgt,(lons.size*lats.size,new_num_days),order='F')
#mean_hgt_series = np.mean(y,0) #spatial mean of height, at every time


def princomp(A):
    """ performs principal components analysis 
    (PCA) on the n-by-p data matrix A
    Rows of A correspond to observations, columns to variables. 
    
    Returns :  
    coeff :
    is a p-by-p matrix, each column containing coefficients 
    for one principal component.
    score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
    latent : 
    a vector containing the eigenvalues 
    of the covariance matrix of A.
    """
    # computing eigenvalues and eigenvectors of covariance matrix
    M = (A-np.mean(A.T,axis=1)).T # subtract the mean (along columns)
    [latent,coeff] = np.linalg.eig(np.cov(M)) # attention:not always sorted
    score = np.real(np.dot(coeff.T,M)) # projection of the data in the new space
    explained = 100.*np.real(latent/sum(latent)) #percent explained by each PC
    return coeff,score,latent,explained

print 'Performing PCA!'
t1 = time.time()
coeff,score,latent,explained = princomp(y.T)
print 'PCA calc took '+str(time.time() - t1)

thresh = 90.
percent_explained = 0
for num_retained,i in enumerate(explained):
    percent_explained = percent_explained + i
    if percent_explained >= thresh:
        break
num_retained += 1 #since num_retained starts at 0, it is currently 48, which means 49 are kept
##Create Index:
retain_coeffs = coeff[:,0:num_retained] #the ~49 coefficients which give 90% of variance.
score = score.T  #not sure why, but score comes out in a weird shape.

for last_index in range(1,17):
    counter = list(range(20))
    print counter[first_index:last_index]
    print 'keeping '+str(len(counter[first_index:last_index]))+' PCs'
    out_dir = '/home/driebel/Desktop/gfdl_pc_subsets/gfdl_post1979_'+str(last_index)+'/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir) #makedirs can do recursive costruction
    waterfall_file = out_dir+base+'.dat'

    variance = score[:,first_index:last_index]**2 
    #pull out the first 16 PCs, and sqaure them to turn into variances not std dev
    index = np.sum(variance,axis=1) # sum them.
    
    num_years = len(index)/daysperyear
    index_waterfall = np.reshape(index,(daysperyear,num_years),order='F')
    
    np.savetxt(waterfall_file,index_waterfall,delimiter=',')

    #clear some RAM
    hgt = 0
    hgt4d = 0
    #score = 0
    coeff = 0
    latent = 0
    explained = 0
    index_waterfall = 0

    print 'starting waterfall, '+str(boot_num)+' boots'
    os.system('python -m waterfall_fitting.py '+waterfall_file+' -boot_num '+str(boot_num))















