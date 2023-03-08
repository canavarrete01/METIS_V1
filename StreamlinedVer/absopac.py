# code for interpolating adam's tables
import numpy as np
import pandas
from scipy.interpolate import RectBivariateSpline, interp1d, RegularGridInterpolator
import constants

# ----------------------------------------------------------------------------------------------    
# abundance file input functions:
# ----------------------------------------------------------------------------------------------
def read_abund_file(filename):
    '''
     adams tables are in the form 805 temps * 108 pressures = 86940 rows
     32 columns, with the first corresponding to log(Temp), 
     the second corresponding to log(pressure)
     the 30 species are: H2, He, CH4, CO, N2, NH3, H2O, PH3, H2S, TiO, VO, CaH, MgH, Li, 
     Na, K, Rb, Cs, FeH, CrH, H-, H, H+, e-, Fe, SiO, CaOH,TiH, Al, Ca
     the first row of the table is specifying the bounding T & P
     '''
    dataframe = pandas.read_csv(filename,sep='\s+',skiprows=1,header=None)
    data = dataframe.values
    x,y = data.shape # array shape should be (86940, 32)
    if x == 86940 and y == 32:
        print('success reading: %s'%filename)
        return data 
    else: 
        print('check format of input table: %s'%filename)
        return 1

def convert_2d_to_3d_array(data):
    ''' 
     returns the temperatures in K,
     the pressures in bars,
     the 3-d array of data in the following form:
     30 diff species, 805 diff temperatures, 108 diff pressures
     ***NOTE*** that the temps are descending from high to low, 
     and the pressures are decreasing from high to low
     ''' 
    sep = np.split(data,108,axis=0)
    full = np.dstack(sep)
    temps = 10.0**full[:,0,0]
    pressures = 10.0**full[0,1,:]
    cut = full[:,2:32,:]
    arrange = np.swapaxes(cut,0,1)
    return temps, pressures, arrange

def load_abund_table(filename):
    '''returns t in K
               p in atm
               cube with abundances of 30 species, for grid of 
               805 temperatures, and 106 pressures       
               cube shape - (30,805,106)
    '''
    data = read_abund_file(filename)
    t,p,cube = convert_2d_to_3d_array(data)
    return t,p,cube

# ----------------------------------------------------------------------------------------------
# the next few functions
# create general interpolators to be used for mean molecular weight, chemical abundances
# gas absorption opacities and rayleigh scattering opacities which work in log space
# so that we can use RegularGridInterpolator and still cover a wide range of temperature/pressure
# you must call these once, then their output interpolation functions
# are saved as global variables and
# wrapped inside abund_func; mu_func; gasopac_func; and rayleigh_func
# ----------------------------------------------------------------------------------------------
def initiate_mu_abund_table_interpolators():
    '''
    user will need to hardcode the path to the chemical abundance
    files they want to use
    
    returns mufunc -- interpolates in log10(Z solar) log10(t K), log10(p atm)  
                      to get the mean weight of molecule in grams per mole
            afunc --- interpolates in log10(Z solar) log10(t K), log10(p atm) 
                      to get the mixing rations of 30 species
    '''
    mu_cube, abund_cube = [], [] # these will end up with shape (nZ,nt,np,nspec)
    for Z in [0.1,0.316,1.0,3.16]:
        # ---------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------
        # HARDCODED  PATH TO CHEMICAL ABUNDANCES HERE
        filename = constants.chempath+'/abundance_%.3fsolar_rainoutTiVH2OFe.dat.gz'%Z
        # ---------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------
        t,p,cube = load_abund_table(filename)  
        # load_abund_table outputs t in descending order, units of K
        # p in descending order, units of atm
        # cube of molar mixing ratios with shape (nspecies, ntemps, npressures)
        # so need to flip the data around so that the grid's points
        # of t and p are in ascending order for RegularGridInterpolator       
        cube2 = np.swapaxes(cube.copy(),0,1)
        cube3 = np.swapaxes(cube2,1,2)
        for k in range(30):
            cube3[:,:,k] = np.fliplr(cube3[:,:,k])
            cube3[:,:,k] = np.flipud(cube3[:,:,k])
        abund_cube.append(cube3)        
        
        # now use abundances to get the corresponding mean molecular weight of atmo
        mu_grid = np.zeros((len(t),len(p)))
        for k in range(30):
            mu_grid[:,:] = mu_grid[:,:] + constants.molar_masses[k]*cube3[:,:,k] 
        mu_cube.append(mu_grid)
        
    log_Z = np.log10(np.logspace(-1,1,4))    
    log_t = np.log10(t)[::-1]
    log_p = np.log10(p)[::-1] 
    mufunc = RegularGridInterpolator((log_Z,log_t,log_p),np.array(mu_cube),method='linear')
    afunc = RegularGridInterpolator((log_Z,log_t,log_p),np.array(abund_cube),method='linear')

    return mufunc, afunc # (log10_Z in Z sun, log10_T in K , log10_P in atm)

def initiate_opac_table_interpolator():
    '''
    user will need to hardcode the path to the gaseous absorption opacity
    files they want to use
    
    returns  a function which interpolates in (log10_Z, ln_freq, ln_densities, ln_temperatures) 
            and returns the opacities in cm^2/gram

    note that opacity reference tables are saved as cm^2/g on a 50 x 50 grid of 
    densities and temperatures where the densities and temperatures 
    are evenly spaced in natural log. The grid points are just
    hardcoded in here rather than read off of the table
    ''' 
    ln_densities = np.array([ 
       -27.63000000000000,        -27.16000000000000,        -26.69000000000000,      
       -26.22000000000000,        -25.75000000000000,        -25.28000000000000,      
       -24.81000000000000,        -24.34000000000000,        -23.87000000000000,      
       -23.40000000000000,        -22.93000000000000,        -22.46000000000000,      
       -21.99000000000000,        -21.52000000000000,        -21.05000000000000,      
       -20.58000000000000,        -20.11000000000000,        -19.64000000000000,      
       -19.17000000000000,        -18.70000000000000,        -18.23000000000000,      
       -17.76000000000000,        -17.29000000000000,        -16.82000000000000,      
       -16.35000000000000,        -15.88000000000000,        -15.41000000000000,      
       -14.94000000000000,        -14.47000000000000,        -14.00000000000000,      
       -13.53000000000000,        -13.06000000000000,        -12.59000000000000,      
       -12.12000000000000,        -11.65000000000000,        -11.18000000000000,      
       -10.71000000000000,        -10.24000000000000,        -9.773999999999999,      
       -9.304000000000000,        -8.834000000000000,        -8.364000000000001,      
       -7.895000000000000,        -7.425000000000000,        -6.955000000000000,      
       -6.485000000000000,        -6.015000000000000,        -5.545000000000000,      
       -5.075000000000000,        -4.605000000000000])  # densities are in ln of g per cubic centimeter...

    ln_temperatures = np.array([
        3.912000000000000,         4.006000000000000,         4.100000000000000,      
        4.194000000000000,         4.288000000000000,         4.382000000000000,      
        4.476000000000000,         4.570000000000000,         4.664000000000000,      
        4.758000000000000,         4.852000000000000,         4.946000000000000,      
        5.040000000000000,         5.134000000000000,         5.228000000000000,      
        5.322000000000000,         5.416000000000000,         5.510000000000000,     
        5.604000000000000,         5.698000000000000,         5.792000000000000,      
        5.886000000000000,         5.980000000000000,         6.074000000000000,      
        6.168000000000000,         6.262000000000000,         6.356000000000000,      
        6.450000000000000,         6.544000000000000,         6.638000000000000,      
        6.732000000000000,         6.825000000000000,         6.919000000000000,      
        7.013000000000000,         7.107000000000000,         7.201000000000000,      
        7.295000000000000,         7.389000000000000,         7.483000000000000,      
        7.577000000000000,         7.671000000000000,         7.765000000000000,      
        7.859000000000000,         7.953000000000000,         8.047000000000001,      
        8.141000000000000,         8.234999999999999,         8.329000000000001,      
        8.423000000000000,         8.516999999999999]) # temperature ranges up to ~ 5000 K


    cspeed = 2.99792458E10 # cm/s
    f0 = 999308193333.3333  # freq 1 in Hz
    ff = 999308193333334.0  # last freq in Hz
    ln_freq = np.linspace(np.log(f0),np.log(ff),5000) # lower freq =  longer wl higher freq=shorter wl
    
    log10_Z = np.log10(np.logspace(-1,1,4)) # np.log10(np.array([0.1,0.316,1.0,3.16]))
    values = []
    for Z in [0.1,0.316,1.0,3.16]:
        # ---------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------
        # HARDCODED  PATH TO GAS ABSORPTION OPACITIES HERE
        fname = constants.gaspath+'absopac.noTiOVO.%.3fsolar.dat'%Z
        # ---------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------
        dataframe = pandas.read_csv(fname,sep='\s+',header=None) 
        print('success reading %s'%fname)
        data = dataframe.values
        values.append(data.reshape((5000,50,50)))
        
    opac_func = RegularGridInterpolator((log10_Z, ln_freq,ln_densities,ln_temperatures), 
                                        np.array(values), method='linear', 
                                        bounds_error=False, fill_value=-10.0)

    return opac_func # (log10_Z, ln_freq, ln_densities, ln_temperatures)

def initiate_rayleigh_table_interpolator():
    '''
    Rayleigh tables have same temperature and density grid as absopac files
    but they only store the opacity in cm^2/g at a single wavelength
    which is then scaled with wavelength^-4, so there is no
    frequency interpolation here
    
    user again needs to hardcode in a path to the desired rayleigh tables
    
    function returns a function which will interpolate the reference rayleigh
    opacity in cm^2/g at a given log10_Z, ln_density, ln_temperature
    '''

    ln_densities = np.array([ 
       -27.63000000000000,        -27.16000000000000,        -26.69000000000000,      
       -26.22000000000000,        -25.75000000000000,        -25.28000000000000,      
       -24.81000000000000,        -24.34000000000000,        -23.87000000000000,      
       -23.40000000000000,        -22.93000000000000,        -22.46000000000000,      
       -21.99000000000000,        -21.52000000000000,        -21.05000000000000,      
       -20.58000000000000,        -20.11000000000000,        -19.64000000000000,      
       -19.17000000000000,        -18.70000000000000,        -18.23000000000000,      
       -17.76000000000000,        -17.29000000000000,        -16.82000000000000,      
       -16.35000000000000,        -15.88000000000000,        -15.41000000000000,      
       -14.94000000000000,        -14.47000000000000,        -14.00000000000000,      
       -13.53000000000000,        -13.06000000000000,        -12.59000000000000,      
       -12.12000000000000,        -11.65000000000000,        -11.18000000000000,      
       -10.71000000000000,        -10.24000000000000,        -9.773999999999999,      
       -9.304000000000000,        -8.834000000000000,        -8.364000000000001,      
       -7.895000000000000,        -7.425000000000000,        -6.955000000000000,      
       -6.485000000000000,        -6.015000000000000,        -5.545000000000000,      
       -5.075000000000000,        -4.605000000000000])  

    ln_temperatures = np.array([
        3.912000000000000,         4.006000000000000,         4.100000000000000,      
        4.194000000000000,         4.288000000000000,         4.382000000000000,      
        4.476000000000000,         4.570000000000000,         4.664000000000000,      
        4.758000000000000,         4.852000000000000,         4.946000000000000,      
        5.040000000000000,         5.134000000000000,         5.228000000000000,      
        5.322000000000000,         5.416000000000000,         5.510000000000000,     
        5.604000000000000,         5.698000000000000,         5.792000000000000,      
        5.886000000000000,         5.980000000000000,         6.074000000000000,      
        6.168000000000000,         6.262000000000000,         6.356000000000000,      
        6.450000000000000,         6.544000000000000,         6.638000000000000,      
        6.732000000000000,         6.825000000000000,         6.919000000000000,      
        7.013000000000000,         7.107000000000000,         7.201000000000000,      
        7.295000000000000,         7.389000000000000,         7.483000000000000,      
        7.577000000000000,         7.671000000000000,         7.765000000000000,      
        7.859000000000000,         7.953000000000000,         8.047000000000001,      
        8.141000000000000,         8.234999999999999,         8.329000000000001,      
        8.423000000000000,         8.516999999999999])

    log10_Z = np.log10(np.logspace(-1,1,4)) #np.log10(np.array([0.1,0.316,1.0,3.16]))
    
    values = []
    for Z in [0.1,0.316,1.0,3.16]:
        # ---------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------
        # HARDCODED  PATH TO RAYLEIGH SCATTERING OPACITIES HERE
        fname = constants.rayleighpath+'/rayleigh.%.3fsolar.dat.gz'%Z
        # ---------------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------------
        dataframe = pandas.read_csv(fname,sep='\s+',header=None) 
        print('success reading %s'%fname)
        values.append(dataframe.values)
        
    rayleigh_table = RegularGridInterpolator((log10_Z,ln_densities,ln_temperatures), 
                                        np.array(values), method='linear', bounds_error=False, fill_value=-10.0)
    return rayleigh_table # (log10_Z, ln_densities, ln_temperatures)

#  ----------------------------------------------------------------------------------
#  ----------------------------------------------------------------------------------
# Initiate the interpolation tables in log space using functions defined above then 
# Make wrapper chemistry and opacity related
# interpolation functions with nicer units (i.e. not log values, wavelengths instead
# of frequencies etc.) which also have methods of treating points that fall
# outside the bounds of the saved tables
#  ----------------------------------------------------------------------------------
#  ----------------------------------------------------------------------------------

MU_TABLE, ABUND_TABLE = initiate_mu_abund_table_interpolators() # interpolates grids in (log10Z, log10_t, log10_p)  
GASOPAC_TABLE = initiate_opac_table_interpolator() # interpolates grid in (log10Z, ln_freq, ln_rho, ln_T)
RAYLEIGH_TABLE = initiate_rayleigh_table_interpolator() 

def abund_func(Z,T,P):
    # T - temperature in kelvin
    # P - pressure in pascals
    # returns 30 x nTP grid of molar mixing raitos
    # species are in order:
    #  H2, He, CH4, CO, N2, NH3, H2O, PH3, H2S, TiO, VO, CaH, MgH, Li, 
    #  Na, K, Rb, Cs, FeH, CrH, H-, H, H+, e-, Fe, SiO, CaOH,TiH, Al, Ca
    # and their indices can be easily obtained from the speciesdict
    # created in the constants module
    # defined in absopac.py
    log10_p = np.log10(np.abs(P*constants.pascals_to_bars*constants.bars_to_atm))
    log10_t = np.log10(T)
    log10_p = np.log10(np.abs(P*constants.pascals_to_bars*constants.bars_to_atm))
    log10_t = np.log10(np.abs(T))
    pmin = 8e-9
    mask = np.where(log10_p < np.log10(pmin))
    log10_p[mask] = np.log10(pmin)
    pmax = 400
    mask2 = np.where(log10_p > np.log10(pmax))
    log10_p[mask2] = np.log10(pmax)
    tmin = 50
    mask3 = np.where(log10_t < np.log10(tmin))
    log10_t[mask3] = np.log10(tmin)
    tmax = 5000
    mask4 = np.where(log10_t > np.log10(tmax))
    log10_t[mask4] = np.log10(tmax) 
    log10_Z = np.log10(Z)+np.zeros(len(log10_t))
    return  ABUND_TABLE((log10_Z,log10_t,log10_p)) # molar mixing ratio    

def mu_func(Z,T,P):
    # T - temperature in kelvin
    # P - pressure in pascals
    # returns the mean molecular weight of the atmosphere (gas only... ignores particulates)
    log10_p = np.log10(np.abs(P*constants.pascals_to_bars*constants.bars_to_atm))
    log10_t = np.log10(np.abs(T))
    pmin = 8e-9
    mask = np.where(log10_p < np.log10(pmin))
    log10_p[mask] = np.log10(pmin)
    pmax = 400
    mask2 = np.where(log10_p > np.log10(pmax))
    log10_p[mask2] = np.log10(pmax)
    tmin = 50
    mask3 = np.where(log10_t < np.log10(tmin))
    log10_t[mask3] = np.log10(tmin)
    tmax = 5000
    mask4 = np.where(log10_t > np.log10(tmax))
    log10_t[mask4] = np.log10(tmax)   
    log10_Z = np.log10(Z)+np.zeros(len(log10_t))
    return MU_TABLE((log10_Z,log10_t,log10_p))

def gasopac_func(Z,wl,T,rho):
    # Z is the metallicity in multiples of solar metallicity
    # wl -  wavelength in microns
    # T - temperature in kelvin
    # rho -  density in kg/m^3
    # returns the gas opacity
    # in units of cm^2/gram
    log10Z = np.log10(Z)
    ln_rho_T = np.vstack((np.log(np.abs(rho*constants.kgcm_to_gccm)),np.log(T))) # now have (2,50) 
                                                                       # [0] are the densities 
                                                                       # [1] are the temperatures  

    ln_freq = np.log(constants.c*10.0**6.0/wl)        
    nwl = len(wl)
    nt = len(T)
    ln_rho_T_repeated = np.tile(ln_rho_T,nwl) # repeat T-rho pairs nwl times, 
                                              # first ntp go through all pairs then sequence repeats
    ln_freq_repeated = np.repeat(ln_freq,nt)  # repeat frequencies ntp times, 
                                              # first ntp are fr1 1, next ntp are fr 2 etc.
    tuples = np.zeros((nwl*nt,4))  
    tuples[:,0] = log10Z
    tuples[:,1] = ln_freq_repeated[:] 
    tuples[:,2] = ln_rho_T_repeated[0] 
    tuples[:,3] = ln_rho_T_repeated[1]
    results = GASOPAC_TABLE(tuples) 
    return (np.exp(results.reshape(nwl,nt))*10**-1*rho).T # has shape (nT, nwl), units are 1/m    

def rayleigh_func(Z,wl,T,rho):
    nwl = len(wl)
    nt = len(T)
    wl0 = 1.0E6*constants.c/5.0872638E14 # reference wavlength for the rayleigh tables in microns
    ln_rho = np.log(np.abs(rho*constants.kgcm_to_gccm))
    ln_T = np.log(T)
    log10Z = np.zeros(nt)+np.log10(Z)
    sig0 = RAYLEIGH_TABLE((log10Z,ln_rho,ln_T))
    wlscaling = (wl/wl0)**-4
    rayleigh = np.repeat(wlscaling,nt).reshape(nwl,nt)*np.exp(sig0) # has shape (nT, nwl), units are cm^2/g
    return ((rho*10.0**-1)*rayleigh).T  # has shape (nT, nwl), units are 1/m



