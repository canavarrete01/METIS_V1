#Packages -------------------------------------------------------------
using Plots
using Pandas
uusing SciPy
using CSV
using DataFrames

include("Constants.jl")

#Functions--------------------------------------------------------------


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
function eos_idealgas(P,mu,T)
    """
    equation of state
    relates rho to P assuming an ideal gas
    assumes P, mu and T are all provided in SI units
    output is density in kg/m^3
    """
    return P*mu/(kb*T)
end 

function fixed_mu_func(Z,T,P,fixedval=2.66)
    """
    function for implementing assumption 
    of a constant mean molecular weight throughout the
    atmosphere while still interfacing correctly with
    code which assumes mu func can vary with Z, T, and P
    
    output is array of mean molecular weight = fixedval  
    with same shape as array of temperatures
    units for fixedval should be in g/mol
    """
    return np.zeros(length(T)) + fixedval
end 

# ----------------------------------------------------------------------------------------------
# actual atmosphere set up for an isothermal atmosphere
# ----------------------------------------------------------------------------------------------

function atm_setup_1d(Temp,Mp,R0,P0,nwl,nalt,nlong,nlat,mu_func,Z)
    # Temp K
    # beta in RADIANS!!!
    # Mp in kg
    # R0 in m
    # P0 in Pascals
    # mu_func takes in atmosphere's Z, temp array, pressure array
    # and returns an array of mean molecular weights
    # Z is the metallicity of the atmosphere in terms of Zsun
    # function returns meshgrid of temps and pressures as one might 
    # get from a GCM, the corresponding ranges of altitude in m, 
    # longitude and latitude in radians, and those grid values
    # as a meshgrid
    
    # make up a meshgrid with appropriate extent
   
    long_range = np.linspace(0.0,2.0*np.pi,nlong)
    lat_range = np.linspace(0.0,np.pi,nlat)
    mu = 2.36 # just using this for a first guess at appropriate extent
    geff = G*Mp/R0^2
    Hday = kb*Temp/(mu*gmol_to_kg*geff)
    zmin, zmax = 0,20.0*Hday
    zrange = np.linspace(zmin,zmax,nalt)
    alt_range = zrange .+R0 

    atmosphere_grid =  np.meshgrid(alt_range,long_range,lat_range)
    #print(atmosphere_grid[1])
    
    Temps = np.zeros(size(atmosphere_grid[:1])) .+ Temp 
    
    # Make the matching pressure mapping on the 
    # atmosphere meshgrid    
    Pressures = np.zeros(size(atmosphere_grid[:1]))
    pressures = [P0]
    g = geff
    p = P0
    z = 0
    M = Mp
    mu = mu_func(Z,np.array([Temp]),np.array([p]))[0]*gmol_to_kg 
    r = R0
    for j in range(1,nalt,1)
        z1 = alt_range[j]-R0
        p1 = p*np.exp(-1.0*g*mu/(kb*Temp) * (z1-z)/(1+(z1-z)/(r+z)))
        pressures.append(p1)
        mu1 = mu_func(Z,np.array([Temp]),np.array([p1]))[0]*gmol_to_kg
        rho1 = eos_idealgas(p1,mu1,Temp)
        M1 = 4/3*np.pi*(z1+R0)^3.0*rho1 - 4/3*np.pi*(z+R0)^3.0*rho1 + M
        g1 = G*M1/(R0+z1)^2.0
        g, p, z, M, mu = g1, p1, z1, M1, mu1
    end 
    for k in range(nlat)        
        Pressures[:,:,k] = np.ones((nlong,nalt))*np.array(pressures)
    end 
        
    return Temps, Pressures, alt_range, long_range, lat_range, atmosphere_grid  
end 
