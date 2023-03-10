#integrate_transits

#Packages------------------------------------------------------------
using PyCall
np = pyimport("numpy")
using SciPy

#Functions

# -- general geometry and transformations --------------------
function cylinder_to_cartesian(rho, theta, x)
    # rho and x in same unit of length
    # theat in radians
    zz = x
    xx = rho*np.cos(theta)
    yy = rho*np.sin(theta)
    return np.array([xx,yy,zz])
end

function cylinder_to_sphere(rho, theta, x)
    # rho and x in same unit of length
    # theat in radians
    xx,yy,zz = cylinder_to_cartesian(rho,theta,x)    
    r,long,lat = cartesian_to_sphere(xx,yy,zz) 
    return np.array([r,long,lat])  
end 

function cartesian_to_sphere(x,y,z)
    # x, y, z in same units of length
    # r same unit of length
    # long and lat in radians
    r = np.sqrt(x^2.0+y^2.0+z^2.0)
    lat = np.arccos(z/r) 
    long = np.arctan2(y,x) # arctan domain is - pi/2 to pi/2, arctan2 uses quadrants of y and x  to get right value 
    long = (long + 2.0*np.pi)%(2.0*np.pi)
    return np.array([r, long, lat])
end 

function calc_path_lengths(alt, long, lat, rho, theta)
    """
    # given a ray (defined in cylindrical coordinates by rho, theta)
    # determine the path length traversed in each of your
    # spherical grid cells
    # alt - in meters, expected to be the full value of z + Rp  for now
    # long - in radians, 0 and 2pi are at "north pole" if planet moves
    #        west to east, which pi is at "south pole" *(might be opposite, should check)*
    # lat - in radians, small latitudes face the star, large latitudes 
    # rho - expected in meters, rho = 0 is the center of the planet 
    # theta - expected in radians, goes along with longitude essentially
    """
    # put your ray into spherical coordinates with evenly divided dx
    nx = 1000 # THIS VALUE NEEDS TO BE TESTED... 
    if np.max(rho) > np.max(alt)
        print("ray falls outside atmosphere completely")
        return None
    end 
    
    # choose xrange carefully so that you don"t exceed atmosphere bounds 
    xmag = np.sqrt(np.max(alt)^2.0-rho^2.0)
    x = np.linspace(-1.0*xmag, xmag,nx)  # get the inwards and outwards directions
    rhog, thg, xg = np.meshgrid(rho,theta,x) 
    ray_sph_meshgrid = cylinder_to_sphere(rhog,thg,xg) 
    D=3
    ray_sph_tuples = np.reshape(ray_sph_meshgrid, (D,nx))
    ray_sph_tuples = ray_sph_tuples.T  # dimensions are npoints x 3 for griddata to work
    
    # now assign ray segments to spherical reference cell they belong to... 
    assigned_alt = griddata(alt.T, alt.T, ray_sph_tuples[:,0], method="nearest")  
    assigned_long = griddata(long.T, long.T, ray_sph_tuples[:,1], method="nearest")
    assigned_lat = griddata(lat.T, lat.T, ray_sph_tuples[:,2], method="nearest") 
    assignments = np.array([assigned_alt, assigned_long, assigned_lat]).T  # dimensions are also npoints x 3 
    
    # use these assignments to get the path lengths and coresponding spherical reference cells
    # first pick out unique cells
    assigned = [(assignments[k,0],assignments[k,1],assignments[k,2]) for k in range(len(assignments))]
    unique_cells = []
    for k in range(length(assigned))
        if assigned[k] not in unique_cells
            unique_cells.append(assigned[k])
        end 
    end 
    
    # then count how many ray segments fall into each
    counts = np.zeros(len(unique_cells))
    r_i, long_i, lat_i = np.zeros(len(unique_cells)), np.zeros(len(unique_cells)), np.zeros(len(unique_cells))

    for k in range(len(unique_cells))
        r_i[k], long_i[k], lat_i[k] = unique_cells[k][0],unique_cells[k][1],unique_cells[k][2]
        for j in range(length(assigned))
            if assigned[j] == unique_cells[k]
                counts[k] =  counts[k] + 1 
            end 
        end 
    end 

    # sum up segments within same cell to get correct path length
    dx_i = counts*(xmag*2.0/nx)

    # recover the indices within alt, lat, long of the unique cell values
    long_ind = [ np.where(long==long_i[k])[0][0] for k in range(len(dx_i))]
    alt_ind = [ np.where(alt==r_i[k])[0][0] for k in range(len(dx_i))]
    lat_ind = [ np.where(lat==lat_i[k])[0][0] for k in range(len(dx_i))]
    unique_cell_indices = np.array([np.array(alt_ind),np.array(long_ind),np.array(lat_ind)])
    
    # return the path lengths + corresponding spherical reference cell values, and the indices within the 
    # atmosphere grid
    return assignments, np.array([dx_i, r_i, long_i, lat_i]), unique_cell_indices  
end 

function compute_transmittance(alt, long, lat, Tmap, Pmap, OpacFunc, rho, theta, Z, wlrange)
    """
    TRANSMITANCE = integrate along x to get tau(rho,theta) 
    # alt, long, lat are 1d specfiying a spherical grid 
    # alt = z + Rp
    # long = theta both in radians
    # lat = 0 at substellar point, Pi at center of night side
    # Tmap and Pmap contain temperatures and pressures at points in np.meshgrid(alt, long, lat)
    # OpacFunc takes in T,P pairs and returns the appropriate opacity
    # rho and theta specify the ray from star through atmosphere towards observer
    # rho is zero at center of planet, theta goes with longitude
    """ 
    assign_tuples, unique_cell_values, unique_cell_indices = calc_path_lengths(alt, long, lat, rho, theta)
    dx_i, r_i, long_i, lat_i = unique_cell_values
    r_ind, long_ind, lat_ind = unique_cell_indices
    
    # retrieve appropriate atmosphere properties for each cell
    T_i = [Tmap[ long_ind[k],r_ind[k],lat_ind[k] ] for k in range(len(dx_i))]
    P_i = [Pmap[ long_ind[k],r_ind[k],lat_ind[k] ] for k in range(len(dx_i))]
    opac_i = OpacFunc(Z, np.array(T_i), np.array(P_i), wlrange)
    
    # sum up the optical depth 
    tau =  np.sum(opac_i.T*dx_i,axis=1) # opac_i may have multiple wavelengths, then tau has multiple wavelengths
    return tau
end


function effective_area_at_full_depth_no_long_dep(alt, long, lat, Tmap, Pmap, OpacFunc, nwl, nrho, Z, wlrange)
    """ 
    assumes planet is right in center of transit, entire disk over star
    and that there is no longitudinal dependence
    integrate through rho to get the amount of light absorbed
    at each wavelength
    """
    tau = []
    rho_range = np.linspace(alt[0],alt[-1],nrho)
    theta = np.array([np.pi]) # since we have no longitude dependence don"t actually
                              # worry about theta
    for rho in rho_range 
        tau.append(compute_transmittance(alt, long, lat, Tmap, Pmap, OpacFunc, np.array([rho]), theta, Z, wlrange))
    end 

    tau_array =  np.array(tau)                              # shape (Nrho, Nwl)
    tiled_rho = np.tile(rho_range, (nwl,1)).T               # shape (Nrho, Nwl) 
    annulus_obscurations = 2.0*np.pi*tiled_rho*(1-np.exp(-1.0*tau_array)) # shape (Nrho, Nwl) 
    atmos_obscuration = np.trapz(annulus_obscurations,tiled_rho,axis=0) # shape (Nwl,)
    return (atmos_obscuration+np.pi*alt[0]^2.0)      
end 
