using PyCall
np = pyimport("numpy")
# ----------------------------------------------------------------------------------------------
# useful constants and conversions :
# ----------------------------------------------------------------------------------------------
G = 6.67408*10^-11       # gravitational constant, SI units
kb = 1.38064852*10^-23   # boltzmann k, SI units
c = 299792458.0           # speed of light, m/s
Mj = 1.898*10^27         # mass of jupiter,  kg
Rj = 6.9911*10^7.0       # radius of jupiter,  m
Rsun = 6.9551*10^8       # radius of the sun, m

gmol_to_kg = 10.0^-3.0 / (6.022140857*10^23) 
pascals_to_bars = 10^-5
bars_to_atm = 0.986923
kgcm_to_gccm = 0.001 
au_to_m = 1.496*10^11
amu_to_kg = 1.66054e-27
gcc_to_kgcm = 1000.0

# ----------------------------------------------------------------------------------------------
# information needed to get mean molecular weight from abundance tables :
# ----------------------------------------------------------------------------------------------

fullspecieslist = ["H2", "He", "CH4", "CO", "N2", "NH3", "H2O", 
                  "PH3", "H2S", "TiO", "VO", "CaH", "MgH", "Li", 
                  "Na", "K", "Rb", "Cs", "FeH", "CrH", "H-", "H", 
                  "H+", "e-", "Fe", "SiO", "CaOH","TiH", "Al", "Ca"]
    
speciesdict = Dict(fullspecieslist .=> range(0, length = length(fullspecieslist)))


# molar masses in g/mol
molarmassdict= Dict( "H2"=>2.01588, "He"=>4.002602, "CH4"=>16.04, "CO"=>28.01, 
                "N2"=>28.013, "NH3"=>17.031, "H2O"=>18.01528, 
                "PH3"=> 33.99758, "H2S"=>34.1, "TiO"=>63.866, "VO"=>66.9409, 
                "CaH"=>42.094, "MgH"=>25.313, "Li"=>6.941, 
                "Na"=>22.989769, "K"=>39.0983, "Rb"=>85.4678, "Cs"=>132.90545, 
                "FeH"=>55.845+1.00794, "CrH"=>54.0040, 
                "H-"=>1.00794+5.48579909070*10^-4, "H"=>1.00794, 
                "H+"=>1.00794-5.48579909070*10^-4, "e-"=>5.48579909070*10^-4, 
                "Fe"=>55.845, "SiO"=>44.08, "CaOH"=>74.093,"TiH"=>47.867+1.00794, 
                "Al"=>26.981539, "Ca"=>40.078)

molar_masses = np.zeros(30)
[(x,y) for x in fullspecieslist for y in molarmassdict[x]]
 

# ----------------------------------------------------------------------------------------------
# directories where pre-computed quantities are stored :
# ----------------------------------------------------------------------------------------------
chempath = "Tables/EqChemAbund"
gaspath = "Tables/GasOpacity/"
miepath = "Tables/Mie"
rayleighpath = "Tables/Rayleigh"

