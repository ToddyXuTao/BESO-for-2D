"""
Parameters for a Michell structure with one simple support and one rollor.
"""
if __name__ == "__main__":
    # Material properties
    E = 1
    nu = 0.3
    
    # Mesh defination
    nelx = 100
    nely = 50
    
    # Optimization parameters
    vol_frac = 0.3
    penal = 3
    rmin = 3
    er = 0.02
    
    # Applying load
    load = Michell_One(nelx, nely, E, nu)
    
    # FEA solver
    fesolver = CvxFEA()
    
    # Whether to plot the images every iteration
    Plotting = True
    
    # Whether to save the final images
    Saving = True
    
    # BESO optimization
    optimization = BESO2D(load, fesolver)
    
    # Execute the data
    t = time.time()
    x = np.ones((nely, nelx))
    ke = load.lk(load.E, load.nu)
    u = fesolver.displace(load, x, ke, penal)
    
    # Topology optimization
    optimization.topology(vol_frac, er, rmin, penal, Plotting, Saving)
    
    # Print the time cost
    print('Time cost: ', time.time() - t, 'seconds.')