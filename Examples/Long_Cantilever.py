"""
Parameters for a long cantilever.
"""
if __name__ == "__main__":
    # Material properties
    E = 1
    nu = 0.3
    
    # Mesh defination
    nelx = 160
    nely = 40
    
    # Optimization parameters
    vol_frac = 0.5
    penal = 3
    rmin = 4
    er = 0.02
    
    # Applying load
    load = Cantilever(nelx, nely, E, nu)
    
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