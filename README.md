Hello! If you are reading this you must have an interest in picking up my research.
In short I have been initializing line defects in a channel and seeing what happens when I allow the system to evolve over time. My experiment involved modifying the defect spacing and activity levels, but of course whatever you do will probably be different.
There are two components of this code: The simulation code you will run in HPCC and the analysis code you will run locally in Python (If you choose)

# Simulation Code
  Our simulation code sets up a 3D active nematic in which 2 line defects are initialized at a variable distance from the center of a rectangular channel. This code utilizes the multigrid method in order to solve the nematohydrodynamic equations over our 3D grid. All of this is run on HPCC. 
  There are two main files you will be modifying in this: example.c and interface.c (mainly example.c)
  example.c handles the bulk of our calculations, whereas interface is a backend with functions that example calls. This simulation code is all in a folder called "Template". You can run a simulation in this folder (or preferably a copy of this folder) with the command "sbatch run.sh". This will generate a folder called "test" which contains Q tensor and velocity (if enabled) data files for each timestep. After this code is done running, you can should download it. 
  ## example.c
  In example.c you can set the dimensions of the channel as well as the dimensions of the grid that will be numerically integrated over by modifying LX, LY, LZ and NX,NY,NZ respectively. The ratio LX/NX, LY/NY, and LZ/NZ will determine our grid spacing. A lower number means finer integration, while a higher number is coarser.  You can also modify the activity number ALPHA (Modifying system height LZ will have a similar effect on nondimensionalized activity number H=   $\sqrt\frac{ \alpha}{k}$). You can also modify the timestep and total time integrated.\
  Deeper into the code our director field is defined. The locations of the defects and their radii are set, and then the director field is defined by angle $\theta(r)=-\frac{1}{2}\tan(\frac{y}{x})+\frac{1}{2}\tan(\frac{y}{x-R})+$ $\frac{\delta \theta}{2} [1+\frac{\log(x^2+y^2)-\log((x-R)^2+y^2)}{\log(R^2) − \log(r^2)}]$. Finally we set our initial director field values $nx=cos(\theta)$, $ny=sin(\theta)$, and $nz=0$.


  ## interface.c
  Serves as a backend for example.c to interact with. You will be modifying this much less than example.c but there are a few things to keep in mind
  In interface.c the boundary conditions for the velocities have been set such that the two walls along the x axis are periodic and every other wall is no slip. The boundary conditions are set in two different places, beginning on lines 104 and 310 respectively. You must modify the mob variables in each for loop to fit whatever boundary conditions you wish to create.
  <img width="1256" height="597" alt="Image" src="https://github.com/user-attachments/assets/7c2a8828-a928-4204-8476-68a2e0058526" /> Additionally the director field is defined in the same manner as in example.c in 2 places. Remember to modify sections starting at lines 92 and 295. each instance of the variables dist and rcore should be the same across example.c and interface.c. 
## run.sh
This code tells the HPCC how to run example. Simply make sure that cpus-per-task is equal to your NUMPROCS in example.c. Also change the mail-user variable from my email to whatever your email is.

  
 

# Analysis Code: 
 This code extracts our data from the test folder and gives us arrays representing the Q tensor at each point in space and time and velocity at each point in space and time. The entire template folder is required for this code to run, not just test. This is because it extracts other parameters such as channel size, activity, grid spacing, etc, from our example.c file. This data is also fed into an hdf5 file system. In practice we do not use the hdf5 filesysten but the infrastrucure is there if you want to use it.:

  ## Base.py   
  We build most of our quiver plotting functions off of this code. It unpacks our Q and u data files and puts them into usable arrays. You can find a more        rigorously comnmented and more robust version of this code in https://github.com/rabiddonkey33/nematicscode.git if you desire. This extra code can help us quantify turbulence by calling an autocorellation function. Base is much of what I actually ended up using and it is much uglier than the github code but it gets the job done. Make sure to change the variable "root" to wherever you download your copy of "Template". 

 ## xyquiver, xzquiver, yzquiver
 Generates a velocity quiver plot (with a color map for vorticity) in the respective planes for a preset timepoint. I would suggest looking at the xyquiver code to understand this.
  <img width="467" height="227" alt="Image" src="https://github.com/user-attachments/assets/e4c555cf-5597-4445-a8a1-fb71afc34f4b" />
  
  
  ## Splot.py
  Generates a plot of the order parameter S in whichever crossection you choose. Options for x-z, y-z, and x-y planes are commented.
  <img width="448" height="223" alt="Image" src="https://github.com/user-attachments/assets/a5ab8265-34f5-45d1-8edc-d57cc9d2dfdf" />
 ## Frank.py
 Calculates and plots each type of Frank deformation (bend, twist, splay) of a system at each time point
 <img width="500" height="250" alt="Screenshot 2026-03-23 101545" src="https://github.com/user-attachments/assets/01cde219-609f-4f97-8dc6-761674255e7e" />
  
## Visualizer.py
Generates figures representing 3D disclination lines and nematics at boundaries for each time point looped throught (currently from 1 to 300). This can easily be compiled into a video. 
  <img width="637" height="180" alt="image" src="https://github.com/user-attachments/assets/329b347d-7c21-41b5-b290-330ca42db15e" />


  
  
