Hello! If you are reading this you must have an interest in picking up my research.
In short I have been initializing line defects in a channel. My experiment involved modifying the defect spacing and activity levels, but of course whatever you do will probably be different.
There are two components of this code: The simulation code you will run in HPCC and the analysis code you will run locally in Python (If you choose)

# Simulation Code
  Our simulation code sets up a 3D active nematic in which 2 line defects are initialized at a variable distance from the center of a rectangular channel. This code utilizes the multigrid method in order to solve the nematohydrodynamic equations over our 3D grid. All of this is run on HPCC.
  There are two main files you will be modifying in this: example.c and interface.c (mainly example.c)
  example.c handles the bulk of our calculations, whereas interface is a backend with functions that example calls.
  ## example.c
  In example.c you can set the dimensions of the channel as well as the dimensions of the grid that will be numerically integrated over by modifying LX, LY, LZ and NX,NY,NZ respectively. The ratio LX/NX, LY/NY, and LZ/NZ will determine our grid spacing, lower number means finer results, higher number is coarser.  You can also modify the activity number ALPHA (Modifying system height LZ will have a similar effect on nondimensionalized activity number H=   $\sqrt\frac{ \alpha}{k}$. You can also modify the timestep and total time integrated.\
  Deeper into the code our director field is defined. The locations of the defects and their radii are set, and then the director field is defined by angle $\theta(r)=-\frac{1}{2}\tan(\frac{y}{x})+\frac{1}{2}\tan(\frac{y}{x-R})+$ $\frac{\delta \theta}{2} [1+\frac{\log(x^2+y^2)-\log((x-R)^2+y^2)}{\log(R^2) − \log(r^2)}]$. Finally we set our initial director field values $nx=cos(\theta)$, $ny=sin(\theta)$, and $nz=0$.
 After running a simulations (with the command sbatch run.sh) a folder labeled "test" will appear in your directory with data files for each component of the Q tensor and velocity (if you have that enabled) for each time step

  ## interface.c
  Serves as a backend for example.c to interact with. You will be modifying this much less than example.c but there are a few things to keep in mind
  In interface.c the boundary conditions for the velocities have been set such that the two walls along the x axis are periodic and every other wall is no slip. This begins on line 104. Additionally the director field is defined in the same manner as in example.c in 2 places. Remember to modify sections starting at lines 92 and 295. each instance of the variables dist and rcore should be the same across example.c and interface.c. 
## run.sh
This code tells the HPCC how to run example. Simply make sure that cpus-per-task is equal to your NUMPROCS in example.c. Also change the mail-user variable from my email to whatever your email is.

  
 

# Analysis Code: 
 This code extracts our data from the test folder and puts it into an hdf5 file system (in practice we do not use the hdf5 filesysten but the infrastrucure is there if  you want to use it). In mypythonlib you will find a list of basic functions that we call in our analysis code. Our analysis code consists of:

  ## Base.py   
  We build most of our quiver plotting functions off of this code. It unpacks our Q and u data files and puts them into usable arrays. You can find a more        rigorously comnmented and more robust version of this code in myfunctions.py if you desire. Base is much of what I actually ended up using and it is ugly

  ## Splot.py
  Generates a plot of the order parameter S in whichever crossection you choose. Very simple code
  <img width="448" height="223" alt="Image" src="https://github.com/user-attachments/assets/a5ab8265-34f5-45d1-8edc-d57cc9d2dfdf" />
 ## Frank.py
 Calculates and plots each type of Frank deformation (bend, twist, splay)  of a system at each time point
 <img width="500" height="250" alt="Screenshot 2026-03-23 101545" src="https://github.com/user-attachments/assets/01cde219-609f-4f97-8dc6-761674255e7e" />
  
## Visualizer.py
Generates figures representing 3D disclination lines and nematics at boundaries for each time point looped throught (currently from 1 to 300)
  <img width="637" height="180" alt="image" src="https://github.com/user-attachments/assets/329b347d-7c21-41b5-b290-330ca42db15e" />

 ## xyquiver, xzquiver, yzquiver
 Generates a velocity quiver plot (with a color map for vorticity) in the respective planes for a preset timepoint:
  <img width="467" height="227" alt="Image" src="https://github.com/user-attachments/assets/e4c555cf-5597-4445-a8a1-fb71afc34f4b" />
  
  
  
  
