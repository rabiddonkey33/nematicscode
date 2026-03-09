Hello! If you are reading this you must have an interest in picking up my research.
In short I have been initializing line defects in a channel. My experiment involved modifying the defect spacing and activity levels, but of course whatever you do will probably be different.
There are two components of this code: The simulation code you will run in HPCC and the analysis code you will run locally in Python (If you choose)

Simulation Code: This code sets up an initial condition and numerically integrates the nematic Stokes equations
There are two main files you will be modifying in this: example.c and interface.c.
In example.c you can set the dimensions of the channel as well as the dimensions of the grid that will be numerically integrated over by modifying LX,LY,LZ and NX,NY,NZ respectively. You can also modify the activity number ALPHA. Deeper into example.c there is code to anal

In interface.c the boundary conditions for the velocities have been set such that the two walls along the x axis are periodic and every othber wall is no slip. This begins on line 104.

Analysis Code: 
