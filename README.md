# Stochastic-Saddle-Point-Dynamics
This is the code associated with the numerical expriements from the paper:
Using Witten Laplacians to locate index-1 saddle points
Tony Lelièvre, Panos Parpas
https://arxiv.org/abs/2212.10135

# Algorithm 3.1 is in main.py.
The main.py file contraints a parameter structure called alg_params.
These are set to some sensible defaults. Next to each parameter we added a small explanation.
Note that the code has more parameters/features than we report in the paper.
The different benchmark problems are specified in potentials_2d.py
When a problem is specified then the parameters are set to the ones we used in the paper.
This is done in the function called setup_problem in potentials_2d.
To run Algorithm 2.1 (i.e. with out the local search), set the two parameters as follows:

prms.run_dimmer = False
prms.run_particle_dimmer=False

These two parameters control if the local algorithm 3.2 is used or 3.3.
The local algorithms are implemented in dimmer.py and dimmer_utils.py.
The file opt_algorithms.py contains the implementation of gradient descent (used to check the basin of attraction of the different saddles.)
Gradient descent is only used for the purpose of analysis and visualization.

The initial points for the problems in sections 4.1, 4.2, and 4.2 are specified in the potentials_2d.py file.
For the vacancy diffusion case (section 4.4) the initial conditions are in the directory vd_init/
For the Lennard-Jones case (section 4.5) the initial conditions are in the directory lj72_init/

The output of the algorithms are saved in pickle files. The algorithm saves information for both X, Y and if the dimer algorithm is used of u.
The directory to save the files is specified in the parameter prms.str_file.
Unless you want to save all the files in the same directory then it is advised to use a separate directory (you have to create this before running the algorithm).
You can switch off this feature by setting "save_file":False.

# Reproducing the results of the paper:
To run the code for the example in section 4.1 proceed as follows:
Set the prms.prb_index == 1
Then the algorithm will load the default parameters (as used in the paper) from the file potentials_2d (see line 143).
The output is saved in a standard pickle file and can be visualized with any standard python library.

The rest of the examples are run in a similar fashion, just change the parameter prms.prb_index and the code will run a different test case.
