from pypower.api import case9
import numpy as np
import copy, math

# Define a function to create a 4-bus distribution system with configurable parameters
def distrSystem4(baseMVA=100, basekV=7.2, d1CorrectionFactor=1, d2CorrectionFactor=1, powerCorrectionFactor=1.0):
    # Calculate base impedance using the given baseMVA and basekV
    baseZ = basekV * basekV / baseMVA

    # Define the bus data: [bus_id, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin]
    BusData = np.array([
        [1, 3, 0, 0, 0, 0, 1, 1, 0, basekV, 1, 1.1, 0.9],  # Slack bus
        [2, 1, 0, 0, 0, 0, 1, 1, 0, basekV, 1, 1.1, 0.9],  # Load bus
        [3, 1, 4.5 * powerCorrectionFactor, 1.5 * powerCorrectionFactor, 0, 0, 1, 1, 0, basekV, 1, 1.1, 0.9],  # Load bus
        [4, 1, 4.5 * powerCorrectionFactor, 1.5 * powerCorrectionFactor, 0, 0, 1, 1, 0, basekV, 1, 1.1, 0.9],  # Load bus
    ]).astype(np.float64)

    # Define generator data for the slack bus (bus 1): [bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, ...]
    GenData = np.array([
        [1, 0, 0, 1e20, -(1e20), 1, baseMVA, 1, 1e20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]).astype(np.float64)

    # Define lengths of the branches (in miles) and convert them to per-unit by dividing by 5280
    branchLength = np.array([
        [1, 2, (2000 + 000) / 5280],
        [2, 3, (1500 + 000) / 5280],
        [3, 4, (2500 + 000) / 5280]
    ]).astype(np.float64)

    # Define branch impedance data Z1 (R and X) per mile with correction factor d1
    branchZ1 = np.array([
        [1, 2, 0.45 / d1CorrectionFactor, 1.07 / d1CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360],
        [2, 3, 0.45 / d1CorrectionFactor, 1.07 / d1CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360],
        [3, 4, 0.45 / d1CorrectionFactor, 1.07 / d1CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360]
    ]).astype(np.float64)

    # Define branch impedance data Z2 (R and X) per mile with correction factor d2
    branchZ2 = np.array([
        [1, 2, 0.36 / d2CorrectionFactor, 0.53 / d2CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360],
        [2, 3, 0.36 / d2CorrectionFactor, 0.32 / d2CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360],
        [3, 4, 0.36 / d2CorrectionFactor, 0.64 / d2CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360]
    ]).astype(np.float64)

    # Initialize branch data 1 with zeros for R and X
    BranchData1 = np.array([
        [1, 2, 0, 0, 0, 500, 500, 500, 0, 0, 1, -360, 360],
        [2, 3, 0, 0, 0, 500, 500, 500, 0, 0, 1, -360, 360],
        [3, 4, 0, 0, 0, 500, 500, 500, 0, 0, 1, -360, 360]
    ]).astype(np.float64)

    # Initialize branch data 2 with zeros for R and X
    BranchData2 = np.array([
        [1, 2, 0.0, 0, 0, 500, 500, 500, 0, 0, 1, -360, 360],
        [2, 3, 0.0, 0, 0, 500, 500, 500, 0, 0, 1, -360, 360],
        [3, 4, 0.0, 0, 0, 500, 500, 500, 0, 0, 1, -360, 360]
    ]).astype(np.float64)

    # Update branch data 1 R and X values using Z1 and line lengths, converting to per-unit
    BranchData1[:, 2] = branchZ1[:, 2] * branchLength[:, 2] / baseZ
    BranchData1[:, 3] = branchZ1[:, 3] * branchLength[:, 2] / baseZ

    # Update branch data 2 R and X values using Z2 and line lengths, converting to per-unit
    BranchData2[:, 2] = branchZ2[:, 2] * branchLength[:, 2] / baseZ
    BranchData2[:, 3] = branchZ2[:, 3] * branchLength[:, 2] / baseZ

    # Create two versions of the system: one using branchZ1 and another using branchZ2
    case1 = {'baseMVA': baseMVA, 'bus': BusData, 'gen': GenData, 'branch': BranchData1}
    case2 = {'baseMVA': baseMVA, 'bus': BusData, 'gen': GenData, 'branch': BranchData2}

    # Return both system configurations
    return case1, case2


# Define a function that creates two power distribution system cases
# The entire load is on 4th bus
def distrSystem4_p1(baseMVA=100, basekV=7.2,d1CorrectionFactor=1,d2CorrectionFactor=1,powerCorrectionFactor=1.0):
    # Calculate base impedance using basekV and baseMVA
    baseZ = basekV * basekV / baseMVA
    # Define a multiplication factor for load values
    fact = 2.25
    # Create BusData array with parameters for each bus (4 buses in this system)
    BusData = np.array([
	[1,	3,	0,	0,	0,	0,	1,	1,	0,	basekV,	1,	1.1,	0.9],  # Bus 1 (slack bus)
	[2,	1,	0,	0,	0,	0,	1,	1,	0,	basekV,	1,	1.1,	0.9],  # Bus 2 (PQ bus)
	[3,	1,	0,	0,	0,	0,	1,	1,	0,	basekV,	1,	1.1,	0.9],  # Bus 3 (PQ bus)
    [4,	1,	fact*4*powerCorrectionFactor,	fact*(4/3)*powerCorrectionFactor,	0,	0,	1,	1,	0,	basekV,	1,	1.1,	0.9],  # Bus 4 (PQ bus with load)
    ]).astype(np.float64)
    
    
    # Create GenData array with generator parameters (only 1 generator at bus 1)
    GenData = np.array([
        [1  ,0  ,0  ,1e20,  -(1e20),  1,  baseMVA,  1,  1e20,  0,  0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0]
    ]).astype(np.float64)
    
    
    # Define branch lengths in miles (converted from feet by dividing by 5280)
    branchLength = np.array([
        [1, 2, (2000+000)/5280],  # Length between bus 1 and 2
        [2, 3, (1500+000)/5280],  # Length between bus 2 and 3
        [3, 4, (2500+000)/5280]   # Length between bus 3 and 4
    ]).astype(np.float64)
    
    # Define branch impedance parameters (type 1) with correction factors
    branchZ1 = np.array([
        [1, 2, 0.45/d1CorrectionFactor, 1.07/d1CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360],  # Impedance between bus 1-2
        [2, 3, 0.45/d1CorrectionFactor, 1.07/d1CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360],  # Impedance between bus 2-3
        [3, 4, 0.45/d1CorrectionFactor, 1.07/d1CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360]   # Impedance between bus 3-4
    ]).astype(np.float64)
    
    # Define branch impedance parameters (type 2) with different correction factors
    branchZ2 = np.array([
        [1, 2, 0.36/d2CorrectionFactor, 0.53/d2CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360],  # Impedance between bus 1-2
        [2, 3, 0.36/d2CorrectionFactor, 0.32/d2CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360],  # Impedance between bus 2-3
        [3, 4, 0.36/d2CorrectionFactor, 0.64/d2CorrectionFactor, 1e20, 1e20, 1e20, 0, 0, 1, -360, 360]   # Impedance between bus 3-4
    ]).astype(np.float64)
    
    
    # Initialize BranchData1 with basic parameters
    BranchData1 = np.array([
        [1,  2,  0,  0,  0,  500,  500,  500,  0,  0,  1,  -360,  360],  # Branch 1-2
        [2,  3,  0,  0,  0,  500,  500,  500,  0,  0,  1,  -360,  360],  # Branch 2-3
        [3,  4,  0,  0,  0,  500,  500,  500,  0,  0,  1,  -360,  360]   # Branch 3-4
    ]).astype(np.float64)
    
    # Initialize BranchData2 with basic parameters
    BranchData2 = np.array([
        [1,  2,  0.0,  0,  0,  500,  500,  500,  0,  0,  1,  -360,  360],  # Branch 1-2
        [2,  3,  0.0,  0,  0,  500,  500,  500,  0,  0,  1,  -360,  360],  # Branch 2-3
        [3,  4,  0.0,  0,  0,  500,  500,  500,  0,  0,  1,  -360,  360]   # Branch 3-4
    ]).astype(np.float64)
    
    # Calculate actual impedance values for BranchData1 using branch lengths and base impedance
    BranchData1[:,2] = branchZ1[:,2] * branchLength[:,2] / baseZ  # Real part of impedance
    BranchData1[:,3] = branchZ1[:,3] * branchLength[:,2] / baseZ  # Imaginary part of impedance
    
    # Calculate actual impedance values for BranchData2 using branch lengths and base impedance
    BranchData2[:,2] = branchZ2[:,2] * branchLength[:,2] / baseZ  # Real part of impedance
    BranchData2[:,3] = branchZ2[:,3] * branchLength[:,2] / baseZ  # Imaginary part of impedance

    # Create case1 dictionary with baseMVA, bus data, generator data, and branch data (type 1)
    case1 = {'baseMVA' : baseMVA , 'bus' : BusData, 'gen' : GenData, 'branch' : BranchData1}
    # Create case2 dictionary with baseMVA, bus data, generator data, and branch data (type 2)
    case2 = {'baseMVA' : baseMVA , 'bus' : BusData, 'gen' : GenData, 'branch' : BranchData2}
    
    # Return both cases
    return case1, case2




# Define a function to create a transmission-distribution system
def TDSystem4(requiredBuses=10, d1CorrectionFactor=1, d2CorrectionFactor=1, powerCorrectionFactor=1):
    # Set the bus number where distribution systems will be attached
    busToAttach = 5
    # Get base transmission system case (case9)
    trans = case9()
    # Create two copies of the base transmission system
    trans1 = case9()
    trans2 = case9()
    
    # Create distribution systems (two types for each case)
    distr1, distr2 = distrSystem4(d1CorrectionFactor=d1CorrectionFactor, d2CorrectionFactor=d2CorrectionFactor, powerCorrectionFactor=powerCorrectionFactor)
    # Create additional distribution systems (for distr4)
    distr3, _ = distrSystem4(d1CorrectionFactor=d1CorrectionFactor, d2CorrectionFactor=d2CorrectionFactor, powerCorrectionFactor=powerCorrectionFactor)
    distr4, _ = distrSystem4(d1CorrectionFactor=d1CorrectionFactor, d2CorrectionFactor=d2CorrectionFactor, powerCorrectionFactor=powerCorrectionFactor)

    # Define parameters for the transformer bus (bus 10)
    transformer_bus = [10, 1, 0, 0, 0, 0, 1, 1, 0, 7.2, 1, 2, 0]
    # Define parameters for the transformer branch connecting busToAttach to bus 10
    transformer_branch = [[busToAttach, 10, 0.01, 0.06, 0, 500, 500, 500, 1, 0, 1, -360, 360]]
    
    # Add transformer bus and branch to both transmission system copies
    trans1['bus'] = np.concatenate((trans1['bus'], [transformer_bus]))
    trans2['bus'] = np.concatenate((trans2['bus'], [transformer_bus]))
    trans1['branch'] = np.concatenate((trans1['branch'], transformer_branch))
    trans2['branch'] = np.concatenate((trans2['branch'], transformer_branch))
    
    # Initialize distribution bus numbering starting from 11
    busNo = 11
    # Initialize another bus numbering starting from 5 (for distr4)
    busNo1 = 5
    # Counter for number of distribution systems added
    systemsAdded = 0
    # Set the first bus of distr1 to be bus 10 (connected to transformer)
    distr1['bus'][0][0] = 10
    
    # Add distribution buses to trans1 and trans2 until requiredBuses is reached
    while systemsAdded < requiredBuses:
        # Assign unique bus numbers to each bus in the distribution systems
        distr1['bus'][1][0] = busNo
        distr2['bus'][1][0] = busNo
        busNo += 1
        distr1['bus'][2][0] = busNo
        distr2['bus'][2][0] = busNo
        busNo += 1
        distr1['bus'][3][0] = busNo
        distr2['bus'][3][0] = busNo
        busNo += 1
        
        # For the first 9 systems, also add to distr4
        if(systemsAdded < 9):
            distr3['bus'][1][0] = busNo1
            busNo1 += 1
            distr3['bus'][2][0] = busNo1
            busNo1 += 1
            distr3['bus'][3][0] = busNo1
            busNo1 += 1
            # Add these buses to distr4
            distr4['bus'] = np.concatenate((distr4['bus'], distr3['bus'][1:]))
            # Update branch connections for distr3
            distr3['branch'][0][0] = 1
            distr3['branch'][0][1] = distr3['bus'][1][0]
            distr3['branch'][1][0] = distr3['bus'][1][0]
            distr3['branch'][1][1] = distr3['bus'][2][0]
            distr3['branch'][2][0] = distr3['bus'][2][0]
            distr3['branch'][2][1] = distr3['bus'][3][0]
            # Add these branches to distr4
            distr4['branch'] = np.concatenate((distr4['branch'], distr3['branch']))

        # Add distribution buses to transmission systems
        trans1['bus'] = np.concatenate((trans1['bus'], distr1['bus'][1:]))
        trans2['bus'] = np.concatenate((trans2['bus'], distr2['bus'][1:]))
        
        # Update branch connections for distr1
        distr1['branch'][0][0] = 10
        distr1['branch'][0][1] = distr1['bus'][1][0]
        distr1['branch'][1][0] = distr1['bus'][1][0]
        distr1['branch'][1][1] = distr1['bus'][2][0]
        distr1['branch'][2][0] = distr1['bus'][2][0]
        distr1['branch'][2][1] = distr1['bus'][3][0]
        # Add these branches to trans1
        trans1['branch'] = np.concatenate((trans1['branch'], distr1['branch']))
        
        # Update branch connections for distr2
        distr2['branch'][0][0] = 10
        distr2['branch'][0][1] = distr2['bus'][1][0]
        distr2['branch'][1][0] = distr2['bus'][1][0]
        distr2['branch'][1][1] = distr2['bus'][2][0]
        distr2['branch'][2][0] = distr2['bus'][2][0]
        distr2['branch'][2][1] = distr2['bus'][3][0]
        # Add these branches to trans2
        trans2['branch'] = np.concatenate((trans2['branch'], distr2['branch']))
        
        # Increment counter
        systemsAdded += 1

    # Return all created systems
    return trans, distr1, distr2, distr4, trans1, trans2



# Using Reduced Order Model to lump the distribution system
def attach_lumped_distribution(trans_case, S_ROM, Z_ROM, lump_bus_index=5, baseMVA=100):
    """
    Attaches the lumped distribution system to a transmission system at the specified bus.
    
    Parameters:
    - trans_case: dict of the transmission system (e.g., case9)
    - S_ROM: complex apparent power of the lumped load (in pu)
    - Z_ROM: complex impedance of the branch (in pu)
    - lump_bus_index: 1-based index of the transmission bus to attach to (default = 5)
    - baseMVA: base power (to match distribution base)
    
    Returns:
    - Modified transmission system (new case dict)
    """

    # Set the bus number where the lumped distribution will be attached
    busToAttach = 5
    
    # Create base transmission system case (case9)
    trans = case9()
    
    # Create two copies of the base transmission system
    trans1 = case9()
    trans2 = case9()
    
    # Clear any existing load at the attachment bus (bus 5, index 4 since it's 0-based)
    trans1['bus'][4][2] = 0  # Set real power load to 0
    trans1['bus'][4][3] = 0  # Set reactive power load to 0
    
    # Extract real and imaginary parts from the impedance
    r, x = Z_ROM.real, Z_ROM.imag
    
    # Create the lumped distribution bus (bus 10) with the specified load
    # Format: [bus number, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin]
    lumped_distr_bus = [
        10,                         # Bus number (10)
        1,                           # Bus type (PQ bus)
        S_ROM.real * baseMVA,        # Real power demand (converted from pu to MW)
        S_ROM.imag * baseMVA,        # Reactive power demand (converted from pu to MVAr)
        0, 0,                        # Shunt conductance/susceptance
        1, 1,                        # Area, voltage magnitude
        0,                           # Voltage angle
        7.2,                         # Base kV
        1,                           # Zone
        1.1, 0.9                     # Voltage limits
    ]
    
    # Create the branch connecting the transmission bus to the lumped distribution bus
    # Format: [fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax]
    lumped_distr_branch = [
        [
            busToAttach,             # From bus (transmission bus 5)
            10,                      # To bus (new distribution bus 10)
            r, x,                    # Resistance and reactance
            0,                       # Total line charging susceptance
            500, 500, 500,           # MVA ratings
            0, 0,                    # Tap ratio and phase shift angle
            1,                       # Initial branch status (in service)
            -360, 360                # Angle difference limits
        ]
    ]
    
    # Add the new distribution bus to the transmission system
    trans1['bus'] = np.concatenate((trans1['bus'], [lumped_distr_bus]))
    
    # Add the new branch to the transmission system
    trans1['branch'] = np.concatenate((trans1['branch'], lumped_distr_branch))

    # Return the modified transmission system
    return trans1

