
from pypower.api import case9, ppoption, runpf, printpf
import math, copy
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve
np.set_printoptions(precision=4,suppress=True,linewidth=np.inf)
print("Running...")

def Calculate_Zbus(BusData1, BranchData1):
    # Get number of buses in the system
    N = len(BusData1)  # number of buses

    # Create working copies of input data
    BusData = BusData1
    BranchData = BranchData1
    
    # Create mapping from original bus numbers to sequential numbers (1-based)
    mp = {}
    for i in range(N):
        mp[BusData1[i][0]] = i + 1  # Map original bus number to sequential index
        BusData[i][0] = i + 1       # Update bus number in working copy
    
    # Update branch data with sequential bus numbers
    for i in range(len(BranchData1)):
        a, b = BranchData1[i][0], BranchData1[i][1]  # Original from/to bus numbers
        c, d = mp[a], mp[b]          # Get sequential bus numbers
        BranchData[i][0], BranchData[i][1] = c, d    # Update branch data

    # Find reference bus (bus with type 3)
    ref_bus = None
    for bus in BusData:
        if bus[1] == 3:  # Bus type 3 is reference/slack bus
            ref_bus = bus[0]
            break

    # Create list of non-reference buses and sort them
    non_ref_buses = [bus[0] for bus in BusData if bus[0] != ref_bus]
    non_ref_buses_sorted = sorted(non_ref_buses)
    n = len(non_ref_buses_sorted)  # Number of non-reference buses
    
    # Create mapping from bus numbers to matrix indices (reference bus = 0)
    bus_remap = {bus: idx + 1 for idx, bus in enumerate(non_ref_buses_sorted)}
    bus_remap[ref_bus] = 0  # Reference bus is mapped to 0

    # Process branches: remap bus numbers and extract impedance
    branches = []
    for branch in BranchData:
        from_orig = int(branch[0])  # Original from bus number
        to_orig = int(branch[1])    # Original to bus number
        r = branch[2]               # Resistance
        x = branch[3]              # Reactance
        z = r + 1j * x             # Complex impedance
        from_bus = bus_remap[from_orig]  # Remapped from bus index
        to_bus = bus_remap[to_orig]      # Remapped to bus index
        branches.append((from_bus, to_bus, z))  # Store processed branch

    # Initialize Zbus matrix (n x n complex matrix) and status array
    Zbus = np.zeros((n, n), dtype=complex)
    status = np.zeros(n, dtype=int)  # Tracks which buses have been processed

    # Process each branch to build Zbus matrix
    for idx, (from_bus, to_bus, z) in enumerate(branches):
        # Type 1: Branch between reference bus (0) and new bus (not in Zbus yet)
        if (from_bus == 0 and to_bus != 0) or (to_bus == 0 and from_bus != 0):
            non_ref_bus = to_bus if from_bus == 0 else from_bus
            status_idx = non_ref_bus - 1  # Convert to 0-based index

            if status[status_idx] == 0:  # If bus hasn't been processed yet
                Zbus[status_idx, status_idx] = z  # Add diagonal element
                status[status_idx] = 1           # Mark bus as processed
                continue

        # Type 2: Branch between processed bus and new bus
        if from_bus != 0 and to_bus != 0:
            from_status = status[from_bus - 1]
            to_status = status[to_bus - 1]

            # Determine which bus is new (unprocessed)
            if (from_status == 1 and to_status == 0):
                old_bus = from_bus
                new_bus = to_bus
            elif (to_status == 1 and from_status == 0):
                old_bus = to_bus
                new_bus = from_bus
            else:
                old_bus = None

            if old_bus is not None:
                old_idx = old_bus - 1  # Convert to 0-based index
                new_idx = new_bus - 1
                # Copy column and row from old bus to new bus
                Zbus[:, new_idx] = Zbus[:, old_idx]
                Zbus[new_idx, :] = Zbus[old_idx, :]
                # Add branch impedance to diagonal
                Zbus[new_idx, new_idx] += z
                status[new_idx] = 1  # Mark new bus as processed
                continue

        # Type 3: Branch between processed bus and reference bus
        if (from_bus == 0 or to_bus == 0) and (from_bus != 0 or to_bus != 0):
            non_ref_bus = from_bus if to_bus == 0 else to_bus
            status_idx = non_ref_bus - 1

            if status[status_idx] == 1:  # If bus has been processed
                m = Zbus[status_idx, status_idx] + z
                if m != 0:
                    # Kron reduction to eliminate reference bus
                    ztemp = np.outer(Zbus[:, status_idx], Zbus[status_idx, :]) / m
                    Zbus -= ztemp
                continue

        # Type 4: Branch between two processed buses
        if from_bus != 0 and to_bus != 0:
            a_idx = from_bus - 1
            b_idx = to_bus - 1

            if status[a_idx] == 1 and status[b_idx] == 1:
                z_ab = Zbus[a_idx, a_idx] + Zbus[b_idx, b_idx] - 2 * Zbus[a_idx, b_idx] + z
                if z_ab != 0:
                    # Update matrix to account for mutual coupling
                    diff_col = Zbus[:, a_idx] - Zbus[:, b_idx]
                    diff_row = Zbus[a_idx, :] - Zbus[b_idx, :]
                    ztemp = np.outer(diff_col, diff_row) / z_ab
                    Zbus -= ztemp
                continue

    return Zbus



def Calculate_Ybus(BusData1, BranchData1):
    """Calculate the admittance matrix (Ybus) for power flow analysis"""
    
    # Get number of buses in the system
    N = len(BusData1)  # number of buses
    
    # Initialize Ybus matrix with zeros (complex numbers)
    Ybus = np.zeros((N,N), dtype=complex)

    # Create working copies of input data
    BusData = BusData1
    BranchData = BranchData1
    
    # Create mapping from original bus numbers to sequential numbers (1-based)
    mp = {}
    for i in range(N):
        mp[BusData1[i][0]] = i+1  # Map original bus number to sequential index
        BusData[i][0] = i+1       # Update bus number in working copy
    
    # Update branch data with sequential bus numbers
    for i in range(len(BranchData1)):
        a, b = BranchData1[i][0], BranchData1[i][1]  # Original from/to bus numbers
        c, d = mp[a], mp[b]          # Get sequential bus numbers
        BranchData[i][0], BranchData[i][1] = c, d    # Update branch data

    # Calculate off-diagonal elements (mutual admittances)
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            # Find branches connecting buses i and j (in either direction)
            pos = ((BranchData[:, 0] == i) & (BranchData[:, 1] == j)) | \
                  ((BranchData[:, 0] == j) & (BranchData[:, 1] == i))
            if np.any(pos):
                a = BranchData[pos, 8]  # Check if it's a transformer (tap ratio > 0)
                if a >= 0:  # If not a transformer (regular transmission line)
                    # Calculate mutual admittance (-1/impedance)
                    Ybus[i-1, j-1] = -1 / (BranchData[pos, 2][0] + 1j * BranchData[pos, 3][0])
    
    # Make Ybus symmetric (since Yij = Yji)
    Ybus = Ybus + np.transpose(Ybus)
    
    # Calculate diagonal elements (self admittances)
    # Start with negative sum of off-diagonal elements
    Ybus = Ybus + np.diag(-np.sum(Ybus, axis=1))
    
    # Add shunt admittances (line charging and bus shunts)
    for i in range(1, N+1):
        pos = (BranchData[:, 0] == i) | (BranchData[:, 1] == i)
        if np.any(pos):
            # Add half line charging susceptance (B/2) and bus shunt admittance (Gs + jBs)
            Ybus[i-1, i-1] = Ybus[i-1, i-1] + \
                            np.sum(0.5j * BranchData[pos, 4]) + \
                            np.sum(BusData[i-1, 4] + 1j * BusData[i-1, 5])

    # Handle transformer data
    pos = BranchData[:, 8] > 0  # Find transformer branches (tap ratio > 0)
    pos = pos * np.arange(0, len(pos))  # Get indices of transformers
    pos = pos[pos != 0]  # Remove zeros
    
    # Process each transformer
    for n in pos:
        i = int(BranchData[n, 0]) - 1  # From bus (0-based index)
        j = int(BranchData[n, 1]) - 1  # To bus (0-based index)
        t = 1 / BranchData[n, 8]       # Tap ratio (1/a)
        Y = 1 / (BranchData[n, 2] + 1j * BranchData[n, 3])  # Series admittance
    
        # Update Ybus elements for transformer
        Ybus[i, i] += Y * (abs(t) ** 2)  # Self admittance at from bus
        Ybus[i, j] -= np.conj(t) * Y     # Mutual admittance (from-to)
        Ybus[j, i] -= t * Y              # Mutual admittance (to-from)
        Ybus[j, j] += Y                  # Self admittance at to bus

    return Ybus


def calculate_Pcalc_Qcalc(V, d, Ybus, Psch, Qsch, posSL, posPV):
    """Calculate power injections and mismatches for power flow"""
    
    # Calculate complex voltage phasors
    V_complex = V * np.exp(1j * d)
    
    # Calculate current injections
    I_complex = Ybus @ V_complex
    
    # Calculate complex power injections (S = V × I*)
    Scalc = np.conj(V_complex) * I_complex
    
    # Separate into active (P) and reactive (Q) power
    Pcalc = np.real(Scalc)   # Real part is active power
    Qcalc = -np.imag(Scalc)  # Negative imaginary part is reactive power
    
    # Calculate power mismatches
    dP = Psch - Pcalc  # Active power mismatch
    dQ = Qsch - Qcalc  # Reactive power mismatch
    
    # Reduce mismatches based on bus types:
    # - For PV buses, we only care about active power mismatch
    # - For PQ buses, we care about both P and Q mismatches
    # - For slack bus, we ignore both mismatches
    dPred = dP[posSL == 0]        # Active power mismatches (excluding slack)
    dQred = dQ[(posSL + posPV) == 0]  # Reactive power mismatches (PQ buses only)
    dPdQred = np.concatenate([dPred, dQred])  # Combined mismatch vector
    
    return Pcalc, Qcalc, dP, dQ, dPred, dQred, dPdQred


def Jacobian_NRLF(Ybus, V, d, posSL, posPV):
    """Calculate the Jacobian matrix for Newton-Raphson power flow"""
    
    N = len(V)  # Number of buses
    
    # Create position mask for Jacobian reduction
    # (we eliminate rows/columns for slack and PV buses)
    pos = np.concatenate([posSL, posSL + posPV])
    
    # Create matrix of angle differences (θi - θj) + angle(Ybus)
    angle_mat = np.tile(d, (len(d), 1))  # Create matrix with θi in each row
    angle_mat -= np.transpose(angle_mat)  # θi - θj
    angle_mat += np.angle(Ybus)          # + angle(Ybus_ij)

    # Calculate J11 submatrix (∂P/∂θ)
    J11 = -np.abs(V)[:, np.newaxis] * np.abs(V) * np.abs(Ybus) * np.sin(angle_mat)
    J11 = J11 * (1 - np.eye(N))  # Zero out diagonal
    J11 = J11 - np.diag(np.sum(J11, axis=1))  # Fill diagonal with negative sum
    
    # Calculate J21 submatrix (∂Q/∂θ)
    J21 = -np.abs(V)[:, np.newaxis] * np.abs(V) * np.abs(Ybus) * np.cos(angle_mat)
    J21 = J21 * (1 - np.eye(N))  # Zero out diagonal
    J21 = J21 - np.diag(np.sum(J21, axis=1))  # Fill diagonal with negative sum
    
    # Calculate J12 submatrix (∂P/∂V)
    J12 = J21 * (2 * np.eye(N) - np.ones((N, N)))  # Transform J21
    J12 = J12 + np.diag(2 * (np.abs(V) ** 2) * np.real(np.diag(Ybus)))  # Add diagonal terms
    
    # Calculate J22 submatrix (∂Q/∂V)
    J22 = J11 * (-2 * np.eye(N) + np.ones((N, N)))  # Transform J11
    J22 = J22 - np.diag(2 * (np.abs(V) ** 2) * np.imag(np.diag(Ybus)))  # Add diagonal terms
    
    # Assemble full Jacobian matrix
    J = np.concatenate([
        np.concatenate([J11, J12], axis=1),
        np.concatenate([J21, J22], axis=1)
    ], axis=0)
    
    WholeJ = J  # Store complete Jacobian before reduction
    
    # Reduce Jacobian by eliminating rows/columns for slack and PV buses
    J = np.delete(J, np.where(pos > 0), axis=0)
    J = np.delete(J, np.where(pos > 0), axis=1)

    return WholeJ, J


def NRLF(V, d, Ybus, Psch, Qsch, posSL, posPV, Toler, Max_It):
    """Newton-Raphson power flow solver"""
    
    it = 0  # Iteration counter
    N = len(V)  # Number of buses
    learn_rate = 1  # Learning rate (can be adjusted for better convergence)
    
    # Get initial Jacobian
    _, J = Jacobian_NRLF(Ybus, V, d, posSL, posPV)
    
    while True:
        # Calculate power injections and mismatches
        Pcalc, Qcalc, dP, dQ, dPred, dQred, dPdQred = \
            calculate_Pcalc_Qcalc(V, d, Ybus, Psch, Qsch, posSL, posPV)
    
        # Check convergence criteria
        if not (np.any(np.abs(dPdQred) > Toler) and (it <= Max_It)):
            break
    
        it += 1  # Increment iteration counter
        
        # Calculate Jacobian matrix
        _, J = Jacobian_NRLF(Ybus, V, d, posSL, posPV)
        
        # Check for singular matrix
        determinant = np.linalg.det(J)
        if determinant == 0:
            print('Singular Matrix in NRLF')
            return V, d, Pcalc, Qcalc, dP, dQ, dPred, dQred, dPdQred, J, it, 0
        
        # Solve for voltage and angle corrections
        inv_J = np.linalg.inv(J)
        dddVred = inv_J @ dPdQred
        
        # Create full correction vector (with zeros for slack/PV buses)
        dddV = 1 - np.concatenate([posSL, posSL + posPV])
        dddV = dddV.astype(np.float64)
        dddV[dddV == 1] = dddVred  # Insert calculated corrections
        
        # Update voltage angles and magnitudes
        d = d + learn_rate * dddV[:N]  # Angle corrections (first N elements)
        V = V * (1 + learn_rate * dddV[N:2*N])  # Voltage magnitude corrections
        
    return V, d, Pcalc, Qcalc, dP, dQ, dPred, dQred, dPdQred, J, it, 1


def NRLF1(V, d, Ybus, Psch, Qsch, posSL, posPV, Toler, Max_It):
    """Alternative Newton-Raphson power flow solver (returns full Jacobian)"""
    
    # This function is nearly identical to NRLF, but returns the complete Jacobian
    # See NRLF comments for detailed explanation
    
    it = 0
    N = len(V)
    learn_rate = 1
    WholeJ, J = Jacobian_NRLF(Ybus, V, d, posSL, posPV)
    
    while True:
        Pcalc, Qcalc, dP, dQ, dPred, dQred, dPdQred = \
            calculate_Pcalc_Qcalc(V, d, Ybus, Psch, Qsch, posSL, posPV)
    
        if not (np.any(np.abs(dPdQred) > Toler) and (it <= Max_It)):
            break
    
        it += 1
        WholeJ, J = Jacobian_NRLF(Ybus, V, d, posSL, posPV)
        
        determinant = np.linalg.det(J)
        if determinant == 0:
            print('Singular Matrix in NRLF')
            return V, d, Pcalc, Qcalc, dP, dQ, dPred, dQred, dPdQred, J, it, 0
        
        inv_J = np.linalg.inv(J)
        dddVred = inv_J @ dPdQred
        
        dddV = 1 - np.concatenate([posSL, posSL + posPV])
        dddV = dddV.astype(np.float64)
        dddV[dddV == 1] = dddVred
        
        d = d + learn_rate * dddV[:N]
        V = V * (1 + learn_rate * dddV[N:2 * N])
        
    return V, d, Pcalc, Qcalc, dP, dQ, dPred, dQred, dPdQred, WholeJ, J, it, 1

def runNRLF(casedata, lambda_val=0 ,tolerance=1e-3, maxiterations=25):
    #preprocessing-data
    # Convert bus data to numpy array with float64 precision
    BusData=np.array(casedata['bus']).astype(np.float64())
    # Convert generator data to numpy array with float64 precision
    GenData=np.array(casedata['gen']).astype(np.float64())
    # Convert branch data to numpy array with float64 precision
    BranchData=np.array(casedata['branch']).astype(np.float64())
    # Get base MVA value from case data
    baseMVA = casedata['baseMVA']
    # Convert bus active power demand to per unit
    BusData[:,2]/=baseMVA
    # Convert bus reactive power demand to per unit
    BusData[:,3]/=baseMVA
    # Convert generator power values to per unit (columns 1-5)
    GenData[:,1:5]/=baseMVA    
    # Convert additional generator parameters to per unit (columns 8-21)
    GenData[:,8:21]/=baseMVA
    
    #convert angles from degrees to radians
    # Convert bus voltage angles from degrees to radians
    BusData[:,8]=BusData[:,8]*np.pi/180
    # Convert branch angles from degrees to radians
    BranchData[:,9]=BranchData[:,9]*np.pi/180

    #Evaluating variables for NRLF
    # Calculate Ybus matrix using bus and branch data
    Ybus=Calculate_Ybus(BusData,BranchData)
    
    #reordering buses to line them as Slack, PV and PQ
    # Create boolean mask for PV buses (type 2)
    posPV = BusData[:,1] == 2
    # Create boolean mask for Slack buses (type 3)
    posSL = BusData[:,1] == 3
    # Create auxiliary array for PQ buses
    aux1=1-posPV-posSL
    # Assign indices to PQ buses
    aux1[aux1 == 1] = np.transpose(np.arange(1, np.sum(aux1 == 1) + 1))
    # Create position array combining all bus types
    auxPos = np.column_stack([np.arange(1, len(posPV) + 1), posPV + posSL, aux1])
    
    #Flat Start initialization
    # Get total number of buses
    N=len(BusData)
    # Initialize voltage magnitudes to 1.0 pu
    V=np.ones(N)   #voltage magnitude
    # Initialize voltage angles to 0 radians
    d=np.zeros(N)  #voltage angle
    
    #setting voltages of slack and PV bus
    # Set generator bus voltages from generator data
    for generator in GenData:
        V[int(generator[0]-1)]=generator[5]
    
    #Setting P and Q values in pu
    # Initialize scheduled active power with negative load (demand)
    Psch=-BusData[:,2]
    # Initialize scheduled reactive power with negative load (demand)
    Qsch=-BusData[:,3]
    # Add generator power to scheduled values
    for generator in GenData:
        Psch[int(generator[0]-1)]+=generator[1]
        Qsch[int(generator[0]-1)]+=generator[2]
    
    #picking just the required values of P,Q,V,d for NR
    # Reduced P schedule (excluding slack buses)
    Pschred = Psch[posSL==0]
    # Reduced Q schedule (only PQ buses)
    Qschred = Qsch[(posSL+posPV)==0]
    # Reduced angle array (excluding slack buses)
    dred = d[posSL==0]
    # Reduced voltage array (only PQ buses)
    Vred=V[(posSL+posPV)==0]

    # Run Newton-Raphson load flow with given parameters
    return NRLF(V, d, Ybus, (1+lambda_val)*Psch, (1+lambda_val)*Qsch, posSL, posPV, tolerance, maxiterations)

def runNRLF1(casedata, lambda_val=0 ,tolerance=1e-3, maxiterations=25):
    #preprocessing-data
    # Convert bus data to numpy array with float64 precision
    BusData=np.array(casedata['bus']).astype(np.float64())
    # Convert generator data to numpy array with float64 precision
    GenData=np.array(casedata['gen']).astype(np.float64())
    # Convert branch data to numpy array with float64 precision
    BranchData=np.array(casedata['branch']).astype(np.float64())
    # Get base MVA value from case data
    baseMVA = casedata['baseMVA']
    # Convert bus active power demand to per unit
    BusData[:,2]/=baseMVA
    # Convert bus reactive power demand to per unit
    BusData[:,3]/=baseMVA
    # Convert generator power values to per unit (columns 1-5)
    GenData[:,1:5]/=baseMVA    
    # Convert additional generator parameters to per unit (columns 8-21)
    GenData[:,8:21]/=baseMVA
    
    #convert angles from degrees to radians
    # Convert bus voltage angles from degrees to radians
    BusData[:,8]=BusData[:,8]*np.pi/180
    # Convert branch angles from degrees to radians
    BranchData[:,9]=BranchData[:,9]*np.pi/180

    #Evaluating variables for NRLF
    # Calculate Ybus matrix using bus and branch data
    Ybus=Calculate_Ybus(BusData,BranchData)
    
    #reordering buses to line them as Slack, PV and PQ
    # Create boolean mask for PV buses (type 2)
    posPV = BusData[:,1] == 2
    # Create boolean mask for Slack buses (type 3)
    posSL = BusData[:,1] == 3
    # Create auxiliary array for PQ buses
    aux1=1-posPV-posSL
    # Assign indices to PQ buses
    aux1[aux1 == 1] = np.transpose(np.arange(1, np.sum(aux1 == 1) + 1))
    # Create position array combining all bus types
    auxPos = np.column_stack([np.arange(1, len(posPV) + 1), posPV + posSL, aux1])
    
    #Flat Start initialization
    # Get total number of buses
    N=len(BusData)
    # Initialize voltage magnitudes to 1.0 pu
    V=np.ones(N)   #voltage magnitude
    # Initialize voltage angles to 0 radians
    d=np.zeros(N)  #voltage angle
    
    #setting voltages of slack and PV bus
    # Set generator bus voltages from generator data
    for generator in GenData:
        V[int(generator[0]-1)]=generator[5]
    
    #Setting P and Q values in pu
    # Initialize scheduled active power with negative load (demand)
    Psch=-BusData[:,2]
    # Initialize scheduled reactive power with negative load (demand)
    Qsch=-BusData[:,3]
    # Add generator power to scheduled values
    for generator in GenData:
        Psch[int(generator[0]-1)]+=generator[1]
        Qsch[int(generator[0]-1)]+=generator[2]
    
    #picking just the required values of P,Q,V,d for NR
    # Reduced P schedule (excluding slack buses)
    Pschred = Psch[posSL==0]
    # Reduced Q schedule (only PQ buses)
    Qschred = Qsch[(posSL+posPV)==0]
    # Reduced angle array (excluding slack buses)
    dred = d[posSL==0]
    # Reduced voltage array (only PQ buses)
    Vred=V[(posSL+posPV)==0]

    # Run Newton-Raphson load flow (version 1) with given parameters
    return NRLF1(V, d, Ybus, (1+lambda_val)*Psch, (1+lambda_val)*Qsch, posSL, posPV, tolerance, maxiterations)


#data of buses and branches must be given with type as float64
def runCPF(casedata, BusForCPF, step1=0.1, step2=0.005, step3=0.01, tolerance=1e-3, maxiterations=25, changefactor=0.75):
    """
    Continuation Power Flow (CPF) analysis to trace PV curves and find maximum loadability
    
    Parameters:
    - casedata: Power system case data (MATPOWER format)
    - BusForCPF: Bus number to monitor for PV curve
    - step1/2/3: Step sizes for different CPF phases
    - tolerance: Convergence tolerance
    - maxiterations: Maximum NR iterations allowed
    - changefactor: Factor to determine when to switch phases
    
    Returns:
    - X values (lambda values - loading parameter)
    - Y values (voltage magnitudes at monitored bus)
    - Jacobian matrices at each step
    """
    
    # ===== DATA PREPROCESSING =====
    # Convert input data to numpy arrays
    BusData=np.array(casedata['bus']).astype(np.float64)
    BranchData=np.array(casedata['branch']).astype(np.float64)
    GenData=np.array(casedata['gen']).astype(np.float64)
    basePower=np.array(casedata['baseMVA']).astype(np.float64)
    
    # Convert powers to per unit values
    BusData[:,2]/=basePower  # Active power demand
    BusData[:,3]/=basePower  # Reactive power demand
    GenData[:,1:5]/=basePower  # Generator power limits
    
    # Convert angles from degrees to radians
    BusData[:,8]=BusData[:,8]*np.pi/180  # Bus voltage angles
    BranchData[:,9]=BranchData[:,9]*np.pi/180  # Branch angles
    BranchData[:,11:13]=BranchData[:,11:13]*np.pi/180  # Transformer angles

    # Calculate Ybus matrix
    Ybus=Calculate_Ybus(BusData,BranchData)

    # ===== BUS REORDERING =====
    # Identify bus types (Slack=3, PV=2, PQ=1)
    posPV = BusData[:,1] == 2  # PV bus mask
    posSL = BusData[:,1] == 3  # Slack bus mask
    aux1=1-posPV-posSL  # PQ bus mask
    aux1[aux1 == 1] = np.transpose(np.arange(1, np.sum(aux1 == 1) + 1))
    auxPos = np.column_stack([np.arange(1, len(posPV) + 1), posPV + posSL, aux1])
    
    # ===== FLAT START INITIALIZATION =====
    N=len(BusData)  # Number of buses
    V=np.ones(N)    # Initialize voltage magnitudes to 1.0 pu
    d=np.zeros(N)   # Initialize voltage angles to 0 radians
    
    # Set known voltages for PV and Slack buses
    for generator in GenData:
        V[int(generator[0]-1)]=generator[5]  # Set voltage from generator data
    
    # ===== POWER SCHEDULE CALCULATION =====
    # Initialize with negative load (demand)
    Psch=-BusData[:,2]  # Active power schedule
    Qsch=-BusData[:,3]  # Reactive power schedule
    
    # Add generator injections
    for generator in GenData:
        Psch[int(generator[0]-1)]+=generator[1]  # Add generator P
        Qsch[int(generator[0]-1)]+=generator[2]  # Add generator Q
    
    # Reduced vectors (excluding slack/PV buses where appropriate)
    Pschred = Psch[posSL==0]  # P schedule without slack
    Qschred = Qsch[(posSL+posPV)==0]  # Q schedule for PQ buses only
    dred = d[posSL==0]  # Angles without slack
    Vred=V[(posSL+posPV)==0]  # Voltages for PQ buses only
    
    # Combined vector of scheduled values
    K=np.concatenate([Pschred,Qschred])
    
    # Initialize storage for PV curve points
    Y_Vph1=[]; Y_Vph2=[]; Y_Vph3=[]  # Voltage points
    X_LambdaPh1=[]; X_LambdaPh2=[]; X_LambdaPh3=[]  # Lambda points
    J_list=[]  # Jacobian matrices
    
    # ===== PHASE 1: LOAD INCREASE (NORMAL OPERATION) =====
    print("Phase - 1 -- Running")
    phase1_counter=0
    sigma=step1  # Step size
    lambda_val=1  # Loading parameter (1 = base case)
    
    # Initial power flow solution
    V, d, _, _, _, _, _, _, _, temp, status = NRLF(V, d, Ybus, 
        Psch * (Psch >= 0) * lambda_val + Psch * (Psch < 0),
        Qsch * (Qsch >= 0) * lambda_val + Qsch * (Psch < 0),
        posSL, posPV, tolerance, maxiterations)
    
    if status==0:  # Singular matrix
        print('Singular Matrix in Initial NRLF')
        return [],[],[]
    if temp>=maxiterations:  # Non-convergence
        print('Initial NRLF did not converge')
        return [],[],[]
    
    # Store initial state
    Last_d=d; Last_V=V; Last_LambdaPh1=lambda_val
    
    while True:
        phase1_counter+=1
        # PREDICTOR STEP (tangent vector calculation)
        d_V_L = np.concatenate([dred, Vred, [lambda_val]])
        _,J = Jacobian_NRLF(Ybus,V,d,posSL,posPV)
        ek = np.concatenate([np.zeros(J.shape[1]), [1]])  # Direction vector
        
        # Form extended Jacobian matrix
        JKe = np.column_stack((J,-K))  # Augmented with -K column
        JKe = np.vstack((JKe,ek))     # And direction row
        
        if np.linalg.det(JKe)==0:  # Singular matrix check
            print(f'Matrix singular at phase 1 iteration {phase1_counter}')
            break
            
        # Calculate tangent vector
        tangent_vector = np.linalg.solve(JKe, ek.T)
        
        # Update state variables
        d_V_L = d_V_L + sigma * tangent_vector
        dred = d_V_L[:len(dred)]
        d[posSL == 0] = dred
        Vred = d_V_L[len(dred):(len(dred) + len(Vred))]
        V[posSL + posPV == 0] = Vred
        lambda_val = d_V_L[-1]
        
        # CORRECTOR STEP (full power flow solution)
        V, d, _, _, _, _, _, _, _, J, temp, status = NRLF(V, d, Ybus, 
            Psch * (Psch >= 0) * lambda_val + Psch * (Psch < 0),
            Qsch * (Qsch >= 0) * lambda_val + Qsch * (Psch < 0),
            posSL, posPV, tolerance, maxiterations)
        
        # Store results if successful
        if (status==1 and temp<maxiterations and 
            tangent_vector[len(dred)+auxPos[BusForCPF-1][2]-1] > -1):
            Y_Vph1.append(V[BusForCPF - 1])
            X_LambdaPh1.append(lambda_val-1)
            J_list.append(J)
            Last_d = d; Last_V = V; Last_LambdaPh1 = lambda_val
        else:  # Revert to last good state if failed
            d = Last_d; V = Last_V; lambda_val = Last_LambdaPh1
            dred = d[posSL == 0]; Vred = V[(posSL + posPV) == 0]
            break
            
    print(f"Phase1 completed with iterations = {phase1_counter}")
    
    # ===== PHASE 2: APPROACHING MAX LOADABILITY =====
    print("Phase2 -- Running")
    phase2_counter=0
    sigma=step2  # Smaller step size near limit
    
    while True:
        phase2_counter+=1
        # PREDICTOR STEP (voltage collapse direction)
        d_V_L = np.concatenate([dred, Vred, [lambda_val]])
        _, J = Jacobian_NRLF(Ybus, V, d, posSL, posPV)
        
        # Special direction vector targeting monitored bus voltage
        ek = np.zeros(len(dred)+len(Vred)+1)
        ek[len(dred)+auxPos[BusForCPF-1][2]-1]=-1
        
        # Form extended Jacobian
        JKe = np.column_stack((J,-K))
        JKe = np.vstack((JKe,ek))
        ZerosOne = np.concatenate([np.zeros(len(dred) + len(Vred)), [1]])
        
        if np.linalg.det(JKe)==0:
            print(f'Matrix singular at phase 2 iteration {phase2_counter}')
            break
            
        tangent_vector = np.linalg.solve(JKe, ZerosOne.T)
        d_V_L = d_V_L + sigma * tangent_vector
        
        # Update state
        dred = d_V_L[:len(dred)]; d[posSL == 0] = dred
        Vred = d_V_L[len(dred):(len(dred) + len(Vred))]
        V[posSL + posPV == 0] = Vred
        lambda_val = d_V_L[-1]
        
        # CORRECTOR STEP (modified power flow)
        it=0
        while True:
            _, J = Jacobian_NRLF(Ybus, V, d, posSL, posPV)
            Pcalc, Qcalc, dP, dQ, _, _, dPdQred = calculate_Pcalc_Qcalc(
                V, d, Ybus, lambda_val * Psch, lambda_val * Qsch, posSL, posPV)
    
            if it>=maxiterations:
                print("NR did not converge")
                break
                
            if not (np.any(np.abs(dPdQred) > tolerance)):
                break
            
            it+=1
            
            # Solve modified system
            RHS = np.concatenate([dPdQred, [0]])
            JPh2 = np.column_stack((J,-K))
            JPh2 = np.vstack((JPh2,ek))

            if np.linalg.det(JPh2)==0:
                print(f'Phase 2 correction singular at iteration {phase2_counter}')
                it=maxiterations+1
                break
                
            dddVredL = np.linalg.solve(JPh2, RHS)
            
            # Update variables
            d[posSL == 0] += dddVredL[:N - 1]
            V[posSL + PV == 0] *= (1 + dddVredL[N-1:(len(dddVredL) - 1)])
            lambda_val += dddVredL[-1]
    
        # Store results
        dred = d[posSL == 0]; Vred = V[(posSL + posPV) == 0]
        Y_Vph2.append(V[BusForCPF - 1])
        X_LambdaPh2.append(lambda_val-1)
        J_list.append(J)
    
        # Check termination conditions
        if lambda_val>Last_Lambda:
            Last_Lambda = lambda_val
            LastV = V[BusForCPF-1]
        if tangent_vector[-1] < -1 or it>=maxiterations:
            break

    print(f"Phase2 completed with iterations = {phase2_counter}")
    
    # ===== PHASE 3: VOLTAGE COLLAPSE (BEYOND NOSE POINT) ===== 
    print("Phase3 -- Running")
    phase3_counter=0
    sigma=step3  # Smallest step size
    
    while True:
        phase3_counter+=1
        # PREDICTOR STEP (reverse direction)
        d_V_L = np.concatenate([dred, Vred, [lambda_val]])
        _, J = Jacobian_NRLF(Ybus, V, d, posSL, posPV)
        ek = np.concatenate([np.zeros(J.shape[1]), [-1]])  # Note negative direction
        
        # Form extended Jacobian
        JKe = np.column_stack((J, -K))
        JKe = np.vstack((JKe,ek))
        
        if np.linalg.det(JKe) == 0:
            print(f'Matrix singular at phase 3 iteration {phase3_counter}')
            break
            
        tangent_vector = np.linalg.solve(JKe, np.abs(ek))
        d_V_L = d_V_L + sigma * tangent_vector
        
        # Update state
        dred = d_V_L[:len(dred)]; d[posSL == 0] = dred
        Vred = d_V_L[len(dred):(len(dred) + len(Vred))]
        V[posSL + posPV == 0] = Vred
        lambda_val = d_V_L[-1]
    
        # CORRECTOR STEP
        V, d, _, _, _, _, _, _, _, J, temp, status = NRLF(
            V, d, Ybus, lambda_val * Psch, lambda_val * Qsch, 
            posSL, posPV, tolerance, maxiterations)
        
        # Store results if successful
        if (status == 1 and temp < maxiterations and lambda_val >= 0 and 
            tangent_vector[len(dred)+auxPos[BusForCPF-1][2]-1] < 0):
            Y_Vph3.append(V[BusForCPF - 1])
            X_LambdaPh3.append(lambda_val-1)
            J_list.append(J)
        else:
            break

    print(f"Phase3 completed with iterations = {phase3_counter}")
    
    # Return concatenated results from all phases
    return (np.concatenate((X_LambdaPh1,X_LambdaPh2,X_LambdaPh3)), 
           np.concatenate((Y_Vph1,Y_Vph2,Y_Vph3)), 
           J_list)




def calculate_load_impedance_lamda(casedata, busIndex, lambda_val):
    """
    Calculate load impedance at a specific bus considering a loading factor (lambda)
    
    Args:
        casedata: Power system case data
        busIndex: Index of the bus to analyze
        lambda_val: Loading factor (1 = base case, >1 = overload)
        
    Returns:
        Complex load impedance at the specified bus
    """
    # Get bus data and base MVA
    bus = casedata['bus'][busIndex]
    baseMVA = casedata['baseMVA']
    
    # Scale load with lambda (P' = P*(1+lambda))
    P_load = bus[2] * (1 + lambda_val) / baseMVA  # Active power in pu
    Q_load = bus[3] * (1 + lambda_val) / baseMVA  # Reactive power in pu
    S_load = P_load + 1j*Q_load  # Complex power
    
    # Get bus voltage (column 7 is voltage magnitude)
    V_bus = bus[7]  
    
    # Calculate load impedance (Z = V²/S*)
    Z_load = (V_bus**2) / np.conj(S_load) if S_load != 0 else np.inf
    
    return Z_load

def calculate_load_impedance(casedata, busIndex):
    """
    Calculate load impedance at a specific bus for base case loading
    
    Args:
        casedata: Power system case data
        busIndex: Index of the bus to analyze
        
    Returns:
        Complex load impedance at the specified bus
    """
    # Convert data to numpy arrays
    BusData = np.array(casedata['bus']).astype(np.float64)
    BranchData=np.array(casedata['branch']).astype(np.float64)
    GenData=np.array(casedata['gen']).astype(np.float64)
    basePower=np.array(casedata['baseMVA']).astype(np.float64)
        
    # Convert powers to per unit values
    BusData[:,2]/=basePower  # Active power demand
    BusData[:,3]/=basePower  # Reactive power demand
    GenData[:,1:5]/=basePower  # Generator power limits
    
    # Convert additional parameters to per unit
    GenData[:,8:21]/=basePower
    BranchData[:,5:8]/=basePower
    
    # Convert angles from degrees to radians
    BusData[:,8]=BusData[:,8]*np.pi/180  # Bus voltage angles
    BranchData[:,9]=BranchData[:,9]*np.pi/180  # Branch angles
    BranchData[:,11:13]=BranchData[:,11:13]*np.pi/180  # Transformer angles

    # Get bus voltage magnitude (column 7)
    V_bus = BusData[busIndex, 7]  
    
    # Get load powers (columns 2 and 3)
    P_load = BusData[busIndex, 2]  # Active power
    Q_load = BusData[busIndex, 3]  # Reactive power
    S_load = P_load + 1j * Q_load  # Complex power
    
    # Return infinity for no-load case
    if S_load == 0:
        return float('inf')  
    
    # Calculate load impedance (Z = V²/S*)
    Z_load = (V_bus ** 2) / np.conj(S_load)
    return Z_load

def calculate_thevenin_impedance(casedata, busIndex):
    """
    Calculate Thevenin impedance at a specific bus
    
    Args:
        casedata: Power system case data
        busIndex: Index of the bus to analyze
        
    Returns:
        Complex Thevenin impedance at the specified bus
    """
    # Load data from case
    BusData = np.array(casedata['bus']).astype(np.float64)
    BranchData = np.array(casedata['branch']).astype(np.float64)
    basePower = float(casedata['baseMVA'])

    # Normalize values to per unit system
    BusData[:, 2:4] /= basePower  # Normalize load powers (P and Q)
    BusData[:, 8] *= np.pi / 180  # Convert angles to radians

    # Calculate Zbus matrix (inverse of Ybus)
    Zbus = Calculate_Zbus(BusData, BranchData)

    # Thevenin impedance is the diagonal element
    Zth = Zbus[busIndex][busIndex]
    return Zth

def calculateVSI(casedata, busIndex):
    """
    Calculate Voltage Stability Index (VSI) for a bus
    
    Args:
        casedata: Power system case data
        busIndex: Index of the bus to analyze
        
    Returns:
        VSI value (|Zth|/|Zload|)
    """
    # Get load and Thevenin impedances
    Z_load = calculate_load_impedance(casedata, busIndex)
    Z_th = calculate_thevenin_impedance(casedata, busIndex)
    
    # VSI is the ratio of impedances
    VSI = abs(Z_th) / abs(Z_load)
    return VSI

def calculate_impedance_average(casedata):
    """
    Calculate the average Thevenin impedance for the entire system
    
    Args:
        casedata: Power system case data
        
    Returns:
        Average system impedance magnitude
    """
    # Convert data to numpy arrays
    BusData=np.array(casedata['bus']).astype(np.float64)
    BranchData=np.array(casedata['branch']).astype(np.float64)
    GenData=np.array(casedata['gen']).astype(np.float64)
    basePower=np.array(casedata['baseMVA']).astype(np.float64)
        
    # Convert powers to per unit
    BusData[:,2]/=basePower
    BusData[:,3]/=basePower
    GenData[:,1:5]/=basePower
    
    # Convert additional parameters
    GenData[:,8:21]/=basePower
    BranchData[:,5:8]/=basePower
    
    # Convert angles to radians
    BusData[:,8]=BusData[:,8]*np.pi/180
    BranchData[:,9]=BranchData[:,9]*np.pi/180
    BranchData[:,11:13]=BranchData[:,11:13]*np.pi/180

    total_impedance = 0
    num_buses = len(BusData)

    # Sum impedances for all buses
    for i in range(num_buses):
        Zth = calculate_thevenin_impedance(casedata, i) 
        total_impedance += abs(Zth)

    # Calculate average
    Z = total_impedance / num_buses
    return Z

def calculate_impedance_average1(Ybus):
    """
    Calculate average impedance directly from Ybus matrix
    
    Args:
        Ybus: System admittance matrix
        
    Returns:
        Average system impedance magnitude
    """
    total_impedance = 0
    num_buses = len(Ybus)
    
    # Calculate Zbus (inverse of Ybus)
    Zbus = np.linalg.inv(Ybus)

    # Sum diagonal elements (self-impedances)
    for i in range(num_buses):
        Zth = Zbus[i,i] 
        total_impedance += abs(Zth)

    # Calculate average
    Z = total_impedance / num_buses
    return Z

def calculate_BPF(casedata):
    """
    Calculate Bus Participation Factors (BPF) for voltage stability analysis
    
    Args:
        casedata: Power system case data
        
    Returns:
        bpf: Dictionary of participation factors per bus
        eigenvalues: System eigenvalues
        WholeJ: Complete Jacobian matrix
        JR: Reduced Jacobian matrix
    """
    # Convert data to numpy arrays
    BusData=np.array(casedata['bus']).astype(np.float64)
    BranchData=np.array(casedata['branch']).astype(np.float64)
    GenData=np.array(casedata['gen']).astype(np.float64)
    basePower=np.array(casedata['baseMVA']).astype(np.float64)
    
    # Convert powers to per unit
    BusData[:,2]/=basePower
    BusData[:,3]/=basePower
    GenData[:,1:5]/=basePower
    
    # Convert additional parameters
    GenData[:,8:21]/=basePower
    BranchData[:,5:8]/=basePower
    
    # Convert angles to radians
    BusData[:,8]=BusData[:,8]*np.pi/180
    BranchData[:,9]=BranchData[:,9]*np.pi/180
    BranchData[:,11:13]=BranchData[:,11:13]*np.pi/180

    # Calculate Ybus matrix
    Ybus=Calculate_Ybus(BusData,BranchData)

    # Identify bus types (Slack=3, PV=2, PQ=1)
    posPV = BusData[:,1] == 2
    posSL = BusData[:,1] == 3

    # Count PV and PQ buses
    PV_num = 0
    PQ_num = 0
    for i in BusData:
        if(i[1] == 2):
            PV_num +=1
        elif(i[1] == 1):
            PQ_num+=1
    
    # Initialize voltages
    N=len(BusData)
    V=np.ones(N)   # Voltage magnitude
    d=np.zeros(N)  # Voltage angle

    # Set known voltages for PV and Slack buses
    for generator in GenData:
        V[int(generator[0]-1)]=generator[5]

    # Calculate Jacobian matrices
    WholeJ, J = Jacobian_NRLF(Ybus, V, d, posSL, posPV)
    
    # Create index sets for different bus types
    T_indices = []
    D_indices = []
    for i in range(PV_num+PQ_num,len(J)):
        D_indices.append(i)
    for i in range(PV_num+PQ_num):
        T_indices.append(i)
        
    # Partition Jacobian matrix
    JPT = J[np.ix_(T_indices, T_indices)]
    JPV = J[np.ix_(T_indices, D_indices)]
    JQT = J[np.ix_(D_indices, T_indices)]
    JQV = J[np.ix_(D_indices, D_indices)]

    # Calculate reduced Jacobian
    inv_JPT = np.linalg.inv(JPT)
    JR = JQV - JQT @ inv_JPT @ JPV
    
    # Eigenvalue decomposition
    eigenvalues, right_eigenvectors = np.linalg.eig(JR)
    left_eigenvectors = np.linalg.inv(right_eigenvectors)

    # Calculate participation factors
    bpf = {}
    for bus in range(len(V)-3):
        bpf[bus+3] = []
        for mode in range(len(eigenvalues)):
            participation = abs(right_eigenvectors[bus, mode] * left_eigenvectors[mode, bus])
            bpf[bus+3].append(participation)
    
    return bpf, eigenvalues, WholeJ, JR

def calculate_TDDI(casedata, buses_in_T):
    """
    Calculate Transmission-Distribution Dependency Index (TDDI)
    
    Args:
        casedata: Power system case data
        buses_in_T: Number of buses in transmission system
        
    Returns:
        TDDI value (logarithm of impedance ratio)
    """
    # Convert data to numpy arrays
    BusData=np.array(casedata['bus']).astype(np.float64)
    BranchData=np.array(casedata['branch']).astype(np.float64)
    GenData=np.array(casedata['gen']).astype(np.float64)
    basePower=np.array(casedata['baseMVA']).astype(np.float64)
    
    # Convert powers to per unit
    BusData[:,2]/=basePower
    BusData[:,3]/=basePower
    GenData[:,1:5]/=basePower
    
    # Convert additional parameters
    GenData[:,8:21]/=basePower
    BranchData[:,5:8]/=basePower
    
    # Convert angles to radians
    BusData[:,8]=BusData[:,8]*np.pi/180
    BranchData[:,9]=BranchData[:,9]*np.pi/180
    BranchData[:,11:13]=BranchData[:,11:13]*np.pi/180

    # Calculate Ybus matrix
    Ybus = Calculate_Ybus(BusData,BranchData)
    
    # Create index sets for transmission and distribution
    T_indices = []
    for i in range(0,buses_in_T):
        T_indices.append(i)
    D_indices = []
    for i in range(buses_in_T,len(Ybus)):
        D_indices.append(i)
        
    # Get submatrices
    YTT = Ybus[np.ix_(T_indices, T_indices)]
    YDD = Ybus[np.ix_(D_indices, D_indices)]

    # Calculate average impedances
    ZT = calculate_impedance_average1(YTT)
    ZD = calculate_impedance_average1(YDD)
    
    # Calculate TDDI (log of impedance ratio)
    return math.log(ZT/ZD)
