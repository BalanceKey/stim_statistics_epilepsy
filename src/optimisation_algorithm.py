'''
Setting up an optimisation algorithm

Input: Empirical binary vector (seizure/no seizure) and corresponding stimulation parameters
Output: Optimal epileptogenic values that best match the empirical binary vector 

'''

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from tvb.simulator.lab import *
from utils import load_stimulation_parameters, plot_stimulation_responses
import sys
sys.path.append('/Users/dollomab/MyProjects/Stimulation/VirtualEpilepsySurgery/VEP/core/')
import vep_prepare 

#%% Empirical data 
pid = 1
patients = {1: 'sub-603cf699f88f', 2: 'sub-2ed87927ff76', 3: 'sub-0c7ab65949e1', 4: 'sub-9c71d0dbd98f', 5: 'sub-4b4606a742bd'}
subject_dir = f'/Users/dollomab/MyProjects/Epinov_trial/patients/{patients[pid]}/vep'
stimulation_data = f'~/MyProjects/Stimulation/Project_DBS_neural_fields/stimulation_data_manager_{patients[pid]}.csv'
# Load stimulation parameters
stimulation_parameters_list = load_stimulation_parameters(pid, patients, stimulation_data)
stim_emp_response_list = [stimulation['induced_seizure'] for stimulation in stimulation_parameters_list]

# Global connectivity
con = connectivity.Connectivity.from_file(f'{subject_dir}/tvb/connectivity.vep.zip')
con.tract_lengths = np.zeros((con.tract_lengths.shape))  # no time-delays
con.weights[np.diag_indices(con.weights.shape[0])] = 0
con.weights /= con.weights.max()
con.configure()
assert con.number_of_regions == 162
roi = con.region_labels.tolist() 
n_regions = len(roi)  # number of regions in the connectivity matrix

# Load SEEG electrodes
seeg_xyz = vep_prepare.read_seeg_xyz(subject_dir)                # read seeg electrodes information
seeg_xyz_names = [channel_name for channel_name, _ in seeg_xyz]  # read monopolar electrode names
#%% Load gain matrix
gain = np.loadtxt(f'{subject_dir}/elec/gain_inv-square.vep.txt')  # import gain matrix
assert len(seeg_xyz_names) == gain.shape[0]
bip_gain, bip_xyz, bip_names = vep_prepare.bipolarize_gain_minus(gain, seeg_xyz, seeg_xyz_names)  # bipolarize gain
bip_gain_prior, _, _ = vep_prepare.bipolarize_gain_minus(gain, seeg_xyz, seeg_xyz_names, is_minus=False)
# Remove cereberall cortex influence
bip_gain[:, roi.index('Left-Cerebellar-cortex')] = bip_gain.min()
bip_gain[:, roi.index('Right-Cerebellar-cortex')] = bip_gain.min()
bip_gain_prior[:, roi.index('Left-Cerebellar-cortex')] = bip_gain_prior.min()
bip_gain_prior[:, roi.index('Right-Cerebellar-cortex')] = bip_gain_prior.min()
# Normalize gain matrix
bip_gain_norm = (bip_gain - bip_gain.min()) / (bip_gain.max() - bip_gain.min())
bip_gain_prior_norm = (bip_gain_prior - bip_gain_prior.min()) / (bip_gain_prior.max() - bip_gain_prior.min())

# EZ hypothesis
if patients[pid] == 'sub-603cf699f88f':
     EZ_clinical = ['Right-Temporal-pole', 'Right-Hippocampus-anterior', 'Right-Amygdala', 'Right-Rhinal-cortex',
              'Right-Insula-gyri-brevi', 'Right-Insula-gyri-longi']
elif patients[pid] == 'sub-2ed87927ff76':
    EZ_clinical = ['Right-Hippocampus-anterior', 'Right-Hippocampus-posterior', 'Right-Rhinal-cortex',
                   'Right-Amygdala', 'Right-Parahippocampal-cortex', 'Left-Hippocampus-anterior']
    EZ_Huifang = ['Right-Amygdala','Right-Hippocampus-anterior','Right-Rhinal-cortex','Right-Collateral-sulcus','Right-Temporal-pole',
                  'Right-T3-anterior','Right-T2-anterior', 'Left-Hippocampus-anterior']
elif patients[pid] == 'sub-0c7ab65949e1':
     EZ_clinical = ['Right-Anterior-cingulate-cortex', 'Right-Orbito-frontal-cortex', 'Right-Insula-gyri-brevi', 
                       'Right-F3-Pars-triangularis', 'Right-Temporal-pole', 'Right-Amygdala', 'Right-Hippocampus-anterior', 'Right-Rhinal-cortex']
elif patients[pid] == 'sub-9c71d0dbd98f':
    EZ_clinical = ['Right-Temporal-pole', 'Right-Rhinal-cortex', 'Right-Amygdala', 'Right-Hippocampus-anterior',
                       'Right-Hippocampus-posterior']
elif patients[pid] == 'sub-4b4606a742bd':
    EZ_clinical = ['Right-Orbito-frontal-cortex']
else:   
    print(f"Unknown type {type} for patient {patients[pid]} !")
EZ = EZ_clinical
print(f"Using EZ hypothesis for patient {patients[pid]}: {EZ}")

# Compute seizure threshold from EZ vector
x0_vector = np.ones(len(roi)) * -2.5
ez_idx = [roi.index(ez) for ez in EZ]
x0_vector[ez_idx] = -1.6
epileptors_threshold = 20 / (1 + np.exp(10*(x0_vector + 2.1))) + 1 # sigmoid function centred around -2.1
assert np.all(epileptors_threshold > 0)  # check thresholds>0, otherwise seizure starts automatically
m_thresholds_initial=epileptors_threshold

#%% Model
# Example: params -> simulated vector
epileptors_r2 = np.ones(n_regions) * (0.002)
m_thresholds = m_thresholds_initial
def run_model(m_thresholds):

    stim_sim_response_list = []
    # compute for each stimulus if a seizure is triggered or not in the model
    for stim_parameter in stimulation_parameters_list:
        stim_weights = bip_gain_prior_norm[bip_names.index(stim_parameter['choi'])]  # stimulation weights for the chosen channel
        sfreq = stim_parameter['sfreq']/4                         # how many steps there are in one second
        T = 1 / stim_parameter['freq'] * sfreq                    # pulse repetition period [s]
        tau = stim_parameter['tau']/1000000 * sfreq               # pulse duration, number of steps [microsec]
        I = stim_parameter['amp']                                 # pulse intensity [mA]

        A = stim_weights * I                                              # amplitude of the stimulus for each region
        d = tau * 2                                                       # duration of the stimulus in seconds
        N = stim_parameter['duration'] * stim_parameter['freq']           # number of pulses
        m_max = np.zeros(n_regions)                 # compute m max analytically from the stimulus timeseries
        m_max[:] = (20 * A[:] / 0.3) * (1 - np.exp(-0.3 * epileptors_r2[:] * d)) * (1 - np.exp(-0.3 * epileptors_r2[:] * N * T)) / (1 - np.exp(-0.3*epileptors_r2[:]*T))

        # Check if a seizure was triggered or not
        induced_seizure_sim = False
        for i in range(m_thresholds.size):
            if m_max[i] > m_thresholds[i]:
                print(f"Seizure induced in region index {i}: {roi[i]}")
                # regions_induced_seizure.append(i)  # add regions where seizure was induced
                induced_seizure_sim = True
        stim_sim_response_list.append(induced_seizure_sim)

    return stim_sim_response_list

# stim_sim_response_list = run_model(m_thresholds, stimulation_parameters_list)
# plot_stimulation_responses(stim_sim_response_list, stim_emp_response_list)

#%%  Loss function
# Similarity metric: Hamming loss (proportion of mismatches)
def hamming_loss(empirical, simulated):
    return np.mean(empirical != simulated)
# TODO why is it 1 ??? 

#%%  Optimisation algorithm
# Objective function for optimization
empirical = stim_emp_response_list
def objective(params):
    # Run multiple simulations to smooth out randomness
    sims = [run_model(params) for _ in range(5)]
    losses = [hamming_loss(empirical, s) for s in sims]
    return np.mean(losses)

# Parameter bounds (example: probability between 0 and 1)
bounds = [(0.1, 21)] * n_regions

result = differential_evolution(objective, bounds, maxiter=50, popsize=15, tol=1e-3)
print("Best parameters:", result.x)
print("Best loss:", result.fun)