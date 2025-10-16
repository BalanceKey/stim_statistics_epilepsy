# from tvb.simulator.lab import *
from nibabel.freesurfer.io import read_geometry
import matplotlib.pyplot as plt
import pyvista as pv
import pandas as pd
import numpy as np
import sys
import re
import os
sys.path.append('/Users/dollomab/MyProjects/Stimulation/VirtualEpilepsySurgery/VEP/core/')
import vep_prepare
roi = vep_prepare.read_vep_mrtrix_lut()
sys.path.append('/Users/dollomab/OtherProjects/epi_visualisation/')
from util.utils import convert_to_pyvista_mesh, load_surfaces, load_bip_coords, plot_cortex_and_subcortex

# pid = 'sub-603cf699f88f' # 1
# pid = 'sub-2ed87927ff76' # 2
# pid = 'sub-0c7ab65949e1' # 3
# pid = 'sub-9c71d0dbd98f' # 4
pid = 'sub-4b4606a742bd' # 5
if pid == 'sub-4b4606a742bd':
    subject_dir = f'/Users/dollomab/MyProjects/Epinov_trial/stimulated_patients/{pid}/vep/'
else:
    subject_dir = f'/Users/dollomab/MyProjects/Epinov_trial/patients/{pid}/vep/'
util_dir = '/Users/dollomab/OtherProjects/epi_visualisation/util'
stimulation_data = f'~/MyProjects/Stimulation/Project_DBS_neural_fields/stimulation_data_manager_{pid}.csv'
df = pd.read_csv(stimulation_data)


# Load lut Mrtrix and Freesurfer
vep_mrtrix_lut         = np.genfromtxt(os.path.join(util_dir,"VepMrtrixLut.txt"),
                                        dtype="str", usecols=1, skip_header=1) # skip first row -> mandatory "Unknown" region
vep_freesurf_lut_names = np.genfromtxt(os.path.join(util_dir,"VepFreeSurferColorLut.txt"),
                                        comments="#", usecols=1, dtype=str)
vep_freesurf_lut_ind   = np.genfromtxt(os.path.join(util_dir,"VepFreeSurferColorLut.txt"),
                                        comments="#", usecols=0, dtype=int)

# Extract stimulation cases where a seizure was A) induced or B) NOT induced
induced_seizure = False
induced_seizure_rows = df.index[df['induced_seizure'] == induced_seizure]
stim_params = []
for stim_index in induced_seizure_rows: # TODO change back
    channels = re.findall(r'[a-zA-Z\']+', df['stim_electrodes'][stim_index])
    channel_nr = re.findall(r'[0-9]+', df['stim_electrodes'][stim_index])

    if channels[0][0] == 'I':
        channels = [channels[0][0] + channels[0][1].lower(), channels[1][0] + channels[1][1].lower()]
    if pid == 'sub-4b4606a742bd' or pid == 'sub-2ed87927ff76' or pid == 'sub-0c7ab65949e1':
        if channels[0] == 'LES':
            channels = [channels[0][0] + channels[0][1:].lower(), channels[1][0] + channels[1][1:].lower()]
        if channels[0] == 'LESA' or channels[0] == 'LESB':
            channels = [channels[0][0] + channels[0][1:3].lower() + channels[0][-1],
                        channels[1][0] + channels[1][1:3].lower() + channels[0][-1]]
        if int(channel_nr[0]) == int(channel_nr[1]) + 1 :  # e.g. channel A3-2 instead of A2-3
            channel_nr = [channel_nr[1], channel_nr[0]]    # quick fix: inverting channel numbers here

    if pid == 'sub-4b4606a742bd':
        freq_val = float(df['frequency'][stim_index].replace(',', '.')) # Hz
        amp_val = float(df['intensity'][stim_index].replace(',', '.'))  # mA
    else:
        freq_val = float(df['frequency'][stim_index]) # Hz
        amp_val = float(df['intensity'][stim_index]) # mA

    stimulation_parameters = {'choi': channels[0] + channel_nr[0] + '-' + channel_nr[1],
                    'freq': freq_val,  # Hz
                    'amp': amp_val,  # mA
                    'duration': float(df['duration'][stim_index]),  # seconds
                    'tau': float(df['pulse_width'][stim_index]),  # microseconds
                    }
    stim_params.append(stimulation_parameters)

# Compute stimulation field
seeg_xyz = vep_prepare.read_seeg_xyz(subject_dir)                   # read seeg electrodes information
seeg_xyz_names = [channel_name for channel_name, _ in seeg_xyz]     # read monopolar electrode names
gain = np.loadtxt(f'{subject_dir}/elec/gain_inv-square.vep.txt')    # import gain matrix
assert len(seeg_xyz_names) == gain.shape[0]
bip_gain, bip_xyz, bip_names = vep_prepare.bipolarize_gain_minus(gain, seeg_xyz, seeg_xyz_names)  # bipolarize gain
bip_gain_prior, _, _ = vep_prepare.bipolarize_gain_minus(gain, seeg_xyz, seeg_xyz_names, is_minus=False)
# Remove cereberall cortex influence
bip_gain_prior[:, roi.index('Left-Cerebellar-cortex')] = bip_gain_prior.min()
bip_gain_prior[:, roi.index('Right-Cerebellar-cortex')] = bip_gain_prior.min()
bip_gain[:, roi.index('Left-Cerebellar-cortex')] = bip_gain.min()
bip_gain[:, roi.index('Right-Cerebellar-cortex')] = bip_gain.min()


#%% Plot seizure thresholds as EZ heatmap
# Import seizure thresholds data
data_path = '/Users/dollomab/MyProjects/Stimulation/stim_statistics_epilepsy/results/'
seizure_thresholds_all = np.load(f'{data_path}/{pid}_diff_evolution_results_100_times.npz')['parameters_all']
seizure_thresholds = np.median(seizure_thresholds_all, axis=0)
# seizure_thresholds = np.load(f'{data_path}/{pid}_diff_evolution_results.npz')['parameters']

seizure_thresholds_normalized = (seizure_thresholds - np.nanmin(seizure_thresholds)) / (np.nanmax(seizure_thresholds) - np.nanmin(seizure_thresholds))
excitability = 1 - seizure_thresholds_normalized

n_regions = 162
plt.figure(figsize=(20, 2), tight_layout=True)
# heatmap = np.abs(m_thresholds-m_thresholds.max())
plt.bar(np.r_[0:n_regions], (excitability), color='purple', alpha=0.5)
plt.xticks(np.r_[:len(roi)], roi, rotation=90, fontsize=8)
plt.title(f'EZ heatmap', fontsize=20)
# plt.hlines(0.9, xmin=0, xmax=n_regions)
plt.xlim(0, n_regions)
plt.show()

#%% Plot brain mesh and stim field
# Load cortical + subcortical surfaces
vep_subcort_aseg_file = f'{util_dir}/subcort.vep.txt'
vep_subcort_aseg = np.genfromtxt(vep_subcort_aseg_file, dtype=int)
py_vista_mesh, cort_parc = load_surfaces(subject_dir, vep_subcort_aseg)         # Load pyvista mesh

# Plot
cmap = plt.cm.get_cmap("RdPu")  # "jet"                                         # define a colormap to use
p = pv.Plotter(notebook=False)
p.set_background(color="white")
plot_cortex_and_subcortex(p, py_vista_mesh, vep_subcort_aseg, subcortex=True)   # Plot cortical and subcortical mesh
for region_idx in range(162):                                                   # Plot stim weights for each region
    idx = np.where(vep_freesurf_lut_names==vep_mrtrix_lut[region_idx])[0]
    aseg = vep_freesurf_lut_ind[idx][0]
    file = os.path.join(subject_dir,"aseg2srf","vep","aseg_"+str(aseg).zfill(5))
    vertices, tris = read_geometry(file)
    py_vista_mesh[aseg] = convert_to_pyvista_mesh(vertices, tris)
    # plot mesh of region, set color by VEP_median, set opacity by seizure threshold
    # if excitability[region_idx] > np.mean(excitability):
    p.add_mesh(py_vista_mesh[aseg], color=cmap(excitability[region_idx]), opacity=excitability[region_idx])
p.set_scale(xscale=10, yscale=10, zscale=10, reset_camera=False)
p.show(interactive=True, full_screen=False)
p.close()