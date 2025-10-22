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

pid = 'sub-603cf699f88f' # 1
# pid = 'sub-2ed87927ff76' # 2
# pid = 'sub-0c7ab65949e1' # 3
# pid = 'sub-9c71d0dbd98f' # 4
# pid = 'sub-4b4606a742bd' # 5
if pid == 'sub-4b4606a742bd' or pid == 'sub-9c71d0dbd98f':
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

    if pid == 'sub-4b4606a742bd' or pid == 'sub-9c71d0dbd98f' or pid == 'sub-603cf699f88f':
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
N = 500
data_path = '/Users/dollomab/MyProjects/Stimulation/stim_statistics_epilepsy/results/'
results = np.load(f'{data_path}/{pid}_diff_evolution_results_{N}_times.npz')
seizure_thresholds_all = results['parameters_all']
seizure_thresholds = np.median(seizure_thresholds_all, axis=0)
# seizure_thresholds = np.load(f'{data_path}/{pid}_diff_evolution_results.npz')['parameters']
losses = results['loss']

seizure_thresholds_normalized = (seizure_thresholds - np.nanmin(seizure_thresholds)) / (np.nanmax(seizure_thresholds) - np.nanmin(seizure_thresholds))
excitability = 1 - seizure_thresholds_normalized

n_regions = 162
plt.figure(figsize=(20, 2), tight_layout=True)
# heatmap = np.abs(m_thresholds-m_thresholds.max())
plt.bar(np.r_[0:n_regions], (excitability), color='purple', alpha=0.5)
plt.xticks(np.r_[:len(roi)], roi, rotation=90, fontsize=8)
plt.title(f'EZ heatmap {pid}', fontsize=20)
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

#%% Plot jaccard similarity performance across methods
project_dir = '/Users/dollomab/MyProjects/Stimulation/VirtualEpilepsySurgery/'

# Load data
emp_binary_responses = np.load(f'{project_dir}/similarity_metric/{pid}/{pid}_EZ_empirical_responses.npz')['binary_responses']
if pid == 'sub-4b4606a742bd':
    sim1_binary_responses = np.load(f'{project_dir}/similarity_metric/{pid}/{pid}_Clinical_EZ_sim1_responses.npz', allow_pickle=True)['binary_responses']
    sim2_binary_responses = np.load(f'{project_dir}/similarity_metric/{pid}/{pid}_Custom_EZ_sim2_responses.npz', allow_pickle=True)['binary_responses']
    sim3_binary_responses = np.load(f'{project_dir}/similarity_metric/{pid}/{pid}_Custom_EZ_sim3_responses.npz', allow_pickle=True)['binary_responses']
else:
    sim1_binary_responses = np.load(f'{project_dir}/similarity_metric/{pid}/{pid}_EZ_sim1_responses.npz', allow_pickle=True)['binary_responses']
    sim2_binary_responses = np.load(f'{project_dir}/similarity_metric/{pid}/{pid}_EZ_sim2_responses.npz', allow_pickle=True)['binary_responses']
    sim3_binary_responses = np.load(f'{project_dir}/similarity_metric/{pid}/{pid}_EZ_sim3_responses.npz', allow_pickle=True)['binary_responses']

# methods = ['clinical \n EZ', 'model \n based \n EZ', 'differential \n evolution \n EZ']
methods = ['Clinical \n EZ', 'Updated \n EZ', 'Hierarchical \n EZ', 'Differential \n evolution \n EZ']

# Plot bar plot with jaccard similarities 
# compute transformed DE losses (percent Jaccard)
transformed_losses = -losses * 100
if transformed_losses.size == 0:
    raise RuntimeError("No DE losses available to plot.")

# Compute Jaccard index (returns 1D array of % values)
def per_run_jaccard(emp, sim):
    emp = np.asarray(emp)
    sim = np.asarray(sim)
    inter = np.sum((emp == 1) & (sim == 1))
    union = np.sum((emp == 1) | (sim == 1))
    return np.array([inter / union * 100])

# per-run jaccards for methods 1-3
j1_run = per_run_jaccard(emp_binary_responses, sim1_binary_responses)
j2_run = per_run_jaccard(emp_binary_responses, sim2_binary_responses)
j3_run = per_run_jaccard(emp_binary_responses, sim3_binary_responses)

# central estimate + uncertainty for DE results
mean4 = np.nanmean(transformed_losses)
sem4 = np.nanstd(transformed_losses) / np.sqrt(len(transformed_losses)) # Standard Error of the Mean
std4 = np.nanstd(transformed_losses)

# prepare data for the 4 bars (means)
means = [j1_run[0], j2_run[0], j3_run[0], mean4]
errs = [0, 0, 0, sem4]  # only show uncertainty for DE (SEM)

plt.figure(figsize=(8,5), tight_layout=True)

# main bars
x_pos = np.arange(4)
bars = plt.bar(x_pos, means, width=0.6, color='midnightblue', alpha=0.7, yerr=errs, capsize=10)

# overlay jittered individual DE points
rng = np.random.default_rng(2)
def jittered_x(base_x, n):
    return base_x + rng.normal(0, 0.1, size=n)
x_de = jittered_x(3, len(transformed_losses))
plt.scatter(x_de, transformed_losses, color='purple', alpha=0.55, s=26, label=f'DE runs (N={len(transformed_losses)})', edgecolors='none')

# violin for the DE distribution (method 4)
violin = plt.violinplot(transformed_losses, positions=[3], widths=0.6,
                        showmeans=False, showmedians=False, showextrema=False)
# style violin: fill and remove black outline
for pc in violin['bodies']:
    pc.set_facecolor('lightgrey')
    pc.set_edgecolor('black')
    pc.set_alpha(0.7)

# ensure any median line (if present) is hidden
if 'cmedians' in violin and violin['cmedians'] is not None:
    try:
        violin['cmedians'].set_visible(False)
    except Exception:
        pass

# labels, ticks, limits
plt.xticks(x_pos, methods, fontsize=21)
plt.yticks(np.arange(0, 101, 10), fontsize=12)
plt.ylabel('Jaccard index (%)', fontsize=24)
plt.xlabel('EZ definition', fontsize=24)
plt.ylim([0, 105])
plt.xlim([-0.35, 3.35])
plt.legend(frameon=False, loc='upper left', fontsize=16)
plt.tight_layout()
plt.show()
