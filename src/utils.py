
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np

def load_stimulation_parameters(pid, patients, stimulation_data_path):
    df = pd.read_csv(stimulation_data_path)
    iterable_params = []
    for i in range(0, df.shape[0]):
        stim_index = i  # taking each stimulation one by one
        channels = re.findall(r'[a-zA-Z\']+', df['stim_electrodes'][stim_index])
        channel_nr = re.findall(r'[0-9]+', df['stim_electrodes'][stim_index])

        if channels[0][0] == 'I':
            channels = [channels[0][0] + channels[0][1].lower(), channels[1][0] + channels[1][1].lower()]
        if patients[pid] == 'sub-4b4606a742bd' or patients[pid] == 'sub-2ed87927ff76' or patients[pid] == 'sub-0c7ab65949e1':
            if channels[0] == 'LES':
                channels = [channels[0][0] + channels[0][1:].lower(), channels[1][0] + channels[1][1:].lower()]
            if channels[0] == 'LESA' or channels[0] == 'LESB':
                channels = [channels[0][0] + channels[0][1:3].lower() + channels[0][-1],
                            channels[1][0] + channels[1][1:3].lower() + channels[0][-1]]
            if int(channel_nr[0]) == int(channel_nr[1]) + 1:  # e.g. channel A3-2 instead of A2-3
                channel_nr = [channel_nr[1], channel_nr[0]]  # quick fix: inverting channel numbers here

        if patients[pid] == 'sub-603cf699f88f' or patients[pid]  == 'sub-9c71d0dbd98f' or patients[pid] == 'sub-4b4606a742bd':
            stimulation_parameters = {'choi': channels[0] + channel_nr[0] + '-' + channel_nr[1],
                                    'freq': float(df['frequency'][stim_index].replace(',', '.')),  # Hz
                                    'amp': float(df['intensity'][stim_index].replace(',', '.')),  # mA
                                    'duration': float(df['duration'][stim_index]),  # seconds
                                    'tau': float(df['pulse_width'][stim_index]),  # microseconds
                                    'sfreq': 512,  # Hz
                                    'stim_index': stim_index,
                                    'induced_seizure': df['induced_seizure'][stim_index]  # whether seizure was induced or not
                                    }
        else:
            stimulation_parameters = {'choi': channels[0] + channel_nr[0] + '-' + channel_nr[1],
                                    'freq': float(df['frequency'][stim_index]),  # Hz
                                    'amp': float(df['intensity'][stim_index]),  # mA
                                    'duration': float(df['duration'][stim_index]),  # seconds
                                    'tau': float(df['pulse_width'][stim_index]),  # microseconds
                                    'sfreq': 512,  # Hz
                                    'stim_index': stim_index,
                                    'induced_seizure': df['induced_seizure'][stim_index]  # whether seizure was induced or not
                                    }
        iterable_params.append(stimulation_parameters)
    return iterable_params

def plot_stimulation_responses(stim_sim_response_list, stim_emp_response_list):
    emp_sim_responses = np.column_stack((stim_sim_response_list, stim_emp_response_list))
    xticks = [0, 1], ['Sim', 'Emp']
    plt.figure(figsize=(4, 10), tight_layout=True)
    plt.imshow(emp_sim_responses, aspect='auto', cmap=plt.cm.Blues)
    plt.vlines(x=0.5, ymin=0, ymax=emp_sim_responses.shape[0], color='darkslategrey')
    plt.vlines(x=1.5, ymin=0, ymax=emp_sim_responses.shape[0], color='darkslategrey')
    plt.xticks(xticks[0], xticks[1], fontsize=12)
    plt.ylim([0, emp_sim_responses.shape[0]])
    plt.show()
