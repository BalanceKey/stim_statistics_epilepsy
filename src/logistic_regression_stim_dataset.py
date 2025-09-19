"Perform a simple logistic regression on a stimulus dataset"
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm

pid = 3
patients = {1: 'sub-603cf699f88f', 2: 'sub-2ed87927ff76', 3: 'sub-0c7ab65949e1', 4: 'sub-9c71d0dbd98f', 5: 'sub-4b4606a742bd'}
stim_data_path = '/Users/dollomab/MyProjects/Stimulation/Project_DBS_neural_fields/'


iterable_params_choi_all_patients = []
iterable_params_all_patients = []
for pid in range(1, 6):
    print(f'Patient ID: {pid}, BIDS ID: {patients[pid]}')
    # %% Load stimulation parameters
    stimulation_data = f'{stim_data_path}/stimulation_data_manager_{patients[pid]}.csv'
    df = pd.read_csv(stimulation_data)
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
                                    'induced_seizure': df['induced_seizure'][stim_index],
                                    'patient': patients[pid]
                                    }
        else:
            stimulation_parameters = {'choi': channels[0] + channel_nr[0] + '-' + channel_nr[1],
                                    'freq': float(df['frequency'][stim_index]),  # Hz
                                    'amp': float(df['intensity'][stim_index]),  # mA
                                    'duration': float(df['duration'][stim_index]),  # seconds
                                    'tau': float(df['pulse_width'][stim_index]),  # microseconds
                                    'sfreq': 512,  # Hz
                                    'stim_index': stim_index,
                                    'induced_seizure': df['induced_seizure'][stim_index],
                                    'patient': patients[pid]
                                    }
        iterable_params.append(stimulation_parameters)
        
    iterable_params_all_patients.extend(iterable_params)

    #%% Take only list of stimulations where the pair of electrodes triggered at least one seizure
    induced_seizures = df['induced_seizure']
    idx = np.where(induced_seizures == True)[0]
    stim_electrodes = df['stim_electrodes'] # all stimulated electrodes 
    seizure_inducing_electrodes = df['stim_electrodes'][idx].tolist() # seizures were induced from these electrodes

    # Search all other cases with stimulus applied in these same channels
    iterable_params_choi = []
    for choi in seizure_inducing_electrodes:
        print(f'All stimulation cases with {choi}:')
        idx = np.where(stim_electrodes == choi)[0]
        for i in idx:
            iterable_params_choi.append(iterable_params[i])
            print(iterable_params[i])

    iterable_params_choi_all_patients.extend(iterable_params_choi)

#%% Logistic regression
# X = np.array([[item['freq'], item['amp'], item['duration']] for item in iterable_params_choi_all_patients])

X = np.array([[item['freq'], item['amp']] for item in iterable_params_choi_all_patients])
X = sm.add_constant(X)  # adding a constant
y = np.array([item['induced_seizure'] for item in iterable_params_choi_all_patients])

model = sm.Logit(y, X)
result = model.fit()
print(result.summary())

'''
The output gives you coefficients (β) and p-values.
Positive β → parameter increases seizure odds.
Negative β → parameter decreases seizure odds.
Significant p-value → parameter is statistically associated.
Convert β to odds ratio:
odds ratio = e ^ β

Example:
Frequency coefficient β = 2.3 → OR = 10 → 50 Hz stimulations are 10×
 more likely to cause seizures than 1 Hz (controlling for other factors).'''

#%% Firth penalized logistic regression to reduce bias in small datasets and handle quasi-separation
from firthlogist import FirthLogisticRegression

X = np.array([[item['freq'], item['amp']] for item in iterable_params_choi_all_patients])
y = np.array([item['induced_seizure'] for item in iterable_params_choi_all_patients])
model = FirthLogisticRegression().fit(X, y)
print(model.summary())

# Doing the same regression with the entire dataset 
X = np.array([[item['freq'], item['amp']] for item in iterable_params_all_patients])
y = np.array([item['induced_seizure'] for item in iterable_params_all_patients])
model = FirthLogisticRegression().fit(X, y)
print(model.summary())

#%% Conditional logistic regression to account for repeated measures from the same electrode pairs
# Cox regression with constant time, stratified by electrode
# 'status' is seizure outcome, 'duration' is dummy (e.g., all 1)
from statsmodels.duration.hazard_regression import PHReg

data = pd.DataFrame(iterable_params_choi_all_patients)
# data = pd.DataFrame(iterable_params_all_patients)


data["stratum"] = data["patient"].astype(str) + "_" + data["choi"].astype(str)
data["duration"] = 1

model = PHReg.from_formula("duration ~ freq + amp", status="induced_seizure", strata="stratum", data=data)
result = model.fit()
print(result.summary())

# NOTE Conditional logistic regression automatically ignores strata with no events: CLR naturally focuses only on electrodes where at least one seizure occurred

 #%% TODO idea 
 # Do linear regression on the same dataset but with the response variable being stimulation frequency based on 
 # whether a seizure was induced or not. # This would give an idea of what frequency is more likely to induce a seizure.

 # add to the regression maybe: distance to the SOZ # if the distance is smaller, the probability of inducing a seizure is higher