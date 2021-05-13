import pickle
import os

var_dict = ['energy_lcc_dict',
            'energy_lcc_final_dict',
            'lcc_final_dict',
            'pv_dict',
            'npv_dict',
            'market_share_dict',
            'renovation_rate_dict',
            'flow_renovation_label_dict',
            'flow_renovation_label_energy_dict',
            'flow_demolition_dict'
            ]

name_folder = '20210512_18XXXX'
folder_output = os.path.join(os.getcwd(), 'project', 'output', name_folder)

variable = 'lcc_final_dict'
name_file = os.path.join(folder_output, '{}.pkl'.format(variable))
with open(name_file, 'rb') as handle:
    var_dict = pickle.load(handle)
