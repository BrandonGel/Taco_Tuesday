""" Let's filter out the dataset to 1. separate out the categorical and non-categorical features """
import numpy as np
from simulator import initialize_prisoner_environment

def get_continuous(np_data, feature_name, obs_dict):
    continuous = []
    for i in [j for j in obs_dict.keys() if feature_name in j]:
        idxes = obs_dict[i]
        c = (np_data[:, idxes[0]:idxes[1]])
    continuous.append(c)
    continuous = np.concatenate(continuous, axis=1)
    return continuous
    # if feature_name in obs_dict:
    #     return np_data[:, obs_dict[feature_name][0]:obs_dict[feature_name][1]]
    # else:
        # raise ValueError('Feature name not found in obs_dict')

def get_continuous_and_discrete(np_data, feature_name, obs_dict):
    """ Separate out the boolean for the features and the locations as the continuous """
    discrete = []
    continuous = []
    for i in [j for j in obs_dict.keys() if feature_name in j]:
        idxes = obs_dict[i]
        # bool is between 0 and 1
        d = (np_data[:, idxes[0]:idxes[0]+1])
        # location is between 0 and 1
        c = (np_data[:, idxes[0]+1:idxes[1]])
        discrete.append(d)
        continuous.append(c)
    discrete = np.concatenate(discrete, axis=1)
    continuous = np.concatenate(continuous, axis=1)
    return discrete, continuous

def filter_dataset(np_data, obs_dict, feature_list, split_categorical=False):
    continuous = []
    discrete = []
    
    if 'time' in feature_list:
        continuous.append(get_continuous(np_data, 'time', obs_dict))
    
    if 'prisoner_loc' in feature_list:
        continuous.append(get_continuous(np_data, 'prisoner_loc', obs_dict))

    if 'prev_action' in feature_list:
        continuous.append(get_continuous(np_data, 'prev_action', obs_dict))
    
    if 'search_party_detect' in feature_list:
        d, c = get_continuous_and_discrete(np_data, 'search_party_detect', obs_dict)
        discrete.append(d)
        continuous.append(c)

    if 'helicopter_detect' in feature_list:
        d, c = get_continuous_and_discrete(np_data, 'helicopter_detect', obs_dict)
        discrete.append(d)
        continuous.append(c)

    if 'camera_loc' in feature_list:
        d, c = get_continuous_and_discrete(np_data, 'camera_loc', obs_dict)
        discrete.append(d)
        continuous.append(c)

    if 'hideout_loc' in feature_list:
        continuous.append(get_continuous(np_data, 'hideout_loc', obs_dict))


    continuous = np.concatenate(continuous, axis=1)
    discrete = np.concatenate(discrete, axis=1)
    if split_categorical:
        return continuous, discrete
    else:
        return np.concatenate((continuous, discrete), axis=1)

if __name__ == "__main__":
    # env = initialize_prisoner_environment(0)
    # prediction_obs_dict= env.prediction_obs_names._idx_dict
    # blue_obs_dict = env.blue_obs_names._idx_dict
    dataset_path = "/nethome/sye40/PrisonerEscape/shared_latent/dataset/map_0_run_300_eps_0.npz"
    np_file = np.load(dataset_path, allow_pickle=True)
    # np_file['prediction_dict'] = np.array(prediction_obs_dict)
    # np_file['blue_dict'] = np.array(blue_obs_dict)
    # np.savez("/nethome/sye40/PrisonerEscape/shared_latent/dataset/map_0_run_100_eps_0_test.npz", **np_file)
    
    # print(np_file['red_observations'].shape)

    prediction_obs_dict = np_file['prediction_dict'].item()
    blue_obs_dict = np_file['blue_dict'].item()

    # Include time, prisoner_location
    feature_names = ['time', 'prisoner_loc', 'search_party_detect', 'helicopter_detect', 'prev_action', 'hideout_loc']
    # print(prediction_obs_names._idx_dict)

    dataset = filter_dataset(np_file['red_observations'], prediction_obs_dict, feature_names, split_categorical=False)
    print(dataset.shape)
    # print(np_file['red_observations'].shape)