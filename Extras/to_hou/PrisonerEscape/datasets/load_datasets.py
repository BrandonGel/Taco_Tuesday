import numpy as np
# from datasets.dataset_old import RedBlueDataset, RedBlueSequence, RedBlueSequenceOriginal, RedBlueSequenceSkip
# from datasets.multi_head_dataset_old import MultiHeadDataset
from torch.utils.data import DataLoader
from datasets.dataset import VectorPrisonerDataset, GNNPrisonerDataset, HeterogeneousGNNDataset

def load_dataset(config, path_type):
    file_path = config[path_type]
    seq_len = config["seq_len"]
    num_heads = config["num_heads"]
    step_length = config["step_length"]
    include_current = config["include_current"]
    view = config["view"]
    multi_head = config["multi_head"]

    if config["dataset_type"] == "gnn":
        dataset = GNNPrisonerDataset(file_path, 
            seq_len, 
            num_heads, 
            step_length, 
            include_current=include_current, 
            multi_head = multi_head,
            one_hot=config["one_hot_agents"], 
            timestep=config["timestep"], 
            detected_location=config["detected_location"],
            get_start_location=config["get_start_location"])
        
        print("one_hot_agents:", config["one_hot_agents"], 
            "timestep:", config["timestep"],
            "detected_location:", config["detected_location"])

    elif config["dataset_type"] == "het_gnn":
        dataset = HeterogeneousGNNDataset(file_path, 
            seq_len, 
            num_heads, 
            step_length, 
            include_current=include_current, 
            multi_head = multi_head,
            one_hot=config["one_hot_agents"], 
            timestep=config["timestep"], 
            detected_location=config["detected_location"], 
            get_start_location=config["get_start_location"])
        
        print("one_hot_agents:", config["one_hot_agents"], 
            "timestep:", config["timestep"],
            "detected_location:", config["detected_location"])
    else:
        dataset = VectorPrisonerDataset(file_path, 
            seq_len, 
            num_heads, 
            step_length, 
            include_current=include_current, 
            multi_head = multi_head, 
            view=view)
    return dataset

def load_dataset_with_config_and_file_path(config, file_path):
    """ Utilize a configuration file but use a different model path """
    seq_len = config["seq_len"]
    num_heads = config["num_heads"]
    step_length = config["step_length"]
    include_current = config["include_current"]
    view = config["view"]
    multi_head = config["multi_head"]

    if config["dataset_type"] == "gnn":
        dataset = GNNPrisonerDataset(file_path, 
            seq_len, 
            num_heads, 
            step_length, 
            include_current=include_current, 
            multi_head = multi_head,
            one_hot=config["one_hot_agents"], 
            timestep=config["timestep"], 
            detected_location=config["detected_location"],
            get_start_location=config["get_start_location"])
        
        print("one_hot_agents:", config["one_hot_agents"], 
            "timestep:", config["timestep"],
            "detected_location:", config["detected_location"])

    elif config["dataset_type"] == "het_gnn":
        dataset = HeterogeneousGNNDataset(file_path, 
            seq_len, 
            num_heads, 
            step_length, 
            include_current=include_current, 
            multi_head = multi_head,
            one_hot=config["one_hot_agents"], 
            timestep=config["timestep"], 
            detected_location=config["detected_location"], 
            get_start_location=config["get_start_location"])
        
        print("one_hot_agents:", config["one_hot_agents"], 
            "timestep:", config["timestep"],
            "detected_location:", config["detected_location"])
    else:
        dataset = VectorPrisonerDataset(file_path, 
            seq_len, 
            num_heads, 
            step_length, 
            include_current=include_current, 
            multi_head = multi_head, 
            view=view)
    return dataset

def load_datasets(config, batch_size):
    """
    Load datasets from config
    """
    # load train dataloader
    # reb_obs, blue_obs, red_locs, dones, sequence_length, view
    # num_workers = config["num_workers"]
    train_dataset = load_dataset(config, "train_path")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # load test dataset
    test_dataset = load_dataset(config, "test_path")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    return train_dataloader, test_dataloader