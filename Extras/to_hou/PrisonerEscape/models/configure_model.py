from models.decoder import SingleGaussianDecoderStd, SingleGaussianDecoderStdParameter, MixtureDensityDecoder
from models.encoders import EncoderRNN
from models.model import Model
from models.multi_head_mog import MixureDecoderMultiHead
from connected_filtering_prediction.model import MixtureInMiddle
import torch.nn as nn
# from models.gnn.gnn_post_lstm import GNNPostLSTM

def configure_decoder(conf, dim, num_heads, multi_head):
    number_gaussians = conf["number_gaussians"]
    if multi_head:
        decoder = MixureDecoderMultiHead(
            input_dim=dim,
            num_heads = num_heads,
            output_dim=2,
            num_gaussians=number_gaussians,
        )
    else:
        decoder = MixtureDensityDecoder(
            input_dim=dim,
            # input_dim = 16,
            output_dim=2, # output dimension is always 2 since this is in the middle
            num_gaussians=number_gaussians,
        )

    return decoder

def configure_gnn_model(conf, num_heads, total_input_dim, multi_head, concat_dim):
    """
    Concat dimension is what we concatenate to the final gnn pooled state that we feed into decoder
    """
    from models.gnn.gnn import GNNLSTM
    # hidden_dim = conf["hidden_dim"]
    # input_dim = 3
    # hidden_dim = 8
    hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]

    encoder = GNNLSTM(total_input_dim, hidden_dim, gnn_hidden_dim)
    dim = gnn_hidden_dim + concat_dim
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_hetero_gnn_lstm_front_model(conf, num_heads, total_input_dim, multi_head, start_location):
    from models.gnn.decoupled_hetero_lstm import LSTMHeteroPost 
    # hidden_dim = conf["hidden_dim"]
    # input_dim = 3
    # hidden_dim = 8
    hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    # hideout_timestep_dim = 3
    # hideout_timestep_location_dim = 5

    encoder = LSTMHeteroPost(total_input_dim, hidden_dim, gnn_hidden_dim, 1, start_location)
    # dim = gnn_hidden_dim + hideout_timestep_dim
    dim = gnn_hidden_dim
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_gc_lstm_gnn(conf, num_heads, total_input_dim, multi_head):
    from models.gnn.gclstm import GCLSTMPrisoner
    gnn_hidden_dim = conf["gnn_hidden_dim"]

    hideout_timestep_dim = 3

    encoder = GCLSTMPrisoner(total_input_dim, gnn_hidden_dim, 1)
    dim = gnn_hidden_dim + hideout_timestep_dim
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_hybrid_gnn(conf, num_heads, total_input_dim, multi_head):
    from models.gnn.hybrid_gnn import HybridGNN
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    hidden_dim = conf["hidden_dim"]

    encoder = HybridGNN(total_input_dim, gnn_hidden_dim, hidden_dim)
    decoder = configure_decoder(conf, hidden_dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model


def configure_connected_model(conf, num_heads):
    encoder_type = conf["encoder_type"]
    hidden_dim = conf["hidden_dim"]
    decoder_type = conf["decoder_type"]
    number_gaussians = conf["number_gaussians"]
    input_dim = conf["input_dim"]
    hidden_connected_bool = conf["hidden_connected"]

    if encoder_type == "lstm":
        encoder = EncoderRNN(input_dim, hidden_dim)
    mixture_decoder = MixtureDensityDecoder(
            input_dim=hidden_dim,
            output_dim=2, # output dimension is always 2 since this is in the middle
            num_gaussians=number_gaussians,
        )

    if hidden_connected_bool:
        decoder_input_dim = 2 + hidden_dim
    else:
        decoder_input_dim = 2

    # location_decoder = SingleGaussianDecoderStd(2, output_dim=2*num_heads) # could switch to mixture as well
    if decoder_type == "single_gaussian":
        # We multiply the number of heads by 2 for each dimension in (x, y)
        location_decoder = SingleGaussianDecoderStd(decoder_input_dim, output_dim=2*num_heads)
    elif decoder_type == "mixture":
        location_decoder = MixureDecoderMultiHead(
            input_dim=hidden_dim,
            num_heads = num_heads,
            output_dim=2,
            num_gaussians=number_gaussians,
        )

    
    model = MixtureInMiddle(encoder, mixture_decoder, location_decoder, hidden_connected_bool)
    return model

def configure_regular(conf, num_heads, multi_head):
    """_summary_

    Args:
        conf (dict): The model configuration dictionary from the yaml config file. 
        num_heads (int): Represents the number of heads of the model.

    Returns:
        _type_: pytorch model
    """
    encoder_type = conf["encoder_type"]
    hidden_dim = conf["hidden_dim"]
    decoder_type = conf["decoder_type"]
    number_gaussians = conf["number_gaussians"]
    input_dim = conf["input_dim"]

    if encoder_type == "lstm":
        encoder = EncoderRNN(input_dim, hidden_dim)


    decoder = configure_decoder(conf, hidden_dim, num_heads, multi_head)
    # if decoder_type == "single_gaussian":
    #     # We multiply the number of heads by 2 for each dimension in (x, y)
    #     decoder = SingleGaussianDecoderStd(hidden_dim, output_dim=2*num_heads)
    # elif decoder_type == "mixture":
    #     decoder = MixureDecoderMultiHead(
    #         input_dim=hidden_dim,
    #         num_heads = num_heads,
    #         output_dim=2,
    #         num_gaussians=number_gaussians,
    #     )
    #     # decoder = MixtureDensityDecoder(
    #     #     input_dim=hidden_dim,
    #     #     # input_dim = 16,
    #     #     output_dim=2, # output dimension is always 2 since this is in the middle
    #     #     num_gaussians=number_gaussians,
    #     # )

    print(decoder_type)

    model = Model(encoder, decoder)

    return model

def configure_gnn_lstm_model(conf):
    # hidden_dim = conf["hidden_dim"]
    input_dim = 3
    hidden_dim = 8
    hideout_timestep_dim = 3

    number_gaussians = conf["number_gaussians"]

    encoder = GNNPostLSTM(input_dim, hidden_dim)
    decoder = MixtureDensityDecoder(
        input_dim=hidden_dim + hideout_timestep_dim,
        # input_dim = 16,
        output_dim=2, # output dimension is always 2 since this is in the middle
        num_gaussians=number_gaussians,
    )

    model = Model(encoder, decoder)
    return model

def configure_stn(conf, num_heads, total_input_dim, multi_head):
    from models.gnn.stn import STGCNEncoder
    # hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    hideout_timestep_dim = 3
    periods=16
    batch_size=128

    encoder = STGCNEncoder(total_input_dim, gnn_hidden_dim, periods, batch_size)
    dim = gnn_hidden_dim + hideout_timestep_dim
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_temporal_gcn(conf, num_heads, total_input_dim, multi_head):
    from experimental.temporal_gcn import TemporalGNN
    # hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]
    hideout_timestep_dim = 3
    periods = 16
    batch_size = 128

    encoder = TemporalGNN(node_features=total_input_dim,
                          hidden_dim=gnn_hidden_dim,
                          periods=periods,
                          batch_size=batch_size)

    dim = gnn_hidden_dim + hideout_timestep_dim
    decoder = configure_decoder(conf, dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_hetero_lstm(conf, num_heads, total_input_dim, multi_head):
    from models.gnn.hetero_gc_lstm_batch import HeteroLSTM
    # hidden_dim = conf["hidden_dim"]
    gnn_hidden_dim = conf["gnn_hidden_dim"]

    encoder = HeteroLSTM(gnn_hidden_dim)
    decoder = configure_decoder(conf, gnn_hidden_dim, num_heads, multi_head)
    model = Model(encoder, decoder)
    return model

def configure_cvae(conf, num_heads, multi_head):
    from models.cvae.model_continuous import CVAEContinuous
    gmm_bool = conf["gmm_bool"]
    z_dim = conf["z_dim"]
    encoder_hidden_dim = conf["encoder_hidden_dim"]
    future_hidden_dim = conf["future_hidden_dim"]
    z_dim = conf["z_dim"]
    x_dim = conf["input_dim"]
    # decoder_input_dim = z_dim * 2
    # decoder = configure_decoder(conf, decoder_input_dim, num_heads, multi_head)
    cvae = CVAEContinuous(x_dim, encoder_hidden_dim, future_hidden_dim, z_dim, gmm_decoder=gmm_bool)

    return cvae

def configure_gmm_cvae(conf, num_heads, multi_head):
    from models.cvae.model_mixture import CVAEContinuous
    encoder_hidden_dim = conf["encoder_hidden_dim"]
    future_hidden_dim = conf["future_hidden_dim"]
    x_dim = conf["input_dim"]
    # decoder_input_dim = z_dim * 2
    # decoder = configure_decoder(conf, decoder_input_dim, num_heads, multi_head)
    cvae = CVAEContinuous(x_dim, encoder_hidden_dim, future_hidden_dim)

    return cvae

def configure_gmm_cvae_z(conf, num_heads, multi_head):
    from models.cvae.model_mixture import CVAEMixture
    decoder_type = conf["decoder_type"]
    z_dim = conf["z_dim"]
    encoder_hidden_dim = conf["encoder_hidden_dim"]
    future_hidden_dim = conf["future_hidden_dim"]
    z_dim = conf["z_dim"]
    x_dim = conf["input_dim"]
    # decoder_input_dim = z_dim * 2
    # decoder = configure_decoder(conf, decoder_input_dim, num_heads, multi_head)
    cvae = CVAEMixture(x_dim, encoder_hidden_dim, future_hidden_dim, z_dim, num_heads, decoder_type=decoder_type)

    return cvae

def configure_gmm_gnn_cvae(conf, num_heads, multi_head, total_input_dim):
    from models.cvae.model_continuous import CVAEContinuous
    encoder_hidden_dim = conf["encoder_hidden_dim"]
    future_hidden_dim = conf["future_hidden_dim"]
    z_dim = conf["z_dim"]
    gmm_decoder = conf["gmm_bool"]

    model = CVAEContinuous(total_input_dim, encoder_hidden_dim, future_hidden_dim, z_dim, gmm_decoder=gmm_decoder, input_type = "gnn")
    return model

def get_input_dim(config):
    total_input_dim = 3
    if config["datasets"]["one_hot_agents"]:
        total_input_dim += 3
    if config["datasets"]["detected_location"]:
        total_input_dim += 2
    if config["datasets"]["timestep"]:
        total_input_dim += 1
    return total_input_dim

def configure_model(config):
    conf = config["model"]
    
    num_heads = config["datasets"]["num_heads"]
    if config["datasets"]["include_current"]:
        num_heads += 1

    multi_head = config["datasets"]["multi_head"]

    if conf["model_type"] == "connected":
        print("connected model")
        return configure_connected_model(conf, num_heads)
    elif conf["model_type"] == "gnn":
        total_input_dim = get_input_dim(config)
        if config["datasets"]["get_start_location"]:
            concat_dim = 5
        else:
            concat_dim = 3
        return configure_gnn_model(conf, num_heads, total_input_dim, multi_head, concat_dim)
    elif conf["model_type"] == "hetero_gnn_lstm_front":
        total_input_dim = get_input_dim(config)

        start_location_bool = config["datasets"]["get_start_location"]
        return configure_hetero_gnn_lstm_front_model(conf, num_heads, total_input_dim, multi_head, start_location_bool)
    elif conf["model_type"] == "stn":
        total_input_dim = get_input_dim(config)
        return configure_stn(conf, num_heads, total_input_dim, multi_head)
    elif conf["model_type"] == "temp_gcn":
        total_input_dim = get_input_dim(config)
        return configure_temporal_gcn(conf, num_heads, total_input_dim, multi_head)
    elif conf["model_type"] == "hetero_lstm":
        total_input_dim = get_input_dim(config)
        return configure_hetero_lstm(conf, num_heads, total_input_dim, multi_head)
    elif conf["model_type"] == "cvae":
        return configure_cvae(conf, num_heads, multi_head)
    elif conf["model_type"] == "gmm_cvae":
        return configure_gmm_cvae(conf, num_heads, multi_head)
    elif conf["model_type"] == "gmm_cvae_z":
        return configure_gmm_cvae_z(conf, num_heads, multi_head)
    elif conf["model_type"] == "gmm_cvae_gnn":
        total_input_dim = get_input_dim(config)
        return configure_gmm_gnn_cvae(conf, num_heads, multi_head, total_input_dim)
    elif conf["model_type"] == "hybrid_gnn":
        total_input_dim = get_input_dim(config)
        return configure_hybrid_gnn(conf, num_heads, total_input_dim, multi_head)
    else:
        return configure_regular(conf, num_heads, multi_head)