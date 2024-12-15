import argparse
import json
from itertools import product


def args_generator(args_grid):
    for arg_values in product(*args_grid.values()):
        yield f"{json.dumps({key: val for key, val in zip(args_grid.keys(), arg_values)})}"


def parse_args():

    # # resnet
    # net_args_grid = {
    #     "ch_list": [
    #         [64, 64, 128, 128],
    #     ],
    #     "dropout": [0.0, 0.25, 0.5],
    #     "numclass": [1],
    # }

    # gnnstack
    net_args_grid1 = {
        "gnn_model_type": ['dyGIN2d'],
        "num_layers": [3],
        "groups": [6],
        "pool_ratio": [0.2],
        "kern_size": [[9, 7, 5]],
        "in_dim": [64],
        "hidden_dim": [128],
        "out_dim": [256],
        "seq_len": [24],
        "num_nodes": [64],
        "num_classes": [1],
    }
    net_args_grid2 = {
        "gnn_model_type": ['dyGIN2d'],
        "num_layers": [3],
        "groups": [2],
        "pool_ratio": [0.2],
        "kern_size": [[9, 7, 5]],
        "in_dim": [64],
        "hidden_dim": [128],
        "out_dim": [256],
        "seq_len": [24],
        "num_nodes": [64],
        "num_classes": [1],
    }
    net_args_grid3 = {
        "gnn_model_type": ['dyGIN2d'],
        "num_layers": [3],
        "groups": [1],
        "pool_ratio": [0.2],
        "kern_size": [[9, 7, 5]],
        "in_dim": [64],
        "hidden_dim": [128],
        "out_dim": [256],
        "seq_len": [24],
        "num_nodes": [64],
        "num_classes": [1],
    }

    # # RNN
    # net_args_grid = {
    #     "c_emb": [32],
    #     "c_out": [1],
    #     "hidden_size": [64],
    #     "n_layers": [1, 2, 3],
    #     "bias": [True],
    #     "rnn_dropout": [0.0, 0.25, 0.5],
    #     "bidirectional": [False],
    #     "fc_dropout": [0.0, 0.25, 0.5],
    #     "init_weights": [True]
    # }

    # # transformer
    # net_args_grid = {
    #     "c_emb": [32],
    #     "c_out": [1],
    #     "d_model": [128],
    #     "n_head": [4],
    #     "d_ffn": [128],
    #     "dropout": [0.25, 0.5],
    #     "activation": ["gelu"],
    #     "n_layers": [3, 4, 5],
    # }
    
    # # resnetpluswwwww
    # net_args_grid = {
    #     "c_in": [32],
    #     "c_out": [1],
    #     "seq_len": [None],
    #     "nf": [64],
    #     "sa": [False, True],
    #     "se": [None],
    #     "fc_dropout": [0.0, 0.25, 0.5],
    #     "concat_pool": [False, True],
    #     "flatten": [False],
    #     "custom_head": [None],
    #     "y_range": [None],
    #     "ks": [[7, 5, 3]],
    #     "coord": [False, True],
    #     "separable": [False, True],
    #     "bn_1st": [True],
    #     "zero_norm": [False, True],
    #     "act": ["relu"],
    # }
    
    # # inceptiontimeplus
    # net_args_grid = {
    #     "c_in": [32],
    #     "c_out": [1],
    #     "seq_len": [None],
    #     "nf": [32, 64, 128],
    #     "nb_filters": [None],
    #     "flatten": [False],
    #     "concat_pool": [False, True],
    #     "fc_dropout": [0.0, 0.25, 0.5],
    #     "bn": [False, True],
    #     "y_range": [None],
    #     "custom_head": [None],
    #     "ks": [40],
    #     "bottleneck": [True, False],
    #     "padding": ['same'],
    #     "coord": [False, True],
    #     "separable": [False, True],
    #     "dilation": [1, 2],
    #     "stride": [1],
    #     "conv_dropout": [0.0],
    #     "sa": [False, True],
    #     "se": [None],
    #     "norm": ['Batch'],
    #     "zero_norm": [False, True],
    #     "bn_1st": [True, False],
    #     "act": ["relu"],
    # }
    
    # # lstmattention
    # net_args_grid = {
    #     "c_in": [32],
    #     "c_out": [1],
    #     "seq_len": [24],
    #     "hidden_size": [64],
    #     "rnn_layers": [1, 2, 3],
    #     "bias": [True],
    #     "rnn_dropout": [0.0, 0.25, 0.5],
    #     "bidirectional": [False],
    #     "encoder_layers": [3, 6],
    #     "n_heads": [16, 32],
    #     "d_k": [None, 32],
    #     "d_v": [None, 32],
    #     "d_ff": [256],
    #     "encoder_dropout": [0.1, 0.2, 0.3],
    #     "act": ['gelu'],
    #     "fc_dropout": [0.0, 0.25, 0.5],
    #     "y_range": [None],
    #     "verbose": [False],
    #     "custom_head": [None],
    # }
    
    # # mlp
    # net_args_grid = {
    #     "c_emb": [32],
    #     "c_out": [1],
    #     "seq_len": [24],
    #     "layers": [[500, 500, 500]],
    #     "ps": [[0.1, 0.2, 0.2]],
    #     "act": ["relu"],
    #     "use_bn": [False],
    #     "bn_final": [False],
    #     "lin_first": [False],
    #     "fc_dropout": [0.0],
    #     "y_range": [None],
    # }
    
    # # gmlp
    # net_args_grid = {
    #     "c_emb": [32],
    #     "c_out": [1],
    #     "seq_len": [24],
    #     "patch_size": [1],
    #     "d_model": [256],
    #     "d_ffn": [512],
    #     "depth": [6],
    # }
    
    opt_args_grid = {
        "lr": [1e-3],
        "weight_decay": [1e-5],
    }
    loss_args_grid = {"pos_weight": [[9.0]]}

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mimic4", help="Config dataset")
    parser.add_argument(
        "--data",
        type=str,
        default="DPGap4SPGap48Len24Str12Art2Grp8_hf",
        help="Config data key",
    )
    parser.add_argument("--model", type=str, default="gnnstack", help="Config model key")
    parser.add_argument("--net_args", type=str, help="Net arguments")
    parser.add_argument("--opt_args", type=str, help="Optimizer arguments")
    parser.add_argument("--loss_args", type=str, help="LossFn arguments")
    parser.add_argument("--n_fold", type=int, default=5, help="Number of folds")
    args = parser.parse_args()
    model_args = []
    for (net_args_str1, net_args_str2, net_args_str3, opt_args_str), loss_args_str in product(
        product(
            args_generator(net_args_grid1),
            args_generator(net_args_grid2),
            args_generator(net_args_grid3),
            args_generator(opt_args_grid),
        ),
        args_generator(loss_args_grid),
    ):
    
        model_args1 = {
            "net": json.loads(net_args_str1),
            # "net": json.loads('\'{"ch_list": [64, 64, 128, 128], "dropout": 0.5, "numclass": 1}\''),
            "opt": json.loads(opt_args_str),
            "loss": json.loads(loss_args_str),
        }
        model_args2 = {
            "net": json.loads(net_args_str2),
            # "net": json.loads('\'{"ch_list": [64, 64, 128, 128], "dropout": 0.5, "numclass": 1}\''),
            "opt": json.loads(opt_args_str),
            "loss": json.loads(loss_args_str),
        }
        model_args3 = {
            "net": json.loads(net_args_str3),
            # "net": json.loads('\'{"ch_list": [64, 64, 128, 128], "dropout": 0.5, "numclass": 1}\''),
            "opt": json.loads(opt_args_str),
            "loss": json.loads(loss_args_str),
        }
        model_args.append(model_args1)
        model_args.append(model_args2)
        model_args.append(model_args3)

    return args, model_args