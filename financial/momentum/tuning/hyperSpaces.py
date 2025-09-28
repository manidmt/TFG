'''
Spaces for hyperparameter tuning of different neural network architectures (CNN, LSTM, Transformer) using Keras.

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''

from financial.lab.tuning.space import HyperparameterSearchSpace

def build_cnn_space():
    return HyperparameterSearchSpace(
        name="keras_cnn_sp500",
        hyperparameters={
            "arch": {
                "cnn": { "model": { "architecture": "cnn" } }
            },

            "topology": {
                "t64x32":     { "model": { "topology": {"layers":[64,32],     "activation":{"hidden":"relu","output":"linear"}}}},
                "t128x64":    { "model": { "topology": {"layers":[128,64],   "activation":{"hidden":"relu","output":"linear"}}}},
                "t64x64x32":  { "model": { "topology": {"layers":[64,64,32], "activation":{"hidden":"relu","output":"linear"}}}},  
            },

            "opt": {
                "adam_huber_e120_b32": { "model": {
                    "optimization": {
                        "optimizer":"adam","loss":"huber","epochs":120,"batch_size":32,"validation_split":0.1,
                        "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                    }
                }},
                "adam_huber_e80_b32": { "model": {   
                    "optimization": {
                        "optimizer":"adam","loss":"huber","epochs":80,"batch_size":32,"validation_split":0.1,
                        "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                    }
                }},
                "adam_mse_e120_b32": { "model": {    
                    "optimization": {
                        "optimizer":"adam","loss":"mean_squared_error","epochs":120,"batch_size":32,"validation_split":0.1,
                        "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                    }
                }},
                "adam_huber_e120_b16": { "model": {  
                    "optimization": {
                        "optimizer":"adam","loss":"huber","epochs":120,"batch_size":16,"validation_split":0.1,
                        "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                    }
                }},
            },

            "cnn_cfg": {
                "lite": { "model": {
                    "n_blocks":2, "filters":[64,64],  "kernel_sizes":[7,3], "padding":"same",
                    "pool_every":1, "pool_size":2, "dropout":0.2, "l2":1e-6, "batch_norm": True, "global_pool": True
                }},
                "deep": { "model": {
                    "n_blocks":3, "filters":[64,64,64], "kernel_sizes":[5,3,3], "padding":"same",
                    "pool_every":1, "pool_size":2, "dropout":0.3, "l2":1e-6, "batch_norm": True, "global_pool": True
                }},
                "dilated": { "model": {
                    "n_blocks":3, "filters":[64,64,64], "kernel_sizes":[3,3,3], "dilations":[1,2,4], "padding":"causal",
                    "pool_every":0, "dropout":0.2, "l2":1e-6, "batch_norm": True, "global_pool": True
                }},
                "wide": { "model": {  
                    "n_blocks":2, "filters":[64,64], "kernel_sizes":[15,7], "padding":"same",
                    "pool_every":1, "pool_size":2, "dropout":0.25, "l2":1e-6, "batch_norm": True, "global_pool": True
                }},
            },
        }
    )


def build_lstm_space():
    return HyperparameterSearchSpace(
        name="keras_lstm_sp500",
        hyperparameters={
            "arch": {
                "lstm": { "model": { "architecture": "lstm" } }
            },
            
            "topology": {
                "t64":         { "model": { "topology": {"layers":[64],        "activation":{"hidden":"relu","output":"linear"}}}},
                "t64x32":      { "model": { "topology": {"layers":[64,32],     "activation":{"hidden":"relu","output":"linear"}}}},
                "t128x64":     { "model": { "topology": {"layers":[128,64],    "activation":{"hidden":"relu","output":"linear"}}}},
            },

            "opt": {
                "adam_huber_e120_b32": { "model": { "optimization": {
                    "optimizer":"adam","loss":"huber","epochs":120,"batch_size":32,"validation_split":0.1,
                    "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                }}},
                "adam_huber_e120_b16": { "model": { "optimization": {
                    "optimizer":"adam","loss":"huber","epochs":120,"batch_size":16,"validation_split":0.1,
                    "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                }}},
            },
            
            "drop": {            
                "do00": { "model": {"dropout":0.0}},
                "do01": { "model": {"dropout":0.1}},
                "do02": { "model": {"dropout":0.2}},
            },
            "rec_drop": {       
                "rdo00": { "model": {"recurrent_dropout":0.0}},
                "rdo01": { "model": {"recurrent_dropout":0.1}},
            },
            "bn": {              
                "bn0":  { "model": {"batch_norm": False}},
                "bn1":  { "model": {"batch_norm": True}},
            },
            "bidir": {           
                "bi0":  { "model": {"bidirectional": False}},
                "bi1":  { "model": {"bidirectional": True}},
            },
            "layer_do": {        
                "ld00": { "model": {"layer_dropout": 0.0}},
                "ld02": { "model": {"layer_dropout": 0.2}},
            },
            "head": {            
                "h0":   { "model": {"dense_head": []}},
                "h32":  { "model": {"dense_head": [32]}},
                "h64_32": { "model": {"dense_head": [64, 32]}},
            },
        }
    )


def build_transformer_space():
    return HyperparameterSearchSpace(
        name="keras_transformer_sp500",
        hyperparameters={
            "arch": {
                "xfmr": { "model": { "architecture": "transformer" } }
            },

            "xfmr_cfg": {
                "h2_f64_d01":   { "model": {"num_heads":2, "ff_dim":64,   "dropout":0.1}},
                "h2_f64_d02":   { "model": {"num_heads":2, "ff_dim":64,   "dropout":0.2}},
                "h2_f128_d01":  { "model": {"num_heads":2, "ff_dim":128,  "dropout":0.1}},
                "h2_f128_d02":  { "model": {"num_heads":2, "ff_dim":128,  "dropout":0.2}},
                "h4_f64_d01":   { "model": {"num_heads":4, "ff_dim":64,   "dropout":0.1}},
                "h4_f64_d02":   { "model": {"num_heads":4, "ff_dim":64,   "dropout":0.2}},
                "h4_f128_d01":  { "model": {"num_heads":4, "ff_dim":128,  "dropout":0.1}},
                "h4_f128_d02":  { "model": {"num_heads":4, "ff_dim":128,  "dropout":0.2}},
                "h8_f128_d01":  { "model": {"num_heads":8, "ff_dim":128,  "dropout":0.1}},
                "h8_f128_d02":  { "model": {"num_heads":8, "ff_dim":128,  "dropout":0.2}},
                "h8_f256_d02":  { "model": {"num_heads":8, "ff_dim":256,  "dropout":0.2}},
                "h8_f256_d03":  { "model": {"num_heads":8, "ff_dim":256,  "dropout":0.3}},
            },

            "opt": {
                "adam_huber_e80_b32":   { "model": { "optimization": {
                    "optimizer":"adam","loss":"huber","epochs":80, "batch_size":32,"validation_split":0.1,
                    "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                }}},
                "adam_huber_e120_b32":  { "model": { "optimization": {
                    "optimizer":"adam","loss":"huber","epochs":120,"batch_size":32,"validation_split":0.1,
                    "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                }}},
                "adam_huber_e120_b16":  { "model": { "optimization": {
                    "optimizer":"adam","loss":"huber","epochs":120,"batch_size":16,"validation_split":0.1,
                    "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                }}},
                "adam_mse_e120_b32":    { "model": { "optimization": {
                    "optimizer":"adam","loss":"mean_squared_error","epochs":120,"batch_size":32,"validation_split":0.1,
                    "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                }}},
            },

            "topology": {
                "t64x32": { "model": {
                    "topology": {"layers":[64,32], "activation":{"hidden":"relu","output":"linear"}}
                }},
            },
        }
    )
