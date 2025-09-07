'''

@author: Manuel Díaz-Meco (manidmt5@gmail.com)
'''


# hp_space_cnn.py
from financial.lab.tuning.space import HyperparameterSearchSpace

# hp_space_cnn_v2.py
from financial.lab.tuning.space import HyperparameterSearchSpace

def build_cnn_space():
    return HyperparameterSearchSpace(
        name="keras_cnn_sp500",
        hyperparameters={
            # Arquitectura (fija: CNN)
            "arch": {
                "cnn": { "model": { "architecture": "cnn" } }
            },

            # Topología de las capas densas de salida
            "topology": {
                "t64x32":     { "model": { "topology": {"layers":[64,32],     "activation":{"hidden":"relu","output":"linear"}}}},
                "t128x64":    { "model": { "topology": {"layers":[128,64],   "activation":{"hidden":"relu","output":"linear"}}}},
                "t64x64x32":  { "model": { "topology": {"layers":[64,64,32], "activation":{"hidden":"relu","output":"linear"}}}},  # NUEVA
            },

            # Optimización (épocas, batch, loss, callbacks)
            "opt": {
                "adam_huber_e120_b32": { "model": {
                    "optimization": {
                        "optimizer":"adam","loss":"huber","epochs":120,"batch_size":32,"validation_split":0.1,
                        "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                    }
                }},
                "adam_huber_e80_b32": { "model": {   # NUEVA (más rápida; ES corta antes si va bien)
                    "optimization": {
                        "optimizer":"adam","loss":"huber","epochs":80,"batch_size":32,"validation_split":0.1,
                        "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                    }
                }},
                "adam_mse_e120_b32": { "model": {    # NUEVA (MSE en lugar de Huber)
                    "optimization": {
                        "optimizer":"adam","loss":"mean_squared_error","epochs":120,"batch_size":32,"validation_split":0.1,
                        "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                    }
                }},
                "adam_huber_e120_b16": { "model": {  # NUEVA (batch más pequeño)
                    "optimization": {
                        "optimizer":"adam","loss":"huber","epochs":120,"batch_size":16,"validation_split":0.1,
                        "callbacks":{"early_stopping":{"patience":8},"reduce_on_plateau":{"patience":4,"factor":0.5}}
                    }
                }},
            },

            # Bloques convolucionales
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
                "wide": { "model": {  # NUEVA (captura patrones más lentos)
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
                "t64":     { "model": { "topology": {"layers":[64],      "activation":{"hidden":"relu","output":"linear"}}}},
                "t64x32":  { "model": { "topology": {"layers":[64,32],   "activation":{"hidden":"relu","output":"linear"}}}},
                "t128x64": { "model": { "topology": {"layers":[128,64],  "activation":{"hidden":"relu","output":"linear"}}}},
            },
            "opt": {
                "adam_huber_e120_b32": { "model": {
                    "optimization": {
                        "optimizer":"adam","loss":"huber","epochs":120,"batch_size":32,"validation_split":0.1,
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
            "reg": {
                "do01": { "model": {"dropout":0.1}},
                "do02": { "model": {"dropout":0.2}},
            },
        }
    )



# hp_space_transformer.py
from financial.lab.tuning.space import HyperparameterSearchSpace

def build_transformer_space():
    return HyperparameterSearchSpace(
        name="keras_transformer_sp500",
        hyperparameters={
            # Arquitectura fija: transformer
            "arch": {
                "xfmr": { "model": { "architecture": "transformer" } }
            },

            # Configs del bloque de atención/FFN (todas usadas por tu initialize_model actual)
            # num_heads, ff_dim y dropout
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

            # Optimización (variantes razonables con tu pipeline actual)
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

            # Mantengo la misma “topology” por compatibilidad (tu transformer usa sólo activation_output)
            "topology": {
                "t64x32": { "model": {
                    "topology": {"layers":[64,32], "activation":{"hidden":"relu","output":"linear"}}
                }},
            },
        }
    )
