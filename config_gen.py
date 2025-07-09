import yaml
import itertools

# Example of a generated config file for fine-tuning
"""
Modules: ['lora','bitfit','compacter','bn','pftuning']

LoRAConfig:
  selfattn_lora: True
  attn_matrices: ['k','v']
  intermediate_lora: True
  output_lora: True
  r: 8
  alpha: 8
  composition_mode: 'add' # 'add' or 'scale'

BnConfig:
  output_adapter: False
  mh_adapter: True
  non_linearity: 'gelu'
  reduction_factor: 8

PrefixTuningConfig:
  prefix_length: 20
"""


def generate_config():
    config = {
        "Modules": ["lora", "bitfit", "compacter", "bn", "pftuning"],
        "LoRAConfig": {
            "selfattn_lora": True,
            "attn_matrices": ["k", "v"],
            "intermediate_lora": True,
            "output_lora": True,
            "r": 8,
            "alpha": 8,
            "composition_mode": "add",  # 'add' or 'scale'
        },
        "BnConfig": {
            "output_adapter": False,
            "mh_adapter": True,
            "non_linearity": "gelu",
            "reduction_factor": 8,
        },
        "PrefixTuningConfig": {"prefix_length": 20},
    }

    with open("config.yaml", "w") as file:
        yaml.dump(config, file)


def generate_lora_configs(
    r_values, attn_lora=True, intermediate_lora=True, output_lora=True, attn_spec=None
):
    attn_matrices = list(attn_spec.strip())
    # config["LoRAConfig"]["attn_matrices"] = [matrix.strip() for matrix in attn_matrices]
    config = {
        "Modules": ["lora"],
        "LoRAConfig": {
            "selfattn_lora": True if attn_lora else False,
            # "attn_matrices": ["k", "v", "q"] if attn_lora else [],
            "attn_matrices": attn_matrices,
            "intermediate_lora": True if intermediate_lora else False,
            "output_lora": True if output_lora else False,
            "r": int(r_values),
            "composition_mode": "add",
        },
    }
    return config


def generate_bn_configs(
    reduction_factor_values, reduction_factor_float, mh_bn, output_bn
):
    config = {
        "Modules": ["bn"],
        "BnConfig": {
            "output_adapter": True if output_bn else False,
            "mh_adapter": False if mh_bn else False,
            "non_linearity": "gelu",
            "reduction_factor": (
                int(reduction_factor_values) / 10
                if reduction_factor_float
                else int(reduction_factor_values)
            ),
        },
    }
    return config


def generate_permutations(n_modules=2):
    modules = ["lora", "bitfit", "compacter", "bn", "pftuning"]
    values = [1, 4, 8, 16]

    permutations = []
    for module1, module2 in itertools.combinations(modules, 2):
        for reduction_factor in values:
            for r in values:
                for alpha in values:
                    config = {
                        "Modules": [module1, module2],
                        "LoRAConfig": {
                            "selfattn_lora": True,
                            "attn_matrices": ["k", "v"],
                            "intermediate_lora": True,
                            "output_lora": True,
                            "r": r,
                            "alpha": alpha,
                            "composition_mode": "add",
                        },
                        "BnConfig": {
                            "output_adapter": False,
                            "mh_adapter": True,
                            "non_linearity": "gelu",
                            "reduction_factor": reduction_factor,
                        },
                        "PrefixTuningConfig": {"prefix_length": 20},
                    }
                    permutations.append(config)

    with open("config.yaml", "w") as file:
        yaml.dump_all(permutations, file)


if __name__ == "__main__":
    generate_config()
