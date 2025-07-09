import argparse
import yaml
import os
from datasets import load_dataset
from huggingface_hub import login

from build import build_model
from finetune import train_model
from config_gen import generate_lora_configs, generate_bn_configs


def get_args(parser):
    parser.add_argument(
        "--model",
        type=str,
        default="google/vit-base-patch16-224-in21k",
        help="name/path for source model",
    )
    parser.add_argument(
        "--config",
        default=None,  # for local testing
        type=str,
        help="name of the config yaml file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset used for training and evaluation",
    )
    parser.add_argument(
        "--domainnet_train_domain",
        type=str,
        default=None,
        help="Specify one of the six domains of DomainNet dataset for training",
    )
    parser.add_argument(
        "--ood_filter_test_dataset",
        type=int,
        default=0,
        help="Whether filter out ood domain from the test dataset (0-No, for dynamic training logging, we want both clean acc and ood attack during trainnig)",
    )
    parser.add_argument(
        "--ood_filter_train_dataset",
        type=int,
        default=0,
        help="Whether filter out ood domain from the training dataset (0-No, for dynamic training logging, we want do the filtering later for memory efficiency)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--epoch", type=int, default=3, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate for training"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay for training"
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="pgd",
        help="Attack name: pgd, corruption, out-of-distribution, etc.",
    )
    parser.add_argument(
        "--corruption_type",
        type=str,
        default=None,
        help="Corruption type for corruption attack",
    )
    parser.add_argument(
        "--alpha", type=float, default=1 / 1020, help="alpha (step size) for attack"
    )
    parser.add_argument(
        "--steps", type=float, default=50, help="number of steps for attack"
    )
    parser.add_argument(
        "--eps",
        type=int,
        default=1 / 255,
        help="norm bound / budget for attack",
    )
    # this is only for hyperparameter sweep (lora, bn) and baseline (scratch, finetune, lp)
    parser.add_argument(
        "--adapter_name",
        type=str,
        default=None,
        help="name of the adapter for hyperparameter sweep or baseline",
    )
    # arguments for LoRA adapter
    parser.add_argument(
        "--r",
        type=int,
        default=1,
        help="r value of lora adapter",
    )
    parser.add_argument(
        "--attn_spec",
        type=str,
        default="kqv",
        help="specify which attention matrices to apply lora on",
    )
    parser.add_argument(
        "--attn_lora",
        type=int,
        default=1,
        help="whether apply lora on attention layers",
    )
    parser.add_argument(
        "--intermediate_lora",
        type=int,
        default=1,
        help="whether apply lora on intermediate layers",
    )
    parser.add_argument(
        "--output_lora",
        type=int,
        default=1,
        help="whether apply lora on output layers",
    )
    parser.add_argument(
        "--reduction_factor",  # for bottleneck, reduction_factor_values=[4:32]
        type=int,
        default=0,
        help="reduction factor of bottleneck adapter",
    )
    parser.add_argument(
        "--reduction_factor_float",  # set reduction factors into decimals
        type=int,
        default=0,  # 0: reduction factors are int, 1: decimals
        help="whether reduction factor is between 0 and 1",
    )
    parser.add_argument(
        "--mh_bn",
        type=int,
        default=1,
        help="whether apply bn after mh layer",
    )
    parser.add_argument(
        "--output_bn",
        type=int,
        default=0,
        help="whether apply bn after output layer",
    )
    parser.add_argument(
        "--rerun_id",
        type=int,
        default=None,
        help="rerun epxeriment id",
    )

    return parser


def main(args):
    """
    1) Build, fine-tune + attack, and save with one fine-tuning method
    2) If configSweep is True, then sweep through hyperparameters r (lora) and
       reduction factor k (adapter bottleneck)
    Note: since we use AdapterHub library, we use the term "adapter" to refer to
    different fine-tuning methods/modules, such as LoRA, bottleneck, etc.
    However, in the paper, "adapter" and "bottleneck" refer to the same
    fine-tuning method.
    """
    if args.adapter_name != None and args.adapter_name != "None":
        # generate config file for the sweep on the fly
        if args.adapter_name == "lora":
            adapter_config = generate_lora_configs(
                args.r,
                args.attn_lora,
                args.intermediate_lora,
                args.output_lora,
                args.attn_spec,
            )
            str1 = "only_attn"
            str2 = "no_attn"
            str3 = "spec"
            str4 = ""
            adapter_config_name = f"lora_{str1 if args.attn_lora else str2}_{str3 if args.attn_spec else str4}_{args.r}"  # unique identifier for the module
        elif args.adapter_name == "bottleneck":
            adapter_config = generate_bn_configs(
                args.reduction_factor,
                args.reduction_factor_float,
                args.mh_bn,
                args.output_bn,
            )
            str1 = "_1"
            str2 = ""
            adapter_config_name = (
                f"bottleneck_mh_{args.reduction_factor}{str1 if args.reduction_factor_float else str2}"
                if args.mh_bn
                else f"bottleneck_{args.reduction_factor}{str1 if args.reduction_factor_float else str2}"
            )
        # baseline: finetune, lp
        elif args.adapter_name == "finetune":
            adapter_config = {"Modules": []}
            adapter_config_name = "finetune"
        elif args.adapter_name == "lp":
            adapter_config = {"Modules": []}
            adapter_config_name = "lp"

    else:
        # read config from the yaml file in current directory
        with open("configs/" + args.config + ".yml", "r") as file:
            adapter_config = yaml.load(file, Loader=yaml.FullLoader)
            adapter_config_name = args.config

    # Load the dataset
    if args.dataset == "food101":
        # different data splits for food101
        train_ds, test_ds = load_dataset(args.dataset, split=["train", "validation"])
    elif args.dataset == "cifar10":
        train_ds, test_ds = load_dataset(args.dataset, split=["train", "test"])
    elif args.dataset == "cifar100":
        train_ds, test_ds = load_dataset("uoft-cs/cifar100", split=["train", "test"])
    elif args.dataset == "domainnet":
        train_ds, test_ds = load_dataset("wltjr1007/DomainNet", split=["train", "test"])

        if args.domainnet_train_domain:
            # if a domain is specified for DomainNet dataset, we only train /
            # fine-tune the model on one domain
            domain_names = {
                "clipart": 0,
                "infograph": 1,
                "painting": 2,
                "quickdraw": 3,
                "real": 4,
                "sketch": 5,
            }

            # define the filter to take into a domain name
            def filter_domain(example, domain_name):
                domain_index = domain_names[domain_name]
                return example["domain"] == domain_index

            # filter the training dataset for a specific domain
            def get_filtered_dataset(dataset, domain_name):
                return dataset.filter(
                    lambda example: filter_domain(example, domain_name)
                )

            if args.ood_filter_train_dataset:
                # Similar reason as below. Filter the data later (with
                # implemented memory-efficient dataloader) for dynamic training
                train_ds = get_filtered_dataset(train_ds, args.domainnet_train_domain)
            if args.ood_filter_test_dataset:
                # Only filter out the ood domain from the test dataset when we
                # are not doing dynamic logging during training. Otherwise, we
                # need both in-domain test data (to get clean acc) and ood test
                # data (to get ood attack acc) during training
                test_ds = get_filtered_dataset(
                    test_ds, args.domainnet_train_domain
                )  # this is not ood attack for domainnet, on the same domain as training

    elif args.dataset == "caltech256":
        train_ds, val_ds, test_ds = load_dataset(
            "ilee0022/Caltech-256", split=["train", "validation", "test"]
        )
    elif args.dataset == "cub200":
        train_ds, test_ds = load_dataset(
            "efekankavalci/CUB_200_2011", split=["train", "test"]
        )
    elif args.dataset == "stanforddogs":
        train_ds, test_ds = load_dataset(
            "amaye15/stanford-dogs", split=["train", "test"]
        )
    elif args.dataset == "places365":
        train_ds = load_dataset("ljnlonoljpiljm/places365-256px", split="train")
        test_ds = load_dataset("dpdl-benchmark/Places365-Validation", split="train")
        test_ds = test_ds.shuffle(seed=42).select(
            range(1000)
        )  # select 1000 samples for evaluation

    # Load the processor and model
    processor, model = build_model(
        args.model, args.dataset, adapter_config_name, adapter_config, train_ds
    )
    print("Model built successfully")

    # Fine-tune the model
    model = train_model(
        processor,
        model,
        args.dataset,
        args.domainnet_train_domain,
        adapter_config_name,
        adapter_config,
        train_ds,
        batch_size=args.batch_size,
        epoch=args.epoch,
        rerun_id=args.rerun_id,
        test_ds=test_ds,
        attack_type=args.attack,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print("Model trained successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = get_args(parser)
    args = parser.parse_args()

    main(args)
