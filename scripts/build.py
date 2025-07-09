from transformers import AutoImageProcessor, ViTConfig
from adapters import (
    ViTAdapterModel,
    BnConfig,
    LoRAConfig,
    IA3Config,
    PrefixTuningConfig,
    CompacterConfig,
    ConfigUnion,
)
import torch
import os
import utils


def build_model(model, dataset, adapter_config_name, adapter_config, train_ds):
    """
    Build a model with fine-tuning modules based on the provided configuration.

    Args:
        model (str): The name or path of the pre-trained model.
        adapter_config (dict): The configuration for the adapters.
        adapter_config_name (str): The name of the adapter configuration.
        train_ds (Dataset): The training dataset.

    Returns:
        processor (AutoImageProcessor): The loaded image processor.
        model (ViTAdapterModel): The built model with adapters.
    """
    # get training, validation, and test data
    if (
        dataset == "cifar10"
        or dataset == "domainnet"
        or dataset == "cub200"
        or dataset == "stanforddogs"
        or dataset == "places365"
    ):
        fetched_labels = train_ds.features["label"].names
        id2label = {id: label for id, label in enumerate(fetched_labels)}
        label2id = {label: id for id, label in id2label.items()}
    elif dataset == "cifar100":
        fetched_labels = train_ds.features["fine_label"].names
        id2label = {id: label for id, label in enumerate(fetched_labels)}
        label2id = {label: id for id, label in id2label.items()}
    elif dataset == "caltech256":
        # Since the dataset is special with label (numerical) and text (class
        # names), labels are from 1-257, instead of 0-256 and the order if
        # specified internally. We create the dictionary accordingly
        ids = train_ds["label"]
        labels = train_ds["text"]
        id2label = dict(map(lambda i, j: (i, j), ids, labels))
        id2label[0] = "dummy_label"
        label2id = {label: id for id, label in id2label.items()}

    # Load the processor
    processor = AutoImageProcessor.from_pretrained(model)

    # Load the model
    if dataset == "caltech256":
        # add 1 to the number of labels, 0-157 instead of 0-156
        # this is because the labels from the dataset starts from 1 instead of 0
        # if we use 0-256, then there is label from the dataset which equals to
        # 257, and it will be out of bound and will cause error during
        # training/backpropagation
        vit_config = ViTConfig.from_pretrained(model, num_labels=258)
    else:
        vit_config = ViTConfig.from_pretrained(model, num_labels=len(id2label))
    vit_config.label2id = label2id
    vit_config.id2label = id2label

    # if the model is pretrained on ImageNet21k without the head, we need to
    # change the architecture to ViTForImageClassification, which places a
    # linear layer on top of a pretrained ViTModel.
    if model == "google/vit-base-patch16-224-in21k":
        vit_config.architectures = ["ViTForImageClassification"]

    # for local testing
    if dataset == "caltech256":
        vit_config.num_labels = 258

    model = ViTAdapterModel.from_pretrained(
        model, config=vit_config, ignore_mismatched_sizes=True
    )

    # Baseline: finetune, lp
    # for finetune: no adapters, all parameters are trainable, no changes needed
    # for lp: no adapters, only head is trainable, set all other parameters to
    # not trainable
    if adapter_config_name == "lp":
        for n, param in model.named_parameters():
            if "head" not in n:
                param.requires_grad = False
        return processor, model

    elif adapter_config_name == "finetune":
        for n, param in model.named_parameters():
            param.requires_grad = True
        return processor, model

    adapter_list = []
    for adapter in adapter_config["Modules"]:
        # Add the adapter to the model
        if adapter == "lora":
            adapter_list.append(LoRAConfig(**adapter_config["LoRAConfig"]))

        elif adapter == "ia3":
            adapter_list.append(IA3Config(**adapter_config["LoRAConfig"]))

        elif adapter == "lokr":
            adapter_list.append(LoRAConfig(**adapter_config["LoRAConfig"]))

        elif adapter == "bn":
            adapter_list.append(BnConfig(**adapter_config["BnConfig"]))

        elif adapter == "compacter":
            adapter_list.append(CompacterConfig())

        elif adapter == "pftuning":
            adapter_list.append(
                PrefixTuningConfig(**adapter_config["PrefixTuningConfig"])
            )

    if adapter_list != []:
        cfg = ConfigUnion(*adapter_list)
        model.add_adapter("{}".format(adapter_config_name), config=cfg)
        model.train_adapter("{}".format(adapter_config_name))

    # Bitfit, adjust the parameter grad after activating adapters
    if "bitfit" in adapter_config["Modules"]:
        for n, param in model.named_parameters():
            if adapter_list != [] and "bias" in n or "head" in n:
                param.requires_grad = True
            if adapter_list == [] and "bias" not in n and "head" not in n:
                param.requires_grad = False

    return processor, model


def load_model(
    model,
    dataset,
    domainnet_domain,
    adapter_config_name,
    adapter_config,
    train_ds,
    local_run,
    epoch=5,
    rerun_id=None,
):
    # get training, validation, and test data
    if (
        dataset == "cifar10"
        or dataset == "domainnet"
        or dataset == "cub200"
        or dataset == "stanforddogs"
        or dataset == "cifar10-stl10"
        or dataset == "places365"
    ):
        fetched_labels = train_ds.features["label"].names
        id2label = {id: label for id, label in enumerate(fetched_labels)}
        label2id = {label: id for id, label in id2label.items()}
    elif dataset == "cifar100":
        fetched_labels = train_ds.features["fine_label"].names
        id2label = {id: label for id, label in enumerate(fetched_labels)}
        label2id = {label: id for id, label in id2label.items()}
    elif dataset == "caltech256":
        # Since the dataset is special with label (numerical) and text (class
        # names), labels are from 1-257, instead of 0-256 and the order if
        # specified internally. We create the dictionary accordingly
        ids = train_ds["label"]
        labels = train_ds["text"]
        id2label = dict(map(lambda i, j: (i, j), ids, labels))
        id2label[0] = "dummy_label"
        label2id = {label: id for id, label in id2label.items()}

    # Load the processor
    processor = AutoImageProcessor.from_pretrained(model)
    # Load the model
    if dataset == "caltech256":
        # add 1 to the number of labels, 0-157 instead of 0-156
        # this is because the labels from the dataset starts from 1 instead of 0
        # if we use 0-256, then there is label from the dataset which equals to
        # 257, and it will be out of bound and will cause error during
        # training/backpropagation
        vit_config = ViTConfig.from_pretrained(model, num_labels=258)
    else:
        vit_config = ViTConfig.from_pretrained(model, num_labels=len(id2label))
    vit_config.label2id = label2id
    vit_config.id2label = id2label

    # If the model is pretrained on ImageNet21k without the head, we need to
    # change the architecture to ViTForImageClassification, which places a
    # linear layer on top of a pretrained ViTModel. Otherwise, the output
    # doesn't have logits but the last hidden state.
    if model == "google/vit-base-patch16-224-in21k":
        vit_config.architectures = ["ViTForImageClassification"]

    model = ViTAdapterModel.from_pretrained(
        model, config=vit_config, ignore_mismatched_sizes=True
    )

    if dataset == "domainnet" and domainnet_domain:
        # while loading the model, the name of the stored model weights include
        # the specific domain name of DomainNet
        dataset = dataset + "_" + domainnet_domain

    # baseline: finetune, lp
    # if finetune: load the entire saved model weights
    # if lp: only load the head weights (no additional action needed)
    if adapter_config_name == "finetune":
        model.load_state_dict(
            torch.load(
                "checkpoints/"
                + "{}_".format(rerun_id)
                + "e{}_".format(epoch)
                + "{}_".format(dataset)
                + "{}.pt".format(adapter_config_name)
            ),
            strict=False,
        )

    adapter_list = []
    for adapter in adapter_config["Modules"]:
        # Add the adapter to the model
        if adapter == "lora":
            adapter_list.append(LoRAConfig(**adapter_config["LoRAConfig"]))

        elif adapter == "ia3":
            adapter_list.append(IA3Config(**adapter_config["LoRAConfig"]))

        elif adapter == "lokr":
            adapter_list.append(LoRAConfig(**adapter_config["LoRAConfig"]))

        elif adapter == "bn":
            adapter_list.append(BnConfig(**adapter_config["BnConfig"]))

        elif adapter == "compacter":
            adapter_list.append(CompacterConfig())

        if adapter == "pftuning":
            adapter_list.append(
                PrefixTuningConfig(**adapter_config["PrefixTuningConfig"])
            )

    cfg = ConfigUnion(*adapter_list)

    if adapter_list != []:
        model.load_adapter(
            "checkpoints/"
            + "{}_".format(rerun_id)
            + "e{}_".format(epoch)
            + "{}_".format(dataset)
            + "{}".format(adapter_config_name),
            config=cfg,
            map_location=(
                torch.device("cpu") if not torch.cuda.is_available() else None
            ),
        )

        # Define which available adapters are used in the forward and backward
        # pass. This is necessary!
        model.set_active_adapters(adapter_config_name)
        print("Active adapters: ", model.active_adapters)

    # Bitfit
    if "bitfit" in adapter_config["Modules"]:
        loaded_weights = torch.load(
            "checkpoints/"
            + "{}_".format(rerun_id)
            + "e{}_".format(epoch)
            + "{}_".format(dataset)
            + "{}_bias.pt".format(adapter_config_name),
            map_location=(
                torch.device("cpu") if not torch.cuda.is_available() else None
            ),
        )

        model.load_state_dict(loaded_weights, strict=False)

    # load head weights
    head_weights = torch.load(
        "checkpoints/"
        + "{}_".format(rerun_id)
        + "e{}_".format(epoch)
        + "{}_".format(dataset)
        + "{}_head.pt".format(adapter_config_name),
        map_location=(torch.device("cpu") if not torch.cuda.is_available() else None),
    )

    model.load_state_dict(head_weights, strict=False)

    # logging number of trainable params to make sure the model is activated correctly
    utils.print_trainable_parameters(model)

    return processor, model
