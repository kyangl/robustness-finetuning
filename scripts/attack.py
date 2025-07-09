# MIT License

# Copyright (c) 2020 Harry Kim

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

"""Code adapted from torchattacks"""
from __future__ import absolute_import, division, print_function, unicode_literals
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim
from collections import abc as container_abcs

import numpy as np
import math


def wrapper_method(func):
    def wrapper_func(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        for atk in self.__dict__.get("_attacks").values():
            eval("atk." + func.__name__ + "(*args, **kwargs)")
        return result

    return wrapper_func


class Attack(object):
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_model_training_mode`.
    """

    def __init__(self, name, model, device):
        r"""
        Initializes internal attack state.

        Arguments:
            name (str): name of attack.
            model (torch.nn.Module): model to attack.
        """

        self.attack = name
        self._attacks = OrderedDict()

        self.set_model(model)
        if device:
            self.device = device
        else:
            try:
                self.device = next(model.parameters()).device
            except Exception:
                self.device = None
                print(
                    "Failed to set device automatically, please try set_device() manual."
                )

        # Controls attack mode.
        self.attack_mode = "default"
        self.supported_mode = ["default"]
        self.targeted = False
        self._target_map_function = None

        # Controls when normalization is used.
        self.normalization_used = {}
        self._normalization_applied = False
        self._set_auto_normalization_used(model)

        # Controls model mode during attack.
        self._model_training = False
        self._batchnorm_training = False
        self._dropout_training = False

    def forward(self, inputs, labels=None, *args, **kwargs):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def _check_inputs(self, images):
        tol = 1e-4
        if self._normalization_applied:
            images = self.inverse_normalize(images)
        if torch.max(images) > 1 + tol or torch.min(images) < 0 - tol:
            raise ValueError(
                "Input must have a range [0, 1] (max: {}, min: {})".format(
                    torch.max(images), torch.min(images)
                )
            )
        return images

    def _check_outputs(self, images):
        if self._normalization_applied:
            images = self.normalize(images)
        return images

    @wrapper_method
    def set_model(self, model):
        self.model = model
        self.model_name = model.__class__.__name__

    def get_logits(self, inputs, labels=None, *args, **kwargs):
        if self._normalization_applied:
            inputs = self.normalize(inputs)
        logits = self.model(inputs).logits
        return logits

    @wrapper_method
    def _set_normalization_applied(self, flag):
        self._normalization_applied = flag

    @wrapper_method
    def set_device(self, device):
        self.device = device

    @wrapper_method
    def _set_auto_normalization_used(self, model):
        if model.__class__.__name__ == "RobModel":
            mean = getattr(model, "mean", None)
            std = getattr(model, "std", None)
            if (mean is not None) and (std is not None):
                if isinstance(mean, torch.Tensor):
                    mean = mean.cpu().numpy()
                if isinstance(std, torch.Tensor):
                    std = std.cpu().numpy()
                if (mean != 0).all() or (std != 1).all():
                    self.set_normalization_used(mean, std)

    #                 logging.info("Normalization automatically loaded from `model.mean` and `model.std`.")

    @wrapper_method
    def set_normalization_used(self, mean, std):
        n_channels = len(mean)
        mean = torch.tensor(mean).reshape(1, n_channels, 1, 1)
        std = torch.tensor(std).reshape(1, n_channels, 1, 1)
        self.normalization_used["mean"] = mean
        self.normalization_used["std"] = std
        self._normalization_applied = True

    def normalize(self, inputs):
        # print("Inputs in normalization ", inputs)
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return (inputs - mean) / std

    def inverse_normalize(self, inputs):
        mean = self.normalization_used["mean"].to(inputs.device)
        std = self.normalization_used["std"].to(inputs.device)
        return inputs * std + mean

    def get_mode(self):
        r"""
        Get attack mode.

        """
        return self.attack_mode

    @wrapper_method
    def set_mode_default(self):
        r"""
        Set attack mode as default mode.

        """
        self.attack_mode = "default"
        self.targeted = False
        print("Attack mode is changed to 'default.'")

    @wrapper_method
    def _set_mode_targeted(self, mode, quiet):
        if "targeted" not in self.supported_mode:
            raise ValueError("Targeted mode is not supported.")
        self.targeted = True
        self.attack_mode = mode
        if not quiet:
            print("Attack mode is changed to '%s'." % mode)

    @wrapper_method
    def set_mode_targeted_by_function(self, target_map_function, quiet=False):
        r"""
        Set attack mode as targeted.

        Arguments:
            target_map_function (function): Label mapping function.
                e.g. lambda inputs, labels:(labels+1)%10.
                None for using input labels as targeted labels. (Default)
            quiet (bool): Display information message or not. (Default: False)

        """
        self._set_mode_targeted("targeted(custom)", quiet)
        self._target_map_function = target_map_function

    @wrapper_method
    def set_mode_targeted_random(self, quiet=False):
        r"""
        Set attack mode as targeted with random labels.

        Arguments:
            quiet (bool): Display information message or not. (Default: False)

        """
        self._set_mode_targeted("targeted(random)", quiet)
        self._target_map_function = self.get_random_target_label

    @wrapper_method
    def set_mode_targeted_least_likely(self, kth_min=1, quiet=False):
        r"""
        Set attack mode as targeted with least likely labels.

        Arguments:
            kth_min (str): label with the k-th smallest probability used as target labels. (Default: 1)
            num_classses (str): number of classes. (Default: False)

        """
        self._set_mode_targeted("targeted(least-likely)", quiet)
        assert kth_min > 0
        self._kth_min = kth_min
        self._target_map_function = self.get_least_likely_label

    @wrapper_method
    def set_mode_targeted_by_label(self, quiet=False):
        r"""
        Set attack mode as targeted.

        Arguments:
            quiet (bool): Display information message or not. (Default: False)

        .. note::
            Use user-supplied labels as target labels.
        """
        self._set_mode_targeted("targeted(label)", quiet)
        self._target_map_function = "function is a string"

    @wrapper_method
    def set_model_training_mode(
        self, model_training=False, batchnorm_training=False, dropout_training=False
    ):
        r"""
        Set training mode during attack process.

        Arguments:
            model_training (bool): True for using training mode for the entire model during attack process.
            batchnorm_training (bool): True for using training mode for batchnorms during attack process.
            dropout_training (bool): True for using training mode for dropouts during attack process.

        .. note::
            For RNN-based models, we cannot calculate gradients with eval mode.
            Thus, it should be changed to the training mode during the attack.
        """
        self._model_training = model_training
        self._batchnorm_training = batchnorm_training
        self._dropout_training = dropout_training

    @wrapper_method
    def _change_model_mode(self, given_training):
        if self._model_training:
            self.model.train()
            for _, m in self.model.named_modules():
                if not self._batchnorm_training:
                    if "BatchNorm" in m.__class__.__name__:
                        m = m.eval()
                if not self._dropout_training:
                    if "Dropout" in m.__class__.__name__:
                        m = m.eval()
        else:
            self.model.eval()

    @wrapper_method
    def _recover_model_mode(self, given_training):
        if given_training:
            self.model.train()

    def save(
        self,
        data_loader,
        save_path=None,
        verbose=True,
        return_verbose=False,
        save_predictions=False,
        save_clean_inputs=False,
        save_type="float",
    ):
        r"""
        Save adversarial inputs as torch.tensor from given torch.utils.data.DataLoader.

        Arguments:
            save_path (str): save_path.
            data_loader (torch.utils.data.DataLoader): data loader.
            verbose (bool): True for displaying detailed information. (Default: True)
            return_verbose (bool): True for returning detailed information. (Default: False)
            save_predictions (bool): True for saving predicted labels (Default: False)
            save_clean_inputs (bool): True for saving clean inputs (Default: False)

        """
        if save_path is not None:
            adv_input_list = []
            label_list = []
            if save_predictions:
                pred_list = []
            if save_clean_inputs:
                input_list = []

        correct = 0
        total = 0
        l2_distance = []

        total_batch = len(data_loader)
        given_training = self.model.training

        for step, (inputs, labels) in enumerate(data_loader):
            start = time.time()
            adv_inputs = self.__call__(inputs, labels)
            batch_size = len(inputs)

            if verbose or return_verbose:
                with torch.no_grad():
                    outputs = self.get_output_with_eval_nograd(adv_inputs)

                    # Calculate robust accuracy
                    _, pred = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    right_idx = pred == labels.to(self.device)
                    correct += right_idx.sum()
                    rob_acc = 100 * float(correct) / total

                    # Calculate l2 distance
                    delta = (adv_inputs - inputs.to(self.device)).view(
                        batch_size, -1
                    )  # nopep8
                    l2_distance.append(
                        torch.norm(delta[~right_idx], p=2, dim=1)
                    )  # nopep8
                    l2 = torch.cat(l2_distance).mean().item()

                    # Calculate time computation
                    progress = (step + 1) / total_batch * 100
                    end = time.time()
                    elapsed_time = end - start

                    if verbose:
                        self._save_print(
                            progress, rob_acc, l2, elapsed_time, end="\r"
                        )  # nopep8

            if save_path is not None:
                adv_input_list.append(adv_inputs.detach().cpu())
                label_list.append(labels.detach().cpu())

                adv_input_list_cat = torch.cat(adv_input_list, 0)
                label_list_cat = torch.cat(label_list, 0)

                save_dict = {
                    "adv_inputs": adv_input_list_cat,
                    "labels": label_list_cat,
                }  # nopep8

                if save_predictions:
                    pred_list.append(pred.detach().cpu())
                    pred_list_cat = torch.cat(pred_list, 0)
                    save_dict["preds"] = pred_list_cat

                if save_clean_inputs:
                    input_list.append(inputs.detach().cpu())
                    input_list_cat = torch.cat(input_list, 0)
                    save_dict["clean_inputs"] = input_list_cat

                if self.normalization_used is not None:
                    save_dict["adv_inputs"] = self.inverse_normalize(
                        save_dict["adv_inputs"]
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.inverse_normalize(
                            save_dict["clean_inputs"]
                        )  # nopep8

                if save_type == "int":
                    save_dict["adv_inputs"] = self.to_type(
                        save_dict["adv_inputs"], "int"
                    )  # nopep8
                    if save_clean_inputs:
                        save_dict["clean_inputs"] = self.to_type(
                            save_dict["clean_inputs"], "int"
                        )  # nopep8

                save_dict["save_type"] = save_type
                torch.save(save_dict, save_path)

        # To avoid erasing the printed information.
        if verbose:
            self._save_print(progress, rob_acc, l2, elapsed_time, end="\n")

        if given_training:
            self.model.train()

        if return_verbose:
            return rob_acc, l2, elapsed_time

    @staticmethod
    def to_type(inputs, type):
        r"""
        Return inputs as int if float is given.
        """
        if type == "int":
            if isinstance(inputs, torch.FloatTensor) or isinstance(
                inputs, torch.cuda.FloatTensor
            ):
                return (inputs * 255).type(torch.uint8)
        elif type == "float":
            if isinstance(inputs, torch.ByteTensor) or isinstance(
                inputs, torch.cuda.ByteTensor
            ):
                return inputs.float() / 255
        else:
            raise ValueError(type + " is not a valid type. [Options: float, int]")
        return inputs

    @staticmethod
    def _save_print(progress, rob_acc, l2, elapsed_time, end):
        print(
            "- Save progress: %2.2f %% / Robust accuracy: %2.2f %% / L2: %1.5f (%2.3f it/s) \t"
            % (progress, rob_acc, l2, elapsed_time),
            end=end,
        )

    @staticmethod
    def load(
        load_path,
        batch_size=128,
        shuffle=False,
        normalize=None,
        load_predictions=False,
        load_clean_inputs=False,
    ):
        save_dict = torch.load(load_path)
        keys = ["adv_inputs", "labels"]

        if load_predictions:
            keys.append("preds")
        if load_clean_inputs:
            keys.append("clean_inputs")

        if save_dict["save_type"] == "int":
            save_dict["adv_inputs"] = save_dict["adv_inputs"].float() / 255
            if load_clean_inputs:
                save_dict["clean_inputs"] = (
                    save_dict["clean_inputs"].float() / 255
                )  # nopep8

        if normalize is not None:
            n_channels = len(normalize["mean"])
            mean = torch.tensor(normalize["mean"]).reshape(1, n_channels, 1, 1)
            std = torch.tensor(normalize["std"]).reshape(1, n_channels, 1, 1)
            save_dict["adv_inputs"] = (save_dict["adv_inputs"] - mean) / std
            if load_clean_inputs:
                save_dict["clean_inputs"] = (
                    save_dict["clean_inputs"] - mean
                ) / std  # nopep8

        adv_data = TensorDataset(*[save_dict[key] for key in keys])
        adv_loader = DataLoader(adv_data, batch_size=batch_size, shuffle=shuffle)
        print(
            "Data is loaded in the following order: [%s]" % (", ".join(keys))
        )  # nopep8
        return adv_loader

    @torch.no_grad()
    def get_output_with_eval_nograd(self, inputs):
        given_training = self.model.training
        if given_training:
            self.model.eval()
        outputs = self.get_logits(inputs)
        if given_training:
            self.model.train()
        return outputs

    def get_target_label(self, inputs, labels=None):
        r"""
        Function for changing the attack mode.
        Return input labels.
        """
        if self._target_map_function is None:
            raise ValueError(
                "target_map_function is not initialized by set_mode_targeted."
            )
        if self.attack_mode == "targeted(label)":
            target_labels = labels
        else:
            target_labels = self._target_map_function(inputs, labels)
        return target_labels

    @torch.no_grad()
    def get_least_likely_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            _, t = torch.kthvalue(outputs[counter][l], self._kth_min)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    @torch.no_grad()
    def get_random_target_label(self, inputs, labels=None):
        outputs = self.get_output_with_eval_nograd(inputs)
        if labels is None:
            _, labels = torch.max(outputs, dim=1)
        n_classses = outputs.shape[-1]

        target_labels = torch.zeros_like(labels)
        for counter in range(labels.shape[0]):
            l = list(range(n_classses))
            l.remove(labels[counter])
            t = (len(l) * torch.rand([1])).long().to(self.device)
            target_labels[counter] = l[t]

        return target_labels.long().to(self.device)

    def __call__(self, images, labels=None, *args, **kwargs):
        if self.device:
            given_training = self.model.training
            self._change_model_mode(given_training)
            images = self._check_inputs(images)
            adv_images = self.forward(images, labels, *args, **kwargs)
            adv_images = self._check_outputs(adv_images)
            self._recover_model_mode(given_training)
            return adv_images
        else:
            print("Device is not set.")

    def __repr__(self):
        info = self.__dict__.copy()

        del_keys = ["model", "attack", "supported_mode"]

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]

        info["attack_mode"] = self.attack_mode
        info["normalization_used"] = (
            True if len(self.normalization_used) > 0 else False
        )  # nopep8

        return (
            self.attack
            + "("
            + ", ".join("{}={}".format(key, val) for key, val in info.items())
            + ")"
        )

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

        attacks = self.__dict__.get("_attacks")

        # Get all items in iterable items.
        def get_all_values(items, stack=[]):
            if items not in stack:
                stack.append(items)
                if isinstance(items, list) or isinstance(items, dict):
                    if isinstance(items, dict):
                        items = list(items.keys()) + list(items.values())
                    for item in items:
                        yield from get_all_values(item, stack)
                else:
                    if isinstance(items, Attack):
                        yield items
            else:
                if isinstance(items, Attack):
                    yield items

        for num, value in enumerate(get_all_values(value)):
            attacks[name + "." + str(num)] = value
            for subname, subvalue in value.__dict__.get("_attacks").items():
                attacks[name + "." + subname] = subvalue


"""PGD Implementation"""


class PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=10, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        device=None,
        eps=8 / 255,
        alpha=1 / 255,
        steps=10,
        random_start=True,
    ):
        super().__init__("PGD", model, device)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.supported_mode = ["default", "targeted"]
        self.prog = (
            []
        )  # record progress: [(delta_1, acc_1), ..., (delta_n_steps, acc_n_steps)]
        self.track_loss = []  # record loss at each step

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()
        adv_images = images.clone().detach()

        if self.random_start:
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        # # record progress: [(delta_1, acc_1), ..., (delta_n_steps, acc_n_steps)]
        self.prog = []  # re-initialize the list
        self.track_loss = np.array([])  # re-initialize the loss record

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # in the end, the len is the number of steps that PGD takes
            self.track_loss = np.append(self.track_loss, cost.item())

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # # get the mean of the batch
            delta_linf_mean_batch = (
                delta.norm(torch.inf, (1, 2, 3)).mean().item()
            )  # l-infinity norm over all pixels for all channel for each image of the batch, then take the average
            delta_l2_mean_batch = (
                # calculate l2 norm over all pixels
                # since we have the delta, it is equivalent to the l2 distance
                # from delta to the origin
                # take the average over all images in the batch
                (delta**2)
                .sum(dim=(1, 2, 3))
                .sqrt()
                .mean()
                .item()
            )
            # delta_mean = (
            #     delta.norm(torch.inf, 1).mean().item()
            # )  # l-infinity norm over 3 channels, then take the average over 224x224 pixels and all batches
            acc_mean = torch.sum(outputs.argmax(dim=1) == labels).item() / len(
                labels
            )  # accuracy of the batch
            self.prog.append([delta_linf_mean_batch, delta_l2_mean_batch, acc_mean])
        return adv_images

    def get_prog(self):
        return self.prog

    def get_track_loss(self):
        return self.track_loss


class FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FGSM(model, eps=8/255)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, device=None, eps=8 / 255):
        super().__init__("FGSM", model, device)
        self.eps = eps
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images.requires_grad = True
        outputs = self.get_logits(images)

        # Calculate loss
        if self.targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(
            cost, images, retain_graph=False, create_graph=False
        )[0]

        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images


class BIM(Attack):
    r"""
    BIM or iterative-FGSM in the paper 'Adversarial Examples in the Physical World'
    [https://arxiv.org/abs/1607.02533]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)

    .. note:: If steps set to 0, steps will be automatically decided following the paper.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
        >>> adv_images = attack(images, labels)
    """

    def __init__(self, model, device=None, eps=8 / 255, alpha=2 / 255, steps=10):
        super().__init__("BIM", model, device)
        self.eps = eps
        self.alpha = alpha
        if steps == 0:
            self.steps = int(min(eps * 255 + 4, 1.25 * eps * 255))
        else:
            self.steps = steps
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        ori_images = images.clone().detach()

        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.get_logits(images)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, images, retain_graph=False, create_graph=False
            )[0]

            adv_images = images + self.alpha * grad.sign()
            a = torch.clamp(ori_images - self.eps, min=0)
            b = (adv_images >= a).float() * adv_images + (
                adv_images < a
            ).float() * a  # nopep8
            c = (b > ori_images + self.eps).float() * (ori_images + self.eps) + (
                b <= ori_images + self.eps
            ).float() * b  # nopep8
            images = torch.clamp(c, max=1).detach()

        return images


class APGD(Attack):
    r"""
    APGD in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of steps. (Default: 10)
        n_restarts (int): number of random restarts. (Default: 1)
        seed (int): random seed for the starting point. (Default: 0)
        loss (str): loss function optimized. ['ce', 'dlr'] (Default: 'ce')
        eot_iter (int): number of iteration for EOT. (Default: 1)
        rho (float): parameter for step-size update (Default: 0.75)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        device=None,
        norm="Linf",
        eps=8 / 255,
        steps=10,
        n_restarts=1,
        seed=0,
        loss="ce",
        eot_iter=1,
        rho=0.75,
        verbose=False,
    ):
        super().__init__("APGD", model, device)
        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        _, adv_images = self.perturb(images, labels, cheap=True)

        return adv_images

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k * k3 * np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(
            x[np.arange(x.shape[0]), y]
            - x_sorted[:, -2] * ind
            - x_sorted[:, -1] * (1.0 - ind)
        ) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        self.steps_2, self.steps_min, self.size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )
        if self.verbose:
            print(
                "parameters: ", self.steps, self.steps_2, self.steps_min, self.size_decr
            )

        if self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                t.reshape([t.shape[0], -1])
                .abs()
                .max(dim=1, keepdim=True)[0]
                .reshape([-1, 1, 1, 1])
            )  # nopep8
        elif self.norm == "L2":
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                (t**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
            )  # nopep8
        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        if self.loss == "ce":
            criterion_indiv = nn.CrossEntropyLoss(reduction="none")
        elif self.loss == "dlr":
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError("unknown loss")

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                # 1 forward pass (eot_iter = 1)
                logits = self.get_logits(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()

            # 1 backward pass (eot_iter = 1)
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = (
            self.eps
            * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
            * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        )  # nopep8
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        # n_reduced = 0
        for i in range(self.steps):
            # gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
                        0.0,
                        1.0,
                    )
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(
                                x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                                x - self.eps,
                            ),
                            x + self.eps,
                        ),
                        0.0,
                        1.0,
                    )

                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size * grad / (
                        (grad**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt(),
                        ),
                        0.0,
                        1.0,
                    )  # nopep8
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12,
                        ),
                        0.0,
                        1.0,
                    )  # nopep8

                x_adv = x_adv_1 + 0.0

            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    # 1 forward pass (eot_iter = 1)
                    logits = self.get_logits(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()

                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                x_adv[(pred == 0).nonzero().squeeze()] + 0.0
            )  # nopep8
            if self.verbose:
                print("iteration: {} - Best loss: {:.6f}".format(i, loss_best.sum()))

            # check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(),
                        i,
                        k,
                        loss_best.detach().cpu().numpy(),
                        k3=self.thr_decr,
                    )
                    fl_reduce_no_impr = (~reduced_last_check) * (
                        loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
                    )  # nopep8
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.steps_min)

        return x_best, acc, loss_best, x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ["Linf", "L2"]
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        acc = self.get_logits(x).max(1)[1] == y
        # loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print(
                "-------------------------- running {}-attack with epsilon {:.4f} --------------------------".format(
                    self.norm, self.eps
                )
            )
            print("initial accuracy: {:.2%}".format(acc.float().mean()))
        startt = time.time()

        if not best_loss:
            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            if not cheap:
                raise ValueError("not implemented yet")

            else:
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = (
                            x[ind_to_fool].clone(),
                            y[ind_to_fool].clone(),
                        )  # nopep8
                        (
                            best_curr,
                            acc_curr,
                            loss_curr,
                            adv_curr,
                        ) = self.attack_single_run(
                            x_to_fool, y_to_fool
                        )  # nopep8
                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        #
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print(
                                "restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s".format(
                                    counter, acc.float().mean(), time.time() - startt
                                )
                            )

            return acc, adv

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(self.device) * (
                -float("inf")
            )  # nopep8
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.0
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.0

                if self.verbose:
                    print("restart {} - loss: {:.5f}".format(counter, loss_best.sum()))

            return loss_best, adv_best


class MultiAttack(Attack):
    r"""
    MultiAttack is a class to attack a model with various attacks agains same images and labels.

    Arguments:
        model (nn.Module): model to attack.
        attacks (list): list of attacks.

    Examples::
        >>> atk1 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
        >>> atk2 = torchattacks.PGD(model, eps=8/255, alpha=2/255, iters=40, random_start=True)
        >>> atk = torchattacks.MultiAttack([atk1, atk2])
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, attacks, device=None, verbose=False):
        super().__init__("MultiAttack", attacks[0].model, device)
        self.attacks = attacks
        self.verbose = verbose
        self.supported_mode = ["default"]

        self.check_validity()

        self._accumulate_multi_atk_records = False
        self._multi_atk_records = [0.0]

    def check_validity(self):
        if len(self.attacks) < 2:
            raise ValueError("More than two attacks should be given.")

        ids = [id(attack.model) for attack in self.attacks]
        if len(set(ids)) != 1:
            raise ValueError(
                "At least one of attacks is referencing a different model."
            )

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        batch_size = images.shape[0]
        fails = torch.arange(batch_size).to(self.device)
        final_images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        multi_atk_records = [batch_size]

        for _, attack in enumerate(self.attacks):
            adv_images = attack(images[fails], labels[fails])

            outputs = self.get_logits(adv_images)
            _, pre = torch.max(outputs.data, 1)

            corrects = pre == labels[fails]
            wrongs = ~corrects

            succeeds = torch.masked_select(fails, wrongs)
            succeeds_of_fails = torch.masked_select(
                torch.arange(fails.shape[0]).to(self.device), wrongs
            )

            final_images[succeeds] = adv_images[succeeds_of_fails]

            fails = torch.masked_select(fails, corrects)
            multi_atk_records.append(len(fails))

            if len(fails) == 0:
                break

        if self.verbose:
            print(self._return_sr_record(multi_atk_records))

        if self._accumulate_multi_atk_records:
            self._update_multi_atk_records(multi_atk_records)

        return final_images

    def _clear_multi_atk_records(self):
        self._multi_atk_records = [0.0]

    def _covert_to_success_rates(self, multi_atk_records):
        sr = [
            ((1 - multi_atk_records[i] / multi_atk_records[0]) * 100)
            for i in range(1, len(multi_atk_records))
        ]
        return sr

    def _return_sr_record(self, multi_atk_records):
        sr = self._covert_to_success_rates(multi_atk_records)
        return "Attack success rate: " + " | ".join(["%2.2f %%" % item for item in sr])

    def _update_multi_atk_records(self, multi_atk_records):
        for i, item in enumerate(multi_atk_records):
            self._multi_atk_records[i] += item

    def save(
        self,
        data_loader,
        save_path=None,
        verbose=True,
        return_verbose=False,
        save_predictions=False,
        save_clean_images=False,
    ):
        r"""
        Overridden.
        """
        self._clear_multi_atk_records()
        prev_verbose = self.verbose
        self.verbose = False
        self._accumulate_multi_atk_records = True

        for i, attack in enumerate(self.attacks):
            self._multi_atk_records.append(0.0)

        if return_verbose:
            rob_acc, l2, elapsed_time = super().save(
                data_loader,
                save_path,
                verbose,
                return_verbose,
                save_predictions,
                save_clean_images,
            )
            sr = self._covert_to_success_rates(self._multi_atk_records)
        elif verbose:
            super().save(
                data_loader,
                save_path,
                verbose,
                return_verbose,
                save_predictions,
                save_clean_images,
            )
            sr = self._covert_to_success_rates(self._multi_atk_records)
        else:
            super().save(
                data_loader,
                save_path,
                False,
                False,
                save_predictions,
                save_clean_images,
            )

        self._clear_multi_atk_records()
        self._accumulate_multi_atk_records = False
        self.verbose = prev_verbose

        if return_verbose:
            return rob_acc, sr, l2, elapsed_time

    def _save_print(self, progress, rob_acc, l2, elapsed_time, end):
        r"""
        Overridden.
        """
        print(
            "- Save progress: %2.2f %% / Robust accuracy: %2.2f %%"
            % (progress, rob_acc)
            + " / "
            + self._return_sr_record(self._multi_atk_records)
            + " / L2: %1.5f (%2.3f it/s) \t" % (l2, elapsed_time),
            end=end,
        )


class APGDT(Attack):
    r"""
    APGD-Targeted in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks.'
    Targeted attack for every wrong classes.
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of steps. (Default: 10)
        n_restarts (int): number of random restarts. (Default: 1)
        seed (int): random seed for the starting point. (Default: 0)
        eot_iter (int): number of iteration for EOT. (Default: 1)
        rho (float): parameter for step-size update (Default: 0.75)
        verbose (bool): print progress. (Default: False)
        n_classes (int): number of classes. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.APGDT(model, norm='Linf', eps=8/255, steps=10, n_restarts=1, seed=0, eot_iter=1, rho=.75, verbose=False, n_classes=10)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        device=None,
        norm="Linf",
        eps=8 / 255,
        steps=10,
        n_restarts=1,
        seed=0,
        eot_iter=1,
        rho=0.75,
        verbose=False,
        n_classes=10,
    ):
        super().__init__("APGDT", model, device)
        self.eps = eps
        self.steps = steps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.target_class = None
        self.n_target_classes = n_classes - 1
        self.supported_mode = ["default"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        _, adv_images = self.perturb(images, labels, cheap=True)

        return adv_images

    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k * k3 * np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss_targeted(self, x, y, y_target):
        x_sorted, ind_sorted = x.sort(dim=1)

        return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (
            x_sorted[:, -1] - 0.5 * x_sorted[:, -3] - 0.5 * x_sorted[:, -4] + 1e-12
        )

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        self.steps_2, self.steps_min, self.size_decr = (
            max(int(0.22 * self.steps), 1),
            max(int(0.06 * self.steps), 1),
            max(int(0.03 * self.steps), 1),
        )  # nopep8
        if self.verbose:
            print(
                "parameters: ", self.steps, self.steps_2, self.steps_min, self.size_decr
            )  # nopep8

        if self.norm == "Linf":
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                t.reshape([t.shape[0], -1])
                .abs()
                .max(dim=1, keepdim=True)[0]
                .reshape([-1, 1, 1, 1])
            )  # nopep8
        elif self.norm == "L2":
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(
                self.device
            ).detach() * t / (
                (t**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
            )  # nopep8
        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.steps, x.shape[0]])
        loss_best_steps = torch.zeros([self.steps + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        output = self.get_logits(x)
        y_target = output.sort(dim=1)[1][:, -self.target_class]

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():
                # 1 forward pass (eot_iter = 1)
                logits = self.get_logits(x_adv)
                loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
                loss = loss_indiv.sum()

            # 1 backward pass (eot_iter = 1)
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()

        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = (
            self.eps
            * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach()
            * torch.Tensor([2.0]).to(self.device).detach().reshape([1, 1, 1, 1])
        )  # nopep8
        x_adv_old = x_adv.clone()
        # counter = 0
        k = self.steps_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)

        # n_reduced = 0
        for i in range(self.steps):
            # gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == "Linf":
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(
                        torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps),
                        0.0,
                        1.0,
                    )  # nopep8
                    x_adv_1 = torch.clamp(
                        torch.min(
                            torch.max(
                                x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                                x - self.eps,
                            ),
                            x + self.eps,
                        ),
                        0.0,
                        1.0,
                    )  # nopep8

                elif self.norm == "L2":
                    x_adv_1 = x_adv + step_size[0] * grad / (
                        (grad**2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12
                    )  # nopep8
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2)
                            .sum(dim=(1, 2, 3), keepdim=True)
                            .sqrt(),
                        ),
                        0.0,
                        1.0,
                    )  # nopep8
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(
                        x
                        + (x_adv_1 - x)
                        / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12
                        )
                        * torch.min(
                            self.eps * torch.ones(x.shape).to(self.device).detach(),
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
                            + 1e-12,
                        ),
                        0.0,
                        1.0,
                    )  # nopep8

                x_adv = x_adv_1 + 0.0

            # get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    # 1 forward pass (eot_iter = 1)
                    logits = self.get_logits(x_adv)
                    loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
                    loss = loss_indiv.sum()

                # 1 backward pass (eot_iter = 1)
                grad += torch.autograd.grad(loss, [x_adv])[0].detach()

            grad /= float(self.eot_iter)

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = (
                x_adv[(pred == 0).nonzero().squeeze()] + 0.0
            )
            if self.verbose:
                print("iteration: {} - Best loss: {:.6f}".format(i, loss_best.sum()))

            # check step size
            with torch.no_grad():
                y1 = loss_indiv.detach().clone()
                loss_steps[i] = y1.cpu() + 0
                ind = (y1 > loss_best).nonzero().squeeze()
                x_best[ind] = x_adv[ind].clone()
                grad_best[ind] = grad[ind].clone()
                loss_best[ind] = y1[ind] + 0
                loss_best_steps[i + 1] = loss_best + 0

                counter3 += 1

                if counter3 == k:
                    fl_oscillation = self.check_oscillation(
                        loss_steps.detach().cpu().numpy(),
                        i,
                        k,
                        loss_best.detach().cpu().numpy(),
                        k3=self.thr_decr,
                    )  # nopep8
                    fl_reduce_no_impr = (~reduced_last_check) * (
                        loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy()
                    )  # nopep8
                    fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                    reduced_last_check = np.copy(fl_oscillation)
                    loss_best_last_check = loss_best.clone()

                    if np.sum(fl_oscillation) > 0:
                        step_size[u[fl_oscillation]] /= 2.0
                        # n_reduced = fl_oscillation.astype(float).sum()

                        fl_oscillation = np.where(fl_oscillation)

                        x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                        grad[fl_oscillation] = grad_best[fl_oscillation].clone()

                    counter3 = 0
                    k = np.maximum(k - self.size_decr, self.steps_min)

        return x_best, acc, loss_best, x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ["Linf", "L2"]
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        adv = x.clone()
        acc = self.get_logits(x).max(1)[1] == y
        # loss = -1e10 * torch.ones_like(acc).float()
        if self.verbose:
            print(
                "-------------------------- running {}-attack with epsilon {:.4f} --------------------------".format(
                    self.norm, self.eps
                )
            )
            print("initial accuracy: {:.2%}".format(acc.float().mean()))
        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        if not cheap:
            raise ValueError("not implemented yet")

        else:
            for target_class in range(2, self.n_target_classes + 2):
                self.target_class = target_class
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = (
                            x[ind_to_fool].clone(),
                            y[ind_to_fool].clone(),
                        )  # nopep8
                        (
                            best_curr,
                            acc_curr,
                            loss_curr,
                            adv_curr,
                        ) = self.attack_single_run(
                            x_to_fool, y_to_fool
                        )  # nopep8
                        ind_curr = (acc_curr == 0).nonzero().squeeze()

                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                        if self.verbose:
                            print(
                                "restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s".format(
                                    counter,
                                    self.target_class,
                                    acc.float().mean(),
                                    self.eps,
                                    time.time() - startt,
                                )
                            )

        return acc, adv


class FAB(Attack):
    r"""
    Fast Adaptive Boundary Attack in the paper 'Minimally distorted Adversarial Examples with a Fast Adaptive Boundary Attack'
    [https://arxiv.org/abs/1907.02044]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2, L1

    Arguments:
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2', 'L1'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        steps (int): number of steps. (Default: 10)
        n_restarts (int): number of random restarts. (Default: 1)
        alpha_max (float): alpha_max. (Default: 0.1)
        eta (float): overshooting. (Default: 1.05)
        beta (float): backward step. (Default: 0.9)
        verbose (bool): print progress. (Default: False)
        seed (int): random seed for the starting point. (Default: 0)
        targeted (bool): targeted attack for every wrong classes. (Default: False)
        n_classes (int): number of classes. (Default: 10)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FAB(model, norm='Linf', steps=10, eps=8/255, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9, loss_fn=None, verbose=False, seed=0, targeted=False, n_classes=10)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        device=None,
        norm="Linf",
        eps=8 / 255,
        steps=10,
        n_restarts=1,
        alpha_max=0.1,
        eta=1.05,
        beta=0.9,
        verbose=False,
        seed=0,
        multi_targeted=False,
        n_classes=10,
    ):
        super().__init__("FAB", model, device)
        self.norm = norm
        self.n_restarts = n_restarts
        Default_EPS_DICT_BY_NORM = {"Linf": 0.3, "L2": 1.0, "L1": 5.0}
        self.eps = eps if eps is not None else Default_EPS_DICT_BY_NORM[norm]
        self.alpha_max = alpha_max
        self.eta = eta
        self.beta = beta
        self.steps = steps
        self.verbose = verbose
        self.seed = seed
        self.target_class = None
        self.multi_targeted = multi_targeted
        self.n_target_classes = n_classes - 1
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.perturb(images, labels)

        return adv_images

    def _get_predicted_label(self, x):
        with torch.no_grad():
            outputs = self.get_logits(x)
        _, y = torch.max(outputs, dim=1)
        return y

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def get_diff_logits_grads_batch(self, imgs, la):
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.get_logits(im)

        g2 = torch.zeros([y.shape[-1], *imgs.size()]).to(self.device)
        grad_mask = torch.zeros_like(y)
        for counter in range(y.shape[-1]):
            zero_gradients(im)
            grad_mask[:, counter] = 1.0
            y.backward(grad_mask, retain_graph=True)
            grad_mask[:, counter] = 0.0
            g2[counter] = im.grad.data

        g2 = torch.transpose(g2, 0, 1).detach()
        # y2 = self.get_logits(imgs).detach()
        y2 = y.detach()
        df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
        df[torch.arange(imgs.shape[0]), la] = 1e10

        return df, dg

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
        u = torch.arange(imgs.shape[0])
        im = imgs.clone().requires_grad_()
        with torch.enable_grad():
            y = self.get_logits(im)
            diffy = -(y[u, la] - y[u, la_target])
            sumdiffy = diffy.sum()

        zero_gradients(im)
        sumdiffy.backward()
        graddiffy = im.grad.data
        df = diffy.detach().unsqueeze(1)
        dg = graddiffy.unsqueeze(1)

        return df, dg

    def attack_single_run(self, x, y=None, use_rand_start=False):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """

        # self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone().float().to(self.device)
        # assert next(self.model.parameters()).device == x.device

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        pred = y_pred == y
        corr_classified = pred.float().sum()
        if self.verbose:
            print("Clean accuracy: {:.2%}".format(pred.float().mean()))
        if pred.sum() == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:
            if use_rand_start:
                if self.norm == "Linf":
                    t = 2 * torch.rand(x1.shape).to(self.device) - 1
                    x1 = (
                        im2
                        + (
                            torch.min(
                                res2, self.eps * torch.ones(res2.shape).to(self.device)
                            ).reshape([-1, *[1] * self.ndims])
                        )
                        * t
                        / (
                            t.reshape([t.shape[0], -1])
                            .abs()
                            .max(dim=1, keepdim=True)[0]
                            .reshape([-1, *[1] * self.ndims])
                        )
                        * 0.5
                    )
                elif self.norm == "L2":
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = (
                        im2
                        + (
                            torch.min(
                                res2, self.eps * torch.ones(res2.shape).to(self.device)
                            ).reshape([-1, *[1] * self.ndims])
                        )
                        * t
                        / (
                            (t**2)
                            .view(t.shape[0], -1)
                            .sum(dim=-1)
                            .sqrt()
                            .view(t.shape[0], *[1] * self.ndims)
                        )
                        * 0.5
                    )
                elif self.norm == "L1":
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = (
                        im2
                        + (
                            torch.min(
                                res2, self.eps * torch.ones(res2.shape).to(self.device)
                            ).reshape([-1, *[1] * self.ndims])
                        )
                        * t
                        / (
                            t.abs()
                            .view(t.shape[0], -1)
                            .sum(dim=-1)
                            .view(t.shape[0], *[1] * self.ndims)
                        )
                        / 2
                    )

                x1 = x1.clamp(0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.steps:
                with torch.no_grad():
                    df, dg = self.get_diff_logits_grads_batch(x1, la2)
                    if self.norm == "Linf":
                        dist1 = df.abs() / (
                            1e-12
                            + dg.abs().view(dg.shape[0], dg.shape[1], -1).sum(dim=-1)
                        )
                    elif self.norm == "L2":
                        dist1 = df.abs() / (
                            1e-12
                            + (dg**2)
                            .view(dg.shape[0], dg.shape[1], -1)
                            .sum(dim=-1)
                            .sqrt()
                        )
                    elif self.norm == "L1":
                        dist1 = df.abs() / (
                            1e-12
                            + dg.abs()
                            .reshape([df.shape[0], df.shape[1], -1])
                            .max(dim=2)[0]
                        )
                    else:
                        raise ValueError("norm not supported")
                    ind = dist1.min(dim=1)[1]
                    dg2 = dg[u1, ind]
                    b = -df[u1, ind] + (dg2 * x1).view(x1.shape[0], -1).sum(dim=-1)
                    w = dg2.reshape([bs, -1])

                    if self.norm == "Linf":
                        d3 = projection_linf(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0),
                        )
                    elif self.norm == "L2":
                        d3 = projection_l2(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0),
                        )
                    elif self.norm == "L1":
                        d3 = projection_l1(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0),
                        )
                    d1 = torch.reshape(d3[:bs], x1.shape)
                    d2 = torch.reshape(d3[-bs:], x1.shape)
                    if self.norm == "Linf":
                        a0 = (
                            d3.abs()
                            .max(dim=1, keepdim=True)[0]
                            .view(-1, *[1] * self.ndims)
                        )
                    elif self.norm == "L2":
                        a0 = (
                            (d3**2)
                            .sum(dim=1, keepdim=True)
                            .sqrt()
                            .view(-1, *[1] * self.ndims)
                        )
                    elif self.norm == "L1":
                        a0 = (
                            d3.abs()
                            .sum(dim=1, keepdim=True)
                            .view(-1, *[1] * self.ndims)
                        )
                    a0 = torch.max(a0, 1e-8 * torch.ones(a0.shape).to(self.device))
                    a1 = a0[:bs]
                    a2 = a0[-bs:]
                    alpha = torch.min(
                        torch.max(
                            a1 / (a1 + a2), torch.zeros(a1.shape).to(self.device)
                        ),
                        self.alpha_max * torch.ones(a1.shape).to(self.device),
                    )
                    x1 = (
                        (x1 + self.eta * d1) * (1 - alpha)
                        + (im2 + d2 * self.eta) * alpha
                    ).clamp(0.0, 1.0)

                    is_adv = self._get_predicted_label(x1) != la2

                    if is_adv.sum() > 0:
                        ind_adv = is_adv.nonzero().squeeze()
                        ind_adv = self.check_shape(ind_adv)
                        if self.norm == "Linf":
                            t = (
                                (x1[ind_adv] - im2[ind_adv])
                                .reshape([ind_adv.shape[0], -1])
                                .abs()
                                .max(dim=1)[0]
                            )
                        elif self.norm == "L2":
                            t = (
                                ((x1[ind_adv] - im2[ind_adv]) ** 2)
                                .view(ind_adv.shape[0], -1)
                                .sum(dim=-1)
                                .sqrt()
                            )
                        elif self.norm == "L1":
                            t = (
                                (x1[ind_adv] - im2[ind_adv])
                                .abs()
                                .view(ind_adv.shape[0], -1)
                                .sum(dim=-1)
                            )
                        adv[ind_adv] = x1[ind_adv] * (
                            t < res2[ind_adv]
                        ).float().reshape([-1, *[1] * self.ndims]) + adv[ind_adv] * (
                            t >= res2[ind_adv]
                        ).float().reshape(
                            [-1, *[1] * self.ndims]
                        )
                        res2[ind_adv] = (
                            t * (t < res2[ind_adv]).float()
                            + res2[ind_adv] * (t >= res2[ind_adv]).float()
                        )
                        x1[ind_adv] = (
                            im2[ind_adv] + (x1[ind_adv] - im2[ind_adv]) * self.beta
                        )

                    counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print(
                "success rate: {:.0f}/{:.0f}".format(
                    ind_succ.float().sum(), corr_classified
                )
                + " (on correctly classified points) in {:.1f} s".format(
                    time.time() - startt
                )
            )

        res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c

    def attack_single_run_targeted(self, x, y=None, use_rand_start=False):
        """
        :param x:    clean images
        :param y:    clean labels, if None we use the predicted labels
        """

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone().float().to(self.device)
        # assert next(self.model.parameters()).device == x.device

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().long().to(self.device)
        else:
            y = y.detach().clone().long().to(self.device)
        pred = y_pred == y
        corr_classified = pred.float().sum()
        if self.verbose:
            print("Clean accuracy: {:.2%}".format(pred.float().mean()))
        if pred.sum() == 0:
            return x
        pred = self.check_shape(pred.nonzero().squeeze())

        output = self.get_logits(x)
        if self.multi_targeted:
            la_target = output.sort(dim=-1)[1][:, -self.target_class]
        else:
            la_target = self.target_class

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[pred].detach().clone()
        la_target2 = la_target[pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = torch.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * torch.ones([bs]).to(self.device)
        res_c = torch.zeros([x.shape[0]]).to(self.device)
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:
            if use_rand_start:
                if self.norm == "Linf":
                    t = 2 * torch.rand(x1.shape).to(self.device) - 1
                    x1 = (
                        im2
                        + (
                            torch.min(
                                res2, self.eps * torch.ones(res2.shape).to(self.device)
                            ).reshape([-1, *[1] * self.ndims])
                        )
                        * t
                        / (
                            t.reshape([t.shape[0], -1])
                            .abs()
                            .max(dim=1, keepdim=True)[0]
                            .reshape([-1, *[1] * self.ndims])
                        )
                        * 0.5
                    )
                elif self.norm == "L2":
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = (
                        im2
                        + (
                            torch.min(
                                res2, self.eps * torch.ones(res2.shape).to(self.device)
                            ).reshape([-1, *[1] * self.ndims])
                        )
                        * t
                        / (
                            (t**2)
                            .view(t.shape[0], -1)
                            .sum(dim=-1)
                            .sqrt()
                            .view(t.shape[0], *[1] * self.ndims)
                        )
                        * 0.5
                    )
                elif self.norm == "L1":
                    t = torch.randn(x1.shape).to(self.device)
                    x1 = (
                        im2
                        + (
                            torch.min(
                                res2, self.eps * torch.ones(res2.shape).to(self.device)
                            ).reshape([-1, *[1] * self.ndims])
                        )
                        * t
                        / (
                            t.abs()
                            .view(t.shape[0], -1)
                            .sum(dim=-1)
                            .view(t.shape[0], *[1] * self.ndims)
                        )
                        / 2
                    )

                x1 = x1.clamp(0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.steps:
                with torch.no_grad():
                    df, dg = self.get_diff_logits_grads_batch_targeted(
                        x1, la2, la_target2
                    )
                    if self.norm == "Linf":
                        dist1 = df.abs() / (
                            1e-12
                            + dg.abs().view(dg.shape[0], dg.shape[1], -1).sum(dim=-1)
                        )
                    elif self.norm == "L2":
                        dist1 = df.abs() / (
                            1e-12
                            + (dg**2)
                            .view(dg.shape[0], dg.shape[1], -1)
                            .sum(dim=-1)
                            .sqrt()
                        )
                    elif self.norm == "L1":
                        dist1 = df.abs() / (
                            1e-12
                            + dg.abs()
                            .reshape([df.shape[0], df.shape[1], -1])
                            .max(dim=2)[0]
                        )
                    else:
                        raise ValueError("norm not supported")
                    ind = dist1.min(dim=1)[1]

                    dg2 = dg[u1, ind]
                    b = -df[u1, ind] + (dg2 * x1).view(x1.shape[0], -1).sum(dim=-1)
                    w = dg2.reshape([bs, -1])

                    if self.norm == "Linf":
                        d3 = projection_linf(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0),
                        )
                    elif self.norm == "L2":
                        d3 = projection_l2(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0),
                        )
                    elif self.norm == "L1":
                        d3 = projection_l1(
                            torch.cat((x1.reshape([bs, -1]), x0), 0),
                            torch.cat((w, w), 0),
                            torch.cat((b, b), 0),
                        )
                    d1 = torch.reshape(d3[:bs], x1.shape)
                    d2 = torch.reshape(d3[-bs:], x1.shape)
                    if self.norm == "Linf":
                        a0 = (
                            d3.abs()
                            .max(dim=1, keepdim=True)[0]
                            .view(-1, *[1] * self.ndims)
                        )
                    elif self.norm == "L2":
                        a0 = (
                            (d3**2)
                            .sum(dim=1, keepdim=True)
                            .sqrt()
                            .view(-1, *[1] * self.ndims)
                        )
                    elif self.norm == "L1":
                        a0 = (
                            d3.abs()
                            .sum(dim=1, keepdim=True)
                            .view(-1, *[1] * self.ndims)
                        )
                    a0 = torch.max(a0, 1e-8 * torch.ones(a0.shape).to(self.device))
                    a1 = a0[:bs]
                    a2 = a0[-bs:]
                    alpha = torch.min(
                        torch.max(
                            a1 / (a1 + a2), torch.zeros(a1.shape).to(self.device)
                        ),
                        self.alpha_max * torch.ones(a1.shape).to(self.device),
                    )
                    x1 = (
                        (x1 + self.eta * d1) * (1 - alpha)
                        + (im2 + d2 * self.eta) * alpha
                    ).clamp(0.0, 1.0)

                    is_adv = self._get_predicted_label(x1) != la2

                    if is_adv.sum() > 0:
                        ind_adv = is_adv.nonzero().squeeze()
                        ind_adv = self.check_shape(ind_adv)
                        if self.norm == "Linf":
                            t = (
                                (x1[ind_adv] - im2[ind_adv])
                                .reshape([ind_adv.shape[0], -1])
                                .abs()
                                .max(dim=1)[0]
                            )
                        elif self.norm == "L2":
                            t = (
                                ((x1[ind_adv] - im2[ind_adv]) ** 2)
                                .view(ind_adv.shape[0], -1)
                                .sum(dim=-1)
                                .sqrt()
                            )
                        elif self.norm == "L1":
                            t = (
                                (x1[ind_adv] - im2[ind_adv])
                                .abs()
                                .view(ind_adv.shape[0], -1)
                                .sum(dim=-1)
                            )
                        adv[ind_adv] = x1[ind_adv] * (
                            t < res2[ind_adv]
                        ).float().reshape([-1, *[1] * self.ndims]) + adv[ind_adv] * (
                            t >= res2[ind_adv]
                        ).float().reshape(
                            [-1, *[1] * self.ndims]
                        )
                        res2[ind_adv] = (
                            t * (t < res2[ind_adv]).float()
                            + res2[ind_adv] * (t >= res2[ind_adv]).float()
                        )
                        x1[ind_adv] = (
                            im2[ind_adv] + (x1[ind_adv] - im2[ind_adv]) * self.beta
                        )

                    counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print(
                "success rate: {:.0f}/{:.0f}".format(
                    ind_succ.float().sum(), corr_classified
                )
                + " (on correctly classified points) in {:.1f} s".format(
                    time.time() - startt
                )
            )

        res_c[pred] = res2 * ind_succ.float() + 1e10 * (1 - ind_succ.float())
        ind_succ = self.check_shape(ind_succ.nonzero().squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c

    def perturb(self, x, y):
        adv = x.clone()
        with torch.no_grad():
            acc = self.get_logits(x).max(1)[1] == y

            startt = time.time()

            torch.random.manual_seed(self.seed)
            torch.cuda.random.manual_seed(self.seed)

            def inner_perturb(targeted):
                for counter in range(self.n_restarts):
                    ind_to_fool = acc.nonzero().squeeze()
                    if len(ind_to_fool.shape) == 0:
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = (
                            x[ind_to_fool].clone(),
                            y[ind_to_fool].clone(),
                        )  # nopep8

                        if targeted:
                            adv_curr = self.attack_single_run_targeted(
                                x_to_fool, y_to_fool, use_rand_start=(counter > 0)
                            )
                        else:
                            adv_curr = self.attack_single_run(
                                x_to_fool, y_to_fool, use_rand_start=(counter > 0)
                            )

                        acc_curr = self.get_logits(adv_curr).max(1)[1] == y_to_fool
                        if self.norm == "Linf":
                            res = (
                                (x_to_fool - adv_curr)
                                .abs()
                                .view(x_to_fool.shape[0], -1)
                                .max(1)[0]
                            )  # nopep8
                        elif self.norm == "L2":
                            res = (
                                ((x_to_fool - adv_curr) ** 2)
                                .view(x_to_fool.shape[0], -1)
                                .sum(dim=-1)
                                .sqrt()
                            )  # nopep8
                        acc_curr = torch.max(acc_curr, res > self.eps)

                        ind_curr = (acc_curr == 0).nonzero().squeeze()
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

                        if self.verbose:
                            if targeted:
                                print(
                                    "restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s".format(
                                        counter,
                                        self.target_class,
                                        acc.float().mean(),
                                        self.eps,
                                        time.time() - startt,
                                    )
                                )
                            else:
                                print(
                                    "restart {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s".format(
                                        counter,
                                        acc.float().mean(),
                                        self.eps,
                                        time.time() - startt,
                                    )
                                )

            if self.multi_targeted:
                for target_class in range(2, self.n_target_classes + 2):
                    self.target_class = target_class
                    inner_perturb(targeted=True)
            elif self.targeted:
                self.target_class = self.get_target_label(x, y)
                inner_perturb(targeted=True)
            else:
                inner_perturb(targeted=False)
        return adv


def projection_linf(points_to_project, w_hyperplane, b_hyperplane):
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane.clone()

    sign = 2 * ((w * t).sum(1) - b >= 0) - 1
    w.mul_(sign.unsqueeze(1))
    b.mul_(sign)

    a = (w < 0).float()
    d = (a - t) * (w != 0).float()

    p = a - t * (2 * a - 1)
    indp = torch.argsort(p, dim=1)

    b = b - (w * t).sum(1)
    b0 = (w * d).sum(1)

    indp2 = indp.flip((1,))
    ws = w.gather(1, indp2)
    bs2 = -ws * d.gather(1, indp2)

    s = torch.cumsum(ws.abs(), dim=1)
    sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)

    b2 = sb[:, -1] - s[:, -1] * p.gather(1, indp[:, 0:1]).squeeze(1)
    c_l = b - b2 > 0
    c2 = (b - b0 > 0) & (~c_l)
    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    indp_, sb_, s_, p_, b_ = indp[c2], sb[c2], s[c2], p[c2], b[c2]
    for counter in range(nitermax):
        counter4 = torch.floor((lb + ub) / 2)

        counter2 = counter4.long().unsqueeze(1)
        indcurr = indp_.gather(1, indp_.size(1) - 1 - counter2)
        b2 = (
            sb_.gather(1, counter2) - s_.gather(1, counter2) * p_.gather(1, indcurr)
        ).squeeze(
            1
        )  # nopep8
        c = b_ - b2 > 0

        lb = torch.where(c, counter4, lb)
        ub = torch.where(c, ub, counter4)

    lb = lb.long()

    if c_l.any():
        lmbd_opt = torch.clamp_min(
            (b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]), min=0
        ).unsqueeze(-1)
        d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

    lmbd_opt = torch.clamp_min((b[c2] - sb[c2, lb]) / (-s[c2, lb]), min=0).unsqueeze(-1)
    d[c2] = torch.min(lmbd_opt, d[c2]) * a[c2] + torch.max(-lmbd_opt, d[c2]) * (
        1 - a[c2]
    )

    return d * (w != 0).float()


def projection_l2(points_to_project, w_hyperplane, b_hyperplane):
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane

    c = (w * t).sum(1) - b
    ind2 = 2 * (c >= 0) - 1
    w.mul_(ind2.unsqueeze(1))
    c.mul_(ind2)

    r = torch.max(t / w, (t - 1) / w).clamp(min=-1e12, max=1e12)
    r.masked_fill_(w.abs() < 1e-8, 1e12)
    r[r == -1e12] *= -1
    rs, indr = torch.sort(r, dim=1)
    rs2 = F.pad(rs[:, 1:], (0, 1))
    rs.masked_fill_(rs == 1e12, 0)
    rs2.masked_fill_(rs2 == 1e12, 0)

    w3s = (w**2).gather(1, indr)
    w5 = w3s.sum(dim=1, keepdim=True)
    ws = w5 - torch.cumsum(w3s, dim=1)
    d = -(r * w)
    d.mul_((w.abs() > 1e-8).float())
    s = torch.cat(
        (-w5 * rs[:, 0:1], torch.cumsum((-rs2 + rs) * ws, dim=1) - w5 * rs[:, 0:1]), 1
    )

    c4 = s[:, 0] + c < 0
    c3 = (d * w).sum(dim=1) + c > 0
    c2 = ~(c4 | c3)

    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, w.shape[1] - 1)
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_, c_ = s[c2], c[c2]
    for counter in range(nitermax):
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long().unsqueeze(1)
        c3 = s_.gather(1, counter2).squeeze(1) + c_ > 0
        lb = torch.where(c3, counter4, lb)
        ub = torch.where(c3, ub, counter4)

    lb = lb.long()

    if c4.any():
        alpha = c[c4] / w5[c4].squeeze(-1)
        d[c4] = -alpha.unsqueeze(-1) * w[c4]

    if c2.any():
        alpha = (s[c2, lb] + c[c2]) / ws[c2, lb] + rs[c2, lb]
        alpha[ws[c2, lb] == 0] = 0
        c5 = (alpha.unsqueeze(-1) > r[c2]).float()
        d[c2] = d[c2] * c5 - alpha.unsqueeze(-1) * w[c2] * (1 - c5)

    return d * (w.abs() > 1e-8).float()


def projection_l1(points_to_project, w_hyperplane, b_hyperplane):
    device = points_to_project.device
    t, w, b = points_to_project, w_hyperplane.clone(), b_hyperplane

    c = (w * t).sum(1) - b
    ind2 = 2 * (c >= 0) - 1
    w.mul_(ind2.unsqueeze(1))
    c.mul_(ind2)

    r = (1 / w).abs().clamp_max(1e12)
    indr = torch.argsort(r, dim=1)
    indr_rev = torch.argsort(indr)

    c6 = (w < 0).float()
    d = (-t + c6) * (w != 0).float()
    ds = torch.min(-w * t, w * (1 - t)).gather(1, indr)
    ds2 = torch.cat((c.unsqueeze(-1), ds), 1)
    s = torch.cumsum(ds2, dim=1)

    c2 = s[:, -1] < 0

    lb = torch.zeros(c2.sum(), device=device)
    ub = torch.full_like(lb, s.shape[1])
    nitermax = math.ceil(math.log2(w.shape[1]))

    s_ = s[c2]
    for counter in range(nitermax):
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long().unsqueeze(1)
        c3 = s_.gather(1, counter2).squeeze(1) > 0
        lb = torch.where(c3, counter4, lb)
        ub = torch.where(c3, ub, counter4)

    lb2 = lb.long()

    if c2.any():
        indr = indr[c2].gather(1, lb2.unsqueeze(1)).squeeze(1)
        u = torch.arange(0, w.shape[0], device=device).unsqueeze(1)
        u2 = torch.arange(0, w.shape[1], device=device, dtype=torch.float).unsqueeze(0)
        alpha = -s[c2, lb2] / w[c2, indr]
        c5 = u2 < lb.unsqueeze(-1)
        u3 = c5[u[: c5.shape[0]], indr_rev[c2]]
        d[c2] = d[c2] * u3.float()
        d[c2, indr] = alpha

    return d * (w.abs() > 1e-8).float()


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, container_abcs.Iterable):
        for elem in x:
            zero_gradients(elem)


class Square(Attack):
    r"""
    Square Attack in the paper 'Square Attack: a query-efficient black-box adversarial attack via random search'
    [https://arxiv.org/abs/1912.00049]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str): Lp-norm of the attack. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 8/255)
        n_queries (int): max number of queries (each restart). (Default: 5000)
        n_restarts (int): number of random restarts. (Default: 1)
        p_init (float): parameter to control size of squares. (Default: 0.8)
        loss (str): loss function optimized ['margin', 'ce'] (Default: 'margin')
        resc_schedule (bool): adapt schedule of p to n_queries (Default: True)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)
        targeted (bool): targeted. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.Square(model, model, norm='Linf', eps=8/255, n_queries=5000, n_restarts=1, eps=None, p_init=.8, seed=0, verbose=False, targeted=False, loss='margin', resc_schedule=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        device=None,
        norm="Linf",
        eps=8 / 255,
        n_queries=5000,
        n_restarts=1,
        p_init=0.8,
        loss="margin",
        resc_schedule=True,
        seed=0,
        verbose=False,
    ):
        super().__init__("Square", model, device)
        self.norm = norm
        self.n_queries = n_queries
        self.eps = eps
        self.p_init = p_init
        self.n_restarts = n_restarts
        self.seed = seed
        self.verbose = verbose
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.supported_mode = ["default", "targeted"]

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self.perturb(images, labels)

        return adv_images

    def margin_and_loss(self, x, y):
        """
        :param y:        correct labels if untargeted else target labels
        """

        logits = self.get_logits(x)
        xent = F.cross_entropy(logits, y, reduction="none")
        u = torch.arange(x.shape[0])
        y_corr = logits[u, y].clone()
        logits[u, y] = -float("inf")
        y_others = logits.max(dim=-1)[0]

        if not self.targeted:
            if self.loss == "ce":
                return y_corr - y_others, -1.0 * xent
            elif self.loss == "margin":
                return y_corr - y_others, y_corr - y_others
        else:
            if self.loss == "ce":
                return y_others - y_corr, xent
            elif self.loss == "margin":
                return y_others - y_corr, y_others - y_corr

    def init_hyperparam(self, x):
        assert self.norm in ["Linf", "L2"]
        assert not self.eps is None
        assert self.loss in ["ce", "margin"]

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def check_shape(self, x):
        return x if len(x.shape) == (self.ndims + 1) else x.unsqueeze(0)

    def random_choice(self, shape):
        t = 2 * torch.rand(shape).to(self.device) - 1
        return torch.sign(t)

    def random_int(self, low=0, high=1, shape=[1]):
        t = low + (high - low) * torch.rand(shape).to(self.device)
        return t.long()

    def normalize_delta(self, x):
        if self.norm == "Linf":
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == "L2":
            t = (x**2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def lp_norm(self, x):
        if self.norm == "L2":
            t = (x**2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def eta_rectangles(self, x, y):
        delta = torch.zeros([x, y]).to(self.device)
        x_c, y_c = x // 2 + 1, y // 2 + 1

        counter2 = [x_c - 1, y_c - 1]
        for counter in range(0, max(x_c, y_c)):
            delta[
                max(counter2[0], 0) : min(counter2[0] + (2 * counter + 1), x),
                max(0, counter2[1]) : min(counter2[1] + (2 * counter + 1), y),
            ] += 1.0 / (
                torch.Tensor([counter + 1]).view(1, 1).to(self.device) ** 2
            )  # nopep8
            counter2[0] -= 1
            counter2[1] -= 1

        delta /= (delta**2).sum(dim=(0, 1), keepdim=True).sqrt()

        return delta

    def eta(self, s):
        delta = torch.zeros([s, s]).to(self.device)
        delta[: s // 2] = self.eta_rectangles(s // 2, s)
        delta[s // 2 :] = -1.0 * self.eta_rectangles(s - s // 2, s)
        delta /= (delta**2).sum(dim=(0, 1), keepdim=True).sqrt()
        if torch.rand([1]) > 0.5:
            delta = delta.permute([1, 0])

        return delta

    def p_selection(self, it):
        """schedule to decrease the parameter p"""

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it:
            p = self.p_init / 512
        else:
            p = self.p_init

        return p

    def attack_single_run(self, x, y):
        with torch.no_grad():
            adv = x.clone()
            c, h, w = x.shape[1:]
            n_features = c * h * w
            n_ex_total = x.shape[0]

            if self.norm == "Linf":
                x_best = torch.clamp(
                    x + self.eps * self.random_choice([x.shape[0], c, 1, w]), 0.0, 1.0
                )
                margin_min, loss_min = self.margin_and_loss(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)
                s_init = int(math.sqrt(self.p_init * n_features / c))

                for i_iter in range(self.n_queries):
                    idx_to_fool = (margin_min > 0.0).nonzero().flatten()

                    if len(idx_to_fool) == 0:
                        break

                    x_curr = self.check_shape(x[idx_to_fool])
                    x_best_curr = self.check_shape(x_best[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr = y_curr.unsqueeze(0)
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]

                    p = self.p_selection(i_iter)
                    s = max(int(round(math.sqrt(p * n_features / c))), 1)
                    vh = self.random_int(0, h - s)
                    vw = self.random_int(0, w - s)
                    new_deltas = torch.zeros([c, h, w]).to(self.device)
                    new_deltas[:, vh : vh + s, vw : vw + s] = (
                        2.0 * self.eps * self.random_choice([c, 1, 1])
                    )

                    x_new = x_best_curr + new_deltas
                    x_new = torch.min(
                        torch.max(x_new, x_curr - self.eps), x_curr + self.eps
                    )
                    x_new = torch.clamp(x_new, 0.0, 1.0)
                    x_new = self.check_shape(x_new)

                    margin, loss = self.margin_and_loss(x_new, y_curr)

                    # update loss if new loss is better
                    idx_improved = (loss < loss_min_curr).float()

                    loss_min[idx_to_fool] = (
                        idx_improved * loss + (1.0 - idx_improved) * loss_min_curr
                    )

                    # update margin and x_best if new loss is better
                    # or misclassification
                    idx_miscl = (margin <= 0.0).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)

                    margin_min[idx_to_fool] = (
                        idx_improved * margin + (1.0 - idx_improved) * margin_min_curr
                    )
                    idx_improved = idx_improved.reshape([-1, *[1] * len(x.shape[:-1])])
                    x_best[idx_to_fool] = (
                        idx_improved * x_new + (1.0 - idx_improved) * x_best_curr
                    )
                    n_queries[idx_to_fool] += 1.0

                    ind_succ = (margin_min <= 0.0).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        print(
                            "{}".format(i_iter + 1),
                            "- success rate={}/{} ({:.2%})".format(
                                ind_succ.numel(),
                                n_ex_total,
                                float(ind_succ.numel()) / n_ex_total,
                            ),
                            "- avg # queries={:.1f}".format(
                                n_queries[ind_succ].mean().item()
                            ),
                            "- med # queries={:.1f}".format(
                                n_queries[ind_succ].median().item()
                            ),
                            "- loss={:.3f}".format(loss_min.mean()),
                        )

                    if ind_succ.numel() == n_ex_total:
                        break

            elif self.norm == "L2":
                delta_init = torch.zeros_like(x)
                s = h // 5
                sp_init = (h - s * 5) // 2
                vh = sp_init + 0
                for _ in range(h // s):
                    vw = sp_init + 0
                    for _ in range(w // s):
                        delta_init[:, :, vh : vh + s, vw : vw + s] += self.eta(s).view(
                            1, 1, s, s
                        ) * self.random_choice([x.shape[0], c, 1, 1])
                        vw += s
                    vh += s

                x_best = torch.clamp(
                    x + self.normalize_delta(delta_init) * self.eps, 0.0, 1.0
                )
                margin_min, loss_min = self.margin_and_loss(x_best, y)
                n_queries = torch.ones(x.shape[0]).to(self.device)
                s_init = int(math.sqrt(self.p_init * n_features / c))

                for i_iter in range(self.n_queries):
                    idx_to_fool = (margin_min > 0.0).nonzero().flatten()

                    if len(idx_to_fool) == 0:
                        break

                    x_curr = self.check_shape(x[idx_to_fool])
                    x_best_curr = self.check_shape(x_best[idx_to_fool])
                    y_curr = y[idx_to_fool]
                    if len(y_curr.shape) == 0:
                        y_curr = y_curr.unsqueeze(0)
                    margin_min_curr = margin_min[idx_to_fool]
                    loss_min_curr = loss_min[idx_to_fool]

                    delta_curr = x_best_curr - x_curr
                    p = self.p_selection(i_iter)
                    s = max(int(round(math.sqrt(p * n_features / c))), 3)
                    if s % 2 == 0:
                        s += 1

                    vh = self.random_int(0, h - s)
                    vw = self.random_int(0, w - s)
                    new_deltas_mask = torch.zeros_like(x_curr)
                    new_deltas_mask[:, :, vh : vh + s, vw : vw + s] = 1.0
                    norms_window_1 = (
                        (delta_curr[:, :, vh : vh + s, vw : vw + s] ** 2)
                        .sum(dim=(-2, -1), keepdim=True)
                        .sqrt()
                    )

                    vh2 = self.random_int(0, h - s)
                    vw2 = self.random_int(0, w - s)
                    new_deltas_mask_2 = torch.zeros_like(x_curr)
                    new_deltas_mask_2[:, :, vh2 : vh2 + s, vw2 : vw2 + s] = 1.0

                    norms_image = self.lp_norm(x_best_curr - x_curr)
                    mask_image = torch.max(new_deltas_mask, new_deltas_mask_2)
                    norms_windows = self.lp_norm(delta_curr * mask_image)

                    new_deltas = torch.ones([x_curr.shape[0], c, s, s]).to(self.device)
                    new_deltas *= self.eta(s).view(1, 1, s, s) * self.random_choice(
                        [x_curr.shape[0], c, 1, 1]
                    )
                    old_deltas = delta_curr[:, :, vh : vh + s, vw : vw + s] / (
                        1e-12 + norms_window_1
                    )
                    new_deltas += old_deltas
                    new_deltas = (
                        new_deltas
                        / (
                            1e-12
                            + (new_deltas**2).sum(dim=(-2, -1), keepdim=True).sqrt()
                        )
                        * (
                            torch.max(
                                (self.eps * torch.ones_like(new_deltas)) ** 2
                                - norms_image**2,
                                torch.zeros_like(new_deltas),
                            )
                            / c
                            + norms_windows**2
                        ).sqrt()
                    )
                    delta_curr[:, :, vh2 : vh2 + s, vw2 : vw2 + s] = 0.0
                    delta_curr[:, :, vh : vh + s, vw : vw + s] = new_deltas + 0

                    x_new = torch.clamp(
                        x_curr + self.normalize_delta(delta_curr) * self.eps, 0.0, 1.0
                    )
                    x_new = self.check_shape(x_new)
                    norms_image = self.lp_norm(x_new - x_curr)

                    margin, loss = self.margin_and_loss(x_new, y_curr)

                    # update loss if new loss is better
                    idx_improved = (loss < loss_min_curr).float()

                    loss_min[idx_to_fool] = (
                        idx_improved * loss + (1.0 - idx_improved) * loss_min_curr
                    )

                    # update margin and x_best if new loss is better
                    # or misclassification
                    idx_miscl = (margin <= 0.0).float()
                    idx_improved = torch.max(idx_improved, idx_miscl)

                    margin_min[idx_to_fool] = (
                        idx_improved * margin + (1.0 - idx_improved) * margin_min_curr
                    )
                    idx_improved = idx_improved.reshape([-1, *[1] * len(x.shape[:-1])])
                    x_best[idx_to_fool] = (
                        idx_improved * x_new + (1.0 - idx_improved) * x_best_curr
                    )
                    n_queries[idx_to_fool] += 1.0

                    ind_succ = (margin_min <= 0.0).nonzero().squeeze()
                    if self.verbose and ind_succ.numel() != 0:
                        print(
                            "{}".format(i_iter + 1),
                            "- success rate={}/{} ({:.2%})".format(
                                ind_succ.numel(),
                                n_ex_total,
                                float(ind_succ.numel()) / n_ex_total,
                            ),
                            "- avg # queries={:.1f}".format(
                                n_queries[ind_succ].mean().item()
                            ),
                            "- med # queries={:.1f}".format(
                                n_queries[ind_succ].median().item()
                            ),
                            "- loss={:.3f}".format(loss_min.mean()),
                        )

                    assert (x_new != x_new).sum() == 0
                    assert (x_best != x_best).sum() == 0

                    if ind_succ.numel() == n_ex_total:
                        break

        return n_queries, x_best

    def perturb(self, x, y=None):
        """
        :param x:           clean images
        :param y:           untargeted attack -> clean labels,
                            if None we use the predicted labels
                            targeted attack -> target labels, if None random classes,
                            different from the predicted ones, are sampled
        """

        self.init_hyperparam(x)

        adv = x.clone()
        if y is None:
            if not self.targeted:
                with torch.no_grad():
                    output = self.get_logits(x)
                    y_pred = output.max(1)[1]
                    y = y_pred.detach().clone().long().to(self.device)
            else:
                with torch.no_grad():
                    y = self.get_target_label(x, None)
        else:
            if not self.targeted:
                y = y.detach().clone().long().to(self.device)
            else:
                y = self.get_target_label(x, y)

        if not self.targeted:
            acc = self.get_logits(x).max(1)[1] == y
        else:
            acc = self.get_logits(x).max(1)[1] != y

        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        for counter in range(self.n_restarts):
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0:
                ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool = x[ind_to_fool].clone()
                y_to_fool = y[ind_to_fool].clone()

                _, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)

                output_curr = self.get_logits(adv_curr)
                if not self.targeted:
                    acc_curr = output_curr.max(1)[1] == y_to_fool
                else:
                    acc_curr = output_curr.max(1)[1] != y_to_fool
                ind_curr = (acc_curr == 0).nonzero().squeeze()

                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                if self.verbose:
                    print(
                        "restart {} - robust accuracy: {:.2%}".format(
                            counter, acc.float().mean()
                        ),
                        "- cum. time: {:.1f} s".format(time.time() - startt),
                    )

        return adv


class AutoAttack(Attack):
    r"""
    AutoAttack in the paper 'Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks'
    [https://arxiv.org/abs/2003.01690]
    [https://github.com/fra31/auto-attack]

    Distance Measure : Linf, L2

    Arguments:
        model (nn.Module): model to attack.
        norm (str) : Lp-norm to minimize. ['Linf', 'L2'] (Default: 'Linf')
        eps (float): maximum perturbation. (Default: 0.3)
        version (bool): version. ['standard', 'plus', 'rand'] (Default: 'standard')
        n_classes (int): number of classes. (Default: 10)
        seed (int): random seed for the starting point. (Default: 0)
        verbose (bool): print progress. (Default: False)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.AutoAttack(model, norm='Linf', eps=8/255, version='standard', n_classes=10, seed=None, verbose=False)
        >>> adv_images = attack(images, labels)

    """

    def __init__(
        self,
        model,
        device=None,
        norm="Linf",
        eps=8 / 255,
        version="standard",
        n_classes=10,
        seed=None,
        verbose=False,
    ):
        super().__init__("AutoAttack", model, device)
        self.norm = norm
        self.eps = eps
        self.version = version
        self.n_classes = n_classes
        self.seed = seed
        self.verbose = verbose
        self.supported_mode = ["default"]

        if version == "standard":  # ['apgd-ce', 'apgd-t', 'fab-t', 'square']
            self._autoattack = MultiAttack(
                [
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=1,
                    ),
                    APGDT(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_classes=n_classes,
                        n_restarts=1,
                    ),
                    FAB(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        multi_targeted=True,
                        n_classes=n_classes,
                        n_restarts=1,
                    ),
                    Square(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_queries=5000,
                        n_restarts=1,
                    ),
                ]
            )

        # ['apgd-ce', 'apgd-dlr', 'fab', 'square', 'apgd-t', 'fab-t']
        elif version == "plus":
            self._autoattack = MultiAttack(
                [
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        n_restarts=5,
                    ),
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="dlr",
                        n_restarts=5,
                    ),
                    FAB(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_classes=n_classes,
                        n_restarts=5,
                    ),
                    Square(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_queries=5000,
                        n_restarts=1,
                    ),
                    APGDT(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        n_classes=n_classes,
                        n_restarts=1,
                    ),
                    FAB(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        multi_targeted=True,
                        n_classes=n_classes,
                        n_restarts=1,
                    ),
                ]
            )

        elif version == "rand":  # ['apgd-ce', 'apgd-dlr']
            self._autoattack = MultiAttack(
                [
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="ce",
                        eot_iter=20,
                        n_restarts=1,
                    ),
                    APGD(
                        model,
                        eps=eps,
                        norm=norm,
                        seed=self.get_seed(),
                        verbose=verbose,
                        loss="dlr",
                        eot_iter=20,
                        n_restarts=1,
                    ),
                ]
            )

        else:
            raise ValueError("Not valid version. ['standard', 'plus', 'rand']")

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        adv_images = self._autoattack(images, labels)

        return adv_images

    def get_seed(self):
        return time.time() if self.seed is None else self.seed


class CW(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1)
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CW(model, c=1, kappa=0, steps=50, lr=0.01)
        >>> adv_images = attack(images, labels)

    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.

    """

    def __init__(
        self,
        model,
        eps=8 / 255,
        norm=None,
        c=1,
        kappa=0,
        steps=50,
        lr=0.01,
        device=None,
    ):
        super().__init__("CW", model, device)
        self.eps = eps
        self.norm = norm  # We added eps and norm here to help compare results of cw with pgd and apgd given our threat model
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.supported_mode = ["default", "targeted"]
        self.l2_distance = 0.0

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        w = self.inverse_tanh_space(images).detach()
        w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction="none")
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        # re-initialize the l2 distance list
        self.l2_distance = 0.0

        for step in range(self.steps):
            # Get adversarial images
            adv_images = self.tanh_space(w)

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.get_logits(adv_images)
            if self.targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            pre = torch.argmax(outputs.detach(), 1)
            if self.targeted:
                # We want to let pre == target_labels in a targeted attack
                condition = (pre == target_labels).float()
            else:
                # If the attack is not targeted we simply make these two values unequal
                condition = (pre != labels).float()

            # Filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = condition * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    # clamp for early stop
                    if self.norm == "Linf":
                        best_adv_images = torch.clamp(
                            torch.min(
                                torch.max(best_adv_images, images - self.eps),
                                images + self.eps,
                            ),
                            0.0,
                            1.0,
                        )
                        print("Linf norm is used for CW attack.")
                    elif self.norm == "L2":
                        # note here the eps for l2 norm is the same as what we have for linf
                        # norm, e.g., 1/255
                        best_adv_images = torch.clamp(
                            images
                            + (best_adv_images - images)
                            / (
                                ((best_adv_images - images) ** 2)
                                .sum(dim=(1, 2, 3), keepdim=True)
                                .sqrt()
                                + 1e-12
                            )
                            * torch.min(
                                # self.eps
                                # * torch.ones(images.shape)
                                # .to(self.device)
                                # .detach(),  # assume eps is the l2 distance budget
                                (
                                    (
                                        (self.eps)
                                        * torch.ones(images.shape)
                                        .to(self.device)
                                        .detach()
                                    )
                                    ** 2
                                )
                                .sum(dim=(1, 2, 3), keepdim=True)
                                .sqrt(),  # assume eps is the perturbation allowed for each pixel
                                ((best_adv_images - images) ** 2)
                                .sum(dim=(1, 2, 3), keepdim=True)
                                .sqrt()
                                + 1e-12,
                            ),
                            0.0,
                            1.0,
                        )

                        # log the l2 distance from the adv images to the original images
                        # average across the batch
                        l2_mean_batch = (
                            ((best_adv_images - images) ** 2)
                            .sum(dim=(1, 2, 3))
                            .sqrt()
                            .mean()
                            .item()
                        )

                        self.l2_distance = l2_mean_batch
                        print("L2 norm is used for CW attack.")
                    return best_adv_images
                prev_cost = cost.item()

        # L2 norm budget for CW attack
        # This is because the perturbation applied might blow up, we want to
        # compare the result of CW attack with PGD, APGD, etc. We set the l2
        # budget here
        if self.norm == "Linf":
            best_adv_images = torch.clamp(
                torch.min(
                    torch.max(best_adv_images, images - self.eps), images + self.eps
                ),
                0.0,
                1.0,
            )
            print("Linf norm is used for CW attack.")
        elif self.norm == "L2":
            # note here the eps for l2 norm is the same as what we have for linf
            # norm, e.g., 1/255
            best_adv_images = torch.clamp(
                images
                + (best_adv_images - images)
                / (
                    ((best_adv_images - images) ** 2)
                    .sum(dim=(1, 2, 3), keepdim=True)
                    .sqrt()
                    + 1e-12
                )
                * torch.min(
                    # self.eps
                    # * torch.ones(images.shape)
                    # .to(self.device)
                    # .detach(),  # assume eps is the l2 distance budget
                    (
                        ((self.eps) * torch.ones(images.shape).to(self.device).detach())
                        ** 2
                    )
                    .sum(dim=(1, 2, 3), keepdim=True)
                    .sqrt(),  # assume eps is the perturbation allowed for each pixel
                    ((best_adv_images - images) ** 2)
                    .sum(dim=(1, 2, 3), keepdim=True)
                    .sqrt()
                    + 1e-12,
                ),
                0.0,
                1.0,
            )

            # log the l2 distance from the adv images to the original images
            # average across the batch
            l2_mean_batch = (
                ((best_adv_images - images) ** 2)
                .sum(dim=(1, 2, 3))
                .sqrt()
                .mean()
                .item()
            )

            self.l2_distance = l2_mean_batch
            print("L2 norm is used for CW attack.")

        return best_adv_images

    def get_l2_distance_batch(self):
        return self.l2_distance

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        # atanh is defined in the range -1 to 1
        return self.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels * outputs, dim=1)[0]

        if self.targeted:
            return torch.clamp((other - real), min=-self.kappa)
        else:
            return torch.clamp((real - other), min=-self.kappa)
