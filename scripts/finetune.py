import torch
import numpy as np
import os

from transformers.trainer_callback import TrainerControl, TrainerState
import wandb
import json

from adapters import AdapterTrainer

from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from attack import PGD, BIM, APGD, CW
from torch.utils.data import DataLoader, Dataset

from typing import Dict


# Define the collate function at module level
class CollateFn:
    def __init__(self, processor, image_key, label_key):
        self.processor = processor
        self.image_key = image_key
        self.label_key = label_key

    def __call__(self, examples):
        pixel_values = torch.stack(
            [
                self.processor(
                    (
                        example[self.image_key]
                        if example[self.image_key].mode == "RGB"
                        else example[self.image_key].convert("RGB")
                    ),
                    return_tensors="pt",
                )["pixel_values"][0]
                for example in examples
            ]
        )

        labels = torch.tensor([example[self.label_key] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}


def create_collate_fn(processor, image_key, label_key):
    """
    Create a collate function that can be pickled.
    Returns a function that collates data for the dataloader.
    """
    return CollateFn(processor, image_key, label_key)


class DomainDataset(Dataset):
    """
    Memory-efficient dataset implementation for OOD
    """

    def __init__(self, original_dataset, domain, filter_in_domain=True):
        self.original_dataset = original_dataset
        self.domain = domain

        domain_names = {
            "clipart": 0,
            "infograph": 1,
            "painting": 2,
            "quickdraw": 3,
            "real": 4,
            "sketch": 5,
        }

        # Create index mapping without loading all data
        self.indices = [
            i
            for i, example in enumerate(original_dataset)
            if (example["domain"] == domain_names[domain]) == filter_in_domain
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.original_dataset[self.indices[idx]]


class EfficientAccuracyLogger:
    """
    Memory-efficient accuracy logging implementation (for OOD)
    """

    def __init__(self, batch_size=128):
        self.batch_size = batch_size

    def compute_accuracy(self, model, dataloader, device):
        model.eval()
        correct = 0
        total = 0

        # Use memory-efficient iteration
        for batch in dataloader:
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model(images)
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            # Explicitly free memory
            del images, labels, outputs, preds
            torch.cuda.empty_cache()

        return correct / total if total > 0 else 0


# Customized call back function to log standard and robust accuracy during
# training. This is planned to be called after each batch of training (including
# before the first batch).
class AccuracyLoggingCallback(TrainerCallback):
    def __init__(
        self,
        test_ds=None,
        attack_type="ood",  # TODO: change to None for non-ood datasets/attacks
        domainnet_train_domain=None,
        collate_fn_custom=None,
        full_config=None,
        dataloaders: Dict[str, DataLoader] = None,
    ):
        self.test_ds = test_ds
        self.attack_type = attack_type

        self.domainnet_train_domain = domainnet_train_domain
        if self.domainnet_train_domain:
            self.accuracy_logger_ood = EfficientAccuracyLogger()
            self.dataloaders = dataloaders

        self.collate_fn_custom = collate_fn_custom
        self.full_config = full_config
        self.log_pgd_loss = False

    def compute_clean_acc(self, model, one_dataloader, metric_name):
        """Going through the dataset, compute the accuracy of the current state
        of the model on the clean examples."""
        # GPU/CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        total_images = 0
        clean_acc = 0.0

        # run the attack algorithms
        for batch in one_dataloader:
            images, labels = batch["pixel_values"], batch["labels"]
            images, labels = images.to(device), labels.to(device)
            total_images += len(labels)

            with torch.no_grad():
                clean_output = model(images)

            # clean accuracy
            clean_acc += torch.sum(
                clean_output.logits.to(device).argmax(dim=1) == labels
            ).item()

        return {metric_name: clean_acc / total_images}

    def compute_test_clean_adv_acc(self, model, test_dataloader):
        """Going through the dataset, generate adversarial examples, and compute
        the accuracy of the current state of the model on the adversarial
        examples."""

        # GPU/CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # initialize the attack algorithms
        self.pgd_attack = PGD(
            model, eps=1 / 255, alpha=1 / 1020, steps=15, random_start=True
        )
        self.pgd_attack.set_normalization_used(
            mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
        )
        total_images = 0
        clean_acc = 0.0
        pgd_adv_acc = 0.0

        batched_pgd_loss = np.array([])
        num_batches = 0

        # run the attack algorithms
        for batch in test_dataloader:
            images, labels = batch["pixel_values"], batch["labels"]
            images, labels = images.to(device), labels.to(device)
            total_images += len(labels)

            adv_images_pgd = self.pgd_attack(images, labels)

            with torch.no_grad():
                clean_output = model(images)
                adv_output_pgd = model(adv_images_pgd)

            # clean accuracy
            clean_acc += torch.sum(
                clean_output.logits.to(device).argmax(dim=1) == labels
            ).item()
            # robust accuracy
            pgd_adv_acc += torch.sum(
                adv_output_pgd.logits.to(device).argmax(dim=1) == labels
            ).item()

            if self.log_pgd_loss:
                num_batches += 1
                print(self.pgd_attack.get_track_loss())
                if len(batched_pgd_loss) == 0:
                    # if it is empty, initialize it (adding doesnt work here)
                    batched_pgd_loss = self.pgd_attack.get_track_loss()
                else:
                    batched_pgd_loss += self.pgd_attack.get_track_loss()

        if self.log_pgd_loss:
            print(f"Averaged pgd loss per batch: {self.batched_pgd_loss / num_batches}")

        return {
            "test_clean_acc": clean_acc / total_images,
            "pgd_adv_acc": pgd_adv_acc / total_images,
            "pgd_loss": batched_pgd_loss / num_batches if self.log_pgd_loss else None,
        }

    def save_metrics_to_json(self, metrics, file_path):
        """Save the metrics to a json file at each step of the training."""
        # create the file if it doesn't exist, or load data
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
        else:
            data = []

        # append the new metrics
        data.append(metrics)

        # save the data back to the file
        with open(file_path, "w") as f:
            json.dump(data, f)

    def get_all_metrics(self, model, train_dataloader):
        """Get all metrics for the current state of the model -- training clean
        acc, test clean acc, test ood/adv acc."""
        res = {}
        # 1. Training accuracy (clean)
        if self.attack_type == "ood":
            train_clean_acc = self.compute_clean_acc(
                model, self.ood_train_clean_dataloader, "train_clean_acc"
            )
            # debugging - log train_dataloader and ood_train_clean_dataloader
            print(
                f"Train dataloader length: {train_dataloader.dataset.num_rows}, and it's unique domains: {train_dataloader.dataset.unique('domain')}"
            )
            print(
                f"Train ood clean dataloder length: {self.ood_train_clean_dataloader.dataset.num_rows}, and it's unique domains: {self.ood_train_clean_dataloader.dataset.unique('domain')}"
            )
        else:
            train_clean_acc = self.compute_clean_acc(
                model, train_dataloader, "train_clean_acc"
            )
        res.update(train_clean_acc)

        # 2. Test accuracy (clean) and robustness
        if self.test_ds is not None:
            if self.attack_type == "ood":
                # test clean accuracy for ood
                test_clean_acc = self.compute_clean_acc(
                    model, self.ood_test_clean_dataloader, "test_clean_acc"
                )
                res.update(test_clean_acc)

                # test ood robustness
                test_ood_acc = self.compute_clean_acc(
                    model, self.ood_test_ood_dataloader, "test_ood_acc"
                )
                res.update(test_ood_acc)
            else:
                # test clean accuracy and adv robustness
                test_clean_ds = self.test_ds

                test_clean_dataloader = DataLoader(
                    test_clean_ds, collate_fn=self.collate_fn_custom, batch_size=64
                )

                test_clean_adv_acc = self.compute_test_clean_adv_acc(
                    model, test_clean_dataloader
                )
                res.update(test_clean_adv_acc)
        return res

    # Determine when to log the metrics during training
    def should_log(self, step):
        if step <= 100:
            return step % 50 == 0
        elif step <= 300:
            return step % 30 == 0
        if step <= 1000:
            return step % 200 == 0
        elif step <= 3000:
            return step % 2000 == 0
        elif step <= 10000:
            return step % 4000 == 0
        elif step <= 30000:
            return step % 6000 == 0
        else:
            return step % 20000 == 0

    def on_train_begin(
        self,
        args,
        state,
        control,
        model=None,
        train_dataloader=None,
        logs=None,
        **kwargs,
    ):
        """This is called before the first batch is trained."""
        super().on_train_begin(
            args, state, control, model=None, train_dataloader=None, logs=None, **kwargs
        )
        print("Training started.")
        print(f"step is {state.global_step}, epoch is {state.epoch}.")
        res = {}
        # print the current state of the model and the logs
        if logs is not None:
            if "loss" in logs:
                res["training_loss"] = logs["loss"]
            if "learning_rate" in logs:
                res["learning_rate"] = logs["learning_rate"]
            if "epoch" in logs:
                res["epoch"] = logs["epoch"]
        res["step"] = state.global_step
        res["epoch"] = state.epoch

        if self.domainnet_train_domain:
            # OOD
            device = next(model.parameters()).device

            # compute accuracy and robustness efficiently
            for name, dataloader in self.dataloaders.items():
                acc = self.accuracy_logger_ood.compute_accuracy(
                    model, dataloader, device
                )
                res[f"{name}_accuracy"] = acc
        else:
            # ADV
            res.update(self.get_all_metrics(model, train_dataloader))

        # save all states, logs, and accuracies to a json file for this step
        # epoch, step, learning rate, loss, train_clean_acc, test_clean_acc,
        # test_ood_acc | pgd_adv_acc
        self.save_metrics_to_json(res, f"{self.full_config}_trade-off_step.json")

    def on_train_end(
        self,
        args,
        state,
        control,
        model=None,
        train_dataloader=None,
        logs=None,
        **kwargs,
    ):
        """This is called at the end of training."""
        super().on_train_end(
            args, state, control, model=None, train_dataloader=None, logs=None, **kwargs
        )
        self.log_pgd_loss = True
        print("Training ended.")
        res = {}
        res["step"] = state.global_step
        res["epoch"] = state.epoch

        if self.domainnet_train_domain:
            # OOD
            device = next(model.parameters()).device

            # compute accuracy and robustness efficiently
            for name, dataloader in self.dataloaders.items():
                acc = self.accuracy_logger_ood.compute_accuracy(
                    model, dataloader, device
                )
                res[f"{name}_accuracy"] = acc
        else:
            # ADV
            res.update(self.get_all_metrics(model, train_dataloader))

        self.save_metrics_to_json(res, f"{self.full_config}_trade-off_step.json")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        train_dataloader=None,
        optimizer=None,
        lr_scheduler=None,
        metrics=None,
        logs=None,
        **kwargs,
    ):
        """This is called after each batch is trained. (for each backpropagation)"""
        super().on_step_end(
            args, state, control, model=None, train_dataloader=None, logs=None, **kwargs
        )
        res = {}
        res["step"] = state.global_step
        res["learning_rate"] = lr_scheduler.base_lrs[-1]
        res["epoch"] = state.epoch

        # log metrics per 10 steps (for faster logging)
        if self.should_log(res["step"]):
            if self.domainnet_train_domain:
                # OOD
                device = next(model.parameters()).device

                # compute accuracy and robustness efficiently
                for name, dataloader in self.dataloaders.items():
                    acc = self.accuracy_logger_ood.compute_accuracy(
                        model, dataloader, device
                    )
                    res[f"{name}_accuracy"] = acc
            else:
                # ADV
                res.update(self.get_all_metrics(model, train_dataloader))
            self.save_metrics_to_json(res, f"{self.full_config}_trade-off_step.json")


def create_efficient_dataloaders(
    train_ds, test_ds, domain, batch_size, collate_fn
) -> Dict[str, DataLoader]:
    """
    Create memory-efficient dataloaders for OOD
    """

    # Create domain-specific datasets using lazy loading
    train_domain_ds = DomainDataset(train_ds, domain, filter_in_domain=True)
    test_clean_ds = DomainDataset(test_ds, domain, filter_in_domain=True)
    test_ood_ds = DomainDataset(test_ds, domain, filter_in_domain=False)

    # Create dataloaders with efficient memory usage
    dataloaders = {
        "train": DataLoader(
            train_domain_ds,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=2,
            prefetch_factor=2,
        ),
        "test_clean": DataLoader(
            test_clean_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=2,
        ),
        "test_ood": DataLoader(
            test_ood_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=2,
        ),
    }

    return dataloaders


def train_model(
    processor,
    model,
    dataset_name,
    domainnet_domain,
    adapter_config_name,
    adapter_config,
    train_ds,
    batch_size=64,
    epoch=5,
    rerun_id=None,
    test_ds=None,
    attack_type=None,
    learning_rate=2e-5,  # for finetuning, 5e-3 for others
    weight_decay=0.01,  # for finetuning, 0 for others
):

    if domainnet_domain:
        # Set the dataset name to domainnet and the specific domain
        dataset_name = f"{dataset_name}_{domainnet_domain}"

    checkpoint_path = (
        "./checkpoints/"
        + f"{dataset_name}_{adapter_config_name}_{str(learning_rate)}_{str(weight_decay)}_{str(rerun_id)}"
    )

    # GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Further split the training data into training and validation data
    splits = train_ds.train_test_split(test_size=0.1)
    train_ds = splits["train"]
    val_ds = splits["test"]

    # Get the key for image and label in the dataset
    if dataset_name == "stanforddogs":
        # specific name for images from stl10 dataset (HF)
        image_key = "pixel_values"
    else:
        # for all other datasets
        image_key = "image" if "image" in train_ds.features else "img"
    # Take the first key that is not image_key
    label_key = [x for x in train_ds.features if x != image_key][0]

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        acc = (preds == p.label_ids).mean()
        return {"accuracy": acc}

    # Create a collate function outside of this training function
    collate_fn = create_collate_fn(processor, image_key, label_key)

    # specific training parameters for finetuning
    if adapter_config_name == "finetune":
        print("Updated fine-tuning training parameters")
        args = TrainingArguments(
            output_dir=checkpoint_path,
            label_names=["labels"],
            save_strategy="epoch",
            save_total_limit=1,  # save the last 3 checkpoints only
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epoch,
            weight_decay=weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir="logs",
            remove_unused_columns=False,
            report_to="none",  # disable wandb for colab
            # fp16=True,  # only on GPUs
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )
    else:
        args = TrainingArguments(
            output_dir=checkpoint_path,
            label_names=["labels"],
            remove_unused_columns=False,
            evaluation_strategy="epoch",
            save_total_limit=1,
            save_strategy="epoch",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,  # how often the model is updated
            per_device_eval_batch_size=batch_size,
            # fp16=True,  # only on GPUs
            num_train_epochs=epoch,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            # output_dir="./training_output",
            report_to="none",  # disable wandb for colab
            # use_cpu=True,  # for local testing
            lr_scheduler_type="cosine",
            warmup_steps=800,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )

    # For OOD, filter the dataset to get test_clean_data and test_ood_data ONCE
    # here. Instead of adv attack (i.e., PGD), OOD doesn't depend on the model
    # weights. Thus, we can get the dataset here and use it for the entire
    # training
    if domainnet_domain:
        dataloaders = create_efficient_dataloaders(
            train_ds, test_ds, domainnet_domain, batch_size, collate_fn
        )
    else:
        dataloaders = None

    if (
        adapter_config_name in ["finetune", "lp"]
        or "bitfit" in adapter_config["Modules"]
    ):
        # conventional trainer for fine-tuning, linear probing, and bitfit
        trainer = Trainer(
            model,
            args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            # tokenizer=processor,
            compute_metrics=compute_metrics,
            callbacks=[
                AccuracyLoggingCallback(
                    test_ds=test_ds,
                    attack_type=attack_type,
                    domainnet_train_domain=domainnet_domain,
                    # ood_train_clean_dataloader=ood_train_clean_dataloader,
                    # ood_test_clean_dataloader=ood_test_clean_dataloader,
                    # ood_test_ood_dataloader=ood_test_ood_dataloader,
                    collate_fn_custom=collate_fn,
                    dataloaders=dataloaders,
                    # full_config=f"{dataset_name}_{domainnet_domain}_{adapter_config_name}_{rerun_id}",
                    full_config=f"{dataset_name}_{adapter_config_name}_{str(learning_rate)}_{str(weight_decay)}_{str(rerun_id)}",
                )
            ],
        )
    else:
        # adapter trainer for other fine-tuning methods
        trainer = AdapterTrainer(
            model,
            args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=collate_fn,
            # tokenizer=processor,
            compute_metrics=compute_metrics,
            callbacks=[
                AccuracyLoggingCallback(
                    test_ds=test_ds,
                    attack_type=attack_type,
                    domainnet_train_domain=domainnet_domain,
                    collate_fn_custom=collate_fn,
                    dataloaders=dataloaders,
                    full_config=f"{dataset_name}_{adapter_config_name}_{str(learning_rate)}_{str(weight_decay)}_{str(rerun_id)}",
                )
            ],
        )

    # Log the number of trainable parameters
    wandb.log(
        {
            "Trainable Parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        }
    )
    wandb.log({"Total Parameters": sum(p.numel() for p in model.parameters())})

    # Train the model (get validation accuracy)
    last_checkpoint = get_last_checkpoint(checkpoint_path)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Print the number of images for training and the training configs
    print(f"Training dataset has {len(train_ds)} images for domain {domainnet_domain}")
    print(
        f"Training dataset is {dataset_name}, with adapter config {adapter_config_name}, learning rate {learning_rate}, and weight decay {weight_decay}, for epochs {epoch}"
    )

    return model
