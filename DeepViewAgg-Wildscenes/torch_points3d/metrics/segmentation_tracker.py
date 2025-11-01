from typing import Dict, Any
import torch
import numpy as np

from torch_points3d.metrics.confusion_matrix import ConfusionMatrix
from torch_points3d.metrics.base_tracker import BaseTracker, meter_value
from torch_points3d.metrics.meters import APMeter
from torch_points3d.datasets.segmentation import IGNORE_LABEL
from torch_points3d.models import model_interface


class SegmentationTracker(BaseTracker):
    def __init__(
        self, dataset, stage="train", wandb_log=False, use_tensorboard: bool = False, ignore_label: int = IGNORE_LABEL
    ):
        """ This is a generic tracker for multimodal tasks.
        It uses a confusion matrix in the back-end to track results.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch

        Arguments:
            dataset  -- dataset to track (used for the number of classes)

        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
            wandb_log {str} --  Log using weight and biases
        """
        super(SegmentationTracker, self).__init__(stage, wandb_log, use_tensorboard)
        self._num_classes = dataset.num_classes
        self._ignore_label = ignore_label
        self._dataset = dataset
        self.reset(stage)
        self._metric_func = {
            "miou": max,
            "macc": max,
            "acc": max,
            "loss": min,
            "map": max,
        }  # Those map subsentences to their optimization functions

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._confusion_matrix = ConfusionMatrix(self._num_classes)
        self._acc = 0
        self._macc = 0
        self._miou = 0
        self._miou_per_class = {}
        self._precision_macro = 0
        self._recall_macro = 0
        self._dice_macro = 0
        self._precision_per_class = {}
        self._recall_per_class = {}
        self._dice_per_class = {}

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """ Add current model predictions (usually the result of a batch) to the tracking
        """
        if not self._dataset.has_labels(self._stage):
            return

        super().track(model)

        outputs = model.get_output()
        targets = model.get_labels()
        self._compute_metrics(outputs, targets)

    def _compute_metrics(self, outputs, labels):
        mask = labels != self._ignore_label
        outputs = outputs[mask]
        labels = labels[mask]

        outputs = self._convert(outputs)
        labels = self._convert(labels)

        if len(labels) == 0:
            return

        assert outputs.shape[0] == len(labels)
        self._confusion_matrix.count_predicted_batch(labels, np.argmax(outputs, 1))

        self._acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        self._miou = 100 * self._confusion_matrix.get_average_intersection_union()
        self._miou_per_class = {
            i: "{:.2f}".format(100 * v)
            for i, v in enumerate(self._confusion_matrix.get_intersection_union_per_class()[0])
        }

        # Precision / Recall / Dice (macro and per-class)
        precision_per_class = self._confusion_matrix.get_precision_per_class()
        recall_per_class = self._confusion_matrix.get_recall_per_class()
        dice_per_class = self._confusion_matrix.get_dice_per_class()

        self._precision_macro = 100 * self._confusion_matrix.get_macro_precision()
        self._recall_macro = 100 * self._confusion_matrix.get_macro_recall()
        self._dice_macro = 100 * self._confusion_matrix.get_macro_dice()

        self._precision_per_class = {i: "{:.2f}".format(100 * v) for i, v in enumerate(precision_per_class)}
        self._recall_per_class = {i: "{:.2f}".format(100 * v) for i, v in enumerate(recall_per_class)}
        self._dice_per_class = {i: "{:.2f}".format(100 * v) for i, v in enumerate(dice_per_class)}

    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """ Returns a dictionnary of all metrics and losses being tracked
        """
        metrics = super().get_metrics(verbose)

        metrics["{}_acc".format(self._stage)] = self._acc
        metrics["{}_macc".format(self._stage)] = self._macc
        metrics["{}_miou".format(self._stage)] = self._miou
        metrics["{}_precision".format(self._stage)] = self._precision_macro
        metrics["{}_recall".format(self._stage)] = self._recall_macro
        metrics["{}_dice".format(self._stage)] = self._dice_macro

        if verbose:
            metrics["{}_miou_per_class".format(self._stage)] = self._miou_per_class
            metrics["{}_precision_per_class".format(self._stage)] = self._precision_per_class
            metrics["{}_recall_per_class".format(self._stage)] = self._recall_per_class
            metrics["{}_dice_per_class".format(self._stage)] = self._dice_per_class
        return metrics

    @property
    def metric_func(self):
        return self._metric_func
