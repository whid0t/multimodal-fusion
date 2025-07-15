import inspect
import os.path as osp
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable

from mmseg.core import eval_metrics
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CustomDatasetEval(CustomDataset):
    """Custom dataset for evaluation that accepts and uses a label_map.

    This allows the evaluation function to use the same label remapping as the
    training pipeline, by passing the `label_map` to the `eval_metrics`
    function during the `evaluate` call. This is the correct way to handle
    evaluation with remapped labels in this version of MMSegmentation.
    """

    def __init__(self, label_map=None, **kwargs):
        # Get the argument names from the parent's __init__ method
        parent_sig = inspect.signature(CustomDataset.__init__)
        parent_arg_names = [p.name for p in parent_sig.parameters.values()]

        # Filter kwargs to only include what the parent expects, to avoid
        # 'unexpected keyword argument' errors.
        parent_kwargs = {
            key: val
            for key, val in kwargs.items() if key in parent_arg_names
        }

        super().__init__(**parent_kwargs)

        # Store the label_map to be used during evaluation
        self.label_map = label_map if label_map is not None else dict()

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 efficient_test=False,
                 **kwargs):
        """Evaluate the dataset with the stored label_map.

        This method is an exact copy of the parent CustomDataset.evaluate, with
        the crucial difference that it passes `self.label_map` (which was
        stored during __init__) to the `eval_metrics` function.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))
        eval_results = {}
        gt_seg_maps = self.get_gt_seg_maps(efficient_test)

        # Manually remap the ground truth labels to match the model's output.
        # This is the most robust way to ensure the metrics are calculated
        # correctly, as we are comparing like with like.
        remapped_gt_seg_maps = []
        for gt_map in gt_seg_maps:
            remapped_gt = np.copy(gt_map)
            for k, v in self.label_map.items():
                remapped_gt[gt_map == k] = v
            remapped_gt_seg_maps.append(remapped_gt)
        gt_seg_maps = remapped_gt_seg_maps

        if self.CLASSES is None:
            num_classes = len(
                reduce(np.union1d, [np.unique(_) for _ in gt_seg_maps]))
        else:
            num_classes = len(self.CLASSES)

        # The ground truth is now remapped, so we don't pass the label_map.
        ret_metrics = eval_metrics(
            results,
            gt_seg_maps,
            num_classes,
            self.ignore_index,
            metric,
            label_map=dict(),  # Pass an empty dict
            reduce_zero_label=self.reduce_zero_label)

        if self.CLASSES is None:
            class_names = tuple(range(num_classes))
        else:
            class_names = self.CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        if mmcv.is_list_of(results, str):
            for file_name in results:
                osp.remove(file_name)
        return eval_results 