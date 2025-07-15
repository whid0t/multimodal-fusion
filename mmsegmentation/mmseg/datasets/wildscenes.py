from mmseg.datasets import DATASETS
from mmseg.datasets import CustomDataset
import os.path as osp


def _get_reversed_custom_label_map(custom_label_map):
        rev_custom_label_map = {}
        label_set = set()
        for k,v in custom_label_map.items():
            if v not in label_set:
                rev_custom_label_map[v] = [k]
            else:
                rev_custom_label_map[v].append(k)
            label_set.add(v)
        
        return rev_custom_label_map


@DATASETS.register_module()
class WildscenesDataset(CustomDataset):
    """The Wildscenes datasets for MMSegmentation 0.14.1.

    Can be pointed to a specific split by setting data_root to a specific directory."""

    CLASSES = (
        "unlabelled",
        "asphalt/concrete",
        "dirt",
        "mud",
        "water",
        "gravel",
        "other-terrain",
        "tree-trunk",
        "tree-foliage",
        "bush",
        "fence",
        "other-structure",
        "pole",
        "vehicle",
        "rock",
        "log",
        "other-object",
        "sky",
        "grass",
    )

    PALETTE = [
        (0, 0, 0),
        (255, 165, 0),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (230, 25, 75),
        (0, 128, 128),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (250, 190, 190),
        (0, 0, 128),
        (128, 128, 128),
    ]

    def __init__(
        self,
        custom_label_map=None,
        **kwargs,
    ):
        # Set default values if not provided
        kwargs.setdefault('img_suffix', '.png')
        kwargs.setdefault('seg_map_suffix', '.png')
        kwargs.setdefault('reduce_zero_label', False)
        
        # Apply custom label map BEFORE calling parent init
        if custom_label_map is not None:
            # Get new classes and update temporarily
            new_classes = self._get_new_labels(custom_label_map)
            new_palette = self._get_updated_palette(custom_label_map)
            self.label_map = self._get_idx_map(custom_label_map)
            
            # Temporarily update CLASSES and PALETTE for parent validation
            self.CLASSES = tuple(new_classes)
            self.PALETTE = new_palette
            
            # Also update classes/palette in kwargs if provided
            if 'classes' in kwargs:
                kwargs['classes'] = new_classes
            if 'palette' in kwargs:
                kwargs['palette'] = new_palette
                
            print(f'Applied custom label map. New classes: {new_classes}')
            print(f'Label mapping: {self.label_map}')
        
        super(WildscenesDataset, self).__init__(**kwargs)

    def _get_idx_map(self, custom_label_map):
        """Get index mapping from original to new classes."""
        new_classes = self._get_new_labels(custom_label_map)
        
        # Filter out unlabelled mappings
        filtered_map = {
            k: v for k, v in custom_label_map.items()
            if k != "unlabelled" and v != "unlabelled"
        }
        
        # Create index mapping
        orig_classes = list(self.__class__.CLASSES)
        idx_map = {}
        
        for orig_name, new_name in filtered_map.items():
            if orig_name in orig_classes and new_name in new_classes:
                orig_idx = orig_classes.index(orig_name)
                new_idx = new_classes.index(new_name)
                idx_map[orig_idx] = new_idx
        
        # Map unmapped classes to ignore_index
        for i, cls in enumerate(orig_classes):
            if i not in idx_map:
                idx_map[i] = 255  # ignore_index
                
        return idx_map

    def _get_new_labels(self, custom_label_map):
        """Get the new class labels after mapping."""
        new_classes = list(set(custom_label_map.values()))
        
        # Remove unlabelled class
        if "unlabelled" in new_classes:
            new_classes.remove("unlabelled")
            
        # Sort alphabetically for consistency
        return sorted(new_classes)

    def _get_updated_palette(self, custom_label_map):
        """Update palette for new classes."""
        rev_map = _get_reversed_custom_label_map(custom_label_map)
        orig_palette = list(self.__class__.PALETTE)
        orig_classes = list(self.__class__.CLASSES)
        new_classes = self._get_new_labels(custom_label_map)
        
        new_palette = []
        for cls in new_classes:
            if cls in orig_classes:
                # Use original palette for existing classes
                idx = orig_classes.index(cls)
                new_palette.append(orig_palette[idx])
            else:
                # Use palette from one of the mapped classes
                mapped_cls = rev_map[cls][-1]
                idx = orig_classes.index(mapped_cls)
                new_palette.append(orig_palette[idx])
                
        return new_palette

    def pre_pipeline(self, results):
        """Pre-pipeline to apply label remapping."""
        super().pre_pipeline(results)
        
        # Apply label remapping if custom mapping exists
        if hasattr(self, 'label_map'):
            results['label_map'] = self.label_map

    def __getitem__(self, idx):
        """Get training/test data after applying transforms."""
        results = super().__getitem__(idx)
        
        # Apply label remapping to segmentation mask
        if hasattr(self, 'label_map') and 'gt_semantic_seg' in results:
            seg = results['gt_semantic_seg']
            # Apply label mapping
            new_seg = seg.copy()
            for old_label, new_label in self.label_map.items():
                new_seg[seg == old_label] = new_label
            results['gt_semantic_seg'] = new_seg
            
        return results 