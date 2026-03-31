"""HDF5 dataset export utilities for RGB, depth, and segmentation data."""

import csv
import json
from pathlib import Path

import numpy as np

try:
    import h5py  # type: ignore[import-not-found]
except ImportError:
    h5py = None


class SegmentationDatasetExporter:
    """Exports synchronized modalities into one HDF5 file per frame."""

    def __init__(self, dataset_root="dataset", split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}):
        """Initialize exporter and folder structure."""
        if h5py is None:
            raise ImportError("h5py is required for HDF5 export. Please install h5py.")

        self.dataset_root = Path(dataset_root)
        self.split_ratios = split_ratios
        self.frame_count = 0
        self.compression = "gzip"
        self.compression_level = 4

        for split in split_ratios.keys():
            (self.dataset_root / "hdf5" / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_root / "metadata" / split).mkdir(parents=True, exist_ok=True)

        self.manifest_file = self.dataset_root / "manifest.csv"
        self._init_manifest()

    def _init_manifest(self):
        """Initialize manifest CSV with headers."""
        if not self.manifest_file.exists():
            with open(self.manifest_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'frame_id', 'split', 'h5_path', 'metadata_path', 'num_instances', 'timestamp'
                ])

    def _get_split(self, frame_idx):
        """Determine which split a frame belongs to."""
        ratios = self.split_ratios
        train_ratio = ratios.get('train', 0.7)
        val_ratio = ratios.get('val', 0.15)

        rand_val = (frame_idx % 100) / 100.0

        if rand_val < train_ratio:
            return 'train'
        elif rand_val < train_ratio + val_ratio:
            return 'val'
        else:
            return 'test'

    def save_frame(
        self,
        frame_id,
        rgb_image,
        seg_mask=None,
        segmentation_rgb=None,
        depth_map=None,
        metadata=None,
        timestamp=None,
    ):
        """Save one frame as a single HDF5 file with optional modalities."""
        split = self._get_split(frame_id)
        frame_name = f"frame_{frame_id:06d}"

        rgb_arr = np.asarray(rgb_image, dtype=np.uint8)
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
            raise ValueError("rgb_image must have shape [H, W, 3]")

        seg_mask_arr = None if seg_mask is None else np.asarray(seg_mask, dtype=np.uint32)
        seg_rgb_arr = None if segmentation_rgb is None else np.asarray(segmentation_rgb, dtype=np.uint8)
        depth_arr = None if depth_map is None else np.asarray(depth_map, dtype=np.float32)

        h5_path = self.dataset_root / "hdf5" / split / f"{frame_name}.h5"
        with h5py.File(h5_path, 'w') as h5f:
            h5f.attrs['frame_id'] = int(frame_id)
            h5f.attrs['split'] = split
            h5f.attrs['timestamp'] = str(timestamp or "")

            h5f.create_dataset(
                "rgb_scene",
                data=rgb_arr,
                compression=self.compression,
                compression_opts=self.compression_level,
            )

            if depth_arr is not None:
                h5f.create_dataset(
                    "depth_planar_m",
                    data=depth_arr,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                )

            if seg_rgb_arr is not None:
                h5f.create_dataset(
                    "segmentation_rgb",
                    data=seg_rgb_arr,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                )

            if seg_mask_arr is not None:
                h5f.create_dataset(
                    "segmentation_instance_id",
                    data=seg_mask_arr,
                    compression=self.compression,
                    compression_opts=self.compression_level,
                )

            h5f.attrs['metadata_json'] = json.dumps(metadata or {})

        metadata_path = self.dataset_root / "metadata" / split / f"{frame_name}.json"
        num_instances = 0
        if seg_mask_arr is not None:
            unique_ids = np.unique(seg_mask_arr)
            num_instances = int(np.sum(unique_ids != 0))

        frame_metadata = {
            'frame_id': frame_id,
            'timestamp': timestamp or "",
            'num_instances': num_instances,
            'h5_path': str(h5_path.relative_to(self.dataset_root)),
            'objects': metadata or {}
        }
        with open(metadata_path, 'w') as f:
            json.dump(frame_metadata, f, indent=2)

        with open(self.manifest_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_id,
                split,
                str(h5_path.relative_to(self.dataset_root)),
                str(metadata_path.relative_to(self.dataset_root)),
                frame_metadata['num_instances'],
                timestamp or ""
            ])

        self.frame_count += 1

    def save_camera_info(self, camera_matrix, img_shape, camera_info_dict=None):
        """Save camera intrinsics and image shape."""
        camera_file = self.dataset_root / "camera_info.json"
        camera_data = {
            'K': camera_matrix.tolist() if isinstance(camera_matrix, np.ndarray) else camera_matrix,
            'image_height': int(img_shape[0]),
            'image_width': int(img_shape[1]),
        }
        if camera_info_dict:
            camera_data.update(camera_info_dict)

        with open(camera_file, 'w') as f:
            json.dump(camera_data, f, indent=2)

    def create_dataset_yaml(self, class_names=None, num_classes=None):
        """Create a lightweight dataset summary yaml."""
        yaml_file = self.dataset_root / "dataset.yaml"

        yaml_content = f"""
# Dataset configuration
path: {str(self.dataset_root)}
train_h5: hdf5/train
val_h5: hdf5/val
test_h5: hdf5/test
nc: {num_classes or 1}
"""

        if class_names:
            yaml_content += f"\nnames:\n"
            for idx, name in enumerate(class_names):
                yaml_content += f"  {idx}: {name}\n"

        with open(yaml_file, 'w') as f:
            f.write(yaml_content)

    def get_dataset_stats(self):
        """Return simple split/frame counts."""
        stats = {
            'total_frames': self.frame_count,
            'splits': {}
        }

        for split in self.split_ratios.keys():
            h5_dir = self.dataset_root / "hdf5" / split
            h5_count = len(list(h5_dir.glob("*.h5")))
            stats['splits'][split] = h5_count

        return stats


def save_yolo_format_labels(dataset_root, frame_id, detections, image_shape):
    """
    Save detections in YOLO format (.txt file with normalized coords).
    
    Args:
        dataset_root (Path): Dataset root directory.
        frame_id (int): Frame ID.
        detections (list): List of dicts with 'bbox', 'class_name'.
        image_shape (tuple): (height, width).
    """
    split = _get_split_for_frame(frame_id)
    labels_dir = Path(dataset_root) / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    label_file = labels_dir / f"frame_{frame_id:06d}.txt"
    
    h, w = image_shape
    
    with open(label_file, 'w') as f:
        for det in detections:
            bbox = det.get('bbox')
            class_name = det.get('class_name', 'unknown')
            class_id = det.get('class_id', 0)
            
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = bbox
            
            # Normalize to [0, 1]
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            
            # Clamp to valid range
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))
            
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def _get_split_for_frame(frame_idx):
    """Helper to get split for a frame."""
    rand_val = (frame_idx % 100) / 100.0
    if rand_val < 0.7:
        return 'train'
    elif rand_val < 0.85:
        return 'val'
    else:
        return 'test'
