# ESPDet 🚀 AGPL-3.0 License
import random
from ultralytics.utils import LOCAL_RANK, TQDM
from ultralytics.data import YOLODataset
from pathlib import Path
import numpy as np
from ultralytics.data.utils import (
    HELP_URL,
    LOGGER,
    get_hash,
    img2label_paths,
    load_dataset_cache_file,
)


DATASET_CACHE_VERSION = "1.0.3"


class YOLOPosNegDataset(YOLODataset):
    """
    Dataset class for loading object detection labels in YOLO format.

    This class supports loading both positive and negative data for object detection using the YOLO format.

    https://blog.csdn.net/qq_40387714/article/details/138996317
    """

    def __init__(self, *args, mode="train", **kwargs):
        """
        Initializes the YOLOPosNegDataset
        """
        self.im_pos_index = []
        self.im_neg_index = []
        self.im_pos_num = 0
        self.im_neg_num = 0
        self.im_neg_path = ""
        self.im_neg_files = []

        super(YOLOPosNegDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Returns transformed label information for given index."""
        if "train" in self.prefix.lower():
            if self.im_pos_num * self.data["negative_setting"]["neg_ratio"] >= self.im_neg_num:
                self.im_neg_num += 1
                index = random.choice(self.im_neg_index)
            else:
                self.im_pos_num += 1
                index = random.choice(self.im_pos_index)
        return self.transforms(self.get_image_and_label(index))

    def __len__(self):
        try:
            if "train" in self.prefix.lower() and self.data["negative_setting"]["fix_dataset_length"] > 0:
                return int(self.data["negative_setting"]["fix_dataset_length"])
        except (ValueError, KeyError, AttributeError) as e:
            print(f"INFO: Failed to set epoch length, adopt raw length：Error: {e}")

        return len(self.labels)

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        try:
            if "train" in self.prefix.lower() and self.data["negative_setting"]["use_extra_neg"]:
                self.im_neg_path = self.data["negative_setting"]["extra_neg_sources"]
                for imp, imn in self.im_neg_path.items():
                    imp_neg_file = self.get_img_files(imp)
                    imn_real = min(len(imp_neg_file), imn)

                    print(f'INFO: extra negative samples:[{imp}], [{len(imp_neg_file)}] images, sample[{imn}]images,'
                          f'sample [{imn_real}] images in fact.')
                    imp_neg_file = random.sample(imp_neg_file, imn_real)
                    self.im_neg_files += imp_neg_file

                print(f"INFO: [{len(self.im_neg_files)}] negative images accessed in total.")

        except (ValueError, KeyError, AttributeError) as e:
            print(f"INFO: Failed to add negative samples. Config [negative_setting] error: {e}")
            print(f"INFO:[{len(self.im_neg_files)}] negative images accessed in total.")

        self.im_files += self.im_neg_files
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache["version"] == DATASET_CACHE_VERSION  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            print(f"")
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        if not labels:
            LOGGER.warning(f"WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}")
        self.im_files = [lb["im_file"] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb["cls"]), len(lb["bboxes"]), len(lb["segments"])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f"WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, "
                f"len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. "
                "To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset."
            )
            for lb in labels:
                lb["segments"] = []
        if len_cls == 0:
            LOGGER.warning(f"WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}")

        if "train" in self.prefix.lower():
            for i, label in enumerate(labels):
                if len(label['cls']) == 0:
                    self.im_neg_index.append(i)
                else:
                    self.im_pos_index.append(i)

        return labels


class YOLOWeightedDataset(YOLODataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    This class supports loading weighted data for object detection using the YOLO format.

    https://y-t-g.github.io/tutorials/yolo-class-balancing/
    """
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """

        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # You can also specify weights manually instead
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts

        # Aggregation function
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
                weights.append(1)
                continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        # Don't use for validation
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))

