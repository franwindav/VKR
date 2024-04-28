import re
import os
import numpy as np
import random
import scipy
import json
import copy
import sys
import pickle
import string
import logging
from functools import lru_cache

import torch
import matplotlib.pyplot as plt

from data_utils import (
    get_emg_features,
    FeatureNormalizer,
    TextTransform,
)

from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_list("remove_channels", [], "channels to remove")
flags.DEFINE_list(
    "silent_data_directories",
    ["./emg_data/silent_parallel_data"],
    "silent data locations",
)
flags.DEFINE_list(
    "voiced_data_directories",
    ["./emg_data/voiced_parallel_data", "./emg_data/nonparallel_data"],
    "voiced data locations",
)
flags.DEFINE_string(
    "testset_file", "testset_largedev.json", "file with testset indices"
)
flags.DEFINE_string(
    "text_align_directory", "text_alignments", "directory with alignment files"
)


def remove_drift(signal, fs):
    b, a = scipy.signal.butter(3, 2, "highpass", fs=fs)
    return scipy.signal.filtfilt(b, a, signal)


def notch(signal, freq, sample_frequency):
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)


def notch_harmonics(signal, freq, sample_frequency):
    for harmonic in range(1, 8):
        signal = notch(signal, freq * harmonic, sample_frequency)
    return signal


def subsample(signal, new_freq, old_freq):
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    result = np.interp(sample_times, times, signal)
    return result


def apply_to_all(function, signal_array, *args, **kwargs):
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:, i], *args, **kwargs))
    return np.stack(results, 1)


def load_utterance(
    base_dir, index, debug=False
):
    index = int(index)
    raw_emg = np.load(os.path.join(base_dir, f"{index}_emg.npy"))

    before = os.path.join(base_dir, f"{index-1}_emg.npy")
    after = os.path.join(base_dir, f"{index+1}_emg.npy")
    if os.path.exists(before):
        raw_emg_before = np.load(before)
    else:
        raw_emg_before = np.zeros([0, raw_emg.shape[1]])
    if os.path.exists(after):
        raw_emg_after = np.load(after)
    else:
        raw_emg_after = np.zeros([0, raw_emg.shape[1]])

    x = np.concatenate([raw_emg_before, raw_emg, raw_emg_after], 0)
    x = apply_to_all(notch_harmonics, x, 60, 1000)
    x = apply_to_all(remove_drift, x, 1000)
    x = x[raw_emg_before.shape[0] : x.shape[0] - raw_emg_after.shape[0], :]
    emg_orig = apply_to_all(subsample, x, 689.06, 1000)

    x = apply_to_all(subsample, x, 516.79, 1000)
    emg = x

    for c in FLAGS.remove_channels:
        emg[:, int(c)] = 0
        emg_orig[:, int(c)] = 0

    emg_features = get_emg_features(emg)

    
    if debug:
        plt.subplot(2,1,1)
        plt.title("До оброботки")
        plt.plot(raw_emg[:,0])
        plt.subplot(2,1,2)
        plt.title("После оброботки")
        plt.plot(x[:,0])
        plt.show()

    with open(os.path.join(base_dir, f"{index}_info.json")) as f:
        info = json.load(f)

    return (
        emg_features,
        info["text"],
        emg_orig.astype(np.float32),
    )


class EMGDirectory(object):
    def __init__(self, session_index, directory, silent, exclude_from_testset=False):
        self.session_index = session_index
        self.directory = directory
        self.silent = silent
        self.exclude_from_testset = exclude_from_testset

    def __lt__(self, other):
        return self.session_index < other.session_index

    def __repr__(self):
        return self.directory


class SizeAwareSampler(torch.utils.data.Sampler):
    def __init__(self, emg_dataset, max_len):
        self.dataset = emg_dataset
        self.max_len = max_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        batch = []
        batch_length = 0
        for idx in indices:
            directory_info, file_idx = self.dataset.example_indices[idx]
            with open(
                os.path.join(directory_info.directory, f"{file_idx}_info.json")
            ) as f:
                info = json.load(f)
            if not np.any([l in string.ascii_letters for l in info["text"]]):
                continue
            length = sum([emg_len for emg_len, _, _ in info["chunks"]])
            if length > self.max_len:
                logging.warning(
                    f"Warning: example {idx} cannot fit within desired batch length"
                )
            if length + batch_length > self.max_len:
                yield batch
                batch = []
                batch_length = 0
            batch.append(idx)
            batch_length += length


class EMGDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dir=None,
        limit_length=False,
        dev=False,
        test=False,
        no_testset=False,
        no_normalizers=False,
    ):
        self.text_align_directory = FLAGS.text_align_directory

        if no_testset:
            devset = []
            testset = []
        else:
            with open(FLAGS.testset_file) as f:
                testset_json = json.load(f)
                devset = testset_json["dev"]
                testset = testset_json["test"]

        directories = []
        if base_dir is not None:
            directories.append(EMGDirectory(0, base_dir, False))
        else:
            for sd in FLAGS.silent_data_directories:
                for session_dir in sorted(os.listdir(sd)):
                    directories.append(
                        EMGDirectory(
                            len(directories), os.path.join(sd, session_dir), True
                        )
                    )

            has_silent = len(FLAGS.silent_data_directories) > 0
            for vd in FLAGS.voiced_data_directories:
                for session_dir in sorted(os.listdir(vd)):
                    directories.append(
                        EMGDirectory(
                            len(directories),
                            os.path.join(vd, session_dir),
                            False,
                            exclude_from_testset=has_silent,
                        )
                    )

        self.example_indices = []
        for directory_info in directories:
            for fname in os.listdir(directory_info.directory):
                m = re.match(r"(\d+)_info.json", fname)
                if m is not None:
                    idx_str = m.group(1)
                    with open(os.path.join(directory_info.directory, fname)) as f:
                        info = json.load(f)
                        if (
                            info["sentence_index"] >= 0
                        ):  # boundary clips of silence are marked -1
                            location_in_testset = [
                                info["book"],
                                info["sentence_index"],
                            ] in testset
                            location_in_devset = [
                                info["book"],
                                info["sentence_index"],
                            ] in devset
                            if (
                                (
                                    test
                                    and location_in_testset
                                    and not directory_info.exclude_from_testset
                                )
                                or (
                                    dev
                                    and location_in_devset
                                    and not directory_info.exclude_from_testset
                                )
                                or (
                                    not test
                                    and not dev
                                    and not location_in_testset
                                    and not location_in_devset
                                )
                            ):
                                self.example_indices.append(
                                    (directory_info, int(idx_str))
                                )

        self.example_indices.sort()
        random.seed(0)
        random.shuffle(self.example_indices)

        self.no_normalizers = no_normalizers
        if not self.no_normalizers:
            self.emg_norm = pickle.load(
                open(FLAGS.normalizers_file, "rb")
            )

        sample_emg, _, _ = load_utterance(
            self.example_indices[0][0].directory, self.example_indices[0][1]
        )
        self.num_features = sample_emg.shape[1]
        self.limit_length = limit_length
        self.num_sessions = len(directories)

        self.text_transform = TextTransform()

    def silent_subset(self):
        result = copy(self)
        silent_indices = []
        for example in self.example_indices:
            if example[0].silent:
                silent_indices.append(example)
        result.example_indices = silent_indices
        return result

    def subset(self, fraction):
        result = copy(self)
        result.example_indices = self.example_indices[
            : int(fraction * len(self.example_indices))
        ]
        return result

    def __len__(self):
        return len(self.example_indices)

    @lru_cache(maxsize=None)
    def __getitem__(self, i):
        directory_info, idx = self.example_indices[i]
        emg, text, raw_emg = load_utterance(
            directory_info.directory,
            idx,
        )
        raw_emg = raw_emg / 20
        raw_emg = 50 * np.tanh(raw_emg / 50.0)

        if not self.no_normalizers:
            emg = self.emg_norm.normalize(emg)
            emg = 8 * np.tanh(emg / 8.0)

        text_int = np.array(self.text_transform.text_to_int(text), dtype=np.int64)

        result = {
            "emg": torch.from_numpy(emg),
            "text": text,
            "text_int": torch.from_numpy(text_int),
            "raw_emg": torch.from_numpy(raw_emg),
        }

        return result

    @staticmethod
    def collate_raw(batch):
        emg = [ex["emg"] for ex in batch]
        raw_emg = [ex["raw_emg"] for ex in batch]
        lengths = [ex["emg"].shape[0] for ex in batch]
        text_ints = [ex["text_int"] for ex in batch]
        text_lengths = [ex["text_int"].shape[0] for ex in batch]

        result = {
            "emg": emg,
            "raw_emg": raw_emg,
            "lengths": lengths,
            "text_int": text_ints,
            "text_int_lengths": text_lengths,
        }
        return result

def make_normalizers():
    dataset = EMGDataset(no_normalizers=True)
    emg_samples = []
    for d in dataset:
    # d[0]
        emg_samples.append(d['emg'])
        if len(emg_samples) > 50:
            break
    emg_norm = FeatureNormalizer(emg_samples, share_scale=False)
    pickle.dump(emg_norm, open(FLAGS.normalizers_file, 'wb'))

if __name__ == "__main__":
    FLAGS(sys.argv)
    d = EMGDataset()
    make_normalizers()
