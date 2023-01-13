# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

"""C4 streaming dataset conversion scripts."""
import gzip
import json
import os
import platform
from argparse import ArgumentParser, Namespace
from typing import Dict, Iterable

import requests
from streaming import MDSWriter
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm

_N_SHARDS_PER_SPLIT = {
    'en': {
        'train': 1024,
        'validation': 8
    },
    'realnewslike': {
        'train': 512,
        'validation': 1
    },
    'en.noblocklist': {
        'train': 1024,
        'validation': 8
    },
    'en.noclean': {
        'train': 7168,
        'validation': 64
    },
}


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    args = ArgumentParser()
    args.add_argument('--out_root', type=str, required=True)
    args.add_argument('--compression', type=str, default=None)
    args.add_argument('--splits',
                      nargs='+',
                      default=['train', 'train-small', 'val'])

    return args.parse_args()


def get_allenai_c4_samples(name, split, shard, n_shards):
    url = f'https://huggingface.co/datasets/allenai/c4/resolve/1ddc917116b730e1859edef32896ec5c16be51d0/{name}/c4-{split}.{shard:05d}-of-{n_shards:05d}.json.gz'
    print(url)
    data = gzip.decompress(requests.get(url).content).decode('utf-8').strip()

    samples = []
    for line in data.split('\n'):
        samples.append(json.loads(line))
    return samples


class ShardedC4(IterableDataset):

    def __init__(self, name, split):
        self.name = name
        self.split = split
        self.n_shards = _N_SHARDS_PER_SPLIT[self.name][self.split]

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            assert self.n_shards % num_workers == 0
        else:
            num_workers = 1
            worker_id = 0

        worker_shards = list(range(self.n_shards))[worker_id::num_workers]
        for shard in worker_shards:
            shard_samples = get_allenai_c4_samples(self.name, self.split, shard,
                                                   self.n_shards)
            for sample in shard_samples:
                yield sample


def generate_samples(dataset: IterableDataset,
                     expected_num_samples: int) -> Iterable[Dict[str, bytes]]:
    """Generator over each dataset sample.

    Args:
        samples (IterableDataset): An iterable dataset that is multi-worker compatible

    Yields:
        Sample dicts.
    """
    # Multiple workers is only supported on linux machines
    if 'linux' in platform.platform().lower():
        num_workers = min(64, dataset.n_shards)  # type: ignore
    else:
        num_workers = 0

    batch_size = 512
    # If using multiple workers, configure each worker to prefetch as many samples as it can, up to the aggregate device batch size
    # If not using workers, the torch DataLoader expects the default value for prefetch_factor, which non-intuitively must be 2.
    prefetch_factor = max(1, 2 * batch_size //
                          num_workers) if num_workers > 0 else 2

    loader = DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            yield {
                key: batch_values[idx].encode('utf-8')
                for key, batch_values in batch.items()
            }
            n_samples += 1
            if n_samples == expected_num_samples:
                return


def main(args: Namespace) -> None:
    """Main: create C4 streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """
    columns = {'text': 'str', 'timestamp': 'str', 'url': 'str'}

    for (split, split_new_name, expected_num_samples) in [
        ('train', 'train', 364868892),
        ('train', 'train-small', 327680),
        ('validation', 'val', 364608),
    ]:
        # Only generate the splits requested
        if split_new_name not in args.splits:
            continue

        # Get samples
        dataset = ShardedC4(name='en', split=split)
        samples = generate_samples(dataset=dataset,
                                   expected_num_samples=expected_num_samples)

        # Write samples
        with MDSWriter(dirname=os.path.join(args.out_root, split_new_name),
                       columns=columns,
                       compression=args.compression) as out:
            for sample in tqdm(samples,
                               desc=split_new_name,
                               total=expected_num_samples):
                out.write(sample)


if __name__ == '__main__':
    main(parse_args())
