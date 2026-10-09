"""MegaLoc descriptor frontend and retrieval-pair generation."""

import logging
import pprint
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from natsort import natsorted
from tqdm import tqdm

from vidmap.frontend.cache import (
    cache_metadata,
    inspect_incremental_items,
    mark_incremental_cache_complete,
    prepare_incremental_cache,
)
from vidmap.frontend.image_dataset import ImageDataset, ImageDatasetOptions
from vidmap.frontend.models.megaloc import MegaLocDescriptorModel, megaloc_cache_identity
from vidmap.utils.device import empty_device_cache
from vidmap.utils.logging import progress_bars_enabled

RETRIEVAL_PAIR_SELECTION_POLICY_VERSION = 1

logger = logging.getLogger(__name__)


def _retrieval_config() -> dict:
    return {
        "model": {},
        "preprocessing": {
            "resize_max": 1024,
            "resize_force": True,
        },
    }


def retrieval_cache_identity(image_list, image_content_fingerprint):
    retrieval_conf = _retrieval_config()
    return cache_metadata(
        stage="retrieval_features",
        config={
            "model": megaloc_cache_identity(),
            "preprocessing": retrieval_conf["preprocessing"],
        },
        ordered_inputs={
            "images": image_list,
            "image_content": image_content_fingerprint,
        },
        payload_format="per-image-global-descriptors",
    )


@torch.no_grad()
def compute_retrieval_features(
    scene_parser,
    retrieval_features_path,
    image_list,
    cache_identity,
    overwrite,
    *,
    device: torch.device,
):
    """
    Compute retrieval features for image matching.

    Args:
        scene_parser: Scene parser with rgb_dir
        retrieval_features_path: Retrieval descriptor artifact
        image_list: List of image names to process
        cache_identity: Cache fingerprint
        overwrite: Overwrite existing cache
        device: Target execution device

    """
    retrieval_conf = _retrieval_config()
    if cache_identity is None:
        raise TypeError("cache_identity is required")
    identity = cache_identity

    logger.debug("Retrieval feature configuration:\n%s", pprint.pformat(retrieval_conf))

    dataset = ImageDataset(
        scene_parser.rgb_dir,
        ImageDatasetOptions(**retrieval_conf["preprocessing"]),
        image_list,
    )
    retrieval_features_path = Path(retrieval_features_path)
    retrieval_features_path.parent.mkdir(parents=True, exist_ok=True)
    expected_names = list(dataset.names)
    prepare_incremental_cache(retrieval_features_path, identity, overwrite=overwrite)
    present_names, _ = inspect_incremental_items(
        retrieval_features_path,
        expected_names,
        identity,
        repair_malformed=True,
    )
    skip_names = set(present_names)

    dataset.names = [name for name in dataset.names if name not in skip_names]
    if len(dataset.names) == 0:
        logger.info("Skipping retrieval frontend because every item is cached")
        mark_incremental_cache_complete(retrieval_features_path, identity, expected_names)
        empty_device_cache(device)
        logger.info("Retrieval features available at %s", retrieval_features_path)
        return

    model = MegaLocDescriptorModel().eval().to(device)
    loader = torch.utils.data.DataLoader(dataset, num_workers=1, shuffle=False, pin_memory=(device.type == "cuda"))
    for idx, data in enumerate(tqdm(loader, disable=not progress_bars_enabled())):
        name = dataset.names[idx]
        pred = model({"image": data["image"].to(device, non_blocking=True)})
        pred = {key: value[0].cpu().numpy() for key, value in pred.items()}
        pred["image_size"] = data["original_size"][0].numpy()

        for key in pred:
            if pred[key].dtype == np.float32:
                pred[key] = pred[key].astype(np.float16)

        with h5py.File(str(retrieval_features_path), "a", libver="latest") as fd:
            try:
                if name in fd:
                    del fd[name]
                group = fd.create_group(name)
                for key, value in pred.items():
                    group.create_dataset(key, data=value)
            except OSError as error:
                if "No space left on device" in error.args[0]:
                    logger.error("Out of disk space while storing retrieval descriptors")
                    del group, fd[name]
                raise error

        del pred

    mark_incremental_cache_complete(retrieval_features_path, identity, expected_names)
    del model

    empty_device_cache(device)

    logger.info("Retrieval features saved to %s", retrieval_features_path)


def _load_descriptors(names, hfile):
    """Load one ordered descriptor set."""
    descriptors = np.stack([hfile[name]["global_descriptor"][()] for name in names])
    return torch.as_tensor(descriptors, dtype=torch.float)


def _excluded_pair_indices(sequence, excluded_pairs):
    """Symmetric (row, col) sequence indices of excluded name pairs, sorted by row."""
    seq_index = {name: idx for idx, name in enumerate(sequence)}
    rows, cols = [], []
    for pair in excluded_pairs:
        if len(pair) != 2:
            continue
        name0, name1 = pair
        if name0 in seq_index and name1 in seq_index:
            rows += [seq_index[name0], seq_index[name1]]
            cols += [seq_index[name1], seq_index[name0]]
    rows, cols = np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    return rows[order], cols[order]


def _retrieve_pairs(
    path,
    sequence,
    excluded_pairs,
    num_matched,
    min_score,
    return_scores,
    *,
    device: torch.device,
):
    """Top-``num_matched`` retrieval candidates per image among the non-excluded, non-self pairs.

    Scores the descriptor matrix in row chunks; candidate order and tie-breaking match a per-image
    stable sort over the remaining images in sequence order.
    """
    with h5py.File(str(path), "r", libver="latest") as hfile:
        names = [name for name in sequence if name in hfile and "global_descriptor" in hfile[name]]
        if not names:
            return []
        descriptors = _load_descriptors(names, hfile).to(device)
    rows, cols = _excluded_pair_indices(names, excluded_pairs)
    count = min(num_matched, len(names))
    chunk = max(1, 2**25 // len(names))
    pairs = []
    for start in range(0, len(names), chunk):
        stop = min(start + chunk, len(names))
        scores = descriptors[start:stop] @ descriptors.T
        invalid = scores < min_score
        local = torch.arange(stop - start, device=device)
        invalid[local, local + start] = True
        lo, hi = np.searchsorted(rows, [start, stop])
        if hi > lo:
            invalid[
                torch.as_tensor(rows[lo:hi] - start, device=device),
                torch.as_tensor(cols[lo:hi], device=device),
            ] = True
        scores.masked_fill_(invalid, float("-inf"))
        indices = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :count]
        values = torch.gather(scores, 1, indices)
        valid = values.isfinite().cpu().numpy()
        indices, values = indices.cpu().numpy(), values.cpu().numpy()
        for i, j in zip(*np.where(valid)):
            reference, query = names[start + i], names[indices[i, j]]
            pairs.append((reference, query, float(values[i, j])) if return_scores else (reference, query))
    return pairs


def generate_retrieval_pairs(
    sequence,
    tcorr,
    sequential_pairs,
    retrieval_path,
    tcorr_min_matches,
    retrieval_min_score,
    nquery,
    lc_pair_nms=False,
    lc_pair_nms_radius=2,
    *,
    device: torch.device,
):
    """Generate retrieval pairs excluding sequential and sufficiently tracked pairs."""
    if not retrieval_path.exists():
        raise ValueError("Retrieval features not found. Run compute_retrieval_features step first.")

    excluded_pairs = {frozenset(pair) for pair in sequential_pairs}
    excluded_pairs |= {frozenset(pair) for pair, matches in tcorr.items() if len(matches) > tcorr_min_matches}
    images = set(sequence)
    num_excluded = sum(1 for pair in excluded_pairs if len(pair) == 2 and pair <= images)
    if num_excluded == len(images) * (len(images) - 1) // 2:
        logger.info("No untracked pairs found for retrieval")
        return []

    retrieval_pairs = _retrieve_pairs(
        retrieval_path,
        sequence,
        excluded_pairs,
        nquery,
        retrieval_min_score,
        lc_pair_nms,
        device=device,
    )
    if lc_pair_nms:
        raw_count = len(retrieval_pairs)
        retrieval_pairs = endpoint_nms_retrieval_pairs(retrieval_pairs, sequence, radius=lc_pair_nms_radius)
        logger.info(
            "LC pair endpoint NMS radius=%d: %d -> %d pairs",
            lc_pair_nms_radius,
            raw_count,
            len(retrieval_pairs),
        )
    else:
        retrieval_pairs = natsorted(tuple(natsorted(pair)) for pair in {frozenset(pair) for pair in retrieval_pairs})

    logger.info("Generated %d retrieval pairs, excluding sequential pairs", len(retrieval_pairs))
    return retrieval_pairs


def endpoint_nms_retrieval_pairs(scored_pairs, sequence, radius):
    """Keep the best retrieval edge per local query/target neighborhood."""
    if radius < 0:
        raise ValueError(f"lc_pair_nms_radius must be non-negative, got {radius}")

    seq_index = {name: idx for idx, name in enumerate(sequence)}
    best_by_pair = {}
    for name0, name1, score in scored_pairs:
        if name0 not in seq_index or name1 not in seq_index or name0 == name1:
            continue
        idx0 = seq_index[name0]
        idx1 = seq_index[name1]
        if idx0 <= idx1:
            pair = (name0, name1)
            endpoints = (idx0, idx1)
        else:
            pair = (name1, name0)
            endpoints = (idx1, idx0)
        if pair not in best_by_pair or score > best_by_pair[pair][0]:
            best_by_pair[pair] = (score, endpoints)

    candidates = sorted(
        ((score, endpoints, pair) for pair, (score, endpoints) in best_by_pair.items()),
        key=lambda item: (-item[0], item[2][0], item[2][1]),
    )
    # Accepted endpoints bucketed into (radius + 1)-sized cells: anything within the radius
    # lies in the 3x3 neighboring cells, so each candidate checks a few entries instead of all.
    cell = radius + 1
    accepted = []
    accepted_cells = defaultdict(list)
    for _score, (idx0, idx1), pair in candidates:
        cell0, cell1 = idx0 // cell, idx1 // cell
        suppress = any(
            abs(idx0 - prev0) <= radius and abs(idx1 - prev1) <= radius
            for d0 in (-1, 0, 1)
            for d1 in (-1, 0, 1)
            for prev0, prev1 in accepted_cells.get((cell0 + d0, cell1 + d1), ())
        )
        if suppress:
            continue
        accepted.append(pair)
        accepted_cells[(cell0, cell1)].append((idx0, idx1))
    return natsorted(accepted)
