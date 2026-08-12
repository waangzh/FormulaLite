#!/usr/bin/env python3
"""Inspect a local FormulaLite manifest without downloading or materializing a dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from formulalite.data.policy import FormulaDataPolicy, OverlengthPolicy, UnknownTokenPolicy
from formulalite.data.streaming import ManifestStreamingDataset
from formulalite.data.tokenizer import FormulaTokenizer


@dataclass
class ImageDimensionStatistics:
    count: int = 0
    width_sum: int = 0
    height_sum: int = 0
    min_width: int | None = None
    max_width: int | None = None
    min_height: int | None = None
    max_height: int | None = None

    def observe(self, width: int, height: int) -> None:
        self.count += 1
        self.width_sum += width
        self.height_sum += height
        self.min_width = width if self.min_width is None else min(self.min_width, width)
        self.max_width = width if self.max_width is None else max(self.max_width, width)
        self.min_height = height if self.min_height is None else min(self.min_height, height)
        self.max_height = height if self.max_height is None else max(self.max_height, height)

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "min_width": self.min_width,
            "max_width": self.max_width,
            "mean_width": self.width_sum / self.count if self.count else None,
            "min_height": self.min_height,
            "max_height": self.max_height,
            "mean_height": self.height_sum / self.count if self.count else None,
        }


def inspect_manifest(
    manifest: str | Path,
    *,
    max_sequence_length: int,
    unknown_policy: UnknownTokenPolicy = UnknownTokenPolicy.KEEP,
    overlength_policy: OverlengthPolicy = OverlengthPolicy.DROP,
) -> dict[str, Any]:
    tokenizer = FormulaTokenizer.build()
    policy = FormulaDataPolicy(
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        unknown_policy=unknown_policy,
        overlength_policy=overlength_policy,
    )
    dataset = ManifestStreamingDataset(manifest, policy=policy)
    dimensions = ImageDimensionStatistics()
    decoded_subset_distribution: Counter[str] = Counter()
    for sample in dataset:
        width, height = sample.image.size  # type: ignore[union-attr]
        dimensions.observe(width, height)
        decoded_subset_distribution[sample.subset or "unspecified"] += 1
    report = dataset.statistics.to_report()
    report["accepted_samples"] = dimensions.count
    report["accepted_subset_distribution"] = dict(sorted(decoded_subset_distribution.items()))
    report["image_dimensions"] = dimensions.to_dict()
    report["policies"] = {
        "unknown": unknown_policy.value,
        "overlength": overlength_policy.value,
        "max_sequence_length": max_sequence_length,
    }
    return report


def _summary(report: dict[str, Any]) -> str:
    lengths = report["sequence_length"]
    return "\n".join(
        (
            f"Dataset: {report['dataset_source']}",
            f"Samples: {report['total_samples']} total, {report['accepted_samples']} accepted",
            f"Subsets: {json.dumps(report['subset_distribution'], sort_keys=True)}",
            "Unknown: "
            f"{report['unknown_samples']} samples / {report['unknown_token_count']} tokens "
            f"({report['unknown_sample_rate']:.2%} of samples)",
            f"Lengths: mean={lengths['mean']:.2f}, max={lengths['max']}, "
            f"p50={lengths['p50']}, p90={lengths['p90']}, p95={lengths['p95']}, "
            f"p99={lengths['p99']}, overlength={lengths['overlength_sample_count']}",
            f"Image dimensions: {json.dumps(report['image_dimensions'], sort_keys=True)}",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-sequence-length", type=int, default=1027)
    parser.add_argument(
        "--unknown-policy",
        choices=[item.value for item in UnknownTokenPolicy],
        default="keep",
    )
    parser.add_argument(
        "--overlength-policy",
        choices=[item.value for item in OverlengthPolicy],
        default="drop",
    )
    parser.add_argument("--output", type=Path, help="write the complete JSON report")
    parser.add_argument(
        "--json", action="store_true", help="print JSON instead of the human summary"
    )
    args = parser.parse_args()
    report = inspect_manifest(
        args.manifest,
        max_sequence_length=args.max_sequence_length,
        unknown_policy=UnknownTokenPolicy(args.unknown_policy),
        overlength_policy=OverlengthPolicy(args.overlength_policy),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    rendered = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        if args.json
        else _summary(report)
    )
    print(rendered)


if __name__ == "__main__":
    main()
