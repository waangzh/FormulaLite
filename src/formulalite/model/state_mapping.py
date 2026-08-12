"""Strict checkpoint mapping for FormulaLite models."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch

_AUTO_PREFIXES = ("module.encoder.", "model.encoder.", "encoder.")
_MODEL_AUTO_PREFIXES = ("module.model.", "module.", "model.")


class StateMappingError(ValueError):
    """Raised when a checkpoint cannot map exactly onto a target state."""


def _select_prefix(
    source_keys: set[str],
    source_prefix: str | None,
    prefixes: tuple[str, ...],
    *,
    require_all: bool,
) -> str:
    if source_prefix is not None:
        return source_prefix
    for prefix in prefixes:
        matches = (key.startswith(prefix) for key in source_keys)
        if (all(matches) if require_all else any(matches)):
            return prefix
    return ""


def _map_exact_state_dict(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    source_prefix: str | None,
    prefixes: tuple[str, ...],
    component: str,
    require_all_prefix: bool,
) -> OrderedDict[str, torch.Tensor]:
    prefix = _select_prefix(
        set(source_state), source_prefix, prefixes, require_all=require_all_prefix
    )
    selected: dict[str, torch.Tensor] = {}
    for source_name, value in source_state.items():
        if prefix and not source_name.startswith(prefix):
            continue
        target_name = source_name[len(prefix) :] if prefix else source_name
        if target_name in selected:
            raise StateMappingError(f"duplicate mapped key: {target_name}")
        selected[target_name] = value

    if not selected:
        raise StateMappingError(f"no {component} tensors found with source prefix {prefix!r}")

    target_keys = set(target_state)
    selected_keys = set(selected)
    missing = sorted(target_keys - selected_keys)
    unexpected = sorted(selected_keys - target_keys)
    if missing or unexpected:
        raise StateMappingError(
            f"state key mismatch: missing={missing}, unexpected={unexpected}"
        )

    shape_mismatches: list[str] = []
    for name, target_value in target_state.items():
        source_value = selected[name]
        if not isinstance(source_value, torch.Tensor):
            raise StateMappingError(f"source value is not a tensor: {name}")
        if source_value.shape != target_value.shape:
            shape_mismatches.append(
                f"{name}: source={tuple(source_value.shape)}, "
                f"target={tuple(target_value.shape)}"
            )
    if shape_mismatches:
        raise StateMappingError("state shape mismatch: " + "; ".join(shape_mismatches))

    return OrderedDict((name, selected[name]) for name in target_state)


def map_encoder_state_dict(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    source_prefix: str | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Strip an encoder prefix and require exact names, shapes, and coverage."""

    return _map_exact_state_dict(
        source_state,
        target_state,
        source_prefix=source_prefix,
        prefixes=_AUTO_PREFIXES,
        component="encoder",
        require_all_prefix=False,
    )


def map_checkpoint_state_dict(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    *,
    source_prefix: str | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Require exact names, shapes, and coverage for a complete model checkpoint."""

    return _map_exact_state_dict(
        source_state,
        target_state,
        source_prefix=source_prefix,
        prefixes=_MODEL_AUTO_PREFIXES,
        component="model",
        require_all_prefix=True,
    )
