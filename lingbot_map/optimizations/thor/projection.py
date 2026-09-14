"""Independent projection caches and combined KV append route validation."""


import os
import types
import weakref
from dataclasses import dataclass

import torch
import torch.nn.functional as F


CANDIDATE022_ENV_FLAG = "LINGBOT_THOR_FA4_QUERY_STAGING"
PROJECTION_SHADOW_ROUTE = "projection_shadow"
DIRECT_KV_APPEND_ROUTE = "direct_kv_append"
PROJECTION_SHADOW_DIRECT_KV_APPEND_ROUTE = (
    "projection_shadow_direct_kv_append"
)


@dataclass(frozen=True)
class Candidate021ProjectionStats:
    modules_patched: int
    buffers_registered: int
    frame_modules: int
    patch_modules: int
    gca_modules: int


def candidate021_runtime_route() -> str | None:
    from .options import housekeeping_route
    return housekeeping_route()


def candidate021_projection_shadow_enabled() -> bool:
    route = candidate021_runtime_route()
    return route in (
        PROJECTION_SHADOW_ROUTE,
        PROJECTION_SHADOW_DIRECT_KV_APPEND_ROUTE,
    )


def validate_candidate021_accepted_stack_env() -> None:
    """Validate public dependencies without enforcing historical experiment order."""
    from .options import ThorOptions
    ThorOptions.from_env().validate()


def _set_nonpersistent_buffer(module, name: str, value: torch.Tensor | None) -> bool:
    if name in module._buffers:
        raise RuntimeError(
            f"Candidate 021 projection attribute collision: buffer {name!r} already exists"
        )
    module.register_buffer(name, value, persistent=False)
    return True


def _refresh_projection_shadows(attn) -> int:
    proj = getattr(attn, "proj", None)
    if not isinstance(proj, torch.nn.Linear):
        raise RuntimeError("Candidate 021 projection route requires torch.nn.Linear proj")
    weight = proj.weight
    bias = proj.bias
    errors = []
    if weight.dtype != torch.float32:
        errors.append(f"weight.dtype={weight.dtype}, expected torch.float32")
    if bias is None or bias.dtype != torch.float32:
        errors.append(f"bias.dtype={None if bias is None else bias.dtype}, expected torch.float32")
    if weight.device.type != "cuda" or (bias is not None and bias.device != weight.device):
        errors.append(f"weight.device={weight.device}, bias.device={None if bias is None else bias.device}")
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        errors.append(f"weight.shape={tuple(weight.shape)}, expected square 2-D projection")
    if errors:
        raise RuntimeError("Candidate 021 projection contract rejected: " + "; ".join(errors))

    buffers_registered = 0
    for source, name in (
        (weight, "proj_weight_bf16"),
        (bias, "proj_bias_bf16"),
    ):
        shadow = source.detach().to(torch.bfloat16).contiguous()
        if _set_nonpersistent_buffer(attn, name, shadow):
            buffers_registered += 1
    attn._candidate021_projection_contract = {
        "weight_id": id(weight),
        "bias_id": id(bias),
        "weight_version": int(weight._version),
        "bias_version": int(bias._version),
        "device": str(weight.device),
        "in_features": int(proj.in_features),
        "out_features": int(proj.out_features),
    }
    return buffers_registered


def _validate_projection_parameters(attn) -> None:
    proj = attn.proj
    weight = proj.weight
    bias = proj.bias
    errors = []
    if weight.dtype != torch.float32:
        errors.append(f"weight.dtype={weight.dtype}, expected torch.float32")
    if bias is None or bias.dtype != torch.float32:
        errors.append(f"bias.dtype={None if bias is None else bias.dtype}, expected torch.float32")
    if weight.device.type != "cuda" or (bias is not None and bias.device != weight.device):
        errors.append(f"weight.device={weight.device}, bias.device={None if bias is None else bias.device}")
    if tuple(weight.shape) != (1024, 1024):
        errors.append(f"weight.shape={tuple(weight.shape)}, expected (1024, 1024)")
    if errors:
        raise RuntimeError("Candidate 021 projection contract rejected: " + "; ".join(errors))


def _candidate021_projection_forward(proj, x):
    if torch.is_grad_enabled():
        raise RuntimeError("Candidate 021 projection route is inference-only")
    attn = proj._candidate021_attn_owner_ref()
    if attn is None:
        raise RuntimeError("Candidate 021 projection attention owner was released")
    errors = []
    if x.device.type != "cuda" or x.device != attn.proj_weight_bf16.device:
        errors.append(f"x.device={x.device}, shadow.device={attn.proj_weight_bf16.device}")
    if x.dtype != torch.bfloat16:
        errors.append(f"x.dtype={x.dtype}, expected torch.bfloat16")
    if x.ndim != 3:
        errors.append(f"x.ndim={x.ndim}, expected 3")
    else:
        if x.shape[0] not in (1, 8):
            errors.append(f"batch={x.shape[0]}, expected streaming 1 or scale 8")
        if x.shape[-1] != 1024:
            errors.append(f"x.shape[-1]={x.shape[-1]}, expected 1024")
    if errors:
        raise RuntimeError("Candidate 021 projection call rejected: " + "; ".join(errors))
    return F.linear(x, attn.proj_weight_bf16, attn.proj_bias_bf16)


def _iter_patch_blocks(model):
    patch_embed = getattr(getattr(model, "aggregator", None), "patch_embed", None)
    blocks = getattr(patch_embed, "blocks", None)
    if blocks is None:
        return []
    flattened = []
    for block in blocks:
        if hasattr(block, "attn"):
            flattened.append(block)
        elif isinstance(block, torch.nn.ModuleList):
            flattened.extend(child for child in block if hasattr(child, "attn"))
    return flattened


def _patch_scope(blocks) -> tuple[int, int]:
    modules_patched = 0
    buffers_registered = 0
    for block in blocks or ():
        attn = getattr(block, "attn", None)
        proj = getattr(attn, "proj", None)
        if attn is None or proj is None:
            continue
        buffers_registered += _refresh_projection_shadows(attn)
        object.__setattr__(proj, "_candidate021_attn_owner_ref", weakref.ref(attn))
        proj.forward = types.MethodType(_candidate021_projection_forward, proj)
        attn._candidate021_projection_shadow = True
        block._candidate021_projection_shadow = True
        modules_patched += 1
    return modules_patched, buffers_registered


def _validated_scope(name: str, blocks) -> list:
    blocks = list(blocks or ())
    if len(blocks) != 24:
        raise RuntimeError(
            f"Candidate 021 projection contract rejected: {name} has "
            f"{len(blocks)} attention blocks, expected 24"
        )
    for index, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        proj = getattr(attn, "proj", None)
        if not isinstance(proj, torch.nn.Linear):
            raise RuntimeError(
                "Candidate 021 projection contract rejected: "
                f"{name}[{index}].attn.proj is not torch.nn.Linear"
            )
        if proj.in_features != 1024 or proj.out_features != 1024:
            raise RuntimeError(
                "Candidate 021 projection contract rejected: "
                f"{name}[{index}] projection is "
                f"{proj.in_features}x{proj.out_features}, expected 1024x1024"
            )
        collisions = [
            attribute
            for attribute in (
                "proj_weight_bf16",
                "proj_bias_bf16",
                "_candidate021_projection_contract",
                "_candidate021_projection_shadow",
            )
            if hasattr(attn, attribute)
        ]
        collisions.extend(
            attribute
            for attribute in (
                "_candidate021_attn_owner_ref",
                "_candidate021_projection_forward",
            )
            if hasattr(proj, attribute)
        )
        if "forward" in proj.__dict__:
            collisions.append("forward")
        if hasattr(block, "_candidate021_projection_shadow"):
            collisions.append("block._candidate021_projection_shadow")
        if collisions:
            raise RuntimeError(
                "Candidate 021 projection contract rejected: "
                f"{name}[{index}] has attribute collisions {collisions}"
            )
    return blocks


def validate_candidate021_projection_shadow(model) -> None:
    """Validate shadow/source ownership without refreshing or reallocating buffers."""
    agg = getattr(model, "aggregator", None)
    if agg is None:
        raise RuntimeError("Candidate 021 projection route requires model.aggregator")
    scopes = (
        ("frame", list(getattr(agg, "frame_blocks", None) or ())),
        ("patch", _iter_patch_blocks(model)),
        ("gca", list(getattr(agg, "global_blocks", None) or ())),
    )
    errors = []
    for scope_name, blocks in scopes:
        if len(blocks) != 24:
            errors.append(f"{scope_name} count={len(blocks)}, expected 24")
            continue
        for index, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            proj = getattr(attn, "proj", None)
            contract = getattr(attn, "_candidate021_projection_contract", None)
            owner_ref = getattr(proj, "_candidate021_attn_owner_ref", None)
            if not isinstance(contract, dict):
                errors.append(f"{scope_name}[{index}] missing source contract")
                continue
            if not callable(owner_ref) or owner_ref() is not attn:
                errors.append(f"{scope_name}[{index}] owner reference mismatch")
            if getattr(proj.forward, "__func__", None) is not _candidate021_projection_forward:
                errors.append(f"{scope_name}[{index}] forward ownership mismatch")
            weight = proj.weight
            bias = proj.bias
            if (
                contract.get("weight_id") != id(weight)
                or contract.get("bias_id") != id(bias)
                or contract.get("weight_version") != int(weight._version)
                or contract.get("bias_version") != int(bias._version)
                or contract.get("device") != str(weight.device)
            ):
                errors.append(f"{scope_name}[{index}] source parameter changed")
            for source, shadow_name in (
                (weight, "proj_weight_bf16"),
                (bias, "proj_bias_bf16"),
            ):
                shadow = getattr(attn, shadow_name, None)
                if (
                    shadow is None
                    or shadow.dtype != torch.bfloat16
                    or shadow.device != source.device
                    or not shadow.is_contiguous()
                    or tuple(shadow.shape) != tuple(source.shape)
                    or shadow_name not in attn._non_persistent_buffers_set
                ):
                    errors.append(f"{scope_name}[{index}] invalid {shadow_name}")
            if not getattr(attn, "_candidate021_projection_shadow", False):
                errors.append(f"{scope_name}[{index}] missing attention marker")
            if not getattr(block, "_candidate021_projection_shadow", False):
                errors.append(f"{scope_name}[{index}] missing block marker")
    if errors:
        raise RuntimeError("Candidate 021 projection validation failed: " + "; ".join(errors))


def validate_candidate021_runtime_route(model) -> None:
    """Attest both legs of an enabled housekeeping route before compile/capture."""
    route = candidate021_runtime_route()
    if route is None:
        return
    validate_candidate021_accepted_stack_env()
    projection_enabled = route in (
        PROJECTION_SHADOW_ROUTE,
        PROJECTION_SHADOW_DIRECT_KV_APPEND_ROUTE,
    )
    direct_append_enabled = route in (
        DIRECT_KV_APPEND_ROUTE,
        PROJECTION_SHADOW_DIRECT_KV_APPEND_ROUTE,
    )
    if projection_enabled:
        validate_candidate021_projection_shadow(model)

    manager = getattr(getattr(model, "aggregator", None), "kv_cache_manager", None)
    if manager is None:
        raise RuntimeError("Candidate 021 runtime route requires a KV cache manager")
    metadata = getattr(manager, "candidate021_metadata", None)
    errors = []
    if getattr(manager, "candidate021_route", None) != route:
        errors.append(
            f"manager.route={getattr(manager, 'candidate021_route', None)!r}, "
            f"expected {route!r}"
        )
    if not isinstance(metadata, dict):
        errors.append("manager metadata is missing")
    else:
        if metadata.get("enabled") is not True or metadata.get("route") != route:
            errors.append("manager metadata route mismatch")
        if route in (
            PROJECTION_SHADOW_ROUTE,
            DIRECT_KV_APPEND_ROUTE,
            PROJECTION_SHADOW_DIRECT_KV_APPEND_ROUTE,
        ):
            candidate022 = metadata.get("candidate022_fa4_b15")
            candidate022_enabled = (
                isinstance(candidate022, dict)
                and candidate022.get("enabled") is True
            )
            candidate022_requested_value = os.environ.get(
                CANDIDATE022_ENV_FLAG, ""
            )
            if candidate022_requested_value not in ("", "0", "1"):
                errors.append(
                    f"{CANDIDATE022_ENV_FLAG} has invalid value "
                    f"{candidate022_requested_value!r}"
                )
            candidate022_requested = candidate022_requested_value == "1"
            if candidate022_requested is not candidate022_enabled:
                errors.append("Candidate 022 environment/manager mismatch")
            expected_kernel_route = candidate022_enabled
            expected_attention_impl = (
                "public_varlen_beta15_qstage2"
                if candidate022_enabled
                else "public_varlen"
            )
            if metadata.get("kernel_route_applied") is not expected_kernel_route:
                errors.append("attention-kernel activation mismatch")
            if metadata.get("attention_impl") != expected_attention_impl:
                errors.append(
                    f"attention implementation={metadata.get('attention_impl')!r}, "
                    f"expected {expected_attention_impl!r}"
                )
            if candidate022 is not None and not candidate022_enabled:
                errors.append("Candidate 022 metadata is malformed")
            if candidate022_enabled:
                if candidate022.get("version") != "4.0.0b15":
                    errors.append("Candidate 022 FA4 version mismatch")
                if candidate022.get("optimization") != "sm110_blackwell_q_stage_2":
                    errors.append("Candidate 022 optimization marker mismatch")
                if candidate022.get("process_route_immutable") is not True:
                    errors.append("Candidate 022 process isolation marker is missing")
                if metadata.get("fa4_version") != "4.0.0b15":
                    errors.append("manager FA4 version does not match Candidate 022")
        if metadata.get("kv_append_route_applied") is not direct_append_enabled:
            errors.append("direct-append activation mismatch")
        if direct_append_enabled:
            if metadata.get("kv_append_impl") != "combined_flashinfer_append":
                errors.append("direct-append implementation mismatch")
            if not isinstance(metadata.get("direct_kv_append_runtime"), dict):
                errors.append("direct-append runtime attestation is missing")
            contract = metadata.get("direct_kv_append_contract")
            if not isinstance(contract, dict):
                errors.append("direct-append manager contract is missing")
            elif contract.get("special_page_policy") != "ascending_from_max_patch_pages":
                errors.append("eager/graph special-page allocation policy mismatch")
        elif (
            metadata.get("direct_kv_append_runtime") is not None
            or metadata.get("direct_kv_append_contract") is not None
        ):
            errors.append("direct-append metadata leaked into a non-direct route")
    if direct_append_enabled:
        manager_validator = getattr(
            manager, "_validate_candidate021_direct_manager_buffers", None
        )
        if not callable(manager_validator):
            errors.append("direct-append manager validator is unavailable")
        else:
            manager_validator()
    if errors:
        raise RuntimeError(
            "Candidate 021 runtime route validation failed: " + "; ".join(errors)
        )


def apply_candidate021_projection_shadow(model) -> Candidate021ProjectionStats:
    """Hoist frame, patch, and GCA projection casts into replay-stable buffers."""
    if model.training:
        raise RuntimeError("Candidate 021 projection route is eval-only")
    agg = getattr(model, "aggregator", None)
    if agg is None:
        raise RuntimeError("Candidate 021 projection route requires model.aggregator")

    frame_blocks = _validated_scope("frame", getattr(agg, "frame_blocks", None))
    patch_blocks = _validated_scope("patch", _iter_patch_blocks(model))
    gca_blocks = _validated_scope("gca", getattr(agg, "global_blocks", None))

    first_weight = frame_blocks[0].attn.proj.weight
    if not torch.cuda.is_available() or first_weight.device.type != "cuda":
        raise RuntimeError("Candidate 021 projection route requires CUDA parameters")
    compute_capability = torch.cuda.get_device_capability(first_weight.device)
    if compute_capability != (11, 0):
        raise RuntimeError(
            "Candidate 021 projection contract rejected: "
            f"compute capability={compute_capability}, expected (11, 0)"
        )

    # Validate every scope before mutating any module, so a rejected topology
    # cannot leave a partially enabled Candidate 021 model behind.
    for scope_name, blocks in (
        ("frame", frame_blocks),
        ("patch", patch_blocks),
        ("gca", gca_blocks),
    ):
        for index, block in enumerate(blocks):
            try:
                _validate_projection_parameters(block.attn)
            except RuntimeError as exc:
                raise RuntimeError(f"{scope_name}[{index}]: {exc}") from exc

    frame_modules, frame_buffers = _patch_scope(frame_blocks)
    patch_modules, patch_buffers = _patch_scope(patch_blocks)
    gca_modules, gca_buffers = _patch_scope(gca_blocks)
    return Candidate021ProjectionStats(
        modules_patched=frame_modules + patch_modules + gca_modules,
        buffers_registered=frame_buffers + patch_buffers + gca_buffers,
        frame_modules=frame_modules,
        patch_modules=patch_modules,
        gca_modules=gca_modules,
    )
