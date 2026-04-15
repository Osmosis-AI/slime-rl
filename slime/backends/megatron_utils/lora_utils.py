"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import json
import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from safetensors.torch import save_file


logger = logging.getLogger(__name__)

LORA_ADAPTER_NAME = "slime_lora"

# ---------------------------------------------------------------------------
# Unified HF <-> Megatron module name mappings
# ---------------------------------------------------------------------------

# Standard LoRA: merged Q/K/V and merged up/gate
_STANDARD_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_qkv",
    "k_proj": "linear_qkv",
    "v_proj": "linear_qkv",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
}

_STANDARD_LORA_ALL_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

# CanonicalLoRA: Split Q/K/V and up/gate
_CANONICAL_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1_gate",
    "up_proj": "linear_fc1_up",
    "down_proj": "linear_fc2",
}

_CANONICAL_LORA_ALL_MODULES = [
    "linear_q",
    "linear_k",
    "linear_v",
    "linear_proj",
    "linear_fc1_up",
    "linear_fc1_gate",
    "linear_fc2",
]

# Megatron -> HF (inverse mapping, one-to-many)
# Covers both standard LoRA (merged) and CanonicalLoRA (split) module names.
_MEGATRON_TO_HF_MODULES = {
    # Standard LoRA (merged layers)
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    # CanonicalLoRA (split layers)
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_gate": ["gate_proj"],
    "linear_fc1_up": ["up_proj"],
}

_DEFAULT_HF_TARGET_MODULES = list(_STANDARD_LORA_HF_TO_MEGATRON)
_HF_MODULE_NAMES = set(_DEFAULT_HF_TARGET_MODULES)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def is_lora_enabled(args: Namespace) -> bool:
    """Check if LoRA is enabled based on arguments."""
    return getattr(args, "lora_rank", 0) > 0 or getattr(args, "lora_adapter_path", None) is not None


def is_lora_model(model: Sequence[torch.nn.Module]) -> bool:
    """Check if model has LoRA layers applied."""
    for model_chunk in model:
        if hasattr(model_chunk.module, "peft_config"):
            return True
        for name, _ in model_chunk.named_parameters():
            if "lora_" in name or "adapter" in name:
                return True
    return False


def is_lora_weight_name(name: str) -> bool:
    """Check if a weight name corresponds to a LoRA adapter weight."""
    return ".lora_A." in name or ".lora_B." in name


def _is_adapter_param_name(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA adapter (Megatron internal naming)."""
    return "lora_" in name or (".adapter." in name and ("linear_in" in name or "linear_out" in name))


def _materialize_tensor_for_safetensors(tensor: torch.Tensor) -> torch.Tensor:
    """Return an independent CPU tensor suitable for safetensors serialization."""
    # Megatron-Bridge can export logical adapter tensors as views into shared storage
    # for fused projections (for example gate_proj/up_proj). safetensors rejects shared
    # storage for dict saves, so each key needs its own contiguous CPU buffer.
    return tensor.detach().to(device="cpu", copy=True).contiguous()


def _dedupe_preserve_order(modules: Sequence[str]) -> list[str]:
    """Return modules without duplicates while preserving their original order."""
    deduped: list[str] = []
    for module in modules:
        if module not in deduped:
            deduped.append(module)
    return deduped


def _resolve_base_model_name_or_path(args: Namespace) -> str | None:
    """Resolve the most portable base model identifier available for adapter metadata."""
    explicit = getattr(args, "base_model_name_or_path", None)
    if explicit:
        return explicit

    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if hf_checkpoint is None:
        return None

    checkpoint_path = Path(hf_checkpoint)
    if not checkpoint_path.is_dir():
        return hf_checkpoint

    config_path = checkpoint_path / "config.json"
    if config_path.exists():
        try:
            config_json = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            config_json = {}

        for key in ("base_model_name_or_path", "_name_or_path", "name_or_path", "model_name_or_path"):
            value = config_json.get(key)
            if isinstance(value, str) and value and value != hf_checkpoint:
                return value

    return hf_checkpoint


def _read_hf_config_json(args: Namespace) -> dict[str, Any]:
    """Best-effort load of the HF config JSON for the current checkpoint."""
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if hf_checkpoint is None:
        return {}

    checkpoint_path = Path(hf_checkpoint)
    if not checkpoint_path.is_dir():
        return {}

    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        return {}

    try:
        return json.loads(config_path.read_text())
    except json.JSONDecodeError:
        return {}


def _read_hf_text_config_json(args: Namespace) -> dict[str, Any]:
    """Return the text_config block if present, otherwise the top-level config."""
    config_json = _read_hf_config_json(args)
    text_config = config_json.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return config_json


def _resolve_modules_arg_to_hf(
    modules: str | Sequence[str] | None,
    *,
    lora_type: type | object | None,
) -> list[str] | None:
    """Resolve an arbitrary module list/string to canonical HF module names."""
    normalized_modules = _normalize_modules_arg(modules)
    if normalized_modules is None:
        return None

    megatron_modules = convert_target_modules_to_megatron(normalized_modules, lora_type=lora_type)
    return _dedupe_preserve_order(convert_target_modules_to_hf(megatron_modules))


def _resolve_modules_arg_to_megatron(
    modules: str | Sequence[str] | None,
    *,
    lora_type: type | object | None,
) -> list[str] | None:
    """Resolve an arbitrary module list/string to Megatron module names."""
    normalized_modules = _normalize_modules_arg(modules)
    if normalized_modules is None:
        return None

    return convert_target_modules_to_megatron(normalized_modules, lora_type=lora_type)


# ---------------------------------------------------------------------------
# Module name conversion
# ---------------------------------------------------------------------------


def _get_lora_class_name(lora_type: type | object | None) -> str:
    """Resolve LoRA type to its class name string."""
    if lora_type is None:
        return "CanonicalLoRA"
    if isinstance(lora_type, str):
        return "CanonicalLoRA" if lora_type.lower() == "canonical_lora" else "LoRA"
    if isinstance(lora_type, type):
        return lora_type.__name__
    return type(lora_type).__name__


def convert_target_modules_to_megatron(
    hf_modules: str | list[str],
    lora_type: type | object | None = None,
) -> list[str]:
    """Convert HuggingFace LoRA target module names to Megatron format.

    HF:  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    Megatron (LoRA):          linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron (CanonicalLoRA): linear_q, linear_k, linear_v, linear_proj,
                              linear_fc1_up, linear_fc1_gate, linear_fc2

    Special values: "all", "all-linear", "all_linear" -> all standard linear modules.
    If input is already in Megatron format, returns as-is.
    """
    class_name = _get_lora_class_name(lora_type)
    is_canonical = class_name == "CanonicalLoRA"

    all_modules = _CANONICAL_LORA_ALL_MODULES if is_canonical else _STANDARD_LORA_ALL_MODULES
    hf_to_megatron = _CANONICAL_LORA_HF_TO_MEGATRON if is_canonical else _STANDARD_LORA_HF_TO_MEGATRON

    # Handle special "all-linear" variants
    if isinstance(hf_modules, str):
        if hf_modules in ("all", "all-linear", "all_linear"):
            return list(all_modules)
        hf_modules = [hf_modules]
    elif isinstance(hf_modules, list) and len(hf_modules) == 1:
        if hf_modules[0] in ("all", "all-linear", "all_linear"):
            return list(all_modules)

    # Check if already in Megatron format
    if all(m not in _HF_MODULE_NAMES for m in hf_modules if "*" not in m):
        return hf_modules

    # Convert HF names to Megatron names (dedup while preserving order)
    megatron_modules: list[str] = []
    for module in hf_modules:
        megatron_name = hf_to_megatron.get(module, module)
        if megatron_name not in megatron_modules:
            megatron_modules.append(megatron_name)

    return megatron_modules


def convert_target_modules_to_hf(megatron_modules: list[str]) -> list[str]:
    """Convert Megatron LoRA target module names to HuggingFace format.

    Supports both standard LoRA and CanonicalLoRA module names.

    Megatron standard:   linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron canonical:  linear_q, linear_k, linear_v, linear_proj,
                         linear_fc1_up, linear_fc1_gate, linear_fc2
    HF:                  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    """
    hf_modules: list[str] = []
    for module in megatron_modules:
        if module in _MEGATRON_TO_HF_MODULES:
            hf_modules.extend(_MEGATRON_TO_HF_MODULES[module])
        else:
            hf_modules.append(module)
    return hf_modules


# ---------------------------------------------------------------------------
# Model setup helpers (used by model.py)
# ---------------------------------------------------------------------------


def create_lora_instance(args: Namespace):
    """Create a LoRA or CanonicalLoRA instance based on args.

    Returns:
        A LoRA/CanonicalLoRA dataclass instance ready to be applied to a model.
    """
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.lora import LoRA

    lora_type_name = getattr(args, "lora_type", "lora").lower()

    if lora_type_name == "canonical_lora":
        lora_cls = CanonicalLoRA
    else:
        lora_cls = LoRA

    target_modules = _resolve_modules_arg_to_megatron(args.target_modules, lora_type=lora_cls)
    exclude_modules = _resolve_modules_arg_to_megatron(getattr(args, "exclude_modules", None), lora_type=lora_cls) or []

    lora = lora_cls(
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        lora_A_init_method=getattr(args, "lora_A_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )

    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}, target_modules={target_modules}, "
        f"exclude_modules={exclude_modules}"
    )
    return lora


def _normalize_modules_arg(modules: str | Sequence[str] | None) -> list[str] | None:
    """Normalize a module-list argument to a clean list without changing semantics."""
    if modules is None:
        return None
    if isinstance(modules, str):
        normalized = [module.strip() for module in modules.split(",") if module.strip()]
        return normalized or None
    normalized = [str(module).strip() for module in modules if str(module).strip()]
    return normalized or None


def resolve_target_modules_to_hf(args: Namespace) -> list[str]:
    """Resolve args.target_modules to canonical HF module names for saved/runtime configs."""
    target_modules = _resolve_modules_arg_to_hf(
        getattr(args, "target_modules", None),
        lora_type=getattr(args, "lora_type", None),
    )
    if target_modules is None:
        return list(_DEFAULT_HF_TARGET_MODULES)
    return target_modules


def resolve_target_modules_from_exported_weights(weight_names: Sequence[str]) -> list[str]:
    """Infer canonical HF target modules from exported LoRA tensor names."""
    discovered: list[str] = []
    for module_name in _DEFAULT_HF_TARGET_MODULES:
        needle = f".{module_name}."
        if any(needle in weight_name for weight_name in weight_names):
            discovered.append(module_name)
    return discovered or list(_DEFAULT_HF_TARGET_MODULES)


def resolve_exclude_modules_to_hf(args: Namespace) -> list[str] | None:
    """Resolve args.exclude_modules to canonical HF module names for PEFT metadata."""
    return _resolve_modules_arg_to_hf(
        getattr(args, "exclude_modules", None),
        lora_type=getattr(args, "lora_type", None),
    )


def build_lora_adapter_config(
    args: Namespace,
    *,
    target_modules: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build the shared PEFT-compatible adapter config for save and live sync."""
    base_model_name_or_path = _resolve_base_model_name_or_path(args)
    target_modules = list(target_modules) if target_modules is not None else resolve_target_modules_to_hf(args)
    exclude_modules = resolve_exclude_modules_to_hf(args)
    revision = getattr(args, "revision", None)

    config = {
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "peft_type": "LORA",
        "r": args.lora_rank,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }

    if exclude_modules is not None:
        config["exclude_modules"] = exclude_modules
    if revision is not None:
        config["revision"] = revision

    return config


def _should_drop_from_external_lora_export(args: Namespace, hf_name: str) -> bool:
    """Return True when a weight should be omitted from the external HF adapter export.

    Stock upstream SGLang currently does not correctly serve Qwen3.5 MoE LoRA
    weights on the ``shared_expert`` path. Keep the Megatron-native resume
    checkpoint intact, but omit those tensors from the external HF adapter so
    the saved artifact remains directly loadable by unmodified SGLang.
    """
    text_config = _read_hf_text_config_json(args)
    model_type = text_config.get("model_type")
    if model_type != "qwen3_5_moe_text":
        return False

    # The external adapter is a text-generation LoRA artifact. Vision-only LoRA
    # tensors are ignored by SGLang's text path and do not match the saved PEFT
    # metadata, so omit them from the exported inference adapter.
    if hf_name.startswith("model.visual."):
        return True

    return ".mlp.shared_expert." in hf_name or ".mlp.shared_expert_gate." in hf_name


def iter_exported_lora_weights(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    *,
    bridge: Any | None = None,
    cpu: bool,
    quantization_config: dict[str, Any] | None = None,
):
    """Yield LoRA weights in the same HF-facing form used by SGLang weight sync."""
    from megatron.bridge import AutoBridge

    from slime.utils import megatron_bridge_utils

    from .megatron_to_hf import postprocess_hf_param
    from .megatron_to_hf.processors import quantize_params

    if bridge is None:
        import slime_plugins.megatron_bridge  # noqa: F401

        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)

    with megatron_bridge_utils.patch_megatron_model(model):
        for hf_name, weight, megatron_name in bridge.export_adapter_weights(
            model,
            cpu=cpu,
            show_progress=False,
        ):
            processed_weight = postprocess_hf_param(
                args=args,
                megatron_param_name=megatron_name,
                hf_param_name=hf_name,
                param=weight,
            )
            yield from quantize_params(
                args=args,
                megatron_name=megatron_name,
                converted_named_params=[(hf_name, processed_weight)],
                quantization_config=quantization_config,
            )


# ---------------------------------------------------------------------------
# Checkpoint save/load
# ---------------------------------------------------------------------------


def save_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
) -> str:
    """Save LoRA adapter checkpoint to disk.

    Saves in two formats:
    1. **HF PEFT format** (``adapter_model.bin`` + ``adapter_config.json``) for
       external tool compatibility. Uses Megatron-Bridge's ``export_adapter_weights``
       which correctly handles fused QKV / gate-up weight splitting and TP gathering.
    2. **Megatron-native format** (``adapter_megatron_tp{tp}_pp{pp}.pt``) for fast
       checkpoint resume without name/weight conversion. Each TP/PP rank saves its
       own shard with original parameter names.

    When ``optimizer`` is provided, training state (optimizer + LR scheduler) is
    also saved per-rank for checkpoint resume. Base model weights are frozen and
    never change, so they are not saved.

    This function is collective: **all ranks must call it** because the bridge
    export performs TP all-gather internally. Only ``dp_rank == 0`` writes files.
    """
    save_path = Path(save_dir)
    is_dp_rank_0 = mpu.get_data_parallel_rank() == 0
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    # Create directory on dp_rank=0, then synchronize
    if is_dp_rank_0:
        save_path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # ---- Megatron-native format (per TP/PP rank, fast resume) ----
    if is_dp_rank_0:
        adapter_state: dict[str, torch.Tensor] = {}
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if _is_adapter_param_name(name):
                    adapter_state[name] = param.data.cpu()

        native_path = save_path / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
        torch.save(adapter_state, native_path)
        logger.info(f"Saved {len(adapter_state)} adapter tensors (native) to {native_path}")

    # ---- HF PEFT format (uses the same export path as live SGLang sync) ----
    lora_state_dict: dict[str, torch.Tensor] = {}
    dropped_hf_names: list[str] = []
    for hf_name, weight in iter_exported_lora_weights(
        args,
        model,
        cpu=True,
    ):
        if _should_drop_from_external_lora_export(args, hf_name):
            dropped_hf_names.append(hf_name)
            continue
        lora_state_dict[hf_name] = weight

    # Only one rank writes the HF PEFT files (bridge already gathered across TP)
    if is_dp_rank_0 and tp_rank == 0:
        safetensors_state_dict = {
            name: _materialize_tensor_for_safetensors(weight) for name, weight in lora_state_dict.items()
        }
        torch.save(lora_state_dict, save_path / "adapter_model.bin")
        save_file(safetensors_state_dict, save_path / "adapter_model.safetensors")

        exported_target_modules = resolve_target_modules_from_exported_weights(lora_state_dict.keys())
        with open(save_path / "adapter_config.json", "w") as f:
            json.dump(build_lora_adapter_config(args, target_modules=exported_target_modules), f, indent=2)

        os.sync()
        if dropped_hf_names:
            logger.warning(
                "Dropped %d unsupported HF adapter tensors from external export for stock SGLang compatibility: %s",
                len(dropped_hf_names),
                ", ".join(dropped_hf_names[:8]) + (" ..." if len(dropped_hf_names) > 8 else ""),
            )
        logger.info(f"Saved HF PEFT adapter to {save_path} with {len(lora_state_dict)} tensors")

    # ---- Training state (optimizer + scheduler) for resume ----
    if optimizer is not None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.save(
            {
                "iteration": iteration,
                "optimizer": optimizer.state_dict(),
                "opt_param_scheduler": opt_param_scheduler.state_dict() if opt_param_scheduler else None,
            },
            save_path / f"training_state_rank{rank}.pt",
        )
        logger.info(f"Saved optimizer/scheduler state to {save_path}")

    if dist.is_initialized():
        dist.barrier()

    return str(save_path)


def load_lora_adapter(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
) -> tuple[bool, int | None]:
    """Load LoRA adapter weights from a saved checkpoint into the model.

    Attempts to load from Megatron-native format first (per-rank ``.pt`` files),
    which preserves the exact TP/PP sharding and requires no name conversion.
    Falls back to HF PEFT ``adapter_model.bin`` if native files are not found
    (not yet implemented for HF PEFT format).

    When ``optimizer`` is provided, also restores training state (optimizer +
    LR scheduler) from a co-located ``training_state_rank*.pt`` file.

    Args:
        model: List of DDP-wrapped model chunks with LoRA layers already applied.
        adapter_path: Path to the adapter checkpoint directory.
        optimizer: If provided, restore optimizer state for training resume.
        opt_param_scheduler: If provided, restore LR scheduler state.

    Returns:
        ``(loaded, iteration)`` -- *loaded* is True if adapter weights were
        successfully loaded; *iteration* is the saved iteration number (or None
        if no training state was found).
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logger.warning(f"LoRA adapter path does not exist: {adapter_dir}")
        return False, None

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    # ---- Try Megatron-native format first (fast, no conversion needed) ----
    native_path = adapter_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
    if native_path.exists():
        state_dict = torch.load(native_path, map_location="cpu", weights_only=True)
        loaded = 0
        for model_chunk in model:
            for name, param in model_chunk.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name].to(device=param.device))
                    loaded += 1
        logger.info(f"Loaded {loaded} adapter tensors from Megatron-native checkpoint: {native_path}")

        iteration = _load_training_state(adapter_dir, optimizer, opt_param_scheduler)
        return True, iteration

    # ---- HF PEFT format (future work) ----
    hf_path = adapter_dir / "adapter_model.bin"
    if hf_path.exists():
        logger.warning(
            f"Found HF PEFT adapter at {hf_path} but direct HF PEFT loading into "
            f"Megatron is not yet supported. Please save using Megatron-native format "
            f"(adapter_megatron_tp*_pp*.pt files) for checkpoint resume."
        )
        return False, None

    logger.warning(f"No adapter checkpoint found at {adapter_dir}")
    return False, None


def _load_training_state(
    adapter_dir: Path,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
) -> int | None:
    """Restore optimizer/scheduler state saved alongside a LoRA adapter checkpoint."""
    if optimizer is None:
        return None

    rank = dist.get_rank() if dist.is_initialized() else 0
    state_path = adapter_dir / f"training_state_rank{rank}.pt"
    if not state_path.exists():
        return None

    # Optimizer state dicts may contain non-tensor objects (e.g. step counts,
    # param group metadata), so full unpickling is required here.
    training_state = torch.load(state_path, map_location="cpu", weights_only=False)

    optimizer.load_state_dict(training_state["optimizer"])
    logger.info("Restored optimizer state from LoRA checkpoint")

    if opt_param_scheduler is not None and training_state.get("opt_param_scheduler") is not None:
        opt_param_scheduler.load_state_dict(training_state["opt_param_scheduler"])
        logger.info("Restored LR scheduler state from LoRA checkpoint")

    iteration = training_state.get("iteration")
    if iteration is not None:
        logger.info(f"Resuming LoRA training from iteration {iteration}")
    return iteration


# ---------------------------------------------------------------------------
# LoRA config dict for weight sync to SGLang
# ---------------------------------------------------------------------------
