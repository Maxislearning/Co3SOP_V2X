# Registers carla_V2V's dataset/pipeline on top of the original Co3SOP plugin
# (model definitions, base Co3SOP dataset class, etc. — imported explicitly
# here rather than relying on tools/train.py's separate hardcoded import of
# projects.mmdet3d_plugin.co3sop_base.apis.train, since tools/test.py doesn't
# have that same guarantee).
import projects.mmdet3d_plugin  # noqa: F401  (registers Co3SOPBase, V2VOccHead, Co3SOP, etc.)
from .datasets import CarlaV2VCo3SOP, CarlaV2VBeamCo3SOP, CarlaV2VTargetOccCo3SOP, LoadCarlaOccupancy  # noqa: F401

# mmcv-full 1.7.2 (this env's version, needed for RTX 5090/torch2.7 — Co3SOP
# was written against mmcv-full 1.4.0) still calls
# torch.nn.parallel._functions._get_stream(device: int) — plain int, matching
# torch<2.x's signature. Under torch 2.7 that function instead expects a real
# torch.device (it does device.type internally) and crashes with
# "'int' object has no attribute 'type'" the moment MMDataParallel.scatter()
# runs, i.e. the very first training iteration. Patched here (monkeypatch,
# process-local, does not touch the installed mmcv package other envs/
# processes share) rather than in mmcv itself.
import mmcv.parallel._functions as _mmcv_parallel_functions
import torch as _torch

_orig_get_stream = _mmcv_parallel_functions._get_stream


def _get_stream_compat(device):
    if isinstance(device, int):
        device = _torch.device('cpu') if device == -1 else _torch.device(f'cuda:{device}')
    return _orig_get_stream(device)


_mmcv_parallel_functions._get_stream = _get_stream_compat

# Same story for MMDistributedDataParallel._run_ddp_forward: it reads
# self._use_replicated_tensor_module, an attribute torch's own DDP only had
# for a brief window (~1.12-1.13) around when mmcv-full 1.7.2 was written.
# torch 2.7's DistributedDataParallel doesn't define it at all, so the
# attribute lookup falls through to nn.Module.__getattr__ and raises
# AttributeError on the very first validation pass (distributed=True is
# needed only to get CustomDistEvalHook wired up instead of the incompatible
# stock EvalHook — see carla_v2v_co3sop.py plugin docs). Patched to always
# use self.module (module_to_run), matching modern DDP's simpler behavior;
# process-local monkeypatch, same rationale as the _get_stream patch above.
import mmcv.parallel.distributed as _mmcv_dist


def _run_ddp_forward_compat(self, *inputs, **kwargs):
    module_to_run = self.module
    if self.device_ids:
        inputs, kwargs = self.to_kwargs(inputs, kwargs, self.device_ids[0])
        return module_to_run(*inputs[0], **kwargs[0])
    else:
        return module_to_run(*inputs, **kwargs)


_mmcv_dist.MMDistributedDataParallel._run_ddp_forward = _run_ddp_forward_compat

__all__ = ['CarlaV2VCo3SOP', 'CarlaV2VBeamCo3SOP', 'CarlaV2VTargetOccCo3SOP', 'LoadCarlaOccupancy']
