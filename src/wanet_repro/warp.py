"""Warping-based trigger generation for WaNet.

The trigger is a fixed elastic warping field M.  It is built from a k x k grid
of random control points, normalized by the mean absolute value of its
entries, bicubically upsampled to the image resolution, scaled by the strength
s, added to the identity sampling grid, and clipped to [-1, 1].  The warped
image is obtained with ``torch.nn.functional.grid_sample``.

This is a from-scratch implementation of Equations (2)-(5) in the paper; it is
not a copy of the official repository.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def make_identity_grid(
    height: int,
    width: Optional[int] = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the normalized identity sampling grid of shape (1, H, W, 2).

    ``grid_sample`` expects the last dimension to be (x, y), where x is the
    horizontal coordinate and y is the vertical coordinate.
    """
    width = height if width is None else width
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


def make_warp_field(
    k: int,
    height: int,
    width: Optional[int] = None,
    device: torch.device | str = "cpu",
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate the fixed warping field M of shape (1, H, W, 2).

    Args:
        k: side length of the control-point grid.
        height: image height.
        width: image width (defaults to ``height``).
        device: target device.
        seed: optional seed for reproducible control points.
        dtype: floating-point dtype.
    """
    width = height if width is None else width
    if seed is None:
        control = torch.rand(1, 2, k, k, device=device, dtype=dtype) * 2.0 - 1.0
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        control = torch.rand(1, 2, k, k, generator=generator, dtype=dtype) * 2.0 - 1.0
        control = control.to(device)

    # Equation (4): normalize by the mean absolute value.
    control = control / torch.mean(torch.abs(control))

    # Bicubic interpolation to the full image resolution (Equation (5)).
    field = F.interpolate(control, size=(height, width), mode="bicubic", align_corners=True)
    return field.permute(0, 2, 3, 1).contiguous()


def _sampling_grid(
    identity_grid: torch.Tensor,
    field: torch.Tensor,
    s: float,
    grid_rescale: float,
) -> torch.Tensor:
    """Combine the identity grid and the warping field, then clip."""
    # The official implementation divides by the image height because the
    # field is expressed in pixel units while the grid is normalized.
    height = identity_grid.shape[1]
    grid = (identity_grid + s * field / height) * grid_rescale
    return torch.clamp(grid, -1.0, 1.0)


def warp_batch(
    inputs: torch.Tensor,
    field: torch.Tensor,
    s: float = 0.5,
    identity_grid: Optional[torch.Tensor] = None,
    grid_rescale: float = 1.0,
) -> torch.Tensor:
    """Apply the fixed WaNet trigger to a batch of images."""
    batch, _, height, width = inputs.shape
    if identity_grid is None:
        identity_grid = make_identity_grid(height, width, device=inputs.device, dtype=inputs.dtype)
    grid = _sampling_grid(identity_grid, field, s, grid_rescale)
    grid = grid.repeat(batch, 1, 1, 1)
    return F.grid_sample(inputs, grid, align_corners=True)


def warp_batch_with_noise(
    inputs: torch.Tensor,
    field: torch.Tensor,
    s: float = 0.5,
    identity_grid: Optional[torch.Tensor] = None,
    grid_rescale: float = 1.0,
    noise_type: str = "uniform",
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply a random perturbation of the warping field (noise mode).

    The paper describes Gaussian noise; the official implementation uses
    uniform noise.  Both are supported so the discrepancy can be tested.
    """
    batch, _, height, width = inputs.shape
    if identity_grid is None:
        identity_grid = make_identity_grid(height, width, device=inputs.device, dtype=inputs.dtype)
    grid = _sampling_grid(identity_grid, field, s, grid_rescale).repeat(batch, 1, 1, 1)

    if noise_type == "uniform":
        if generator is None:
            noise = torch.rand(batch, height, width, 2, device=inputs.device, dtype=inputs.dtype) * 2.0 - 1.0
        else:
            noise = torch.rand(batch, height, width, 2, generator=generator, dtype=inputs.dtype) * 2.0 - 1.0
            noise = noise.to(inputs.device)
    elif noise_type == "gaussian":
        if generator is None:
            noise = torch.randn(batch, height, width, 2, device=inputs.device, dtype=inputs.dtype)
        else:
            noise = torch.randn(batch, height, width, 2, generator=generator, dtype=inputs.dtype)
            noise = noise.to(inputs.device)
    else:
        raise ValueError(f"Unsupported noise_type: {noise_type!r}")

    grid = torch.clamp(grid + noise / height, -1.0, 1.0)
    return F.grid_sample(inputs, grid, align_corners=True)
