from __future__ import annotations

from ..models import CaptureProfile
from .base import CaptureSource
from .direct import DirectWindowCapture
from .obs import ObsVirtualCameraCapture


def create_capture_source(profile: CaptureProfile, *, fps: float = 5.0) -> CaptureSource:
    if profile.source == "obs":
        return ObsVirtualCameraCapture(profile.virtual_camera_index, fps=fps)
    if profile.source == "direct":
        return DirectWindowCapture(profile.window_title, fps=fps)
    raise ValueError(f"Unsupported capture source: {profile.source}")
