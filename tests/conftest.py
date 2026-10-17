"""Shared test plumbing.

The face guard (tools/picker/faces.py) runs inside optimize_images.optimize_image, so every
test that publishes an image would start a real face detector. Unless a module is about the
guard itself (test_faces.py), the default detector is replaced by one that finds nothing.
"""
import sys
from pathlib import Path

import pytest

PICKER = Path(__file__).resolve().parents[1] / "tools" / "picker"
if str(PICKER) not in sys.path:
    sys.path.insert(0, str(PICKER))


class _NoFaces:
    name = "stub-no-faces"

    def detect(self, image):
        return []


@pytest.fixture(autouse=True)
def _stub_face_detector(request, monkeypatch):
    if request.module.__name__ == "test_faces":
        return
    import faces
    monkeypatch.setattr(faces, "default_detectors", lambda settings=None: [_NoFaces()])
