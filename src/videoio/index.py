import json
import os
from typing import List

from .dataset_spec import VideoClipSpec


class VideoIndex:
    """
    Minimal JSONL-based index mapping episode_id to video/spec metadata.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def add(self, spec: VideoClipSpec):
        with open(self.path, "a") as f:
            f.write(json.dumps(spec.to_dict()) + "\n")

    def load_all(self) -> List[VideoClipSpec]:
        if not os.path.exists(self.path):
            return []
        specs = []
        with open(self.path, "r") as f:
            for line in f:
                if line.strip():
                    specs.append(VideoClipSpec.from_dict(json.loads(line)))
        return specs
