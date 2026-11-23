"""Conservative online re-identification of recently lost people."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Observation:
    timestamp: float
    box: np.ndarray
    appearance: np.ndarray | None


class TrackReidentifier:
    def __init__(self, max_gap_sec, min_similarity=.95, min_margin=.06, retention_sec=None):
        self.max_gap_sec = max_gap_sec
        self.retention_sec = max_gap_sec if retention_sec is None else retention_sec
        self.min_similarity = min_similarity
        self.min_margin = min_margin
        self.aliases = {}
        self.tracks = {}
        self.next_id = 1

    @staticmethod
    def appearance(frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        dx, dy = x2-x1, y2-y1
        left, right = np.clip([x1+.2*dx, x2-.2*dx], 0, w).astype(int)
        top, bottom = np.clip([y1+.18*dy, y1+.68*dy], 0, h).astype(int)
        if right-left < 8 or bottom-top < 8:
            return None
        crop = frame[top:bottom, left:right]
        hsv = cv2.cvtColor(cv2.resize(crop, (24, 32)), cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
        vector = np.sqrt(hist.ravel() / hist.sum())
        return vector

    def step(self, frame, detections, timestamp):
        self.tracks = {pid: obs for pid, obs in self.tracks.items()
                       if timestamp-obs.timestamp <= self.retention_sec+1e-8}
        self.aliases = {raw: pid for raw, pid in self.aliases.items() if pid in self.tracks}
        active = {self.aliases[raw] for raw, _, _ in detections if raw in self.aliases}
        descriptors = {raw: self.appearance(frame, box) for raw, box, _ in detections}
        candidates = []
        for raw, box, _ in detections:
            descriptor = descriptors[raw]
            if raw in self.aliases or descriptor is None:
                continue
            center = (box[:2]+box[2:])/2
            for pid, old in self.tracks.items():
                if pid in active or old.appearance is None or timestamp-old.timestamp > self.max_gap_sec+1e-8:
                    continue
                scale = max(1., float(np.linalg.norm(old.box[2:]-old.box[:2])))
                old_center = (old.box[:2]+old.box[2:])/2
                if np.linalg.norm(center-old_center) <= .55*scale:
                    similarity = float(descriptor @ old.appearance)
                    candidates.append((similarity, raw, pid))
        links = []
        for score, raw, pid in sorted(candidates, reverse=True):
            if score < self.min_similarity:
                break
            if any((other_raw == raw or other_pid == pid)
                   and (other_raw, other_pid) != (raw, pid)
                   and other_score > score-self.min_margin
                   for other_score, other_raw, other_pid in candidates):
                continue
            # A revived track now belongs to the new raw ID. Drop stale aliases.
            self.aliases = {key: val for key, val in self.aliases.items() if val != pid}
            self.aliases[raw] = pid
            links.append((raw, pid, score))

        result = []
        for raw, box, points in detections:
            if raw not in self.aliases:
                self.aliases[raw] = self.next_id
                self.next_id += 1
            pid = self.aliases[raw]
            descriptor = descriptors[raw]
            previous = self.tracks.get(pid)
            if previous is not None and previous.appearance is not None:
                if descriptor is None:
                    descriptor = previous.appearance
                else:
                    descriptor = .8*previous.appearance + .2*descriptor
                    descriptor /= np.linalg.norm(descriptor)
            self.tracks[pid] = Observation(timestamp, box.copy(), descriptor)
            result.append((pid, box, points))
        return result, links
