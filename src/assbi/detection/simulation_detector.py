"""Synthetic detector that simulates a busy street scene.

This is the secret to a *fully runnable* demo on any machine: it generates
realistic, temporally-coherent detections of pedestrians and vehicles moving
across the frame, so tracking, line-crossing, crowd density, anomaly detection
and forecasting all exercise real data — no GPU, no video file, no torch.

The motion model is deterministic given a seed, so results are reproducible.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..domain.geometry import BoundingBox
from ..domain.interfaces import Frame, ObjectDetector
from ..domain.models import Detection, ObjectClass


@dataclass
class _Agent:
    object_class: ObjectClass
    x: float
    y: float
    vx: float
    vy: float
    w: float
    h: float
    born: int
    lifespan: int

    def position_at(self, frame: int) -> tuple[float, float]:
        dt = frame - self.born
        return self.x + self.vx * dt, self.y + self.vy * dt


@dataclass
class _Scene:
    width: int
    height: int
    rng: random.Random
    agents: list[_Agent] = field(default_factory=list)


class SimulationDetector(ObjectDetector):
    """Generates synthetic detections of pedestrians and vehicles.

    Args:
        width/height: frame size in pixels (match the SimulationVideoSource).
        pedestrian_rate / vehicle_rate: expected new agents spawned per frame.
        surge_frames: optional (start, end) window where pedestrian spawning
            multiplies — used to demonstrate anomaly detection.
        seed: RNG seed for reproducibility.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        pedestrian_rate: float = 0.28,
        vehicle_rate: float = 0.16,
        surge_frames: tuple[int, int] | None = (300, 360),
        surge_multiplier: float = 4.0,
        burn_in: int = 200,
        seed: int = 42,
    ) -> None:
        self.width = width
        self.height = height
        self.pedestrian_rate = pedestrian_rate
        self.vehicle_rate = vehicle_rate
        self.surge_frames = surge_frames
        self.surge_multiplier = surge_multiplier
        self._scene = _Scene(width, height, random.Random(seed))
        # Pre-populate the scene so frame 0 is already a busy street rather than
        # an empty one — otherwise the cold-start ramp looks like an anomaly.
        for f in range(-burn_in, 0):
            self._spawn(f)

    def detect(self, frame: Frame) -> list[Detection]:
        self._spawn(frame.index)
        detections: list[Detection] = []
        survivors: list[_Agent] = []
        for agent in self._scene.agents:
            if frame.index - agent.born > agent.lifespan:
                continue
            cx, cy = agent.position_at(frame.index)
            if cx < -100 or cx > self.width + 100 or cy < -100 or cy > self.height + 100:
                continue
            survivors.append(agent)
            # Small per-frame jitter so the tracker/IoU logic is exercised.
            jitter = self._scene.rng.uniform(-2, 2)
            box = BoundingBox(
                cx - agent.w / 2 + jitter,
                cy - agent.h / 2,
                cx + agent.w / 2 + jitter,
                cy + agent.h / 2,
            )
            conf = round(self._scene.rng.uniform(0.55, 0.95), 3)
            detections.append(Detection(agent.object_class, box, conf))
        self._scene.agents = survivors
        return detections

    # -- scene generation --------------------------------------------------
    def _spawn(self, frame_index: int) -> None:
        rng = self._scene.rng
        ped_rate = self.pedestrian_rate
        if self.surge_frames and self.surge_frames[0] <= frame_index <= self.surge_frames[1]:
            ped_rate *= self.surge_multiplier

        for _ in range(self._poisson(ped_rate, rng)):
            self._scene.agents.append(self._make_pedestrian(frame_index))
        for _ in range(self._poisson(self.vehicle_rate, rng)):
            self._scene.agents.append(self._make_vehicle(frame_index))

    def _make_pedestrian(self, frame_index: int) -> _Agent:
        rng = self._scene.rng
        # Pedestrians traverse vertically (cross a horizontal mid-line).
        going_down = rng.random() < 0.5
        x = rng.uniform(0.1 * self.width, 0.9 * self.width)
        y = -40 if going_down else self.height + 40
        vy = rng.uniform(3.5, 6.5) * (1 if going_down else -1)
        vx = rng.uniform(-1.0, 1.0)
        return _Agent(ObjectClass.PERSON, x, y, vx, vy, 22, 55, frame_index, 400)

    def _make_vehicle(self, frame_index: int) -> _Agent:
        rng = self._scene.rng
        cls = rng.choice([ObjectClass.CAR, ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.BUS])
        # Vehicles traverse horizontally.
        going_right = rng.random() < 0.5
        x = -80 if going_right else self.width + 80
        y = rng.uniform(0.45 * self.height, 0.9 * self.height)
        vx = rng.uniform(7.0, 12.0) * (1 if going_right else -1)
        size = {ObjectClass.CAR: (90, 45), ObjectClass.TRUCK: (140, 60), ObjectClass.BUS: (160, 70)}[cls]
        return _Agent(cls, x, y, vx, rng.uniform(-0.5, 0.5), size[0], size[1], frame_index, 300)

    @staticmethod
    def _poisson(lam: float, rng: random.Random) -> int:
        """Knuth's algorithm — number of arrivals in one interval."""
        import math

        target = math.exp(-lam)
        k, p = 0, 1.0
        while True:
            p *= rng.random()
            if p <= target:
                return k
            k += 1
