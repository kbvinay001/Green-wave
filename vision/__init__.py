"""
Vision module for Green Wave++
YOLOv11-based ambulance detection
"""

from .infer import AmbulanceDetector, Detection, VehicleTracker, LaneAssigner

__all__ = [
    'AmbulanceDetector',
    'Detection',
    'VehicleTracker',
    'LaneAssigner'
]
