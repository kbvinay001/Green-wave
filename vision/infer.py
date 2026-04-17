#!/usr/bin/env python3
"""
YOLOv11 inference with tracking for ambulance detection
Real-time detection + tracking + lane assignment
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml
from collections import defaultdict, deque

from ultralytics import YOLO
from ultralytics.engine.results import Results


class Detection:
    """Single detection result"""
    def __init__(
        self,
        bbox: List[float],
        confidence: float,
        class_id: int,
        class_name: str,
        track_id: Optional[int] = None
    ):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.track_id = track_id
        
        # Computed properties
        self.center = self._compute_center()
        self.area = self._compute_area()
    
    def _compute_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _compute_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def to_dict(self) -> dict:
        return {
            'bbox': self.bbox,
            'center': self.center,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'track_id': self.track_id,
            'area': self.area
        }


class VehicleTracker:
    """
    Track vehicles and estimate speed/direction
    Uses position history to compute velocity
    """
    def __init__(self, history_length: int = 10, fps: float = 25.0):
        self.history_length = history_length
        self.fps = fps
        self.tracks = defaultdict(lambda: deque(maxlen=history_length))
    
    def update(self, detection: Detection, timestamp: float):
        """Update track with new detection"""
        if detection.track_id is not None:
            self.tracks[detection.track_id].append({
                'center': detection.center,
                'timestamp': timestamp,
                'bbox': detection.bbox
            })
    
    def get_velocity(self, track_id: int) -> Optional[Tuple[float, float]]:
        """
        Estimate velocity (vx, vy) in pixels/second
        Returns None if insufficient history
        """
        if track_id not in self.tracks or len(self.tracks[track_id]) < 3:
            return None
        
        track = list(self.tracks[track_id])
        
        # Linear regression on positions
        positions = np.array([t['center'] for t in track])
        times = np.array([t['timestamp'] for t in track])
        
        # Compute velocity
        dt = times[-1] - times[0]
        if dt < 0.1:  # Too short time window
            return None
        
        dx = positions[-1, 0] - positions[0, 0]
        dy = positions[-1, 1] - positions[0, 1]
        
        vx = dx / dt
        vy = dy / dt
        
        return (vx, vy)
    
    def get_speed_kmh(self, track_id: int, pixels_per_meter: float = 10.0) -> Optional[float]:
        """
        Estimate speed in km/h
        Requires calibration: pixels_per_meter
        """
        velocity = self.get_velocity(track_id)
        if velocity is None:
            return None
        
        vx, vy = velocity
        speed_pixels_per_sec = np.sqrt(vx**2 + vy**2)
        speed_m_per_sec = speed_pixels_per_sec / pixels_per_meter
        speed_kmh = speed_m_per_sec * 3.6
        
        return speed_kmh
    
    def is_approaching(self, track_id: int) -> bool:
        """
        Check if vehicle is moving toward camera
        Assumes camera is facing forward, vehicles approach from top of frame
        """
        velocity = self.get_velocity(track_id)
        if velocity is None:
            return False
        
        vx, vy = velocity
        # Approaching if moving down in frame (positive y velocity)
        return vy > 5.0  # threshold: 5 pixels/sec
    
    def cleanup_old_tracks(self, current_ids: set, max_age_sec: float = 2.0):
        """Remove tracks not seen recently"""
        import time
        current_time = time.time()
        
        ids_to_remove = []
        for track_id, track_history in self.tracks.items():
            if track_id not in current_ids and len(track_history) > 0:
                last_seen = track_history[-1]['timestamp']
                if current_time - last_seen > max_age_sec:
                    ids_to_remove.append(track_id)
        
        for track_id in ids_to_remove:
            del self.tracks[track_id]


class LaneAssigner:
    """
    Assign detections to lanes using homography or simple rules
    """
    def __init__(
        self,
        lane_boundaries: Optional[List[Tuple[float, float]]] = None,
        homography_matrix: Optional[np.ndarray] = None
    ):
        self.lane_boundaries = lane_boundaries
        self.H = homography_matrix  # Homography: image -> BEV
        
        # Default: simple vertical division (3 lanes)
        if lane_boundaries is None:
            self.lane_boundaries = [
                (0.0, 0.33),    # Left lane
                (0.33, 0.67),   # Center lane
                (0.67, 1.0)     # Right lane
            ]
    
    def assign_lane(self, detection: Detection, image_width: int) -> int:
        """
        Assign detection to lane (0=left, 1=center, 2=right, -1=unknown)
        Uses center x-coordinate
        """
        cx, cy = detection.center
        
        # Normalize x coordinate
        x_norm = cx / image_width
        
        for lane_id, (x_min, x_max) in enumerate(self.lane_boundaries):
            if x_min <= x_norm < x_max:
                return lane_id
        
        return -1  # Unknown lane
    
    def assign_lane_with_homography(self, detection: Detection) -> int:
        """
        Assign lane using homography transformation to bird's eye view
        More accurate for perspective scenes
        """
        if self.H is None:
            raise ValueError("Homography matrix not set")
        
        # Transform center point
        cx, cy = detection.center
        point = np.array([[cx, cy]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), self.H)
        
        tx, ty = transformed[0, 0]
        
        # Assign based on transformed x-coordinate
        # (Assumes BEV coordinate system is calibrated)
        # This is simplified - real implementation needs proper calibration
        
        return 0  # Placeholder


class AmbulanceDetector:
    """
    Main detector class: YOLOv11 + Tracking + Lane Assignment
    """
    def __init__(
        self,
        model_path: str,
        config_path: str = "../common/config.yaml",
        device: str = "cuda"
    ):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.vision_cfg = self.config['vision']
        self.device = device
        
        # Load YOLO model
        print(f"Loading YOLOv11 model from {model_path}...")
        self.model = YOLO(model_path)
        self.model.to(device)

        # -----------------------------------------------------------------
        # MODIFIED: Get class names from model if possible, else config
        # -----------------------------------------------------------------
        self.class_names = getattr(self.model.model, 'names', self.vision_cfg.get('classes', {}))
        
        print(f"[OK] Model loaded on {device}")
        print(f"  Classes source: {'Model' if hasattr(self.model.model, 'names') else 'Config'}")
        
        # Tracking
        self.tracker = VehicleTracker(history_length=10, fps=25)
        
        # Lane assignment
        self.lane_assigner = LaneAssigner()
        
        # Performance tracking
        self.frame_count = 0
        self.total_inference_time = 0
    
    def detect(
        self,
        frame: np.ndarray,
        timestamp: float = None,
        conf_threshold: float = None,
        track: bool = True
    ) -> List[Detection]:
        """
        Run detection on a single frame
        
        Args:
            frame: BGR image
            timestamp: Current timestamp (for velocity estimation)
            conf_threshold: Confidence threshold (overrides config)
            track: Enable tracking
        
        Returns:
            List of Detection objects
        """
        import time
        
        if timestamp is None:
            timestamp = time.time()
        
        conf_threshold = conf_threshold or self.vision_cfg['conf_threshold']
        
        # Run inference
        start_time = time.time()
        
        if track:
            # With tracking (uses ByteTrack internally)
            results = self.model.track(
                frame,
                conf=conf_threshold,
                iou=self.vision_cfg['iou_threshold'],
                persist=True,  # Keep track IDs across frames
                verbose=False
            )
        else:
            # Detection only
            results = self.model(
                frame,
                conf=conf_threshold,
                iou=self.vision_cfg['iou_threshold'],
                verbose=False
            )
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        self.frame_count += 1
        
        # Parse results
        detections = []
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    class_id = int(boxes.cls[i])
                    
                    # Get track ID if available
                    track_id = None
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        track_id = int(boxes.id[i])
                    
                    # -----------------------------------------------------------------
                    # MODIFIED: Retrieve class name safely using the new self.class_names
                    # -----------------------------------------------------------------
                    if isinstance(self.class_names, dict):
                        class_name = self.class_names.get(class_id, str(class_id))
                    else:
                        class_name = self.vision_cfg['classes'][class_id]
                    
                    detection = Detection(
                        bbox=bbox.tolist(),
                        confidence=conf,
                        class_id=class_id,
                        class_name=class_name,
                        track_id=track_id
                    )
                    
                    detections.append(detection)
                    
                    # Update tracker
                    if track and track_id is not None:
                        self.tracker.update(detection, timestamp)
        
        return detections
    
    def detect_with_lanes(
        self,
        frame: np.ndarray,
        timestamp: float = None
    ) -> Dict:
        """
        Detect and assign to lanes
        
        Returns:
            dict with 'detections', 'lanes', 'metadata'
        """
        detections = self.detect(frame, timestamp, track=True)
        
        # Assign lanes
        image_width = frame.shape[1]
        lanes = {}
        
        for det in detections:
            lane_id = self.lane_assigner.assign_lane(det, image_width)
            
            if lane_id not in lanes:
                lanes[lane_id] = []
            
            # Add velocity info
            det_dict = det.to_dict()
            if det.track_id is not None:
                velocity = self.tracker.get_velocity(det.track_id)
                speed = self.tracker.get_speed_kmh(det.track_id)
                approaching = self.tracker.is_approaching(det.track_id)
                
                det_dict['velocity'] = velocity
                det_dict['speed_kmh'] = speed
                det_dict['approaching'] = approaching
            
            lanes[lane_id].append(det_dict)
        
        # Performance metrics
        avg_fps = self.frame_count / self.total_inference_time if self.total_inference_time > 0 else 0
        
        return {
            'detections': [d.to_dict() for d in detections],
            'lanes': lanes,
            'metadata': {
                'frame_count': self.frame_count,
                'avg_fps': avg_fps,
                'timestamp': timestamp
            }
        }
    
    def visualize(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        show_tracks: bool = True,
        show_velocity: bool = True
    ) -> np.ndarray:
        """
        Draw detections on frame
        """
        vis_frame = frame.copy()
        
        colors = {
            0: (0, 0, 255),    # ambulance: red
            1: (0, 255, 255)   # lightbar: yellow
        }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            color = colors.get(det.class_id, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.track_id is not None and show_tracks:
                label = f"ID{det.track_id} {label}"
            
            cv2.putText(
                vis_frame, label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
            
            # Velocity arrow
            if show_velocity and det.track_id is not None:
                velocity = self.tracker.get_velocity(det.track_id)
                if velocity is not None:
                    vx, vy = velocity
                    cx, cy = det.center
                    
                    # Scale for visualization
                    scale = 0.1
                    end_x = int(cx + vx * scale)
                    end_y = int(cy + vy * scale)
                    
                    cv2.arrowedLine(
                        vis_frame,
                        (int(cx), int(cy)),
                        (end_x, end_y),
                        (0, 255, 0), 2, tipLength=0.3
                    )
        
        return vis_frame
    
    def reset(self):
        """Reset tracking state"""
        self.tracker = VehicleTracker(history_length=10, fps=25)
        self.frame_count = 0
        self.total_inference_time = 0


def test_detector():
    """Test detector with your trained model"""
    print("=" * 60)
    print("YOLOv11 Ambulance Detector Test")
    print("=" * 60)
    
    # Check for model
    model_path = "models/yolov11s-ambulance.pt"
    
    if not Path(model_path).exists():
        print(f"[WARN] Model not found: {model_path}")
        print("\nPlease provide your trained model:")
        print("  1. Copy your trained weights to vision/models/")
        print("  2. Rename to: yolov11s-ambulance.pt")
        print("  3. Or update model_path in test code")
        return
    
    # Initialize detector
    detector = AmbulanceDetector(
        model_path=model_path,
        device='cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    )
    
    # Create test image
    print("\nTesting with synthetic image...")
    test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Detect
    import time
    start = time.time()
    detections = detector.detect(test_img, track=False)
    elapsed = time.time() - start
    
    print(f"[OK] Inference time: {elapsed*1000:.1f} ms")
    print(f"  Detections: {len(detections)}")
    
    # Test tracking over multiple frames
    print("\nTesting tracking over 10 frames...")
    for i in range(10):
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = detector.detect_with_lanes(frame, timestamp=i*0.04)
        
        if i == 9:
            print(f"  Frame {i}: {len(result['detections'])} detections")
            print(f"  Avg FPS: {result['metadata']['avg_fps']:.1f}")
    
    print("\n[OK] Detector test complete!")
    print(f"\nYour model is ready for integration!")


if __name__ == "__main__":
    test_detector()