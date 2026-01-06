# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# SAM3 Server - REST API for LiGuard-Web Integration
"""
SAM3 Server: A FastAPI-based REST API for the Segment Anything Model 3.

This server provides a standardized API for image and video segmentation,
designed for integration with LiGuard-Web's node-based pipeline system.

Usage:
    conda activate sam3
    python server.py [--host 0.0.0.0] [--port 8765] [--session-timeout 300]

The server manages model sessions with automatic cleanup of idle sessions.
All image I/O is done via base64-encoded strings for portability.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

# SAM3 imports - only available in SAM3 environment
SAM3_AVAILABLE = False
Sam3Processor = None
build_sam3_image_model = None
build_sam3_video_predictor = None
render_masklet_frame = None

try:
    from sam3.model_builder import build_sam3_image_model, build_sam3_video_predictor
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.visualization_utils import render_masklet_frame
    SAM3_AVAILABLE = True
    print("SAM3 modules imported successfully")
except ImportError as e:
    print(f"SAM3 import failed: {e}")
except Exception as e:
    print(f"SAM3 import error: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sam3_server")


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8765
DEFAULT_SESSION_TIMEOUT = 300  # 5 minutes
DEFAULT_CLEANUP_INTERVAL = 60  # 1 minute


# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class ModelType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class SessionCreateRequest(BaseModel):
    model_type: ModelType = ModelType.IMAGE
    device: str = "cuda"


class SessionResponse(BaseModel):
    session_id: str
    model_type: ModelType
    created_at: float
    last_accessed: float


class ImageSetRequest(BaseModel):
    image: str  # base64 encoded


class TextPromptRequest(BaseModel):
    text: str
    confidence_threshold: float = 0.5


class BoxPromptRequest(BaseModel):
    x1: float  # normalized [0, 1]
    y1: float
    x2: float
    y2: float
    is_positive: bool = True


class PointPromptRequest(BaseModel):
    x: float  # normalized [0, 1]
    y: float
    is_positive: bool = True


class VisualizeRequest(BaseModel):
    alpha: float = 0.5


class SegmentationResult(BaseModel):
    count: int
    masks: List[str]  # base64 encoded PNGs
    boxes: List[List[float]]  # [[x1, y1, x2, y2], ...]
    scores: List[float]


class VisualizeResponse(BaseModel):
    image: str  # base64 encoded


class VideoStartRequest(BaseModel):
    streaming: bool = True
    video_path: Optional[str] = None


class VideoFrameRequest(BaseModel):
    image: str  # base64 encoded
    frame_idx: int


class VideoPromptRequest(BaseModel):
    frame_idx: int
    text: Optional[str] = None
    points: Optional[List[List[float]]] = None  # [[x, y], ...]
    point_labels: Optional[List[int]] = None  # 1 = positive, 0 = negative
    boxes: Optional[List[List[float]]] = None  # [[x1, y1, x2, y2], ...]
    box_labels: Optional[List[int]] = None


class VideoPropagateRequest(BaseModel):
    direction: str = "forward"  # forward, backward, both
    max_frames: Optional[int] = None


class VideoResultRequest(BaseModel):
    frame_idx: int


class HealthResponse(BaseModel):
    status: str
    sam3_available: bool
    gpu_available: bool
    active_sessions: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# =============================================================================
# Session Management
# =============================================================================

@dataclass
class ImageSession:
    """Holds state for an image segmentation session."""
    session_id: str
    model: Any = None
    processor: Optional[Sam3Processor] = None
    state: Optional[Dict] = None
    device: str = "cuda"
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()


@dataclass
class VideoSession:
    """Holds state for a video segmentation session."""
    session_id: str
    predictor: Any = None
    internal_session_id: Optional[str] = None
    frames: List[np.ndarray] = field(default_factory=list)
    outputs: Dict[int, Dict] = field(default_factory=dict)
    streaming: bool = True
    temp_dir: Optional[str] = None
    device: str = "cuda"
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    # Maps external frame indices to internal SAM3 frame indices
    frame_idx_mapping: Dict[int, int] = field(default_factory=dict)
    
    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()


class SessionManager:
    """
    Manages SAM3 model sessions with automatic cleanup of idle sessions.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, timeout_seconds: int = DEFAULT_SESSION_TIMEOUT):
        self.timeout_seconds = timeout_seconds
        self._image_sessions: Dict[str, ImageSession] = {}
        self._video_sessions: Dict[str, VideoSession] = {}
        self._lock = threading.RLock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Lazy-loaded shared models
        self._image_model = None
        self._video_predictor = None
        self._model_lock = threading.Lock()
        
    def start_cleanup_task(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the background cleanup task."""
        self._running = True
        self._cleanup_task = loop.create_task(self._cleanup_loop())
        
    async def stop_cleanup_task(self) -> None:
        """Stop the background cleanup task."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired sessions."""
        while self._running:
            try:
                await asyncio.sleep(DEFAULT_CLEANUP_INTERVAL)
                self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove sessions that have exceeded the timeout."""
        now = time.time()
        expired_image = []
        expired_video = []
        
        with self._lock:
            for sid, session in self._image_sessions.items():
                if now - session.last_accessed > self.timeout_seconds:
                    expired_image.append(sid)
                    
            for sid, session in self._video_sessions.items():
                if now - session.last_accessed > self.timeout_seconds:
                    expired_video.append(sid)
                    
            for sid in expired_image:
                logger.info(f"Cleaning up expired image session: {sid}")
                self._image_sessions.pop(sid, None)
                
            for sid in expired_video:
                logger.info(f"Cleaning up expired video session: {sid}")
                session = self._video_sessions.pop(sid, None)
                if session and session.predictor and session.internal_session_id:
                    try:
                        session.predictor.close_session(session.internal_session_id)
                    except Exception as e:
                        logger.warning(f"Error closing video session: {e}")
        
        if expired_image or expired_video:
            self._maybe_unload_models()
    
    def _maybe_unload_models(self) -> None:
        """Unload models if no sessions are active."""
        with self._lock:
            if not self._image_sessions and self._image_model is not None:
                logger.info("Unloading image model (no active sessions)")
                with self._model_lock:
                    del self._image_model
                    self._image_model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            if not self._video_sessions and self._video_predictor is not None:
                logger.info("Unloading video predictor (no active sessions)")
                with self._model_lock:
                    if hasattr(self._video_predictor, 'shutdown'):
                        self._video_predictor.shutdown()
                    del self._video_predictor
                    self._video_predictor = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    def _get_image_model(self, device: str = "cuda") -> Tuple[Any, Sam3Processor]:
        """Get or create the shared image model."""
        if not SAM3_AVAILABLE:
            raise RuntimeError("SAM3 is not available in this environment")
            
        with self._model_lock:
            if self._image_model is None:
                logger.info(f"Loading SAM3 image model on {device}...")
                self._image_model = build_sam3_image_model(device=device)
                logger.info("SAM3 image model loaded")
            return self._image_model, Sam3Processor(self._image_model)
    
    def _get_video_predictor(self, device: str = "cuda") -> Any:
        """Get or create the shared video predictor."""
        if not SAM3_AVAILABLE:
            raise RuntimeError("SAM3 is not available in this environment")
            
        with self._model_lock:
            if self._video_predictor is None:
                logger.info(f"Loading SAM3 video predictor on {device}...")
                self._video_predictor = build_sam3_video_predictor()
                logger.info("SAM3 video predictor loaded")
            return self._video_predictor
    
    def create_image_session(self, device: str = "cuda") -> ImageSession:
        """Create a new image segmentation session."""
        session_id = str(uuid.uuid4())[:12]
        model, processor = self._get_image_model(device)
        
        session = ImageSession(
            session_id=session_id,
            model=model,
            processor=processor,
            device=device,
        )
        
        with self._lock:
            self._image_sessions[session_id] = session
            
        logger.info(f"Created image session: {session_id}")
        return session
    
    def create_video_session(self, device: str = "cuda") -> VideoSession:
        """Create a new video segmentation session."""
        session_id = str(uuid.uuid4())[:12]
        predictor = self._get_video_predictor(device)
        
        session = VideoSession(
            session_id=session_id,
            predictor=predictor,
            device=device,
        )
        
        with self._lock:
            self._video_sessions[session_id] = session
            
        logger.info(f"Created video session: {session_id}")
        return session
    
    def get_image_session(self, session_id: str) -> Optional[ImageSession]:
        """Get an image session by ID."""
        with self._lock:
            session = self._image_sessions.get(session_id)
            if session:
                session.touch()
            return session
    
    def get_video_session(self, session_id: str) -> Optional[VideoSession]:
        """Get a video session by ID."""
        with self._lock:
            session = self._video_sessions.get(session_id)
            if session:
                session.touch()
            return session
    
    def close_image_session(self, session_id: str) -> bool:
        """Close an image session."""
        with self._lock:
            session = self._image_sessions.pop(session_id, None)
            if session:
                logger.info(f"Closed image session: {session_id}")
                self._maybe_unload_models()
                return True
            return False
    
    def close_video_session(self, session_id: str) -> bool:
        """Close a video session."""
        with self._lock:
            session = self._video_sessions.pop(session_id, None)
            if session:
                if session.predictor and session.internal_session_id:
                    try:
                        session.predictor.close_session(session.internal_session_id)
                    except Exception as e:
                        logger.warning(f"Error closing internal video session: {e}")
                # Clean up temp directory if it exists
                if session.temp_dir:
                    try:
                        shutil.rmtree(session.temp_dir)
                        logger.info(f"Cleaned up temp directory: {session.temp_dir}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up temp directory: {e}")
                logger.info(f"Closed video session: {session_id}")
                self._maybe_unload_models()
                return True
            return False
    
    def heartbeat(self, session_id: str) -> bool:
        """Keep a session alive."""
        with self._lock:
            if session_id in self._image_sessions:
                self._image_sessions[session_id].touch()
                return True
            if session_id in self._video_sessions:
                self._video_sessions[session_id].touch()
                return True
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get session statistics."""
        with self._lock:
            return {
                "image_sessions": len(self._image_sessions),
                "video_sessions": len(self._video_sessions),
                "total": len(self._image_sessions) + len(self._video_sessions),
            }
    
    def shutdown(self) -> None:
        """Shutdown all sessions and unload models."""
        with self._lock:
            # Close all image sessions
            for sid in list(self._image_sessions.keys()):
                self.close_image_session(sid)
            
            # Close all video sessions
            for sid in list(self._video_sessions.keys()):
                self.close_video_session(sid)


# =============================================================================
# Utility Functions
# =============================================================================

def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode a base64 string to a PIL Image."""
    # Handle data URL format
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    
    image_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def encode_image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """Encode a PIL Image to a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=90)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type = "image/jpeg" if format.upper() == "JPEG" else "image/png"
    return f"data:{mime_type};base64,{base64_str}"


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Encode a binary mask to a base64 PNG string."""
    # Convert boolean/float mask to uint8
    if mask.dtype == bool:
        mask_uint8 = (mask * 255).astype(np.uint8)
    elif mask.dtype in [np.float32, np.float64]:
        mask_uint8 = ((mask > 0.5) * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)
    
    # Handle 3D mask (take first channel if needed)
    if mask_uint8.ndim == 3:
        mask_uint8 = mask_uint8[0]
    
    image = Image.fromarray(mask_uint8, mode='L')
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_str}"


def numpy_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a numpy array to PIL Image."""
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


# =============================================================================
# FastAPI Application
# =============================================================================

# Global session manager
session_manager: Optional[SessionManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global session_manager
    
    # Startup
    logger.info("Starting SAM3 Server...")
    
    # Use custom timeout if set via command line, otherwise use default
    timeout = getattr(app.state, 'session_timeout', DEFAULT_SESSION_TIMEOUT)
    session_manager = SessionManager(timeout_seconds=timeout)
    session_manager.start_cleanup_task(asyncio.get_event_loop())
    
    # Enable TensorFloat-32 for Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SAM3 Server...")
    if session_manager:
        await session_manager.stop_cleanup_task()
        session_manager.shutdown()


app = FastAPI(
    title="SAM3 Server",
    description="REST API for Segment Anything Model 3",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and availability."""
    stats = session_manager.get_stats() if session_manager else {"total": 0}
    return HealthResponse(
        status="healthy",
        sam3_available=SAM3_AVAILABLE,
        gpu_available=torch.cuda.is_available(),
        active_sessions=stats["total"],
    )


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {"name": "sam3_image", "type": "image", "available": SAM3_AVAILABLE},
            {"name": "sam3_video", "type": "video", "available": SAM3_AVAILABLE},
        ]
    }


# =============================================================================
# Session Management Endpoints
# =============================================================================

@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new segmentation session."""
    if not SAM3_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SAM3 is not available in this environment"
        )
    
    try:
        if request.model_type == ModelType.IMAGE:
            session = session_manager.create_image_session(device=request.device)
        else:
            session = session_manager.create_video_session(device=request.device)
        
        return SessionResponse(
            session_id=session.session_id,
            model_type=request.model_type,
            created_at=session.created_at,
            last_accessed=session.last_accessed,
        )
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/sessions/{session_id}")
async def close_session(session_id: str):
    """Close a session and release resources."""
    image_closed = session_manager.close_image_session(session_id)
    video_closed = session_manager.close_video_session(session_id)
    
    if not image_closed and not video_closed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    return {"status": "closed", "session_id": session_id}


@app.post("/sessions/{session_id}/heartbeat")
async def session_heartbeat(session_id: str):
    """Keep a session alive."""
    if not session_manager.heartbeat(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return {"status": "alive", "session_id": session_id}


# =============================================================================
# Image Segmentation Endpoints
# =============================================================================

def _get_image_session(session_id: str) -> ImageSession:
    """Helper to get an image session or raise 404."""
    session = session_manager.get_image_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image session {session_id} not found"
        )
    return session


@app.post("/image/{session_id}/set")
async def set_image(session_id: str, request: ImageSetRequest):
    """Set the image for segmentation."""
    session = _get_image_session(session_id)
    
    try:
        image = decode_base64_image(request.image)
        session.state = session.processor.set_image(image)
        
        # Store original image for visualization (processor may not keep it)
        if session.state is None:
            session.state = {}
        session.state["original_image"] = image
        session.state["original_width"] = image.width
        session.state["original_height"] = image.height
        
        return {
            "status": "ok",
            "width": image.width,
            "height": image.height,
        }
    except Exception as e:
        logger.error(f"Error setting image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.post("/image/{session_id}/prompt/text")
async def add_text_prompt(session_id: str, request: TextPromptRequest):
    """Add a text prompt and run inference."""
    session = _get_image_session(session_id)
    
    if session.state is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image set. Call /image/{session_id}/set first."
        )
    
    try:
        session.state = session.processor.set_confidence_threshold(
            request.confidence_threshold, session.state
        )
        session.state = session.processor.set_text_prompt(request.text, session.state)
        
        masks = session.state.get("masks", [])
        return {"status": "ok", "objects_found": len(masks)}
    except Exception as e:
        logger.error(f"Error adding text prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/image/{session_id}/prompt/box")
async def add_box_prompt(session_id: str, request: BoxPromptRequest):
    """Add a box prompt."""
    session = _get_image_session(session_id)
    
    if session.state is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image set. Call /image/{session_id}/set first."
        )
    
    try:
        # Convert from xyxy normalized to cxcywh normalized
        cx = (request.x1 + request.x2) / 2
        cy = (request.y1 + request.y2) / 2
        w = request.x2 - request.x1
        h = request.y2 - request.y1
        box = [cx, cy, w, h]
        
        session.state = session.processor.add_geometric_prompt(
            box, request.is_positive, session.state
        )
        
        masks = session.state.get("masks", [])
        return {"status": "ok", "objects_found": len(masks)}
    except Exception as e:
        logger.error(f"Error adding box prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/image/{session_id}/prompt/point")
async def add_point_prompt(session_id: str, request: PointPromptRequest):
    """Add a point prompt."""
    session = _get_image_session(session_id)
    
    if session.state is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image set. Call /image/{session_id}/set first."
        )
    
    try:
        # For point prompts, we create a small box centered on the point
        # SAM3 image processor currently supports box prompts, not direct points
        # We simulate a point with a tiny box
        size = 0.01  # 1% of image dimension
        box = [request.x, request.y, size, size]
        
        session.state = session.processor.add_geometric_prompt(
            box, request.is_positive, session.state
        )
        
        masks = session.state.get("masks", [])
        return {"status": "ok", "objects_found": len(masks)}
    except Exception as e:
        logger.error(f"Error adding point prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/image/{session_id}/reset")
async def reset_prompts(session_id: str):
    """Clear all prompts."""
    session = _get_image_session(session_id)
    
    if session.state is None:
        return {"status": "ok", "message": "No state to reset"}
    
    try:
        session.state = session.processor.reset_all_prompts(session.state)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error resetting prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/image/{session_id}/results", response_model=SegmentationResult)
async def get_results(session_id: str):
    """Get segmentation results."""
    session = _get_image_session(session_id)
    
    if session.state is None:
        return SegmentationResult(count=0, masks=[], boxes=[], scores=[])
    
    try:
        masks_tensor = session.state.get("masks", [])
        boxes_tensor = session.state.get("boxes", [])
        scores_list = session.state.get("scores", [])
        
        # Convert masks to base64
        masks_b64 = []
        for mask in masks_tensor:
            mask_np = mask[0].cpu().numpy() if hasattr(mask, 'cpu') else mask
            masks_b64.append(encode_mask_to_base64(mask_np))
        
        # Convert boxes to list
        boxes_list = []
        for box in boxes_tensor:
            box_np = box.cpu().numpy() if hasattr(box, 'cpu') else box
            boxes_list.append(box_np.tolist())
        
        # Convert scores
        scores = []
        for score in scores_list:
            if hasattr(score, 'item'):
                scores.append(score.item())
            else:
                scores.append(float(score))
        
        return SegmentationResult(
            count=len(masks_b64),
            masks=masks_b64,
            boxes=boxes_list,
            scores=scores,
        )
    except Exception as e:
        logger.error(f"Error getting results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/image/{session_id}/visualize", response_model=VisualizeResponse)
async def visualize_image(session_id: str, request: VisualizeRequest = None):
    """Get the image with segmentation overlay."""
    if request is None:
        request = VisualizeRequest()
    
    session = _get_image_session(session_id)
    
    if session.state is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image set"
        )
    
    try:
        import cv2
        
        # Get original image
        original = session.state.get("original_image")
        if original is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Original image not available"
            )
        
        # Convert to numpy if needed
        if isinstance(original, Image.Image):
            img_np = np.array(original)
        else:
            img_np = original
        
        # Create overlay
        overlay = img_np.copy().astype(np.float32)
        
        masks = session.state.get("masks", [])
        boxes = session.state.get("boxes", [])
        scores = session.state.get("scores", [])
        
        # Color palette
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
        ]
        
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            color = colors[i % len(colors)]
            mask_np = mask[0].cpu().numpy() if hasattr(mask, 'cpu') else mask
            
            # Apply mask overlay
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask_np > 0.5,
                    overlay[:, :, c] * (1 - request.alpha) + color[c] * request.alpha,
                    overlay[:, :, c]
                )
            
            # Draw bounding box
            box_np = box.cpu().numpy() if hasattr(box, 'cpu') else box
            x1, y1, x2, y2 = [int(v) for v in box_np]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw score
            score_val = score.item() if hasattr(score, 'item') else float(score)
            label = f"{score_val:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, color, 2)
        
        # Convert to PIL and encode
        overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
        result_image = Image.fromarray(overlay_uint8)
        
        return VisualizeResponse(image=encode_image_to_base64(result_image))
    except Exception as e:
        logger.error(f"Error visualizing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# Video Segmentation Endpoints
# =============================================================================

def _get_video_session(session_id: str) -> VideoSession:
    """Helper to get a video session or raise 404."""
    session = session_manager.get_video_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video session {session_id} not found"
        )
    return session


@app.post("/video/{session_id}/start")
async def start_video(session_id: str, request: VideoStartRequest):
    """Start a video session."""
    session = _get_video_session(session_id)
    
    try:
        session.streaming = request.streaming
        
        if not request.streaming and request.video_path:
            # Start with a video file
            response = session.predictor.handle_request({
                "type": "start_session",
                "resource_path": request.video_path,
            })
            session.internal_session_id = response.get("session_id")
        
        return {"status": "ok", "streaming": session.streaming}
    except Exception as e:
        logger.error(f"Error starting video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/video/{session_id}/frame")
async def add_video_frame(session_id: str, request: VideoFrameRequest):
    """Add a frame to a streaming video session."""
    session = _get_video_session(session_id)
    
    if not session.streaming:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session is not in streaming mode"
        )
    
    try:
        image = decode_base64_image(request.image)
        frame_np = np.array(image)
        
        # Ensure we have enough slots
        while len(session.frames) <= request.frame_idx:
            session.frames.append(None)
        session.frames[request.frame_idx] = frame_np
        
        return {"status": "ok", "frame_idx": request.frame_idx}
    except Exception as e:
        logger.error(f"Error adding frame: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/video/{session_id}/prompt")
async def add_video_prompt(session_id: str, request: VideoPromptRequest):
    """Add a prompt to a video frame."""
    session = _get_video_session(session_id)
    
    try:
        # Lazy initialization for streaming mode
        if session.internal_session_id is None:
            if not session.streaming:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Video session not started. Call /video/{session_id}/start first."
                )
            
            # For streaming mode, check if we have frames to process
            valid_frames = [f for f in session.frames if f is not None]
            if not valid_frames:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No frames added. Call /video/{session_id}/frame first."
                )
            
            # Create temp directory and save frames
            logger.info(f"Initializing streaming session with {len(valid_frames)} frames")
            session.temp_dir = tempfile.mkdtemp(prefix="sam3_video_")
            
            # Save frames with contiguous indices and create mapping
            internal_idx = 0
            for external_idx, frame in enumerate(session.frames):
                if frame is not None:
                    frame_path = os.path.join(session.temp_dir, f"{internal_idx}.jpg")
                    Image.fromarray(frame).save(frame_path, quality=95)
                    session.frame_idx_mapping[external_idx] = internal_idx
                    internal_idx += 1
            
            # Initialize the predictor session with the temp directory
            response = session.predictor.handle_request({
                "type": "start_session",
                "resource_path": session.temp_dir,
            })
            session.internal_session_id = response.get("session_id")
            logger.info(f"Created internal video session: {session.internal_session_id}")
        
        # Map external frame index to internal SAM3 frame index
        internal_frame_idx = session.frame_idx_mapping.get(request.frame_idx)
        if internal_frame_idx is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Frame {request.frame_idx} was not uploaded to this session"
            )
        
        prompt_request = {
            "type": "add_prompt",
            "session_id": session.internal_session_id,
            "frame_index": internal_frame_idx,
        }
        
        if request.text:
            prompt_request["text"] = request.text
        if request.points:
            prompt_request["points"] = request.points
            prompt_request["point_labels"] = request.point_labels or [1] * len(request.points)
        if request.boxes:
            prompt_request["bounding_boxes"] = request.boxes
            prompt_request["bounding_box_labels"] = request.box_labels or [1] * len(request.boxes)
        
        response = session.predictor.handle_request(prompt_request)
        
        # Store outputs
        if "outputs" in response:
            session.outputs[request.frame_idx] = response["outputs"]
        
        return {"status": "ok", "frame_idx": request.frame_idx}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding video prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/video/{session_id}/propagate")
async def propagate_video(session_id: str, request: VideoPropagateRequest):
    """Propagate prompts through the video."""
    session = _get_video_session(session_id)
    
    try:
        if session.internal_session_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video session not started"
            )
        
        # Map direction to SAM3's expected format
        direction_map = {
            "forward": 1,
            "backward": -1,
            "both": 0,
        }
        direction = direction_map.get(request.direction, 1)
        
        response = session.predictor.propagate_in_video(
            session_id=session.internal_session_id,
            propagation_direction=direction,
            start_frame_idx=0,
            max_frame_num_to_track=request.max_frames,
        )
        
        # Store propagation outputs
        for frame_data in response:
            frame_idx = frame_data.get("frame_idx", 0)
            session.outputs[frame_idx] = frame_data
        
        return {"status": "ok", "frames_processed": len(response)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error propagating: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/video/{session_id}/results/{frame_idx}", response_model=SegmentationResult)
async def get_video_results(session_id: str, frame_idx: int):
    """Get segmentation results for a specific frame."""
    session = _get_video_session(session_id)
    
    if frame_idx not in session.outputs:
        return SegmentationResult(count=0, masks=[], boxes=[], scores=[])
    
    try:
        output = session.outputs[frame_idx]
        
        # Extract masks, boxes, scores from output
        masks_b64 = []
        boxes_list = []
        scores = []
        
        if "out_binary_masks" in output:
            for mask in output["out_binary_masks"]:
                masks_b64.append(encode_mask_to_base64(mask))
        
        if "out_boxes_xywh" in output:
            for box in output["out_boxes_xywh"]:
                # Convert xywh to xyxy
                x, y, w, h = box
                boxes_list.append([x, y, x + w, y + h])
        
        if "out_probs" in output:
            scores = [float(p) for p in output["out_probs"]]
        
        return SegmentationResult(
            count=len(masks_b64),
            masks=masks_b64,
            boxes=boxes_list,
            scores=scores,
        )
    except Exception as e:
        logger.error(f"Error getting video results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/video/{session_id}/visualize/{frame_idx}", response_model=VisualizeResponse)
async def visualize_video_frame(session_id: str, frame_idx: int, request: VisualizeRequest = None):
    """Get a video frame with segmentation overlay."""
    if request is None:
        request = VisualizeRequest()
    
    session = _get_video_session(session_id)
    
    try:
        # Get frame
        if frame_idx >= len(session.frames) or session.frames[frame_idx] is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Frame {frame_idx} not found"
            )
        
        frame = session.frames[frame_idx]
        output = session.outputs.get(frame_idx, {})
        
        if output and SAM3_AVAILABLE:
            # Use SAM3's visualization utility
            overlay = render_masklet_frame(frame, output, frame_idx=frame_idx, alpha=request.alpha)
        else:
            overlay = frame
        
        result_image = numpy_to_pil(overlay)
        return VisualizeResponse(image=encode_image_to_base64(result_image))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error visualizing video frame: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="SAM3 REST API Server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    parser.add_argument("--session-timeout", type=int, default=DEFAULT_SESSION_TIMEOUT,
                        help="Session timeout in seconds")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    # Store timeout in app state for lifespan to use
    app.state.session_timeout = args.session_timeout
    
    logger.info(f"Starting SAM3 Server on {args.host}:{args.port}")
    logger.info(f"Session timeout: {args.session_timeout}s")
    logger.info(f"SAM3 available: {SAM3_AVAILABLE}")
    logger.info(f"GPU available: {torch.cuda.is_available()}")
    
    uvicorn.run(
        "server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
