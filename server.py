# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# SAM3 Server - Unified REST API for LiGuard-Web Integration
"""
SAM3 Server: A FastAPI-based REST API for the Segment Anything Model 3.

This server provides a unified API for image and video segmentation,
designed for integration with LiGuard-Web's node-based pipeline system.

Usage:
    conda activate sam3
    python server.py [--host 0.0.0.0] [--port 8765] [--session-timeout 300]

The server manages sessions with automatic cleanup of idle sessions.
All image I/O is done via base64-encoded strings for portability.

Unified API:
    POST /api/connect              - Create a new session
    GET  /api/{session_id}/is-alive - Check if session is alive
    POST /api/{session_id}/set-prompt - Set text prompt (queues if no images)
    POST /api/{session_id}/add-image  - Add image/frame
    GET  /api/{session_id}/get-output/{frame_idx} - Get raw outputs
    POST /api/{session_id}/visualize  - Get visualization
    DELETE /api/{session_id}/disconnect - Close session
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import random
import shutil
import string
import tempfile
import threading
import time
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
SESSION_ID_LENGTH = 14


# =============================================================================
# Pydantic Models for Request/Response
# =============================================================================

class ModelType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


# Unified API Models
class ConnectRequest(BaseModel):
    model_type: str = "image"
    timeout: int = 30
    device: str = "cuda"


class ConnectResponse(BaseModel):
    session_id: str
    error: str = ""


class IsAliveResponse(BaseModel):
    alive: bool


class SetPromptRequest(BaseModel):
    prompt: str


class SetPromptResponse(BaseModel):
    session_id: str
    queued: bool = False


class AddImageRequest(BaseModel):
    image: str  # base64 encoded
    batch: int = 1


class AddImageResponse(BaseModel):
    session_id: str
    added: bool
    frame_idx: int = 0


class GetOutputResponse(BaseModel):
    session_id: str
    count: int = 0
    masks: List[str] = []  # base64 encoded PNGs
    boxes: List[List[float]] = []  # [[x1, y1, x2, y2], ...]
    scores: List[float] = []


class VisualizeRequest(BaseModel):
    frame_idx: int = 0
    alpha: float = 0.5


class VisualizeResponse(BaseModel):
    session_id: str
    image: str = ""  # base64 encoded


class DisconnectResponse(BaseModel):
    disconnected: bool


class HealthResponse(BaseModel):
    status: str
    sam3_available: bool
    gpu_available: bool
    active_sessions: int


# =============================================================================
# Session Management
# =============================================================================

def generate_session_id(length: int = SESSION_ID_LENGTH) -> str:
    """Generate a secure alphanumeric session ID."""
    chars = string.ascii_letters + string.digits
    return ''.join(random.SystemRandom().choice(chars) for _ in range(length))


@dataclass
class UnifiedSession:
    """Unified session for both image and video segmentation."""
    session_id: str
    model_type: str  # "image" or "video"
    device: str = "cuda"
    timeout: int = 30
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    # Model references (shared, not owned)
    image_model: Any = None
    image_processor: Optional[Any] = None
    video_predictor: Any = None
    
    # Shared state
    pending_prompt: Optional[str] = None  # Queued until images arrive
    frames: List[Optional[np.ndarray]] = field(default_factory=list)
    outputs: Dict[int, Dict] = field(default_factory=dict)
    batch_size: int = 1
    current_batch: List[np.ndarray] = field(default_factory=list)
    
    # Image mode state
    image_state: Optional[Dict] = None
    
    # Video mode state
    internal_session_id: Optional[str] = None
    temp_dir: Optional[str] = None
    frame_idx_mapping: Dict[int, int] = field(default_factory=dict)
    frames_at_init: int = 0
    
    def touch(self) -> None:
        """Update last accessed time."""
        self.last_accessed = time.time()
    
    def needs_video_reinit(self) -> bool:
        """Check if video session needs re-initialization due to new frames."""
        if self.internal_session_id is None:
            return False
        current_frame_count = sum(1 for f in self.frames if f is not None)
        return current_frame_count != self.frames_at_init


class SessionManager:
    """
    Manages unified SAM3 sessions with automatic cleanup of idle sessions.
    Thread-safe implementation for concurrent access.
    """
    
    def __init__(self, timeout_seconds: int = DEFAULT_SESSION_TIMEOUT):
        self.timeout_seconds = timeout_seconds
        self._sessions: Dict[str, UnifiedSession] = {}
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
        expired = []
        
        with self._lock:
            for sid, session in self._sessions.items():
                if now - session.last_accessed > self.timeout_seconds:
                    expired.append(sid)
            
            for sid in expired:
                logger.info(f"Cleaning up expired session: {sid}")
                self._close_session_internal(sid)
        
        if expired:
            self._maybe_unload_models()
    
    def _close_session_internal(self, session_id: str) -> None:
        """Close a session (must hold lock)."""
        session = self._sessions.pop(session_id, None)
        if session:
            # Clean up video session resources
            if session.video_predictor and session.internal_session_id:
                try:
                    session.video_predictor.close_session(session.internal_session_id)
                except Exception as e:
                    logger.warning(f"Error closing video session: {e}")
            
            # Clean up temp directory
            if session.temp_dir:
                try:
                    shutil.rmtree(session.temp_dir)
                    logger.info(f"Cleaned up temp directory: {session.temp_dir}")
                except Exception as e:
                    logger.warning(f"Error cleaning up temp directory: {e}")
    
    def _maybe_unload_models(self) -> None:
        """Unload models if no sessions are active."""
        with self._lock:
            if not self._sessions:
                with self._model_lock:
                    if self._image_model is not None:
                        logger.info("Unloading image model (no active sessions)")
                        del self._image_model
                        self._image_model = None
                    
                    if self._video_predictor is not None:
                        logger.info("Unloading video predictor (no active sessions)")
                        if hasattr(self._video_predictor, 'shutdown'):
                            self._video_predictor.shutdown()
                        del self._video_predictor
                        self._video_predictor = None
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    def _get_image_model(self, device: str = "cuda") -> Tuple[Any, Any]:
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
    
    def create_session(self, model_type: str, device: str = "cuda", timeout: int = 30) -> UnifiedSession:
        """Create a new unified session."""
        session_id = generate_session_id()
        
        session = UnifiedSession(
            session_id=session_id,
            model_type=model_type,
            device=device,
            timeout=timeout,
        )
        
        # Load appropriate model
        if model_type == "image":
            model, processor = self._get_image_model(device)
            session.image_model = model
            session.image_processor = processor
        else:
            predictor = self._get_video_predictor(device)
            session.video_predictor = predictor
        
        with self._lock:
            self._sessions[session_id] = session
        
        logger.info(f"Created {model_type} session: {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[UnifiedSession]:
        """Get a session by ID and refresh its expiry."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session
    
    def close_session(self, session_id: str) -> bool:
        """Close a session and release resources."""
        with self._lock:
            if session_id in self._sessions:
                self._close_session_internal(session_id)
                self._maybe_unload_models()
                logger.info(f"Closed session: {session_id}")
                return True
            return False
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        with self._lock:
            return session_id in self._sessions
    
    def get_stats(self) -> Dict[str, int]:
        """Get session statistics."""
        with self._lock:
            image_count = sum(1 for s in self._sessions.values() if s.model_type == "image")
            video_count = sum(1 for s in self._sessions.values() if s.model_type == "video")
            return {
                "image_sessions": image_count,
                "video_sessions": video_count,
                "total": len(self._sessions),
            }
    
    def shutdown(self) -> None:
        """Shutdown all sessions and unload models."""
        with self._lock:
            for sid in list(self._sessions.keys()):
                self._close_session_internal(sid)
            self._maybe_unload_models()


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
    description="Unified REST API for Segment Anything Model 3",
    version="2.0.0",
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
# Unified API Endpoints
# =============================================================================

def _get_session(session_id: str) -> UnifiedSession:
    """Helper to get a session or raise 404."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return session


@app.post("/api/connect", response_model=ConnectResponse)
async def connect(request: ConnectRequest):
    """Create a new segmentation session."""
    if not SAM3_AVAILABLE:
        return ConnectResponse(session_id="", error="SAM3 is not available in this environment")
    
    try:
        model_type = request.model_type.lower()
        if model_type not in ["image", "video"]:
            return ConnectResponse(session_id="", error=f"Invalid model_type: {model_type}")
        
        session = session_manager.create_session(
            model_type=model_type,
            device=request.device,
            timeout=request.timeout,
        )
        
        return ConnectResponse(session_id=session.session_id)
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return ConnectResponse(session_id="", error=str(e))


@app.get("/api/{session_id}/is-alive", response_model=IsAliveResponse)
async def is_alive(session_id: str):
    """Check if a session is alive."""
    # This also refreshes the session expiry
    exists = session_manager.session_exists(session_id)
    if exists:
        session_manager.get_session(session_id)  # Touch to refresh
    return IsAliveResponse(alive=exists)


@app.post("/api/{session_id}/set-prompt", response_model=SetPromptResponse)
async def set_prompt(session_id: str, request: SetPromptRequest):
    """Set the text prompt for segmentation."""
    session = _get_session(session_id)
    
    try:
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prompt cannot be empty"
            )
        
        # Check if we have images to apply prompt to
        has_images = len(session.frames) > 0 or session.image_state is not None
        
        if session.model_type == "image":
            if session.image_state is None:
                # Queue the prompt for when image arrives
                session.pending_prompt = prompt
                logger.info(f"Session {session_id}: Queued prompt '{prompt}' (no image yet)")
                return SetPromptResponse(session_id=session_id, queued=True)
            else:
                # Apply prompt immediately
                session.image_state = session.image_processor.set_text_prompt(prompt, session.image_state)
                logger.info(f"Session {session_id}: Applied text prompt '{prompt}'")
                return SetPromptResponse(session_id=session_id, queued=False)
        else:
            # Video mode - queue prompt, will be applied when frames are processed
            session.pending_prompt = prompt
            logger.info(f"Session {session_id}: Queued video prompt '{prompt}'")
            return SetPromptResponse(session_id=session_id, queued=True)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/{session_id}/add-image", response_model=AddImageResponse)
async def add_image(session_id: str, request: AddImageRequest):
    """Add an image for segmentation."""
    session = _get_session(session_id)
    
    try:
        image = decode_base64_image(request.image)
        image_np = np.array(image)
        batch_size = max(1, request.batch)
        
        session.batch_size = batch_size
        frame_idx = len(session.frames)
        session.frames.append(image_np)
        session.current_batch.append(image_np)
        
        # Check if batch is complete
        batch_complete = len(session.current_batch) >= batch_size
        
        if session.model_type == "image":
            if batch_complete:
                # For image mode, set the image(s)
                if batch_size == 1:
                    # Single image
                    session.image_state = session.image_processor.set_image(image)
                else:
                    # Batched input - use the last complete batch
                    # Note: SAM3 image processor may need custom handling for batches
                    session.image_state = session.image_processor.set_image(image)
                
                # Store original for visualization
                if session.image_state is None:
                    session.image_state = {}
                session.image_state["original_image"] = image
                session.image_state["original_width"] = image.width
                session.image_state["original_height"] = image.height
                
                # Apply pending prompt if any
                if session.pending_prompt:
                    session.image_state = session.image_processor.set_text_prompt(
                        session.pending_prompt, session.image_state
                    )
                    logger.info(f"Session {session_id}: Applied pending prompt '{session.pending_prompt}'")
                    session.pending_prompt = None
                
                session.current_batch = []
                logger.info(f"Session {session_id}: Image set (batch complete, idx={frame_idx})")
                return AddImageResponse(session_id=session_id, added=True, frame_idx=frame_idx)
            else:
                logger.info(f"Session {session_id}: Image added to batch ({len(session.current_batch)}/{batch_size})")
                return AddImageResponse(session_id=session_id, added=False, frame_idx=frame_idx)
        
        else:
            # Video mode - sliding window approach
            if batch_complete:
                # Initialize video session if needed
                if session.internal_session_id is None or session.needs_video_reinit():
                    await _initialize_video_session(session)
                
                # Apply pending prompt to first frame if any
                if session.pending_prompt and session.internal_session_id:
                    internal_frame_idx = session.frame_idx_mapping.get(0, 0)
                    prompt_request = {
                        "type": "add_prompt",
                        "session_id": session.internal_session_id,
                        "frame_index": internal_frame_idx,
                        "text": session.pending_prompt,
                    }
                    response = session.video_predictor.handle_request(prompt_request)
                    if "outputs" in response:
                        session.outputs[0] = response["outputs"]
                    logger.info(f"Session {session_id}: Applied pending prompt '{session.pending_prompt}'")
                    session.pending_prompt = None
                
                session.current_batch = []
                logger.info(f"Session {session_id}: Video batch complete (idx={frame_idx})")
                return AddImageResponse(session_id=session_id, added=True, frame_idx=frame_idx)
            else:
                logger.info(f"Session {session_id}: Frame added to batch ({len(session.current_batch)}/{batch_size})")
                return AddImageResponse(session_id=session_id, added=False, frame_idx=frame_idx)
    
    except Exception as e:
        logger.error(f"Error adding image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


async def _initialize_video_session(session: UnifiedSession) -> None:
    """Initialize the video predictor session."""
    # Close old session if reinitializing
    if session.internal_session_id:
        try:
            session.video_predictor.close_session(session.internal_session_id)
        except Exception as e:
            logger.warning(f"Error closing old session: {e}")
    
    # Clean up old temp directory
    if session.temp_dir:
        try:
            shutil.rmtree(session.temp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temp dir: {e}")
    
    # Reset state
    session.frame_idx_mapping.clear()
    session.outputs.clear()
    
    # Create temp directory and save frames
    valid_frames = [f for f in session.frames if f is not None]
    if not valid_frames:
        return
    
    session.temp_dir = tempfile.mkdtemp(prefix="sam3_video_")
    
    # Save frames with contiguous indices and create mapping
    internal_idx = 0
    for external_idx, frame in enumerate(session.frames):
        if frame is not None:
            frame_path = os.path.join(session.temp_dir, f"{internal_idx}.jpg")
            Image.fromarray(frame).save(frame_path, quality=95)
            session.frame_idx_mapping[external_idx] = internal_idx
            internal_idx += 1
    
    # Initialize the predictor session
    response = session.video_predictor.handle_request({
        "type": "start_session",
        "resource_path": session.temp_dir,
    })
    session.internal_session_id = response.get("session_id")
    session.frames_at_init = len(valid_frames)
    logger.info(f"Created internal video session: {session.internal_session_id}")


@app.get("/api/{session_id}/get-output/{frame_idx}", response_model=GetOutputResponse)
async def get_output(session_id: str, frame_idx: int):
    """Get segmentation outputs for a specific frame."""
    session = _get_session(session_id)
    
    try:
        if session.model_type == "image":
            if session.image_state is None:
                return GetOutputResponse(session_id=session_id, count=0)
            
            masks_tensor = session.image_state.get("masks", [])
            boxes_tensor = session.image_state.get("boxes", [])
            scores_list = session.image_state.get("scores", [])
            
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
            
            return GetOutputResponse(
                session_id=session_id,
                count=len(masks_b64),
                masks=masks_b64,
                boxes=boxes_list,
                scores=scores,
            )
        
        else:
            # Video mode
            if frame_idx not in session.outputs:
                # Try to get output if we have a video session
                if session.internal_session_id and frame_idx in session.frame_idx_mapping:
                    internal_idx = session.frame_idx_mapping[frame_idx]
                    # Propagate if needed
                    try:
                        response = session.video_predictor.propagate_in_video(
                            session_id=session.internal_session_id,
                            propagation_direction=0,  # both
                            start_frame_idx=0,
                        )
                        for frame_data in response:
                            fidx = frame_data.get("frame_idx", 0)
                            session.outputs[fidx] = frame_data
                    except Exception as e:
                        logger.warning(f"Error propagating: {e}")
            
            output = session.outputs.get(frame_idx, {})
            
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
            
            return GetOutputResponse(
                session_id=session_id,
                count=len(masks_b64),
                masks=masks_b64,
                boxes=boxes_list,
                scores=scores,
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting output: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/api/{session_id}/visualize", response_model=VisualizeResponse)
async def visualize(session_id: str, request: VisualizeRequest):
    """Get visualization of segmentation results."""
    session = _get_session(session_id)
    
    try:
        import cv2
        
        if session.model_type == "image":
            if session.image_state is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No image set"
                )
            
            # Get original image
            original = session.image_state.get("original_image")
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
            
            masks = session.image_state.get("masks", [])
            boxes = session.image_state.get("boxes", [])
            scores = session.image_state.get("scores", [])
            
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
            
            return VisualizeResponse(
                session_id=session_id,
                image=encode_image_to_base64(result_image)
            )
        
        else:
            # Video mode
            frame_idx = request.frame_idx
            
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
            return VisualizeResponse(
                session_id=session_id,
                image=encode_image_to_base64(result_image)
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error visualizing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.delete("/api/{session_id}/disconnect", response_model=DisconnectResponse)
async def disconnect(session_id: str):
    """Close a session and release resources."""
    closed = session_manager.close_session(session_id)
    return DisconnectResponse(disconnected=closed)


# =============================================================================
# Legacy Endpoints (for backward compatibility)
# =============================================================================

# Keep the old endpoints working for now
class LegacySessionCreateRequest(BaseModel):
    model_type: ModelType = ModelType.IMAGE
    device: str = "cuda"


class LegacySessionResponse(BaseModel):
    session_id: str
    model_type: ModelType
    created_at: float
    last_accessed: float


@app.post("/sessions", response_model=LegacySessionResponse)
async def create_session_legacy(request: LegacySessionCreateRequest):
    """Create a new segmentation session (legacy endpoint)."""
    if not SAM3_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SAM3 is not available in this environment"
        )
    
    try:
        session = session_manager.create_session(
            model_type=request.model_type.value,
            device=request.device,
        )
        
        return LegacySessionResponse(
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
async def close_session_legacy(session_id: str):
    """Close a session and release resources (legacy endpoint)."""
    if not session_manager.close_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    
    return {"status": "closed", "session_id": session_id}


@app.post("/sessions/{session_id}/heartbeat")
async def session_heartbeat(session_id: str):
    """Keep a session alive (legacy endpoint)."""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )
    return {"status": "alive", "session_id": session_id}


# Image endpoints (legacy)
class LegacyImageSetRequest(BaseModel):
    image: str  # base64 encoded


class LegacyTextPromptRequest(BaseModel):
    text: str
    confidence_threshold: float = 0.5


@app.post("/image/{session_id}/set")
async def set_image_legacy(session_id: str, request: LegacyImageSetRequest):
    """Set the image for segmentation (legacy endpoint)."""
    session = _get_session(session_id)
    
    if session.model_type != "image":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This endpoint is for image sessions only"
        )
    
    try:
        image = decode_base64_image(request.image)
        session.image_state = session.image_processor.set_image(image)
        
        # Store original image
        if session.image_state is None:
            session.image_state = {}
        session.image_state["original_image"] = image
        session.image_state["original_width"] = image.width
        session.image_state["original_height"] = image.height
        
        session.frames.append(np.array(image))
        
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
async def add_text_prompt_legacy(session_id: str, request: LegacyTextPromptRequest):
    """Add a text prompt and run inference (legacy endpoint)."""
    session = _get_session(session_id)
    
    if session.model_type != "image":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This endpoint is for image sessions only"
        )
    
    if session.image_state is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image set. Call /image/{session_id}/set first."
        )
    
    try:
        session.image_state = session.image_processor.set_confidence_threshold(
            request.confidence_threshold, session.image_state
        )
        session.image_state = session.image_processor.set_text_prompt(request.text, session.image_state)
        
        masks = session.image_state.get("masks", [])
        return {"status": "ok", "objects_found": len(masks)}
    except Exception as e:
        logger.error(f"Error adding text prompt: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/image/{session_id}/results")
async def get_results_legacy(session_id: str):
    """Get segmentation results (legacy endpoint)."""
    result = await get_output(session_id, 0)
    return {
        "count": result.count,
        "masks": result.masks,
        "boxes": result.boxes,
        "scores": result.scores,
    }


class LegacyVisualizeRequest(BaseModel):
    alpha: float = 0.5


@app.post("/image/{session_id}/visualize")
async def visualize_image_legacy(session_id: str, request: LegacyVisualizeRequest = None):
    """Get the image with segmentation overlay (legacy endpoint)."""
    if request is None:
        request = LegacyVisualizeRequest()
    
    result = await visualize(session_id, VisualizeRequest(frame_idx=0, alpha=request.alpha))
    return {"image": result.image}


@app.post("/image/{session_id}/reset")
async def reset_prompts_legacy(session_id: str):
    """Clear all prompts (legacy endpoint)."""
    session = _get_session(session_id)
    
    if session.image_state is None:
        return {"status": "ok", "message": "No state to reset"}
    
    try:
        session.image_state = session.image_processor.reset_all_prompts(session.image_state)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error resetting prompts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# Video endpoints (legacy)
class LegacyVideoStartRequest(BaseModel):
    streaming: bool = True
    video_path: Optional[str] = None


class LegacyVideoFrameRequest(BaseModel):
    image: str  # base64 encoded
    frame_idx: int


class LegacyVideoPromptRequest(BaseModel):
    frame_idx: int
    text: Optional[str] = None
    points: Optional[List[List[float]]] = None
    point_labels: Optional[List[int]] = None
    boxes: Optional[List[List[float]]] = None
    box_labels: Optional[List[int]] = None


@app.post("/video/{session_id}/start")
async def start_video_legacy(session_id: str, request: LegacyVideoStartRequest):
    """Start a video session (legacy endpoint)."""
    session = _get_session(session_id)
    
    if session.model_type != "video":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This endpoint is for video sessions only"
        )
    
    try:
        if not request.streaming and request.video_path:
            response = session.video_predictor.handle_request({
                "type": "start_session",
                "resource_path": request.video_path,
            })
            session.internal_session_id = response.get("session_id")
        
        return {"status": "ok", "streaming": request.streaming}
    except Exception as e:
        logger.error(f"Error starting video: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post("/video/{session_id}/frame")
async def add_video_frame_legacy(session_id: str, request: LegacyVideoFrameRequest):
    """Add a frame to a streaming video session (legacy endpoint)."""
    session = _get_session(session_id)
    
    if session.model_type != "video":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This endpoint is for video sessions only"
        )
    
    try:
        image = decode_base64_image(request.image)
        frame_np = np.array(image)
        
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
async def add_video_prompt_legacy(session_id: str, request: LegacyVideoPromptRequest):
    """Add a prompt to a video frame (legacy endpoint)."""
    session = _get_session(session_id)
    
    if session.model_type != "video":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This endpoint is for video sessions only"
        )
    
    try:
        # Re-init if needed
        if session.needs_video_reinit():
            await _initialize_video_session(session)
        
        # Lazy initialization for streaming mode
        if session.internal_session_id is None:
            valid_frames = [f for f in session.frames if f is not None]
            if not valid_frames:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No frames added"
                )
            await _initialize_video_session(session)
        
        # Map frame index
        internal_frame_idx = session.frame_idx_mapping.get(request.frame_idx)
        if internal_frame_idx is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Frame {request.frame_idx} was not uploaded"
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
        
        response = session.video_predictor.handle_request(prompt_request)
        
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


class LegacyVideoPropagateRequest(BaseModel):
    direction: str = "forward"
    max_frames: Optional[int] = None


@app.post("/video/{session_id}/propagate")
async def propagate_video_legacy(session_id: str, request: LegacyVideoPropagateRequest):
    """Propagate prompts through the video (legacy endpoint)."""
    session = _get_session(session_id)
    
    if session.model_type != "video":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This endpoint is for video sessions only"
        )
    
    try:
        if session.internal_session_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Video session not started"
            )
        
        direction_map = {"forward": 1, "backward": -1, "both": 0}
        direction = direction_map.get(request.direction, 1)
        
        response = session.video_predictor.propagate_in_video(
            session_id=session.internal_session_id,
            propagation_direction=direction,
            start_frame_idx=0,
            max_frame_num_to_track=request.max_frames,
        )
        
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


@app.get("/video/{session_id}/results/{frame_idx}")
async def get_video_results_legacy(session_id: str, frame_idx: int):
    """Get segmentation results for a specific frame (legacy endpoint)."""
    result = await get_output(session_id, frame_idx)
    return {
        "count": result.count,
        "masks": result.masks,
        "boxes": result.boxes,
        "scores": result.scores,
    }


@app.post("/video/{session_id}/visualize/{frame_idx}")
async def visualize_video_frame_legacy(session_id: str, frame_idx: int, request: LegacyVisualizeRequest = None):
    """Get a video frame with segmentation overlay (legacy endpoint)."""
    if request is None:
        request = LegacyVisualizeRequest()
    
    result = await visualize(session_id, VisualizeRequest(frame_idx=frame_idx, alpha=request.alpha))
    return {"image": result.image}


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
