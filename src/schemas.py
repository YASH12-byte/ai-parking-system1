from pydantic import BaseModel, Field
from typing import List, Optional


class Slot(BaseModel):
    id: str
    polygon: List[List[int]] = Field(..., description="List of [x,y] points defining ROI")


class DetectionResult(BaseModel):
    slot_id: str
    occupied: bool
    confidence: float


class AssignmentRequest(BaseModel):
    vehicle_size: float = Field(1.0, ge=0.1, le=3.0)
    distance_to_gate: float = Field(10.0, ge=0.0, le=1000.0)
    user_priority: float = Field(0.5, ge=0.0, le=1.0)
    available_slots: List[str]


class AssignmentResponse(BaseModel):
    slot_id: Optional[str]
    score: Optional[float]


