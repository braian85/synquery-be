from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from uuid import UUID, uuid4

# Pydantic models for the API
class BookingBase(BaseModel):
    technician_name: str
    specialty: str
    booking_time: datetime

class BookingCreate(BookingBase):
    pass

class BookingResponse(BookingBase):
    id: UUID
    
    class Config:
        from_attributes = True

class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    response: str

# Models for Google Calendar
class CalendarEvent(BaseModel):
    summary: str
    description: str
    start_time: datetime
    end_time: datetime
    attendees: Optional[List[str]] = None

class CalendarEventResponse(BaseModel):
    event_id: str
    html_link: str 