from sqlalchemy import create_engine, Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID, uuid4
import dateutil.parser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/technician_booking")

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the base for models
Base = declarative_base()

# SQLAlchemy model for bookings
class BookingModel(Base):
    __tablename__ = "bookings"

    id = Column(PostgresUUID, primary_key=True, default=uuid4)
    technician_name = Column(String, nullable=False)
    specialty = Column(String, nullable=False)
    booking_time = Column(DateTime, nullable=False)

# Create tables in the database
Base.metadata.create_all(bind=engine)

# Get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize with sample data
def init_db():
    db = SessionLocal()
    
    # Check if there is already data
    if db.query(BookingModel).count() == 0:
        sample_bookings = [
            BookingModel(
                id=uuid4(),
                technician_name="Nicolas Woollett",
                specialty="Plumber",
                booking_time=dateutil.parser.parse("2022-10-15T10:00:00")
            ),
            BookingModel(
                id=uuid4(),
                technician_name="Franky Flay",
                specialty="Electrician",
                booking_time=dateutil.parser.parse("2022-10-16T18:00:00")
            ),
            BookingModel(
                id=uuid4(),
                technician_name="Griselda Dickson",
                specialty="Welder",
                booking_time=dateutil.parser.parse("2022-10-18T11:00:00")
            )
        ]
        
        for booking in sample_bookings:
            db.add(booking)
        
        db.commit()
    
    db.close()

# CRUD operations
def get_all_bookings(db=None) -> List[BookingModel]:
    if db is None:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False
    
    bookings = db.query(BookingModel).all()
    
    if should_close:
        db.close()
    
    return bookings

def get_booking_by_id(booking_id: UUID, db=None) -> Optional[BookingModel]:
    if db is None:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False
    
    booking = db.query(BookingModel).filter(BookingModel.id == booking_id).first()
    
    if should_close:
        db.close()
    
    return booking

def delete_booking(booking_id: UUID, db=None) -> bool:
    if db is None:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False
    
    booking = db.query(BookingModel).filter(BookingModel.id == booking_id).first()
    if booking:
        db.delete(booking)
        db.commit()
        success = True
    else:
        success = False
    
    if should_close:
        db.close()
    
    return success

def create_booking(technician_name: str, specialty: str, booking_time: datetime, db=None) -> Optional[BookingModel]:
    if db is None:
        db = SessionLocal()
        should_close = True
    else:
        should_close = False
    
    # Define the booking end time (1 hour after start time)
    booking_end_time = booking_time + timedelta(hours=1)
    
    # Check if the technician is already booked during this time period
    # A technician is considered unavailable if:
    # 1. An existing booking starts during our new booking time slot, OR
    # 2. An existing booking ends during our new booking time slot, OR
    # 3. An existing booking completely overlaps our new booking time slot
    existing_bookings = db.query(BookingModel).filter(
        BookingModel.technician_name == technician_name,
        # Check for any overlap between existing bookings and the new booking
        # Existing booking starts before our booking ends AND
        # Existing booking ends after our booking starts
        ((BookingModel.booking_time < booking_end_time) & 
         (BookingModel.booking_time + timedelta(hours=1) > booking_time))
    ).all()
    
    if existing_bookings:
        if should_close:
            db.close()
        return None
    
    # Create new booking
    new_booking = BookingModel(
        id=uuid4(),
        technician_name=technician_name,
        specialty=specialty,
        booking_time=booking_time
    )
    
    db.add(new_booking)
    db.commit()
    db.refresh(new_booking)
    
    if should_close:
        db.close()
    
    return new_booking 