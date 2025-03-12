from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

from app.models import BookingCreate, BookingResponse, MessageRequest, MessageResponse, CalendarEvent, CalendarEventResponse
from app.database import init_db, get_all_bookings, get_booking_by_id, delete_booking, create_booking, get_db, BookingModel
from app.llm_processor import LLMProcessor

# Load environment variables
load_dotenv()

app = FastAPI(title="Technician Booking System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the database with sample data
init_db()

# Initialize LLM processor
llm_processor = LLMProcessor()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Technician Booking System API"}

@app.get("/bookings", response_model=List[BookingResponse])
def list_bookings(db: Session = Depends(get_db)):
    return get_all_bookings(db)

@app.get("/bookings/{booking_id}", response_model=BookingResponse)
def get_booking(booking_id: UUID, db: Session = Depends(get_db)):
    booking = get_booking_by_id(booking_id, db)
    if booking is None:
        raise HTTPException(status_code=404, detail="Booking not found")
    return booking

@app.delete("/bookings/{booking_id}")
def remove_booking(booking_id: UUID, db: Session = Depends(get_db)):
    success = delete_booking(booking_id, db)
    if not success:
        raise HTTPException(status_code=404, detail="Booking not found")
    return {"message": "Booking deleted successfully"}

@app.post("/bookings", response_model=BookingResponse)
def add_booking(booking: BookingCreate, db: Session = Depends(get_db)):
    try:
        new_booking = create_booking(
            booking.technician_name,
            booking.specialty,
            booking.booking_time,
            db
        )
        
        if new_booking:
            return new_booking
        else:
            raise HTTPException(status_code=400, detail="Could not create booking")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/chat", response_model=MessageResponse)
def process_message(request: MessageRequest):
    response = llm_processor.process_message(request.message)
    return MessageResponse(response=response)

@app.post("/api/messages/direct", response_model=MessageResponse)
async def process_message_direct(request: MessageRequest, db: Session = Depends(get_db)):
    """Process a message directly using OpenAI API, bypassing LangGraph."""
    try:
        # Get the message from the request
        message = request.message
        print(f"Processing message directly: {message}")
        
        # Check for direct commands first
        message_lower = message.lower()
        
        # Handle listing bookings
        if "list bookings" in message_lower or "show bookings" in message_lower or "show all bookings" in message_lower:
            bookings = get_all_bookings(db)
            if not bookings:
                return {"response": "You don't have any bookings scheduled."}
            
            response = "Here are your current bookings:\n\n"
            for booking in bookings:
                formatted_date = booking.booking_time.strftime("%A, %B %d at %I:%M %p")
                response += f"- {booking.specialty} with {booking.technician_name} on {formatted_date} (ID: {booking.id})\n"
            
            return {"response": response}
        
        # Handle canceling/deleting a booking
        elif "cancel booking" in message_lower or "delete booking" in message_lower:
            # Try to extract booking ID
            import re
            id_match = re.search(r'(?:id|ID|Id)[:\s]*([a-zA-Z0-9-]+)', message)
            
            if id_match:
                booking_id_str = id_match.group(1).strip()
                try:
                    from uuid import UUID
                    booking_id = UUID(booking_id_str)
                    success = delete_booking(booking_id, db)
                    
                    if success:
                        return {"response": f"Your booking with ID {booking_id} has been successfully cancelled."}
                    else:
                        return {"response": f"I couldn't find a booking with ID {booking_id}. Please check the ID and try again."}
                except ValueError:
                    return {"response": f"The booking ID {booking_id_str} is not valid. Please provide a valid booking ID."}
            else:
                return {"response": "Please provide the booking ID you want to cancel. For example: 'Cancel booking with ID 12345'."}
        
        # Handle booking details
        elif "booking details" in message_lower or "details for booking" in message_lower:
            # Try to extract booking ID
            import re
            id_match = re.search(r'(?:id|ID|Id)[:\s]*([a-zA-Z0-9-]+)', message)
            
            if id_match:
                booking_id_str = id_match.group(1).strip()
                try:
                    from uuid import UUID
                    booking_id = UUID(booking_id_str)
                    booking = get_booking_by_id(booking_id, db)
                    
                    if booking:
                        formatted_date = booking.booking_time.strftime("%A, %B %d at %I:%M %p")
                        response = f"Booking details:\n- ID: {booking.id}\n- Technician: {booking.technician_name}\n- Specialty: {booking.specialty}\n- Date and time: {formatted_date}"
                        return {"response": response}
                    else:
                        return {"response": f"I couldn't find a booking with ID {booking_id}. Please check the ID and try again."}
                except ValueError:
                    return {"response": f"The booking ID {booking_id_str} is not valid. Please provide a valid booking ID."}
            else:
                return {"response": "Please provide the booking ID you want to view details for. For example: 'Show details for booking with ID 12345'."}
        
        # Handle creating a new booking
        elif any(phrase in message_lower for phrase in ["book", "schedule", "reserve", "appointment", "make a booking"]) or (
            "," in message and any(specialty in message_lower for specialty in ["plumber", "electrician", "welder", "gardener", "carpenter"])):
            
            # Try to extract booking information
            import re
            import dateutil.parser
            from datetime import datetime, timedelta
            
            # Extract technician name, specialty, and date/time
            # Look for patterns like "Name, specialty, date time"
            booking_pattern = r'([a-zA-Z\s]+),\s*([a-zA-Z\s]+),\s*([a-zA-Z0-9\s:]+)'
            booking_match = re.search(booking_pattern, message)
            
            if booking_match:
                technician_name = booking_match.group(1).strip()
                specialty = booking_match.group(2).strip()
                date_time_str = booking_match.group(3).strip()
                
                try:
                    # Try to parse the date and time
                    booking_datetime = None
                    try:
                        # Try direct parsing
                        booking_datetime = dateutil.parser.parse(date_time_str)
                    except:
                        # If direct parsing fails, try with more flexible parsing
                        current_year = datetime.now().year
                        # Add current year if not specified
                        if str(current_year) not in date_time_str:
                            date_time_str = f"{date_time_str}, {current_year}"
                        try:
                            booking_datetime = dateutil.parser.parse(date_time_str)
                        except:
                            # If still fails, use tomorrow at the specified time or default to 5 PM
                            tomorrow = datetime.now() + timedelta(days=1)
                            # Try to extract time
                            time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)', date_time_str.lower())
                            if time_match:
                                hour = int(time_match.group(1))
                                minute = int(time_match.group(2) or 0)
                                am_pm = time_match.group(3)
                                
                                # Convert to 24-hour format
                                if am_pm == 'pm' and hour < 12:
                                    hour += 12
                                elif am_pm == 'am' and hour == 12:
                                    hour = 0
                                
                                booking_datetime = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
                            else:
                                # Default to 5 PM tomorrow
                                booking_datetime = tomorrow.replace(hour=17, minute=0, second=0, microsecond=0)
                    
                    # Create the booking
                    new_booking = create_booking(technician_name, specialty.capitalize(), booking_datetime, db)
                    
                    if new_booking:
                        formatted_date = booking_datetime.strftime("%A, %B %d at %I:%M %p")
                        formatted_end_time = (booking_datetime + timedelta(hours=1)).strftime("%I:%M %p")
                        # Eliminamos la integración con Google Calendar
                        return {"response": f"Great! I've booked a {specialty} technician ({technician_name}) for {formatted_date} to {formatted_end_time} (1 hour). Your booking ID is {new_booking.id}."}
                    else:
                        return {"response": f"Sorry, the {specialty} technician ({technician_name}) is not available at that time. They may already have another booking during this hour. Please try a different time or date."}
                
                except Exception as e:
                    print(f"Error creating booking: {e}")
                    return {"response": f"I couldn't create your booking due to an error: {str(e)}. Please try again with a clearer date and time format."}
            
            # If we couldn't extract booking information, ask for it
            return {"response": "To book a technician, please provide the following information in this format: 'Technician Name, Specialty, Date and Time'. For example: 'John Doe, Plumber, October 20 at 3 PM'."}
        
        # If no direct command is recognized, use OpenAI API
        # Get OpenAI API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not openai_api_key:
            print("WARNING: OpenAI API key is not set!")
            return {"response": "API key not configured. Please contact the administrator."}
        
        try:
            # Initialize the OpenAI client
            from openai import OpenAI
            client = OpenAI(api_key=openai_api_key)
            
            # Get all bookings to provide context to the AI
            bookings = get_all_bookings(db)
            bookings_context = "Current bookings:\n"
            if bookings:
                for booking in bookings:
                    formatted_date = booking.booking_time.strftime("%Y-%m-%d at %H:%M")
                    bookings_context += f"- ID: {booking.id}, Technician: {booking.technician_name}, Specialty: {booking.specialty}, Date: {formatted_date}\n"
            else:
                bookings_context += "No bookings found.\n"
            
            # Make a simple API call with context
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"""You are a helpful assistant for a technician booking system. 
You can help users book technicians, cancel bookings, list bookings, and get details about bookings.

{bookings_context}

When users want to:
1. List bookings - Tell them to say "list bookings"
2. Cancel a booking - Tell them to say "cancel booking with ID [booking_id]"
3. Get booking details - Tell them to say "show details for booking with ID [booking_id]"
4. Book a new technician - Tell them to provide the information in this format: "Technician Name, Specialty, Date and Time"
   For example: "John Doe, Plumber, October 20 at 3 PM"

Always be helpful and provide clear instructions."""},
                    {"role": "user", "content": message}
                ],
                temperature=0.7
            )
            
            # Extract the response
            response_text = response.choices[0].message.content.strip()
            print(f"OpenAI API response: {response_text}")
            
            return {"response": response_text}
        
        except Exception as e:
            import traceback
            print(f"Error calling OpenAI API: {e}")
            print(traceback.format_exc())
            return {"response": f"Error processing message: {str(e)}"}
    
    except Exception as e:
        import traceback
        print(f"Error in direct message processing: {e}")
        print(traceback.format_exc())
        return {"response": "Sorry, an error occurred while processing your message. Please try again."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 