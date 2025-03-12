# Technician Booking System - Backend

This is the backend for the Technician Booking System, built with FastAPI.

## Setup and Installation

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the FastAPI server:
   ```
   uvicorn app.main:app --reload
   ```

The backend will be available at http://localhost:8000

## API Endpoints

- `GET /bookings`: List all bookings
- `GET /bookings/{booking_id}`: Get a specific booking
- `DELETE /bookings/{booking_id}`: Delete a booking
- `POST /bookings`: Create a new booking
- `POST /chat`: Process a natural language message

## Features

- In-memory database with sample bookings
- Natural language processing for booking instructions
- RESTful API for booking management 