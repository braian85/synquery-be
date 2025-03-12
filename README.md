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

4. Set up the PostgreSQL database:
   
   The project includes a `docker-compose.yml` file to easily set up a PostgreSQL database:
   
   ```
   # Start the PostgreSQL container
   docker-compose up -d
   ```
   
   This will start a PostgreSQL instance with the following configuration:
   - Username: postgres
   - Password: postgres
   - Database: technician_booking
   - Port: 5432
   
   You can verify the database is running with:
   ```
   docker ps
   ```

5. Set up environment variables:
   Create a `.env` file in the backend directory with:
   ```
   # Database connection string
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/technician_booking
   
   # OpenAI API key for chat functionality
   OPENAI_API_KEY=your_openai_api_key_here
   ```

6. Run the FastAPI server:
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
- `POST /api/messages/direct`: Process messages directly with OpenAI

## Environment Variables

The application requires the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5432/technician_booking` |
| `OPENAI_API_KEY` | API key for OpenAI services | None (Required) |

You can set these variables in a `.env` file or directly in your deployment environment.

## Database Setup with Docker

The application uses PostgreSQL as its database. A `docker-compose.yml` file is included for easy setup:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    container_name: synquery-postgres
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: technician_booking
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Managing the Database

- **Start the database**: `docker-compose up -d`
- **Stop the database**: `docker-compose down`
- **View database logs**: `docker-compose logs postgres`
- **Access the database directly**:
  ```
  docker exec -it synquery-postgres psql -U postgres -d technician_booking
  ```

## Testing OpenAI Integration

The repository includes a test script `test_openai.py` to verify your OpenAI API key and connection:

1. Make sure your `.env` file contains a valid `OPENAI_API_KEY`
2. Run the test script:
   ```
   python test_openai.py
   ```
3. If successful, you'll see a message confirming the API connection and a sample response
4. If there's an error, the script will display detailed error information

This script is useful for:
- Verifying your OpenAI API key is valid
- Testing connectivity to OpenAI services
- Troubleshooting API integration issues before running the full application

## Features

- PostgreSQL database with sample bookings
- Natural language processing for booking instructions
- RESTful API for booking management 