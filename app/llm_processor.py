import re
import json
from datetime import datetime, timedelta
import dateutil.parser
from typing import Dict, Any, Optional, Tuple, List
from uuid import UUID
import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END, START

from app.database import get_all_bookings, get_booking_by_id, delete_booking, create_booking, SessionLocal

# Load environment variables
load_dotenv()

# Define models for the output parser
class BookingInfo(BaseModel):
    action: str = Field(description="The action to perform: 'book', 'cancel', 'list', 'details', 'unknown'")
    technician_type: Optional[str] = Field(None, description="The type of technician to book")
    booking_date: Optional[str] = Field(None, description="The booking date in YYYY-MM-DD format")
    booking_time: Optional[str] = Field(None, description="The booking time in HH:MM format")
    booking_id: Optional[str] = Field(None, description="The booking ID to cancel or view details")
    response: Optional[str] = Field(None, description="The response for the user")

# Configure the output parser
parser = JsonOutputParser(pydantic_object=BookingInfo)

# Define the prompt for the LLM
prompt_template = """
Analyze the following user message and extract relevant information for a technician booking.

User message: {message}

Extract the following information:
1. The action the user wants to perform (book, cancel, list, details, unknown)
2. The type of technician needed (plumber, electrician, etc.)
3. The booking date (YYYY-MM-DD)
4. The booking time (HH:MM)
5. The booking ID (if cancelling or requesting details)

{format_instructions}
"""

# Class for processing messages with LangChain and LangGraph
class LLMProcessor:
    def __init__(self):
        self.chain = None
        self.graph = None
        self.workflow = None
        self.last_booking_id = None
        self.setup_langchain()
        self.setup_langgraph()
    
    def setup_langchain(self):
        """Configure LangChain components."""
        # Configure the prompt with format instructions
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["message"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Get OpenAI API key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        print(f"OpenAI API key: {openai_api_key[:5]}...{openai_api_key[-5:] if openai_api_key else ''}")
        
        if not openai_api_key:
            print("WARNING: OpenAI API key is not set!")
        
        try:
            # Configure the language model with the API key from environment
            from openai import OpenAI as DirectOpenAI
            self.openai_client = DirectOpenAI(api_key=openai_api_key)
            
            # Also set up LangChain for compatibility
            llm = OpenAI(
                openai_api_key=openai_api_key,
                model_name="gpt-3.5-turbo-instruct",  # Use a specific model
                temperature=0.7
            )
            
            # Create a runnable sequence instead of LLMChain
            from langchain_core.runnables import RunnablePassthrough
            self.chain = prompt | llm | parser
            print("LangChain setup completed successfully")
        except Exception as e:
            import traceback
            print(f"Error setting up LangChain: {e}")
            print(traceback.format_exc())
            raise
    
    def setup_langgraph(self):
        """Configure the LangGraph for the processing flow."""
        # Define the state graph
        self.workflow = StateGraph(BookingInfo)
        
        # Add nodes to the graph
        self.workflow.add_node("parse_intent", self._parse_intent)
        self.workflow.add_node("handle_booking", self._handle_booking_request)
        self.workflow.add_node("handle_cancellation", self._handle_cancellation_request)
        self.workflow.add_node("handle_list", self._handle_list_request)
        self.workflow.add_node("handle_details", self._handle_details_request)
        
        # Define the entry point
        self.workflow.add_edge(START, "parse_intent")
        
        # Define transitions between nodes
        self.workflow.add_conditional_edges(
            "parse_intent",
            self._route_from_parse_intent,
            {
                "book": "handle_booking",
                "cancel": "handle_cancellation",
                "list": "handle_list",
                "details": "handle_details",
                "unknown": END
            }
        )
        
        # Connect all handling nodes to the end
        self.workflow.add_edge("handle_booking", END)
        self.workflow.add_edge("handle_cancellation", END)
        self.workflow.add_edge("handle_list", END)
        self.workflow.add_edge("handle_details", END)
        
        # Compile the graph
        self.graph = self.workflow.compile()
    
    def process_message(self, message: str) -> str:
        """Process a user message and return a response."""
        try:
            # Execute the graph with the message as input
            print(f"Processing message: {message}")
            result = self.graph.invoke({"message": message})
            
            # Get the last state of the graph
            final_state = result.get("handle_booking") or result.get("handle_cancellation") or \
                          result.get("handle_list") or result.get("handle_details") or \
                          result.get("parse_intent")
            
            print(f"Final state: {final_state}")
            
            # If there is a response in the final state, return it
            if hasattr(final_state, "response") and final_state.response:
                return final_state.response
            
            # If there is no response, return a generic message
            return "I couldn't understand your request. Could you try again?"
        
        except Exception as e:
            import traceback
            print(f"Error processing message: {e}")
            print(f"Error type: {type(e)}")
            print(traceback.format_exc())
            return "Sorry, an error occurred while processing your message. Please try again."
    
    # Condition functions for the graph
    def _should_book(self, state: BookingInfo) -> bool:
        return state.action == "book"
    
    def _should_cancel(self, state: BookingInfo) -> bool:
        return state.action == "cancel"
    
    def _should_list(self, state: BookingInfo) -> bool:
        return state.action == "list"
    
    def _should_get_details(self, state: BookingInfo) -> bool:
        return state.action == "details"
    
    def _is_unknown(self, state: BookingInfo) -> bool:
        return state.action == "unknown"
    
    # Nodes of the graph
    def _parse_intent(self, state: Dict[str, Any]) -> BookingInfo:
        """Analyze the user's intent using LangChain."""
        message = state.get("message", "")
        print(f"Parsing intent for message: {message}")
        
        # Create a default BookingInfo with 'unknown' action as fallback
        default_booking_info = BookingInfo(
            action="unknown",
            response="I'm not sure what you're asking for. Could you please provide more details about booking, cancelling, or viewing technician appointments?"
        )
        
        try:
            # Use the LangChain chain to analyze the message
            print(f"Analyzing intent with LangChain: {message}")
            result = self.chain.invoke({"message": message})
            
            # Convert the result to a BookingInfo object
            if isinstance(result, str):
                result = json.loads(result)
            
            # Create a BookingInfo object and ensure at least action is set
            booking_info = BookingInfo(**result)
            if not booking_info.action:
                booking_info.action = "unknown"
            
            return booking_info
        
        except Exception as e:
            print(f"Error analyzing intent with LangChain: {e}")
            
            try:
                # Fallback to direct OpenAI API
                print("Falling back to direct OpenAI API")
                
                # Use chat completions instead of completions for better results
                try:
                    response = self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that extracts booking information."},
                            {"role": "user", "content": f"""
                            Analyze the following user message and extract relevant information for a technician booking.

                            User message: {message}

                            Extract the following information:
                            1. The action the user wants to perform (book, cancel, list, details, unknown)
                            2. The type of technician needed (plumber, electrician, etc.)
                            3. The booking date (YYYY-MM-DD)
                            4. The booking time (HH:MM)
                            5. The booking ID (if cancelling or requesting details)

                            Return ONLY a valid JSON object with the following structure:
                            {{
                                "action": "book|cancel|list|details|unknown",
                                "technician_type": "type of technician (if applicable)",
                                "booking_date": "YYYY-MM-DD (if applicable)",
                                "booking_time": "HH:MM (if applicable)",
                                "booking_id": "ID (if applicable)",
                                "response": "A response message for the user"
                            }}
                            """}
                        ],
                        temperature=0.7
                    )
                    result_text = response.choices[0].message.content.strip()
                except Exception as chat_error:
                    print(f"Error with chat completions: {chat_error}")
                    # Fall back to completions API
                    response = self.openai_client.completions.create(
                        model="gpt-3.5-turbo-instruct",
                        prompt=f"""
                        Analyze the following user message and extract relevant information for a technician booking.

                        User message: {message}

                        Extract the following information:
                        1. The action the user wants to perform (book, cancel, list, details, unknown)
                        2. The type of technician needed (plumber, electrician, etc.)
                        3. The booking date (YYYY-MM-DD)
                        4. The booking time (HH:MM)
                        5. The booking ID (if cancelling or requesting details)

                        Return ONLY a valid JSON object with the following structure:
                        {{
                            "action": "book|cancel|list|details|unknown",
                            "technician_type": "type of technician (if applicable)",
                            "booking_date": "YYYY-MM-DD (if applicable)",
                            "booking_time": "HH:MM (if applicable)",
                            "booking_id": "ID (if applicable)",
                            "response": "A response message for the user"
                        }}
                        """,
                        max_tokens=500,
                        temperature=0.7
                    )
                    result_text = response.choices[0].text.strip()
                
                # Extract the JSON from the response
                print(f"OpenAI API response: {result_text}")
                
                # Try to parse the JSON
                try:
                    result_json = json.loads(result_text)
                    booking_info = BookingInfo(**result_json)
                    
                    # Ensure action is set
                    if not booking_info.action:
                        booking_info.action = "unknown"
                    
                    return booking_info
                except json.JSONDecodeError:
                    # If JSON parsing fails, extract the JSON part from the text
                    import re
                    json_match = re.search(r'({.*})', result_text, re.DOTALL)
                    if json_match:
                        try:
                            result_json = json.loads(json_match.group(1))
                            booking_info = BookingInfo(**result_json)
                            
                            # Ensure action is set
                            if not booking_info.action:
                                booking_info.action = "unknown"
                            
                            return booking_info
                        except Exception as json_error:
                            print(f"Error parsing extracted JSON: {json_error}")
                
                # If we get here, return the default
                return default_booking_info
            
            except Exception as fallback_error:
                print(f"Error in fallback to OpenAI API: {fallback_error}")
                import traceback
                print(traceback.format_exc())
            
            # Return the default BookingInfo if all else fails
            return default_booking_info
    
    def _handle_booking_request(self, state: BookingInfo) -> BookingInfo:
        """Handle a booking request."""
        # Extract booking information
        specialty = state.technician_type
        if not specialty:
            state.response = "Please specify what type of technician you need (plumber, electrician, welder, gardener, etc.)"
            return state
        
        # Determine the booking date and time
        booking_date = state.booking_date
        booking_time = state.booking_time
        
        if booking_date and booking_time:
            try:
                booking_datetime = dateutil.parser.parse(f"{booking_date}T{booking_time}")
            except:
                # If the date/time cannot be parsed, use tomorrow at 5 PM
                tomorrow = datetime.now() + timedelta(days=1)
                booking_datetime = tomorrow.replace(hour=17, minute=0, second=0, microsecond=0)
        else:
            # By default, schedule for tomorrow at 5 PM
            tomorrow = datetime.now() + timedelta(days=1)
            booking_datetime = tomorrow.replace(hour=17, minute=0, second=0, microsecond=0)
        
        # Create a technician name based on the specialty
        technician_names = {
            "plumber": "Alex Waters",
            "electrician": "Jamie Sparks",
            "welder": "Sam Steel",
            "gardener": "Pat Green",
            "carpenter": "Chris Wood"
        }
        
        technician_name = technician_names.get(specialty.lower(), f"Professional {specialty.capitalize()}")
        
        # Create the booking in the database
        db = SessionLocal()
        booking = create_booking(technician_name, specialty.capitalize(), booking_datetime, db)
        
        if booking:
            self.last_booking_id = booking.id
            formatted_date = booking_datetime.strftime("%A, %B %d at %I:%M %p")
            state.response = f"Great! I've booked a {specialty} technician ({technician_name}) for {formatted_date}. Your booking ID is {booking.id}."
        else:
            state.response = f"Sorry, the {specialty} technician is not available at that time. Please try a different time or date."
        
        return state
    
    def _handle_cancellation_request(self, state: BookingInfo) -> BookingInfo:
        """Handle a cancellation request."""
        booking_id = state.booking_id
        
        if not booking_id:
            # If no ID is provided, use the last booking ID
            if self.last_booking_id:
                booking_id = str(self.last_booking_id)
            else:
                state.response = "Please provide the booking ID you want to cancel."
                return state
        
        try:
            # Convert the ID to UUID
            booking_id_uuid = UUID(booking_id)
            
            # Delete the booking
            db = SessionLocal()
            success = delete_booking(booking_id_uuid, db)
            
            if success:
                state.response = f"Your booking with ID {booking_id} has been successfully cancelled."
            else:
                state.response = f"I couldn't find a booking with ID {booking_id}. Please check the ID and try again."
        
        except ValueError:
            state.response = f"The booking ID {booking_id} is not valid. Please provide a valid booking ID."
        
        return state
    
    def _handle_list_request(self, state: BookingInfo) -> BookingInfo:
        """Handle a request to list all bookings."""
        db = SessionLocal()
        bookings = get_all_bookings(db)
        
        if not bookings:
            state.response = "You don't have any bookings scheduled."
            return state
        
        response = "Here are your current bookings:\n\n"
        for booking in bookings:
            formatted_date = booking.booking_time.strftime("%A, %B %d at %I:%M %p")
            response += f"- {booking.specialty} with {booking.technician_name} on {formatted_date} (ID: {booking.id})\n"
        
        state.response = response
        return state
    
    def _handle_details_request(self, state: BookingInfo) -> BookingInfo:
        """Handle a request to view details of a specific booking."""
        booking_id = state.booking_id
        
        if not booking_id:
            state.response = "Please provide the booking ID you want to view details for."
            return state
        
        try:
            # Convert the ID to UUID
            booking_id_uuid = UUID(booking_id)
            
            # Get the booking
            db = SessionLocal()
            booking = get_booking_by_id(booking_id_uuid, db)
            
            if booking:
                formatted_date = booking.booking_time.strftime("%A, %B %d at %I:%M %p")
                state.response = f"Booking details:\n- ID: {booking.id}\n- Technician: {booking.technician_name}\n- Specialty: {booking.specialty}\n- Date and time: {formatted_date}"
            else:
                state.response = f"I couldn't find a booking with ID {booking_id}. Please check the ID and try again."
        
        except ValueError:
            state.response = f"The booking ID {booking_id} is not valid. Please provide a valid booking ID."
        
        return state
    
    def _route_from_parse_intent(self, state: BookingInfo) -> str:
        """Route to the appropriate handler based on the action."""
        try:
            print(f"Routing from parse_intent with action: {state.action}")
            
            # Validate that action is one of the expected values
            valid_actions = ["book", "cancel", "list", "details", "unknown"]
            
            if state.action and state.action.lower() in valid_actions:
                return state.action.lower()
            else:
                print(f"Invalid action: {state.action}, defaulting to 'unknown'")
                # If action is None or not in valid_actions, return "unknown"
                return "unknown"
        except Exception as e:
            print(f"Error in _route_from_parse_intent: {e}")
            import traceback
            print(traceback.format_exc())
            # Default to unknown if there's an error
            return "unknown" 