from flask import Flask, request, jsonify
from datetime import datetime, timedelta;
import spacy;
import dateutil.parser
import re

app = Flask(__name__)

# Load English language model
nlp = spacy.load("en_core_web_sm")

# Mock database for appointments, healthcare providers, and availability
appointments = []
healthcare_providers = [
    {"id": 1, "name": "Dr. John Smith", "specialty": "General Practitioner"},
    {"id": 2, "name": "Dr. Emily Johnson", "specialty": "Dermatologist"},
    {"id": 3, "name": "Dr. Michael Davis", "specialty": "Cardiologist"}
]
time_slots = [
    {"date": "2023-03-19", "slots": ["09:00", "10:00", "11:00", "14:00", "15:00"]},
    {"date": "2023-03-20", "slots": ["09:00", "10:00", "11:00", "14:00", "15:00", "16:00"]},
    {"date": "2023-03-21", "slots": ["09:00", "10:00", "14:00", "15:00", "16:00"]}
]

# Function to extract appointment details from user input
def extract_appointment_details(user_input):
    doc = nlp(user_input)
    appointment_date = None
    appointment_time = None
    healthcare_provider = None

    date_pattern = r"""
        (?:
            (?:
                (?:0?[1-9]|[12][0-9]|3[01])  # Day (optional leading zero)
                [\/-]                        # Separator
                (?:0?[1-9]|1[0-2])           # Month (optional leading zero)
                [\/-]                        # Separator
                \d{4}                        # Year
            )
            |
            (?:
                (?:0?[1-9]|[12][0-9]|3[01])  # Day (optional leading zero)
                (?:st|nd|rd|th)?             # Optional ordinal suffix
                \s+                          # Whitespace
                (?:
                    Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
                    Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?
                )                            # Month (full or abbreviated)
                \s+                          # Whitespace
                \d{4}                        # Year
            )
            |
            (?:
                (?:
                    Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|
                    Jul(?:y)?|Aug(?:aust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?
                )                            # Month (full or abbreviated)
                \s+                          # Whitespace
                (?:0?[1-9]|[12][0-9]|3[01])  # Day (optional leading zero)
                (?:st|nd|rd|th)?             # Optional ordinal suffix
                ,?                           # Optional comma
                \s+                          # Whitespace
                \d{4}                        # Year
            )
        )
    """

    date_match = re.search(date_pattern, user_input, re.IGNORECASE | re.VERBOSE)
    if date_match:
            try:
                appointment_date = parser.parse(date_match.group(),fuzzy_with_tokens=True).strftime('%d/%m/%Y')
            except ValueError:
                pass
    else:
            return None,None,None,

    for ent in doc.ents:
        if ent.label_ == "TIME":
            appointment_time = ent.text
        elif ent.label_ == "PERSON":
            healthcare_provider_name = ent.text.replace("Dr.", "Dr")
            healthcare_provider_name = healthcare_provider_name.lower()
            healthcare_provider = next((provider for provider in healthcare_providers if provider["name"].lower() == healthcare_provider_name), None)

    return appointment_date, appointment_time, healthcare_provider


# Function to check if the appointment date is valid
def is_valid_date(appointment_date):
    try:
        parsed_date = datetime.strptime(appointment_date, "%Y-%m-%d")
        if parsed_date.date() < datetime.now().date():
            return False
        return True
    except ValueError:
        return False

# Function to check if the appointment time is valid
def is_valid_time(appointment_time):
    try:
        parsed_time = datetime.strptime(appointment_time, "%H:%M").time()
        business_start_time = datetime.strptime("07:00", "%H:%M").time()
        business_end_time = datetime.strptime("17:00", "%H:%M").time()
        if parsed_time < business_start_time or parsed_time >= business_end_time:
            return False
        return True
    except ValueError:
        return False

# Function to check appointment availability
def check_availability(appointment_date, appointment_time, healthcare_provider):
    # Check against existing appointments
    for appointment in appointments:
        if (appointment["date"] == appointment_date and
            appointment["time"] == appointment_time and
            appointment["provider"]["id"] == healthcare_provider["id"]):
            return False
    
    # Check against available time slots
    available_slots = next((slot["slots"] for slot in time_slots if slot["date"] == appointment_date), [])
    if appointment_time not in available_slots:
        return False
    
    return True

# Function for appointment booking
@app.route('/book-appointment', methods=['POST'])
def book_appointment():
    try:
        # Get user input from the request
        user_input = request.json.get('text')
        
        # Extract appointment details from user input
        appointment_date, appointment_time, healthcare_provider = extract_appointment_details(user_input)
        
        # Check for valid date and time
        if not is_valid_date(appointment_date):
            return jsonify({'error': 'Invalid appointment date. Please provide a future date.'}), 400
        if not is_valid_time(appointment_time):
            return jsonify({'error': 'Invalid appointment time. Please choose a time between 9 AM and 5 PM.'}), 400
        
        # Check appointment availability
        if not check_availability(appointment_date, appointment_time, healthcare_provider):
            return jsonify({'error': 'Appointment slot is not available. Please choose a different date or time.'}), 409
        
        # Book the appointment
        new_appointment = {
            "date": appointment_date,
            "time": appointment_time,
            "provider": healthcare_provider
        }
        appointments.append(new_appointment)
        
        # Return the result
        return jsonify({'result': f"Appointment booked with {healthcare_provider['name']} on {appointment_date} at {appointment_time}."}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)