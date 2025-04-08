import gradio as gr
from datetime import datetime, timedelta
import pytz

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os

SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('calendar', 'v3', credentials=creds)
    return service

# Get busy times from Google Calendar
def get_busy_times():
    service = get_calendar_service()
    now = datetime.now(pytz.UTC).isoformat()
    end = (datetime.now(pytz.UTC) + timedelta(days=3)).isoformat()

    try:
        events_result = service.events().list(
            calendarId='primary',
            timeMin=now,
            timeMax=end,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        events = events_result.get('items', [])
        busy_slots = []

        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))

            # Handle all-day events
            if 'T' not in start:
                start_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            else:
                start_dt = datetime.fromisoformat(start)

            if 'T' not in end:
                end_dt = datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=pytz.UTC)
            else:
                end_dt = datetime.fromisoformat(end)

            busy_slots.append((start_dt, end_dt))

        return busy_slots

    except HttpError as e:
        print(f"An error occurred: {e}")
        return []

# Calculate free times
def calculate_free_times():
    busy_slots = get_busy_times()
    free_slots = []

    start_time = datetime.now(pytz.UTC).replace(minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(days=2)

    current_time = start_time
    while current_time < end_time:
        slot_end = current_time + timedelta(minutes=30)
        is_free = all(not (current_time < busy_end and slot_end > busy_start)
                      for busy_start, busy_end in busy_slots)
        if is_free:
            free_slots.append(current_time.strftime('%Y-%m-%d %H:%M'))
        current_time = slot_end

    return free_slots

# Schedule meeting
def schedule_meeting(selected_time, email):
    service = get_calendar_service()

    start_time = datetime.strptime(selected_time, '%Y-%m-%d %H:%M').replace(tzinfo=pytz.UTC)
    end_time = start_time + timedelta(minutes=30)

    # Beautiful email passage
    email_passage = """
    Dear Candidate,

    We are excited to invite you to discuss a potential career opportunity with our company. This meeting will provide an excellent chance to explore your skills, experience, and how they align with our organization's vision and goals.

    We look forward to speaking with you soon.

    Best regards,
    HR Team
    BerryOnMars
    """

    event = {
        'summary': 'Scheduled Meeting',
        'description': email_passage.strip(),
        'start': {'dateTime': start_time.isoformat()},
        'end': {'dateTime': end_time.isoformat()},
        'attendees': [{'email': email}],
        'reminders': {'useDefault': True},
    }

    try:
        event_result = service.events().insert(
            calendarId='primary',
            body=event,
            sendUpdates='all'
        ).execute()

        return f"Meeting scheduled successfully! Event link: {event_result.get('htmlLink')}"

    except HttpError as e:
        return f"An error occurred while scheduling: {e}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Schedule Your Meeting (All times in UTC)")
    email = gr.Textbox(label="Enter attendee's email", placeholder="example@gmail.com")
    free_times = gr.Dropdown(choices=calculate_free_times(), label="Select a free time slot")
    refresh_btn = gr.Button("Refresh Available Slots")
    schedule_button = gr.Button("Schedule Meeting")
    result = gr.Textbox(label="Result")

    refresh_btn.click(fn=lambda: gr.update(choices=calculate_free_times()), outputs=free_times)
    schedule_button.click(fn=schedule_meeting, inputs=[free_times, email], outputs=result)

demo.launch()