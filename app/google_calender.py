import gradio as gr
from datetime import datetime, timedelta
import pytz

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
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
    now = datetime.utcnow().isoformat() + 'Z'
    end = (datetime.utcnow() + timedelta(days=3)).isoformat() + 'Z'

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
        start = event['start'].get('dateTime')
        end = event['end'].get('dateTime')
        busy_slots.append((start, end))

    return busy_slots

# Calculate free times
def calculate_free_times():
    busy_slots = get_busy_times()
    free_slots = []

    start_time = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    end_time = start_time + timedelta(days=2)

    current_time = start_time
    while current_time < end_time:
        slot_end = current_time + timedelta(minutes=30)
        is_free = all(not (current_time.isoformat() < busy_end and slot_end.isoformat() > busy_start)
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

    event = {
        'summary': 'Scheduled Meeting',
        'description': 'Automated meeting scheduled via Gradio.',
        'start': {'dateTime': start_time.isoformat()},
        'end': {'dateTime': end_time.isoformat()},
        'attendees': [{'email': email}],
        'reminders': {'useDefault': True},
    }

    event_result = service.events().insert(
        calendarId='primary',
        body=event,
        sendUpdates='all'
    ).execute()

    return f"Meeting scheduled successfully! Event link: {event_result.get('htmlLink')}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Schedule Your Meeting")
    email = gr.Textbox(label="Enter attendee's email", placeholder="example@gmail.com")
    free_times = gr.Dropdown(choices=calculate_free_times(), label="Select a free time slot")
    schedule_button = gr.Button("Schedule Meeting")
    result = gr.Textbox(label="Result")

    schedule_button.click(fn=schedule_meeting, inputs=[free_times, email], outputs=result)

demo.launch()
