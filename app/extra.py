# import requests
# import json

# url = "https://dev-hiring-employer.berryonmars.com/JobPost/CreateFromAI"
#     # "https://dev-hiring-candidate.berryonmars.com/Admin/candidate/GetAnonimousByCustomFieldList?skip=0&take=50"

# payload = json.dumps(
#    {
#   "title": "test kasra",
#   "requirementsText": "test kasra",
#   "description": "test kasra",
#   "salaryFrom": 12000,
#   "salaryTo": 55000,
#   "country": "Iran",
#   "city": "Tabriz",
#   "jobType": "fullTime",
#   "locationTypeEnum": "remote",
#   "employerId": "reza_sheshbolooki@yahoo.com"
# }
# )

# headers = {
#     'accept': '*/*',
#     # 'Authorization': 'Bearer',  # Replace with a valid token
#     'Content-Type': 'application/json'
# }

# response = requests.post(url, headers=headers, data=payload)
# print(response.text)


# import requests
# import json
# import json
# from datetime import datetime, timedelta

# import json
# from datetime import datetime, timedelta

# def parse_free_slots(free_time_string):
#     """
#     Parse free time intervals from a JSON string and return structured slot info.
    
#     Args:
#         free_time_string (str): A JSON string of free time intervals, each with start and end datetime.
        
#     Returns:
#         List[dict]: A list of slots with 'date', 'start_time', and 'end_time'
#     """
#     events = json.loads(free_time_string)
#     result = []

#     for event in events:
#         start_dt = datetime.fromisoformat(event["start"]["dateTime"])
#         end_dt = datetime.fromisoformat(event["end"]["dateTime"])

#         result.append({
#             "date": start_dt.strftime("%Y-%m-%d"),
#             "start_time": start_dt.strftime("%H:%M"),
#             "end_time": end_dt.strftime("%H:%M")
#         })

#     return result

# url = "https://dev-hiring-candidate.berryonmars.com/GetFreeEvents/r.sheshbolooki@berryonmars.com/2025-04-01T08:00:00/2025-04-02T23:00:00"
#     # "https://dev-hiring-candidate.berryonmars.com/Admin/candidate/GetAnonimousByCustomFieldList?skip=0&take=50"


# headers = {
#     'accept': '*/*',
#     # 'Authorization': 'Bearer',  # Replace with a valid token
#     'Content-Type': 'application/json'
# }

# response = requests.get(url)
# free_slots = parse_free_slots(response.text)
# print(free_slots)



# url = "https://dev-hiring-candidate.berryonmars.com/Calender/r.sheshbolooki@berryonmars.com/CreateEvent"
#     # "https://dev-hiring-candidate.berryonmars.com/Admin/candidate/GetAnonimousByCustomFieldList?skip=0&take=50"

# payload = json.dumps(
#    {
#     "subject": "Meeting with reza shesh employer from Test",
#     "body": {
#         "contentType": 1,
#         "content": "Let's have a meeting"
#     },
#     "start": {
#         "timeZone": "Asia/Tehran",
#         "dateTime": "2025-04-10T14:00:00"
#     },
#     "end": {
#         "timeZone": "Asia/Tehran",
#         "dateTime": "2025-04-10T14:30:00"
#     },
#     "location": {
#         "displayName": "Recruiting Meeting Scheduler"
#     },
#     "attendees": [
#         {
#             "type": 0,
#             "emailAddress": {
#                 "name": "reza shesh employer",
#                 "address": "reza_sheshbolooki@yahoo.com"
#             }
#         }
#     ],
#     "allowNewTimeProposals": True
# }
# )

# headers = {
#     'accept': '*/*',
#     # 'Authorization': 'Bearer',  # Replace with a valid token
#     'Content-Type': 'application/json'
# }

# response = requests.post(url, payload, headers=headers)
# print(response)





# import requests
# import json
# from datetime import datetime
# from datetime import datetime, timedelta
# from typing import List, Dict
# from langchain.tools import tool


# def get_free_time_slots(email: str) -> List[Dict[str, str]]:
#     """
#     Retrieve available time slots for a given email within the next 7 days.

#     This tool contacts the scheduling API to fetch free time slots for a specific user 
#     based on their email. It queries a date range starting from now and ending 7 days later.

#     Parameters:
#     ----------
#     email : str
#         The email address of the person whose calendar is being queried 
#         (e.g., r.sheshbolooki@berryonmars.com).

#     Returns:
#     -------
#     List[dict]
#         A list of available free time slots with:
#         {
#             "date": "YYYY-MM-DD",
#             "start_time": "HH:MM",
#             "end_time": "HH:MM"
#         }

#     Example:
#     --------
#     >>> get_free_time_slots("r.sheshbolooki@berryonmars.com")
#     [
#         {"date": "2025-04-01", "start_time": "09:00", "end_time": "09:30"},
#         {"date": "2025-04-01", "start_time": "09:30", "end_time": "10:00"},
#         ...
#     ]

#     Notes:
#     ------
#     - The function automatically sets the query range from the current datetime 
#       to 7 days ahead.
#     - Assumes the API returns results in ISO 8601 format.
#     - Returns an empty list on error or if no free slots are found.
#     """

#     # Set date range: today to 7 days later
#     now = datetime.utcnow()
#     start_date = now.strftime("%Y-%m-%dT%H:%M:%S")
#     end_date = (now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%S")

#     url = f"https://dev-hiring-candidate.berryonmars.com/GetFreeEvents/{email}/{start_date}/{end_date}"

#     headers = {
#         "accept": "*/*",
#         "Content-Type": "application/json"
#     }

#     try:
#         response = requests.get(url, headers=headers)
#         response.raise_for_status()

#         events = json.loads(response.text)
#         free_slots = []

#         for event in events:
#             start_dt = datetime.fromisoformat(event["start"]["dateTime"])
#             end_dt = datetime.fromisoformat(event["end"]["dateTime"])

#             free_slots.append({
#                 "date": start_dt.strftime("%Y-%m-%d"),
#                 "start_time": start_dt.strftime("%H:%M"),
#                 "end_time": end_dt.strftime("%H:%M")
#             })

#         return free_slots

#     except Exception as e:
#         print(f"❌ Failed to fetch free time slots: {e}")
#         return []
    
# print(get_free_time_slots(email="r.sheshbolooki@berryonmars.com"))


import requests
import json
from langchain.tools import tool
from typing import Optional


@tool
def schedule_meeting(email: str, reserved_time: str) -> Optional[str]:
    """
    Schedule a calendar meeting with a given attendee at a specific time.

    This tool uses the recruiting meeting scheduling system to create a calendar event.
    It receives the attendee's email and the reserved meeting time, then sends a request
    to create an event in the system.

    Parameters:
    ----------
    email : str
        The email address of the attendee who will receive the calendar invite.
    reserved_time : str
        The meeting start and end time in ISO 8601 format (e.g., "2025-04-01T14:00:00").
        The meeting is assumed to be 30 minutes long.

    Returns:
    -------
    str or None
        A success message if the meeting is scheduled successfully.
        Returns None if the operation fails.

    Example:
    --------
    >>> schedule_meeting("attendee@example.com", "2025-04-01T10:00:00")
    '✅ Meeting scheduled successfully.'

    Notes:
    ------
    - This tool assumes meetings are always 30 minutes.
    - If the reserved_time format is invalid, or the request fails, the function returns None.
    """

    # API endpoint for scheduling
    url = "https://dev-hiring-candidate.berryonmars.com/Calender/r.sheshbolooki@berryonmars.com/CreateEvent"

    # Default meeting duration: 30 minutes
    meeting_duration_minutes = 30
    try:
        from datetime import datetime, timedelta

        start_time = datetime.fromisoformat(reserved_time)
        end_time = (start_time + timedelta(minutes=meeting_duration_minutes)).isoformat()

        payload = json.dumps({
            "subject": f"Meeting with {email.split('@')[0]} from Test",
            "body": {
                "contentType": 1,
                "content": "Let's have a meeting"
            },
            "start": {
                "timeZone": "Asia/Tehran",
                "dateTime": reserved_time
            },
            "end": {
                "timeZone": "Asia/Tehran",
                "dateTime": end_time
            },
            "location": {
                "displayName": "Recruiting Meeting Scheduler"
            },
            "attendees": [
                {
                    "type": 0,
                    "emailAddress": {
                        "name": f"{email.split('@')[0]}",
                        "address": email
                    }
                }
            ],
            "allowNewTimeProposals": True
        })

        headers = {
            'accept': '*/*',
            'Content-Type': 'application/json'
            # 'Authorization': 'Bearer <token>'  # Add if needed
        }

        response = requests.post(url, data=payload, headers=headers)

        if response.status_code in [200, 201]:
            return "✅ Meeting scheduled successfully."
        else:
            print(f"❌ Failed with status {response.status_code}: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Error scheduling meeting: {e}")
        return None