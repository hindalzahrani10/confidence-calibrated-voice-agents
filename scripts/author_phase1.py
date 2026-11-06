"""Author the Phase 1 feasibility dataset (schema phase1.v2): 50 base items, no audio.

Every item is hand-written synthetic text. None comes from SLURP, MASSIVE,
AmbigQA or ClarQ-LLM, and records say so (`meta.external_dataset: null`).
All annotations are an unreviewed AI draft; no human annotator or agreement
score exists.

`data/` and `*.jsonl` are git-ignored, so this script is the committed,
reproducible source of:

    data/phase1/base_items.jsonl
    data/phase1/manifest.json
    notes/phase1_human_review.md        (review checklist, generated)

    python3 scripts/author_phase1.py
    python3 scripts/validate_phase1.py

Design (full spec: notes/phase1_dataset_spec.md)
------------------------------------------------
* Each record has an `agent_visible` block (opaque ids + null audio paths
  only) and a `private` block (everything that reveals the intended task).
* Probes are PRIVATE reference scenarios (question -> expected reply ->
  recovery). They do not tell an agent what to ask or propose; a future user
  simulator answers whatever the agent actually asks, from the hidden gold
  task and the per-item `say` lexicon.
* Recovery is OPEN-WORLD and STRICT by default. A rejected proposal or a
  rejected option set is recorded as exclusion evidence, never as a recovered
  value. A LENIENT reading (unmentioned proposal slots implicitly accepted)
  is stored beside it as a labelled sensitivity annotation.
* `intent` is a reserved slot name. It is in `required_slots` only when the
  intent itself is uncertain (intent items, or acoustic items whose
  confusable changes the intent).
* Acoustic items: the confusable slot is "at risk". The spoken value is in
  `utterance_slots`, but it only counts as known after a reply re-supplies it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "phase1"
REVIEW_PATH = ROOT / "notes" / "phase1_human_review.md"
SCHEMA_VERSION = "phase1.v2"

PLAN = {  # category -> (train, val, test)
    "none": (8, 3, 2),
    "acoustic": (8, 2, 3),
    "intent": (7, 3, 2),
    "underspecified": (7, 2, 3),
}
AXIS_FOR = {"none": "none", "acoustic": "acoustic", "intent": "intent", "underspecified": "slot"}


def A(why, intent=None, say=None, **slots):
    """An alternative to the gold task: the gold task with these changes."""
    return {"why": why, "intent": intent, "slots": slots, "say": say or {}}


def conf(kind, surface, competitors, affects):
    return {"kind": kind, "surface": surface, "competitors": competitors, "affects": affects}


# fmt: off
# Verb phrases used to render a proposed task as a confirmation question.
def tm(t: str) -> str:
    hh, mm = (int(x) for x in t.split(":"))
    h = hh % 12 or 12
    return f"{h}:{mm:02d} {'AM' if hh < 12 else 'PM'}" if mm else f"{h} {'AM' if hh < 12 else 'PM'}"


def _rem(s):
    out = f"set a reminder for '{s['task']}'"
    if "time" in s:
        out += f" at {tm(s['time'])}"
    if "date" in s:
        out += f" {s['date']}"
    if "lead_time_min" in s:
        out += f" {s['lead_time_min']} minutes before"
    return out


def _fc(s):
    out = "get the weather"
    if "city" in s:
        out += f" in {s['city']}"
    if "date" in s:
        out += f" for {s['date']}"
    return out


TEMPLATES = {
    "set_alarm": lambda s: f"set an alarm for {tm(s['time'])}",
    "light_off": lambda s: f"turn off the {s['room']} lights",
    "get_forecast": _fc,
    "next_track": lambda s: "skip to the next track",
    "previous_track": lambda s: "go back to the previous track",
    "pause": lambda s: "pause the music",
    "stop_music": lambda s: "stop the music",
    "play_music": lambda s: "start the music",
    "set_timer": lambda s: f"set a timer for {s['duration_min']} minutes",
    "add_item": lambda s: f"add {s['item']} to your {s['list']} list",
    "call_contact": lambda s: f"call {s['contact']}",
    "set_volume": lambda s: f"set the volume to {s['level']} percent",
    "get_time": lambda s: f"tell you the time in {s['city']}",
    "create_reminder": _rem,
    "send_text": lambda s: f"text {s['recipient']}" + (f" saying '{s['body']}'" if "body" in s else ""),
    "navigate": lambda s: f"navigate to {s['destination']}" if "destination" in s else f"navigate to the {s['place']}",
    "set_brightness": lambda s: f"set the brightness to {s['level']} percent",
    "play_artist": lambda s: f"play music by {s['artist']}",
    "dim_lights": lambda s: f"dim the {s['room']} lights",
    "unlock_door": lambda s: f"unlock the {s['door']} door",
    "lock_door": lambda s: f"lock the {s['door']} door",
    "open_door": lambda s: f"open the {s['door']} door",
    # intent items
    "play_song": lambda s: f"play the song {s['title']}",
    "play_music_video": lambda s: f"play the music video for {s['title']}",
    "play_album": lambda s: f"play the album {s['title']}",
    "create_calendar_event": lambda s: f"add a calendar event for {s['task']} {s['date']} at {tm(s['time'])}",
    "create_note": lambda s: f"save a note about {s['task']} {s['date']} at {tm(s['time'])}",
    "record_audio": lambda s: "record audio only",
    "record_video": lambda s: "record video",
    "record_screen": lambda s: "record the screen",
    "cancel_food_delivery": lambda s: f"cancel your {s['date']} food delivery",
    "cancel_shopping_order": lambda s: f"cancel your {s['date']} shopping order",
    "cancel_restaurant_reservation": lambda s: f"cancel your {s['date']} restaurant reservation",
    "call_business": lambda s: f"call the {s['place']}",
    "show_business_hours": lambda s: f"show opening hours for the {s['place']}",
    "set_thermostat": lambda s: (f"set the {s['room']} temperature to {s['value']}" if "room" in s
                                 else f"set the thermostat to {s['temperature_c']} degrees"),
    "set_light_brightness": lambda s: f"set the {s['room']} lights to {s['value']} percent",
    "set_speaker_volume": lambda s: f"set the {s['room']} speaker volume to {s['value']} percent",
    "book_hotel": lambda s: f"book a hotel in {s['destination']} for {s['month']}",
    "book_flight": lambda s: f"book a flight to {s['destination']} for {s['month']}",
    "book_rental_car": lambda s: f"book a rental car in {s['destination']} for {s['month']}",
    "show_calendar": lambda s: f"show your calendar for {s['date']}",
    "show_reminders": lambda s: f"show your reminders for {s['date']}",
    "show_contact_details": lambda s: f"show {s['contact']}'s contact details",
    "open_chat": lambda s: f"open a chat with {s['contact']}",
    "navigate_to_event": lambda s: f"navigate to the {s['event']} {s['date']}",
    "play_event_broadcast": lambda s: f"put on the {s['event']} {s['date']}",
    "buy_event_tickets": lambda s: f"buy tickets to the {s['event']} {s['date']}",
    "dismiss_alarm": lambda s: "dismiss the ringing alarm",
    "disable_alarm": lambda s: "disable the alarm",
    "disarm_security_system": lambda s: "disarm the security system",
    "mute_chat_notifications": lambda s: f"mute {s['person']}'s chat notifications",
    "mute_call_participant": lambda s: f"mute {s['person']} on the call",
    "mute_social_feed": lambda s: f"mute {s['person']}'s posts",
    # underspecified items
    "play_genre": lambda s: f"play some {s['genre']}",
    "book_table": lambda s: f"book a table for {s['party_size']} at {tm(s['time'])} {s['date']}",
    "adjust_volume": lambda s: f"turn the volume {s['direction']} on the {s['device']}",
    "cancel_appointment": lambda s: f"cancel the {s['appointment']}",
    "book_ticket": lambda s: f"book {s['quantity']} ticket to the {s['event']}",
    "send_email": lambda s: f"email {s['attachment']} to {s['recipient']}",
}

# Coarse action families, used only to check that "intent" items differ in the
# action performed rather than in a parameter of one action.
FAMILY = {
    "play_song": "audio_playback", "play_music_video": "video_playback", "play_album": "album_playback",
    "create_calendar_event": "calendar", "create_reminder": "reminder", "create_note": "notes",
    "record_audio": "audio_capture", "record_video": "video_capture", "record_screen": "screen_capture",
    "cancel_food_delivery": "food_order", "cancel_shopping_order": "retail_order",
    "cancel_restaurant_reservation": "reservation",
    "call_business": "voice_call", "navigate": "navigation", "show_business_hours": "info_lookup",
    "set_thermostat": "climate", "set_light_brightness": "lighting", "set_speaker_volume": "audio_output",
    "book_hotel": "lodging", "book_flight": "air_travel", "book_rental_car": "ground_travel",
    "show_calendar": "calendar_view", "get_forecast": "weather", "show_reminders": "reminder_view",
    "call_contact": "voice_call", "show_contact_details": "info_lookup", "open_chat": "messaging",
    "navigate_to_event": "navigation", "play_event_broadcast": "media_playback", "buy_event_tickets": "purchase",
    "dismiss_alarm": "alarm_control", "disable_alarm": "alarm_schedule", "disarm_security_system": "security",
    "mute_chat_notifications": "notification_control", "mute_call_participant": "call_control",
    "mute_social_feed": "feed_control",
}

IW = "intent"  # shorthand for the reserved slot name inside `say`

ITEMS = [
    # ===== none (13): fully specified, nothing to resolve ====================================
    dict(id="none_01", split="train", cat="none", text="set an alarm for seven thirty in the morning",
         domain="alarm", intent="set_alarm", slots={"time": "07:30"}, say={"time": "seven thirty in the morning"},
         alts=[A("neighbouring time", time="06:30")]),
    dict(id="none_02", split="train", cat="none", text="turn off the kitchen lights",
         domain="smart_home", intent="light_off", slots={"room": "kitchen"}, say={"room": "kitchen"},
         alts=[A("another room", room="dining room")]),
    dict(id="none_03", split="train", cat="none", text="what's the weather in Boston tomorrow",
         domain="weather", intent="get_forecast", slots={"city": "Boston", "date": "tomorrow"},
         say={"city": "Boston"}, alts=[A("Boston/Austin are a known ASR confusion", city="Austin")]),
    dict(id="none_04", split="train", cat="none", text="play the next song",
         domain="music", intent="next_track", slots={}, say={IW: "the next song"},
         alts=[A("opposite direction", intent="previous_track", say={IW: "the previous song"})]),
    dict(id="none_05", split="train", cat="none", text="set a timer for ten minutes",
         domain="timer", intent="set_timer", slots={"duration_min": 10}, say={"duration_min": "ten minutes"},
         alts=[A("another duration", duration_min=20)]),
    dict(id="none_06", split="train", cat="none", text="add eggs to my grocery list",
         domain="lists", intent="add_item", slots={"item": "eggs", "list": "grocery"}, say={"item": "eggs"},
         alts=[A("lexically close item", item="eggplant")]),
    dict(id="none_07", split="train", cat="none", text="call my sister",
         domain="phone", intent="call_contact", slots={"contact": "my sister"}, say={"contact": "my sister"},
         alts=[A("another family contact", contact="my mother")]),
    dict(id="none_08", split="train", cat="none", text="set the volume to fifty percent",
         domain="device", intent="set_volume", slots={"level": 50}, say={"level": "fifty percent"},
         alts=[A("fifty/fifteen confusion", level=15)]),
    dict(id="none_09", split="val", cat="none", text="what time is it in Tokyo",
         domain="time", intent="get_time", slots={"city": "Tokyo"}, say={"city": "Tokyo"},
         alts=[A("Tokyo/Kyoto are close", city="Kyoto")]),
    dict(id="none_10", split="val", cat="none", text="remind me to take my medicine at nine PM",
         domain="reminders", intent="create_reminder", slots={"task": "take my medicine", "time": "21:00"},
         say={"time": "nine PM"}, alts=[A("AM/PM slip", time="09:00")]),
    dict(id="none_11", split="val", cat="none", text="send a text to Jordan saying I'm running late",
         domain="messaging", intent="send_text", slots={"recipient": "Jordan", "body": "I'm running late"},
         say={"recipient": "Jordan"}, alts=[A("similar name", recipient="Jaden")]),
    dict(id="none_12", split="test", cat="none", text="navigate to the nearest pharmacy",
         domain="navigation", intent="navigate", slots={"destination": "the nearest pharmacy"},
         say={"destination": "the nearest pharmacy"}, alts=[A("another nearby place", destination="the nearest grocery store")]),
    dict(id="none_13", split="test", cat="none", text="pause the music",
         domain="music", intent="pause", slots={}, say={IW: "pause the music"},
         alts=[A("stronger action", intent="stop_music", say={IW: "stop the music"})]),

    # ===== acoustic (13): clear text, a designed confusable ==================================
    dict(id="acoustic_01", split="train", cat="acoustic", text="call mom",
         domain="phone", intent="call_contact", slots={"contact": "Mom"}, say={"contact": "Mom"},
         conf=conf("minimal_pair", "mom", ["Tom", "Tim"], "contact"),
         alts=[A("minimal pair /m/ vs /t/", contact="Tom"), A("nearby name", contact="Tim")],
         open=[("Who would you like to call?", ["contact"], "Mom")], stem="Did you mean"),
    dict(id="acoustic_02", split="train", cat="acoustic", text="call Dan",
         domain="phone", intent="call_contact", slots={"contact": "Dan"}, say={"contact": "Dan"},
         conf=conf("name_collision", "dan", ["Don", "Dean"], "contact"),
         alts=[A("one vowel apart", contact="Don"), A("similar name", contact="Dean")],
         open=[("Who should I call?", ["contact"], "Dan")], stem="Did you mean"),
    dict(id="acoustic_03", split="train", cat="acoustic", text="add pears to the shopping list",
         domain="lists", intent="add_item", slots={"item": "pears", "list": "shopping"}, say={"item": "pears"},
         conf=conf("minimal_pair", "pears", ["beers", "peas"], "item"),
         alts=[A("minimal pair /p/ vs /b/", item="beers"), A("nearby word", item="peas")],
         open=[("What should I add?", ["item"], "Pears")], stem="Did you mean"),
    dict(id="acoustic_04", split="train", cat="acoustic", text="set the brightness to fifteen percent",
         domain="device", intent="set_brightness", slots={"level": 15}, say={"level": "fifteen"},
         conf=conf("minimal_pair", "fifteen", ["fifty", "sixteen"], "level"),
         alts=[A("fifteen vs fifty differ in stress only", say={"level": "fifty"}, level=50),
               A("nearby number", say={"level": "sixteen"}, level=16)],
         open=[("What brightness level would you like?", ["level"], "Fifteen percent")], stem="Did you mean"),
    dict(id="acoustic_05", split="train", cat="acoustic", text="set a timer for thirteen minutes",
         domain="timer", intent="set_timer", slots={"duration_min": 13}, say={"duration_min": "thirteen"},
         conf=conf("minimal_pair", "thirteen", ["thirty", "fourteen"], "duration_min"),
         alts=[A("thirteen vs thirty differ in stress only", say={"duration_min": "thirty"}, duration_min=30),
               A("nearby number", say={"duration_min": "fourteen"}, duration_min=14)],
         open=[("How long should the timer run?", ["duration_min"], "Thirteen minutes")], stem="Did you mean"),
    dict(id="acoustic_06", split="train", cat="acoustic", text="play music by Beck",
         domain="music", intent="play_artist", slots={"artist": "Beck"}, say={"artist": "Beck"},
         conf=conf("near_homophone", "beck", ["Bach", "Bush"], "artist"),
         alts=[A("Beck vs Bach", artist="Bach"), A("similar onset and coda", artist="Bush")],
         open=[("Which artist?", ["artist"], "Beck")], stem="Did you mean"),
    dict(id="acoustic_07", split="train", cat="acoustic", text="what's the weather in Portland",
         domain="weather", intent="get_forecast", slots={"city": "Portland"}, say={"city": "Portland"},
         conf=conf("near_homophone", "portland", ["Poland", "Porto"], "city"),
         alts=[A("Portland vs Poland", city="Poland"), A("similar onset", city="Porto")],
         open=[("Which city?", ["city"], "Portland")], stem="Did you mean"),
    dict(id="acoustic_08", split="train", cat="acoustic", text="navigate to Beach Street",
         domain="navigation", intent="navigate", slots={"destination": "Beach Street"},
         say={"destination": "Beach Street"},
         conf=conf("minimal_pair", "beach street", ["Peach Street", "Beech Street"], "destination"),
         alts=[A("minimal pair /b/ vs /p/", destination="Peach Street"), A("homophone-like", destination="Beech Street")],
         open=[("Where would you like to go?", ["destination"], "Beach Street")], stem="Did you mean"),
    dict(id="acoustic_09", split="val", cat="acoustic", text="remind me at six PM to buy flour",
         domain="reminders", intent="create_reminder", slots={"task": "buy flour", "time": "18:00"},
         say={"task": "buy flour"},
         conf=conf("near_homophone", "flour", ["flowers", "flax"], "task"),
         alts=[A("flour vs flower", task="buy flowers"), A("nearby word", task="buy flax")],
         open=[("What should I remind you about?", ["task"], "Buy flour")], stem="Did you mean"),
    dict(id="acoustic_10", split="val", cat="acoustic", text="dim the bedroom lights",
         domain="smart_home", intent="dim_lights", slots={"room": "bedroom"}, say={"room": "bedroom"},
         conf=conf("near_homophone", "bedroom", ["bathroom"], "room"),
         alts=[A("bedroom vs bathroom (weak acoustic pair)", room="bathroom"), A("another room", room="guest room")],
         open=[("Which room?", ["room"], "The bedroom")], stem="Did you mean",
         flags=["weak confusable: must be verified against ASR output in Phase 2"]),
    dict(id="acoustic_11", split="test", cat="acoustic", text="stop the music",
         domain="music", intent="stop_music", slots={}, say={IW: "stop the music"},
         conf=conf("minimal_pair", "stop", ["start", "skip"], "intent"),
         alts=[A("stop vs start; opposite effect", intent="play_music", say={IW: "start the music"}),
               A("stop vs skip", intent="next_track", say={IW: "skip the song"})],
         open=[("What would you like me to do with the music?", ["intent"], "Stop the music")],
         stem="Do you want me to"),
    dict(id="acoustic_12", split="test", cat="acoustic", text="unlock the front door",
         domain="smart_home", intent="unlock_door", slots={"door": "front"},
         say={"door": "front", IW: "unlock the door"},
         conf=conf("minimal_pair", "unlock", ["lock"], "intent"),
         alts=[A("unlock vs lock; safety-relevant", intent="lock_door", say={IW: "lock the door"}),
               A("neighbouring action", intent="open_door", say={IW: "open the door"})],
         open=[("What would you like me to do with the front door?", ["intent"], "Unlock the door")],
         stem="Do you want me to"),
    dict(id="acoustic_13", split="test", cat="acoustic", text="text Jen that I'm outside",
         domain="messaging", intent="send_text", slots={"recipient": "Jen", "body": "I'm outside"},
         say={"recipient": "Jen"},
         conf=conf("name_collision", "jen", ["Jan", "Jenna"], "recipient"),
         alts=[A("one vowel apart", recipient="Jan"), A("longer similar name", recipient="Jenna")],
         open=[("Who is the message for?", ["recipient"], "Jen")], stem="Did you mean"),

    # ===== intent (12): slots given, the action itself is uncertain ==========================
    dict(id="intent_01", split="train", cat="intent", text="play Thriller",
         domain="media", intent="play_song", slots={"title": "Thriller"}, say={IW: "the song"},
         alts=[A("could mean video", intent="play_music_video", say={IW: "the music video"}),
               A("could mean the full album", intent="play_album", say={IW: "the album"})],
         open=[("What would you like me to play?", ["intent"], "The song")], stem="Do you want",
         gold_prior="modal",
         rationale="audio vs video vs album playback are different actions on different outputs",
         flags=["axis boundary: media type could be argued as a parameter"]),
    dict(id="intent_02", split="train", cat="intent", text="dentist tomorrow at ten",
         domain="productivity", intent="create_calendar_event",
         slots={"task": "dentist", "date": "tomorrow", "time": "10:00"}, say={IW: "a calendar event"},
         alts=[A("could be a reminder", intent="create_reminder", say={IW: "a reminder"}),
               A("could be a note", intent="create_note", say={IW: "a note"})],
         open=[("What should I do with that?", ["intent"], "Make it a calendar event")], stem="Do you want",
         gold_prior="modal",
         rationale="a fragment with a time: calendar event, reminder and note are different actions",
         flags=["re-authored: v1 said 'note down' but gold was reminder", "'ten' normalised to 10:00 (AM) from context"]),
    dict(id="intent_03", split="train", cat="intent", text="record this",
         domain="device", intent="record_audio", slots={}, say={IW: "audio only"},
         alts=[A("could mean video", intent="record_video", say={IW: "video"}),
               A("could mean the screen", intent="record_screen", say={IW: "the screen"})],
         open=[("What would you like to record?", ["intent"], "Audio only")], stem="Do you want",
         gold_prior="unclear",
         rationale="audio, video and screen capture are different capture actions",
         flags=["re-authored: v1 'tell me about Jaguar' was entity-sense (slot) ambiguity"]),
    dict(id="intent_04", split="train", cat="intent", text="cancel my order for Friday",
         domain="orders", intent="cancel_food_delivery", slots={"date": "Friday"}, say={IW: "the food delivery"},
         alts=[A("could be retail", intent="cancel_shopping_order", say={IW: "the shopping order"}),
               A("could be a booking", intent="cancel_restaurant_reservation", say={IW: "the restaurant reservation"})],
         open=[("Which order do you mean?", ["intent"], "The food delivery")], stem="Do you mean",
         gold_prior="unclear",
         rationale="each cancellation calls a different backend action",
         flags=["axis boundary: could be argued as 'which object' (slot)"]),
    dict(id="intent_05", split="train", cat="intent", text="the pharmacy on Main Street",
         domain="navigation", intent="call_business", slots={"place": "pharmacy on Main Street"},
         say={IW: "call them"},
         alts=[A("go there", intent="navigate", say={IW: "get directions"}),
               A("look up hours", intent="show_business_hours", say={IW: "see their opening hours"})],
         open=[("What would you like to do?", ["intent"], "Call them")], stem="Do you want to",
         gold_prior="minority", rationale="no verb given: call, navigate and look up hours are different actions",
         flags=["fragment utterance"]),
    dict(id="intent_06", split="train", cat="intent", text="set the living room to seventy",
         domain="smart_home", intent="set_thermostat", slots={"room": "living room", "value": 70},
         say={IW: "the temperature"},
         alts=[A("seventy percent brightness", intent="set_light_brightness", say={IW: "the lights"}),
               A("seventy percent volume", intent="set_speaker_volume", say={IW: "the speaker"})],
         open=[("What would you like to set?", ["intent"], "The temperature")], stem="Do you mean",
         gold_prior="modal", rationale="thermostat, lighting and audio are different device actions",
         flags=["re-authored: v1 had an unresolved 'it'", "unit of 70 (F vs C) is assumed Fahrenheit"]),
    dict(id="intent_07", split="train", cat="intent", text="book Paris for June",
         domain="travel", intent="book_hotel", slots={"destination": "Paris", "month": "June"},
         say={IW: "a hotel"},
         alts=[A("transport instead of lodging", intent="book_flight", say={IW: "a flight"}),
               A("ground transport", intent="book_rental_car", say={IW: "a rental car"})],
         open=[("What would you like to book?", ["intent"], "A hotel")], stem="Do you want",
         gold_prior="minority", rationale="lodging, air travel and car hire are different booking actions",
         flags=["axis boundary: could be argued as 'travel product' (slot)"]),
    dict(id="intent_08", split="val", cat="intent", text="what's Thursday looking like",
         domain="productivity", intent="show_calendar", slots={"date": "Thursday"}, say={IW: "my schedule"},
         alts=[A("could mean weather", intent="get_forecast", say={IW: "the weather"}),
               A("could mean reminders", intent="show_reminders", say={IW: "my reminders"})],
         open=[("What would you like to see for Thursday?", ["intent"], "My schedule")], stem="Do you want",
         gold_prior="modal", rationale="calendar, weather and reminders are different lookups"),
    dict(id="intent_09", split="val", cat="intent", text="Priya's number",
         domain="contacts", intent="call_contact", slots={"contact": "Priya"}, say={IW: "call her"},
         alts=[A("only look it up", intent="show_contact_details", say={IW: "see her number"}),
               A("reach her by message", intent="open_chat", say={IW: "message her"})],
         open=[("What would you like to do?", ["intent"], "Call her")], stem="Do you want to",
         gold_prior="minority",
         rationale="dialling, displaying a number and messaging are different actions",
         flags=["re-authored: v1 text-vs-email was a channel parameter", "fragment utterance"]),
    dict(id="intent_10", split="val", cat="intent", text="the Giants game tonight",
         domain="events", intent="navigate_to_event", slots={"event": "Giants game", "date": "today"},
         say={IW: "navigate there"},
         alts=[A("watch or listen", intent="play_event_broadcast", say={IW: "put it on"}),
               A("attend", intent="buy_event_tickets", say={IW: "buy tickets"})],
         open=[("What would you like to do about it?", ["intent"], "Navigate there")], stem="Do you want me to",
         gold_prior="minority",
         rationale="navigating, broadcasting and ticketing are different actions",
         flags=["re-authored: v1 speaker-vs-TV was a device parameter", "fragment utterance",
                "'tonight' normalised to date=today"]),
    dict(id="intent_11", split="test", cat="intent", text="turn off the alarm",
         domain="alarm", intent="dismiss_alarm", slots={}, say={IW: "dismiss the ringing alarm"},
         alts=[A("switch the schedule off", intent="disable_alarm", say={IW: "disable the alarm"}),
               A("security alarm", intent="disarm_security_system", say={IW: "disarm the security system"})],
         open=[("Which alarm do you mean?", ["intent"], "Dismiss the ringing alarm")], stem="Do you want me to",
         gold_prior="modal",
         rationale="dismissing, disabling and disarming are different actions",
         flags=["re-authored: v1 delivery-vs-pickup was a fulfilment parameter"]),
    dict(id="intent_12", split="test", cat="intent", text="mute Alex",
         domain="messaging", intent="mute_chat_notifications", slots={"person": "Alex"},
         say={IW: "mute Alex's chat notifications"},
         alts=[A("mute on a live call", intent="mute_call_participant", say={IW: "mute Alex on the call"}),
               A("mute their feed", intent="mute_social_feed", say={IW: "mute Alex's posts"})],
         open=[("What would you like me to mute?", ["intent"], "Mute Alex's chat notifications")],
         stem="Do you want me to", gold_prior="modal",
         rationale="notifications, call audio and feed posts are different mute actions",
         flags=["v1 alternative 'mute_call_microphone' was implausible and was replaced"]),

    # ===== underspecified (12): intent clear, a required slot missing ========================
    dict(id="underspec_01", split="train", cat="underspecified", text="set an alarm",
         domain="alarm", intent="set_alarm", slots={"time": "07:00"}, utt={}, say={"time": "seven in the morning"},
         alts=[A("other morning time", say={"time": "six in the morning"}, time="06:00"),
               A("other morning time", say={"time": "eight in the morning"}, time="08:00")],
         open=[("What time should I set it for?", ["time"], "Seven in the morning")], ask="time",
         stem="Should I set it for"),
    dict(id="underspec_02", split="train", cat="underspecified", text="remind me about the meeting",
         domain="reminders", intent="create_reminder", slots={"task": "the meeting", "lead_time_min": 60},
         utt={"task": "the meeting"}, say={"lead_time_min": "an hour before"},
         alts=[A("shorter lead", say={"lead_time_min": "fifteen minutes before"}, lead_time_min=15),
               A("shorter lead", say={"lead_time_min": "thirty minutes before"}, lead_time_min=30)],
         open=[("How long before should I remind you?", ["lead_time_min"], "An hour before")],
         ask="lead_time_min", stem="Should I remind you"),
    dict(id="underspec_03", split="train", cat="underspecified", text="play some music",
         domain="music", intent="play_genre", slots={"genre": "jazz"}, utt={}, say={"genre": "jazz"},
         alts=[A("other genre", genre="rock"), A("other genre", genre="classical")],
         open=[("What would you like to hear?", ["genre"], "Jazz")], ask="genre", stem="Do you want",
         acceptable=[{"intent": "play_genre", "slots": {"genre": "*"}}],
         acceptable_note=("low-stakes: the utterance expresses no genre preference, so any genre completes the "
                          "task; the hidden 'jazz' is only what the simulated user says if asked"),
         flags=["acceptable_tasks wildcard: asking is not required for task success here"]),
    dict(id="underspec_04", split="train", cat="underspecified", text="book a table",
         domain="restaurant", intent="book_table", slots={"party_size": 2, "time": "20:00", "date": "today"},
         utt={}, say={"party_size": "two", "time": "eight", "date": "tonight"},
         alts=[A("different party size", say={"party_size": "four"}, party_size=4),
               A("different party size", say={"party_size": "six"}, party_size=6)],
         open=[("For how many people, and when?", ["party_size", "time", "date"], "Two people at eight tonight"),
               ("How many people?", ["party_size"], "Two")],
         ask="party_size", stem="Is that for",
         flags=["'eight' + 'tonight' normalised to 20:00/today (value normalisation, not inference)"]),
    dict(id="underspec_05", split="train", cat="underspecified", text="send a message to Sam",
         domain="messaging", intent="send_text", slots={"recipient": "Sam", "body": "on my way"},
         utt={"recipient": "Sam"}, say={"body": "on my way"},
         alts=[A("other message", body="running late", say={"body": "running late"}),
               A("other message", body="call me", say={"body": "call me"})],
         open=[("What should the message say?", ["body"], "On my way")], ask="body", stem="Should it say"),
    dict(id="underspec_06", split="train", cat="underspecified", text="turn up the volume",
         domain="device", intent="adjust_volume", slots={"direction": "up", "device": "living room speaker"},
         utt={"direction": "up"}, say={"device": "the living room speaker"},
         alts=[A("other device", device="phone", say={"device": "the phone"}),
               A("other device", device="TV", say={"device": "the TV"})],
         open=[("On which device?", ["device"], "The living room speaker")], ask="device", stem="Do you mean"),
    dict(id="underspec_07", split="train", cat="underspecified", text="add milk to the list",
         domain="lists", intent="add_item", slots={"item": "milk", "list": "grocery"}, utt={"item": "milk"},
         say={"list": "the grocery list"},
         alts=[A("other list", list="to-do", say={"list": "the to-do list"}),
               A("other list", list="baking", say={"list": "the baking list"})],
         open=[("Which list?", ["list"], "The grocery list")], ask="list", stem="Do you mean"),
    dict(id="underspec_08", split="val", cat="underspecified", text="cancel it",
         domain="appointments", intent="cancel_appointment", slots={"appointment": "dentist appointment"},
         utt={}, say={"appointment": "the dentist appointment"},
         alts=[A("other appointment", appointment="haircut", say={"appointment": "the haircut"}),
               A("other appointment", appointment="gym class", say={"appointment": "the gym class"})],
         open=[("What would you like to cancel?", ["appointment"], "The dentist appointment")],
         ask="appointment", stem="Do you mean"),
    dict(id="underspec_09", split="val", cat="underspecified", text="get me a ticket",
         domain="events", intent="book_ticket", slots={"event": "Coldplay concert", "quantity": 1}, utt={},
         say={"event": "the Coldplay concert", "quantity": "one ticket"},
         alts=[A("other event", event="comedy show", say={"event": "the comedy show"}),
               A("other event", event="jazz festival", say={"event": "the jazz festival"})],
         open=[("Which event, and how many tickets?", ["event", "quantity"], "The Coldplay concert, one ticket"),
               ("Which event?", ["event"], "The Coldplay concert")],
         ask="event", stem="Do you mean",
         flags=["v1 open reply silently omitted quantity; now a question that covers only event is partial"]),
    dict(id="underspec_10", split="test", cat="underspecified", text="what's the weather",
         domain="weather", intent="get_forecast", slots={"city": "Chicago", "date": "today"}, utt={},
         say={"city": "Chicago", "date": "today"},
         alts=[A("other city", city="Denver", say={"city": "Denver"}),
               A("other city", city="Seattle", say={"city": "Seattle"})],
         open=[("Which city and which day?", ["city", "date"], "Chicago, today"),
               ("Which city?", ["city"], "Chicago")],
         ask="city", stem="Do you mean"),
    dict(id="underspec_11", split="test", cat="underspecified", text="email the report",
         domain="email", intent="send_email", slots={"attachment": "the report", "recipient": "my manager"},
         utt={"attachment": "the report"}, say={"recipient": "my manager"},
         alts=[A("other recipient", recipient="the team", say={"recipient": "the team"}),
               A("other recipient", recipient="my assistant", say={"recipient": "my assistant"})],
         open=[("Who should I send it to?", ["recipient"], "To my manager")], ask="recipient",
         stem="Should I send it to"),
    dict(id="underspec_12", split="test", cat="underspecified", text="set the thermostat",
         domain="smart_home", intent="set_thermostat", slots={"temperature_c": 21}, utt={},
         say={"temperature_c": "twenty-one"},
         alts=[A("other temperature", say={"temperature_c": "nineteen"}, temperature_c=19),
               A("other temperature", say={"temperature_c": "twenty-three"}, temperature_c=23)],
         open=[("What temperature would you like?", ["temperature_c"], "Twenty-one degrees")],
         ask="temperature_c", stem="Should I set it to"),
]
# fmt: on


# --------------------------------------------------------------------------------------------
def disp_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + (", or " if len(items) > 2 else " or ") + items[-1]


def apply_alt(gold_intent: str, gold_slots: dict, alt: dict) -> dict:
    return {"intent": alt["intent"] or gold_intent, "slots": {**gold_slots, **alt["slots"]}}


def changed_keys(gold_intent: str, gold_slots: dict, task: dict) -> list[str]:
    keys = [k for k, v in task["slots"].items() if gold_slots.get(k) != v]
    return keys + (["intent"] if task["intent"] != gold_intent else [])


def recover(required, known, explicit, assumed=None):
    """Strict (or, with `assumed`, lenient) status. Unresolved = required and not known/explicit."""
    got = set(known) | set(explicit) | set(assumed or {})
    unresolved = [k for k in required if k not in got]
    helped = any(k in required and k not in known for k in list(explicit) + list(assumed or {}))
    status = "full" if not unresolved else ("partial" if helped else "none")
    return {"status": status, "unresolved_slots": unresolved}


def build_record(spec: dict, snr_levels: list, seed: str, gold_pos_counter: dict) -> dict:
    bid = f"p1_{spec['id']}"
    cat, slots, intent = spec["cat"], spec["slots"], spec["intent"]
    say, alts_in = spec["say"], spec["alts"]
    cf = spec.get("conf")
    gold = {**slots, "intent": intent}
    intent_unresolved = cat == "intent" or (cf is not None and cf["affects"] == "intent")
    required = list(slots) + (["intent"] if intent_unresolved else [])
    at_risk = [cf["affects"]] if cf else []
    if "utt" in spec:
        utt = dict(spec["utt"])
    else:
        utt = dict(slots)
        if cf is not None and cf["affects"] == "intent":
            utt["intent"] = intent
    known = {k: gold[k] for k in gold if (k in utt and k not in at_risk) or (k == "intent" and "intent" not in required)}

    alternatives = []
    for a in alts_in:
        task = apply_alt(intent, slots, a)
        alternatives.append({"task": task, "differs_in": changed_keys(intent, slots, task),
                             "say": a["say"], "why": a["why"]})

    def phrase(k):  # literal phrase the user uses for the gold value of k
        return say[k]

    def task_text(t):
        return TEMPLATES[t["intent"]](t["slots"])

    probes: list[dict] = []

    def add_probe(p):
        p["probe_id"] = f"{bid}.{len(probes) + 1:02d}"
        probes.append(p)

    def rec_block(explicit, confirms, exclusions, assumed=None, contradicted=()):
        """A rejected proposal that contradicts utterance-supported slots shows the agent did not hold them,
        so those slots stop counting as known for this probe (and count as required)."""
        reply_info = {**explicit, **confirms}
        known_p = {k: v for k, v in known.items() if k not in contradicted}
        eff = required + [k for k in contradicted if k not in required]
        strict = recover(eff, known_p, reply_info)
        out = {
            "from_utterance": known,
            "contradicted_utterance_support": [k for k in contradicted if k in known],
            "from_reply": reply_info,
            "exclusions": exclusions,
            "strict": strict,
        }
        lenient = recover(eff, known_p, reply_info, assumed)
        out["lenient"] = {**lenient, "assumed_accepted": assumed or {},
                          "assumption": "slots of the rejected proposal that were not contradicted are accepted"
                          if assumed else "none"}
        return out

    # ---- open questions -------------------------------------------------------------------
    if cat != "none":
        for q, covers, reply in spec["open"]:
            supplies = {k: gold[k] for k in covers}
            add_probe({
                "question_type": "open", "question": q, "covers": covers,
                "reply": {"type": "answer", "text": reply, "entailment": "literal",
                          "supplies": supplies, "confirms": {}},
                "recovery": rec_block(supplies, {}, []),
            })

    # ---- closed questions -----------------------------------------------------------------
    ask = None
    if cat != "none":
        ask = spec.get("ask") or ("intent" if cat == "intent" else cf["affects"])
        cands = [a for a in alternatives if ask in a["differs_in"]]
        assert len(cands) >= 2, bid
        cands = cands[:2]

        def alt_val(a):
            return a["task"]["intent"] if ask == "intent" else a["task"]["slots"][ask]

        def alt_txt(a):
            return a["say"].get(ask, str(alt_val(a)))

        gold_val = gold[ask]
        n = 1 + len(cands)
        pos = gold_pos_counter.setdefault(n, 0) % n
        gold_pos_counter[n] += 1
        rng = random.Random(f"{seed}|{bid}|closed_with_gold")
        others = list(cands)
        rng.shuffle(others)
        entries = [(phrase(ask), gold_val)] + [(alt_txt(a), alt_val(a)) for a in others]
        order = entries[1:]
        order.insert(pos, entries[0])
        add_probe({
            "question_type": "closed",
            "question": f"{spec['stem']} {disp_list([t for t, _ in order])}?",
            "covers": [ask], "options": [t for t, _ in order], "option_values": [v for _, v in order],
            "options_contain_gold": True,
            "option_order": {"method": "balanced_gold_position_then_seeded_shuffle",
                             "seed": f"{seed}|{bid}|closed_with_gold", "gold_position": pos},
            "reply": {"type": "select", "text": phrase(ask)[0].upper() + phrase(ask)[1:], "entailment": "literal",
                      "supplies": {ask: gold_val}, "confirms": {}},
            "recovery": rec_block({ask: gold_val}, {}, []),
        })
        rng2 = random.Random(f"{seed}|{bid}|closed_without_gold")
        omit = list(cands)
        rng2.shuffle(omit)
        add_probe({
            "question_type": "closed",
            "question": f"{spec['stem']} {disp_list([alt_txt(a) for a in omit])}?",
            "covers": [ask], "options": [alt_txt(a) for a in omit], "option_values": [alt_val(a) for a in omit],
            "options_contain_gold": False,
            "option_order": {"method": "seeded_shuffle", "seed": f"{seed}|{bid}|closed_without_gold",
                             "gold_position": None},
            "reply": {"type": "none_of_these", "text": "None of those.", "entailment": "literal",
                      "supplies": {}, "confirms": {}},
            "recovery": rec_block({}, {}, [{"kind": "rejected_options", "target": ask,
                                            "values": [alt_val(a) for a in omit]}]),
        })

    # ---- confirmation probes: correct, incorrect-bare, incorrect-corrected -----------------
    gold_task = {"intent": intent, "slots": slots}
    q_correct = f"Do you want me to {task_text(gold_task)}?"
    asserted = {**slots, "intent": intent}
    add_probe({
        "question_type": "confirm", "question": q_correct, "proposed_task": gold_task,
        "proposal_is_correct": True, "proposal_source": "gold", "covers": list(required),
        "reply": {"type": "yes", "text": "Yes.", "entailment": "literal", "supplies": {}, "confirms": asserted},
        "recovery": rec_block({}, asserted, []),
    })
    wrong = alternatives[0]["task"]
    ch = alternatives[0]["differs_in"]
    q_wrong = f"Do you want me to {task_text(wrong)}?"
    rejected = [{"kind": "rejected_proposal", "task": wrong, "joint": True}]
    add_probe({
        "question_type": "confirm", "question": q_wrong, "proposed_task": wrong,
        "proposal_is_correct": False, "proposal_source": "alternative:0", "covers": list(required),
        "reply": {"type": "no_bare", "text": "No.", "entailment": "literal", "supplies": {}, "confirms": {}},
        "recovery": rec_block({}, {}, rejected, contradicted=ch),
    })
    corr = {k: gold[k] for k in ch}
    corr_text = "No, " + " and ".join(phrase(k) for k in ch) + "."
    wrong_asserted = {**wrong["slots"], "intent": wrong["intent"]}
    assumed = {k: v for k, v in wrong_asserted.items() if k in required and k not in corr and gold[k] == v}
    add_probe({
        "question_type": "confirm", "question": q_wrong, "proposed_task": wrong,
        "proposal_is_correct": False, "proposal_source": "alternative:0", "covers": list(required),
        "reply": {"type": "no_correction", "text": corr_text, "entailment": "literal",
                  "supplies": corr, "confirms": {}},
        "recovery": rec_block(corr, {}, rejected, assumed, contradicted=ch),
    })

    # ---- agent-visible block: opaque ids, null audio, nothing else -------------------------
    uid = "u" + hashlib.sha256(f"{seed}|{bid}".encode()).hexdigest()[:8]
    letters = list("abcd"[: len(snr_levels)])
    random.Random(f"{seed}|{bid}|variant_order").shuffle(letters)
    conditions, visible = [], []
    for snr, letter in zip(snr_levels, letters):
        vid = f"{uid}-{letter}"
        conditions.append({"variant_id": vid, "snr_db": snr, "label": "clean" if snr is None else f"{snr}dB"})
        visible.append({"variant_id": vid, "audio_path": None, "audio_status": "planned"})
    visible.sort(key=lambda v: v["variant_id"])

    fams = {i: FAMILY[i] for i in {intent, *[a["task"]["intent"] for a in alternatives]} if i in FAMILY}
    acceptable = spec.get("acceptable") or [gold_task]
    private = {
        "ambiguity_type": cat,
        "distinguishing_axis": AXIS_FOR[cat],
        "axis_rationale": spec.get("rationale", ""),
        "gold_prior": spec.get("gold_prior"),
        "acoustic_confusable": cf,
        "conditions": conditions,
        "reference_text": spec["text"],
        "gold_task": {"domain": spec["domain"], "intent": intent, "slots": slots},
        "utterance_slots": utt,
        "required_slots": required,
        "at_risk_slots": at_risk,
        "say": say,
        "intent_families": fams,
        "alternatives": alternatives,
        "candidate_set_exhaustive": False,
        "acceptable_tasks": acceptable,
        "acceptable_note": spec.get("acceptable_note", ""),
        "probes": probes,
        "not_applicable_questions": ["open", "closed"] if cat == "none" else [],
        "scoring": "structured comparison of (intent, slots) against acceptable_tasks; never substring match",
        "annotation": {"annotator_ids": ["claude_draft"], "agreement": None, "status": "ai_drafted_unreviewed",
                       "flags": spec.get("flags", [])},
        "meta": {"provenance": "authored_synthetic", "external_dataset": None},
    }
    return {"schema_version": SCHEMA_VERSION, "base_id": bid, "split": spec["split"],
            "source": "authored_synthetic", "agent_visible": {"audio_variants": visible}, "private": private}


def review_markdown(records: list[dict]) -> str:
    lines = [
        "# Phase 1 human-review checklist (generated by scripts/author_phase1.py)", "",
        "Every annotation below is an **unreviewed AI draft**. No human annotator, agreement score or",
        "inter-annotator reliability exists. Tick a box only after you have personally checked the item;",
        "do not tick on the validator's behalf: passing `validate_phase1.py` is not scientific validation.", "",
        "Columns: **Gold** defensible? | **Cat/axis** right? | **Alts** plausible? | **Qs** natural? | "
        "**Replies** answer the questions? | **Recovery** justified?", "",
        "| Item | Split | Utterance | Gold task | Axis | Flags for your decision | G | C | A | Q | R | V |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        p = r["private"]
        gt = p["gold_task"]
        gold = f"{gt['intent']}({', '.join(f'{k}={v}' for k, v in gt['slots'].items())})"
        flags = "; ".join(p["annotation"]["flags"]) or "-"
        if p["gold_prior"]:
            flags = f"gold_prior={p['gold_prior']}; " + flags
        lines.append(f"| {r['base_id'][3:]} | {r['split']} | {p['reference_text']} | `{gold}` | "
                     f"{p['ambiguity_type']}/{p['distinguishing_axis']} | {flags} |  |  |  |  |  |  |")
    lines += ["", "## Judgement calls that need adjudication", ""]
    for r in records:
        p = r["private"]
        if p["ambiguity_type"] == "intent":
            lines.append(f"- **{r['base_id'][3:]}** gold `{p['gold_task']['intent']}` "
                         f"(author's unmeasured prior: {p['gold_prior']}): {p['axis_rationale']}.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--review-path", default=str(REVIEW_PATH))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    snr_levels = cfg["data"]["snr_levels"]
    seed = f"phase1.v2:{cfg['seed']}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counter: dict[int, int] = {}
    records = [build_record(s, snr_levels, seed, counter) for s in ITEMS]
    items_path = out_dir / "base_items.jsonl"
    items_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records))

    splits = ("train", "val", "test")
    counts = {c: {sp: sum(1 for r in records if r["private"]["ambiguity_type"] == c and r["split"] == sp)
                  for sp in splits} for c in PLAN}
    probes = [p for r in records for p in r["private"]["probes"]]
    tally = lambda key: {k: sum(1 for p in probes if key(p) == k) for k in sorted({key(p) for p in probes})}  # noqa: E731
    rec_counts = {s: sum(1 for p in probes if p["recovery"]["strict"]["status"] == s) for s in ("full", "partial", "none")}
    na_pairs = sum(len(r["private"]["not_applicable_questions"]) for r in records)
    wild = [r["base_id"] for r in records if any("*" in t["slots"].values() for t in r["private"]["acceptable_tasks"])]
    manifest = {
        "name": "phase1_feasibility",
        "schema_version": SCHEMA_VERSION,
        "status": "draft_unfrozen",
        "description": ("Small feasibility dataset, not a statistically sufficient benchmark. train = development, "
                        "val = tuning and calibration, test = held-out evaluation."),
        "provenance": ("All 50 base items are hand-authored synthetic text; none is drawn from SLURP or any "
                       "external dataset. Annotations are an unreviewed single-annotator AI draft."),
        "audio": "none generated; every audio_path is null and every audio_status is 'planned'",
        "snr_levels_planned": snr_levels,
        "seed": seed,
        "planned_counts": {"base_items": 50, "by_category_split": {c: dict(zip(splits, v)) for c, v in PLAN.items()}},
        "actual_counts": {"base_items": len(records), "by_category_split": counts,
                          "variants": sum(len(r["agent_visible"]["audio_variants"]) for r in records)},
        "prevalence": {"labelled_ambiguous": sum(1 for r in records if r["private"]["ambiguity_type"] != "none"),
                       "labelled_clear": sum(1 for r in records if r["private"]["ambiguity_type"] == "none"),
                       "wildcard_acceptable_items": wild,
                       "note": ("37 of 50 items are labelled as needing clarification; wildcard items accept any "
                                "value, so asking is not required for success there.")},
        "probe_counts": {"total": len(probes), "by_question_type": tally(lambda p: p["question_type"]),
                         "by_reply_type": tally(lambda p: p["reply"]["type"]),
                         "confirm_correct": sum(1 for p in probes if p.get("proposal_is_correct") is True),
                         "confirm_incorrect": sum(1 for p in probes if p.get("proposal_is_correct") is False)},
        "recovery_counts_strict": {**rec_counts, "not_applicable_question_slots": na_pairs},
        "split_ids": {sp: [r["base_id"] for r in records if r["split"] == sp] for sp in splits},
        "items_file": items_path.name,
        "items_sha256": hashlib.sha256(items_path.read_bytes()).hexdigest(),
        "author_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "agent_visible_fields": ["agent_visible.audio_variants[].variant_id/audio_path/audio_status"],
        "private_fields": "everything under `private`, plus `base_id` (its text names the category)",
        "notes": [
            "Probes are private reference scenarios, not instructions to the agent.",
            "Recovery is open-world and strict by default; lenient is a labelled sensitivity annotation.",
            "'intent' is a reserved slot name; see the scripts/author_phase1.py docstring.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    review_path = Path(args.review_path)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(review_markdown(records))
    print(f"Wrote {len(records)} base items, {len(probes)} probes -> {items_path}")


if __name__ == "__main__":
    main()
