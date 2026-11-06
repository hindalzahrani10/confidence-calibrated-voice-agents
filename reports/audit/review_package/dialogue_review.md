# Manual review package: 20 closed-loop dialogues

**Nothing here is human-reviewed.** All fields in `review_checklist.csv` marked *(blank)* are for you to fill in. Data are SYNTHETIC (hand-authored text, Piper TTS, DEMAND cafeteria noise). Dialogues are drawn only from the train/val dev episodes (`data/phase5/dev.v1`); the held-out test split was not opened. Audio files are byte-for-byte copies (sha256 checked against the Phase 2 manifests); originals are untouched.

**Listening in VS Code:** open this file with *Markdown: Open Preview* (Ctrl+Shift+V); the `<audio>` players work in the preview. If a player does not render, click the `.wav` link (VS Code's built-in media preview plays WAV), or open `index.html` in a browser/Live Preview.

Legend of automatic flags (informational, not verdicts): F1 reply ASR differs from intended text; F2 reply unusable (belief not updated); F3 parsed outcome != intended; F4 off-target simulator reply; F5 reply type incompatible with question; F6 near-tie initial belief; F7 blind SNR bucket != nominal; F8 initial ASR != reference; F9 wrong answer despite correctly parsed reply; F10 unnecessary clarification; F13 gold not in agent's initial top-5.


---

## D01 · `p1_acoustic_01` · acoustic · 10 dB · evpi · split=train

- **Episode key:** `evpi|uf38d6672-b` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “call mom”
- **Intended meaning (hidden gold):** call Mom  ·  acceptable set: `[{"intent": "call_contact", "slots": {"contact": "Mom"}}]`
- **Initial audio:** [audio/D01_initial__uf38d6672-b.wav](audio/D01_initial__uf38d6672-b.wav)  
  <audio controls src="audio/D01_initial__uf38d6672-b.wav"></audio>
- **ASR of original (agent saw):** “Come on.”  ·  blind SNR bucket est.: 10
- **Initial belief top-5:** book 1 ticket to the comedy show (0.052); play some classical (0.048); play some jazz (0.048); play some rock (0.048); get the weather in Denver for tomorrow (0.004)  ·  max-prob 0.0525, entropy 4.9442
- **Expected utilities considered:** `{"abstain": 0.0, "answer": -0.895, "ask_closed:body": -0.2, "ask_closed:intent": -0.2, "ask_closed:recipient": -0.2, "ask_confirm": -0.0475, "ask_open": -0.0855}`
- **Initial decision:** **abstain**
- **Final decision:** **abstain**  ·  success=False, wrong=False, reward=+0.0, cost=0.00, **utility=+0.00**
- **Automatic flags:** F8 initial ASR transcript differs from reference text; F13 gold task not in agent's initial top-5
- **Human assessment:** *(blank, see `review_checklist.csv`, row D01)*

---

## D02 · `p1_acoustic_10` · acoustic · 20 dB · nm_voi · split=val

- **Episode key:** `nm_voi|ua69ef8a4-d` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “dim the bedroom lights”
- **Intended meaning (hidden gold):** dim the bedroom lights  ·  acceptable set: `[{"intent": "dim_lights", "slots": {"room": "bedroom"}}]`
- **Initial audio:** [audio/D02_initial__ua69ef8a4-d.wav](audio/D02_initial__ua69ef8a4-d.wav)  
  <audio controls src="audio/D02_initial__ua69ef8a4-d.wav"></audio>
- **ASR of original (agent saw):** “dim the bedroom lights”  ·  blind SNR bucket est.: 20
- **Initial belief top-5:** dim the bedroom lights (0.945); dim the bathroom lights (0.054); dim the guest room lights (0.001); turn off the dining room lights (0.000); turn off the kitchen lights (0.000)  ·  max-prob 0.945, entropy 0.2161
- **Expected utilities considered:** `{"abstain": 0.0, "answer": 0.8899, "ask_closed:intent": 0.6899, "ask_closed:room": 0.7637, "ask_closed:time": 0.6899, "ask_confirm": 0.8934, "ask_open": 0.6781}`
- **Initial decision:** **ask_confirm**
- **Clarification question (confirm):** “Do you want me to dim the bedroom lights?”
- **Intended simulated reply:** “Yes.” (type `yes`)  ·  relevance rule: simulator says Yes only if the proposal is acceptable, else No (bare or corrective)
- **Reply audio:** [audio/D02_reply__rbd3882a7b2.wav](audio/D02_reply__rbd3882a7b2.wav)  
  <audio controls src="audio/D02_reply__rbd3882a7b2.wav"></audio>
- **ASR of reply (agent saw):** “Yes.”  ·  parsed outcome: 0
- **Updated belief top-3:** dim the bedroom lights (0.998); dim the bathroom lights (0.002); dim the guest room lights (0.000)
- **Final decision:** **answer** → “dim the bedroom lights”  ·  success=True, wrong=False, reward=+1.0, cost=0.10, **utility=+0.90**
- **Automatic flags:** F10 clarification asked although initial MAP task was already acceptable
- **Human assessment:** *(blank, see `review_checklist.csv`, row D02)*

---

## D03 · `p1_acoustic_02` · acoustic · 20 dB · nm_voi · split=train

- **Episode key:** `nm_voi|u8044439b-a` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “call Dan”
- **Intended meaning (hidden gold):** call Dan  ·  acceptable set: `[{"intent": "call_contact", "slots": {"contact": "Dan"}}]`
- **Initial audio:** [audio/D03_initial__u8044439b-a.wav](audio/D03_initial__u8044439b-a.wav)  
  <audio controls src="audio/D03_initial__u8044439b-a.wav"></audio>
- **ASR of original (agent saw):** “Cole Dan.”  ·  blind SNR bucket est.: 20
- **Initial belief top-5:** call Dan (0.128); call Dean (0.052); text Jan saying I'm running late (0.038); text Jan saying running late (0.038); call Don (0.038)  ·  max-prob 0.1284, entropy 4.536
- **Expected utilities considered:** `{"abstain": 0.0, "answer": -0.7432, "ask_closed:body": -0.1842, "ask_closed:intent": -0.1943, "ask_closed:recipient": -0.2, "ask_confirm": -0.0019, "ask_open": 0.0259}`
- **Initial decision:** **ask_open**
- **Clarification question (open):** “Could you tell me more precisely what you want?”
- **Intended simulated reply:** “Dan” (type `answer`)  ·  relevance rule: generic open question: simulator restates the unresolved information
- **Reply audio:** [audio/D03_reply__readd46da38.wav](audio/D03_reply__readd46da38.wav)  
  <audio controls src="audio/D03_reply__readd46da38.wav"></audio>
- **ASR of reply (agent saw):** “Then”  ·  parsed outcome: None
- **Updated belief top-3:** call Dan (0.128); call Dean (0.052); text Jan saying I'm running late (0.038)
- **Final decision:** **abstain**  ·  success=False, wrong=False, reward=+0.0, cost=0.30, **utility=-0.30**
- **Automatic flags:** F8 initial ASR transcript differs from reference text; F1 reply ASR transcript differs from intended reply text; F2 reply not usable by the agent (parsed outcome None; belief not updated); F10 clarification asked although initial MAP task was already acceptable
- **Human assessment:** *(blank, see `review_checklist.csv`, row D03)*

---

## D04 · `p1_acoustic_09` · acoustic · clean dB · always_ask · split=val

- **Episode key:** `always_ask|u232a951f-d` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “remind me at six PM to buy flour”
- **Intended meaning (hidden gold):** set a reminder for buy flour at 6 pm  ·  acceptable set: `[{"intent": "create_reminder", "slots": {"task": "buy flour", "time": "18:00"}}]`
- **Initial audio:** [audio/D04_initial__u232a951f-d.wav](audio/D04_initial__u232a951f-d.wav)  
  <audio controls src="audio/D04_initial__u232a951f-d.wav"></audio>
- **ASR of original (agent saw):** “remind me at 6 p.m. to buy flour”  ·  blind SNR bucket est.: clean
- **Initial belief top-5:** set a reminder for buy flour at 6 pm (0.994); set a reminder for buy flax at 6 pm (0.002); set a reminder for buy flowers at 6 pm (0.002); set a reminder for buy flour at 9 pm (0.001); set a reminder for buy flour at 9 am (0.001)  ·  max-prob 0.9944, entropy 0.0419
- **Initial decision:** **ask_closed**
- **Clarification question (closed):** “Which do you mean: buy flour, buy flax, buy flowers, or something else?”  (variable: task; options: ['buy flour', 'buy flax', 'buy flowers'])
- **Intended simulated reply:** “Buy flour” (type `select`)  ·  relevance rule: simulator answers with the offered option equal to its hidden value, else 'None of those.'
- **Reply audio:** [audio/D04_reply__r89693be0fd.wav](audio/D04_reply__r89693be0fd.wav)  
  <audio controls src="audio/D04_reply__r89693be0fd.wav"></audio>
- **ASR of reply (agent saw):** “Bye, flower.”  ·  parsed outcome: None
- **Updated belief top-3:** set a reminder for buy flour at 6 pm (0.994); set a reminder for buy flax at 6 pm (0.002); set a reminder for buy flowers at 6 pm (0.002)
- **Final decision:** **answer** → “set a reminder for buy flour at 6 pm”  ·  success=True, wrong=False, reward=+1.0, cost=0.20, **utility=+0.80**
- **Automatic flags:** F1 reply ASR transcript differs from intended reply text; F2 reply not usable by the agent (parsed outcome None; belief not updated); F10 clarification asked although initial MAP task was already acceptable
- **Human assessment:** *(blank, see `review_checklist.csv`, row D04)*

---

## D05 · `p1_acoustic_05` · acoustic · clean dB · noise_indep_voi · split=train

- **Episode key:** `noise_indep_voi|ue549cc1b-b` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “set a timer for thirteen minutes”
- **Intended meaning (hidden gold):** set a timer for 13 minutes  ·  acceptable set: `[{"intent": "set_timer", "slots": {"duration_min": 13}}]`
- **Initial audio:** [audio/D05_initial__ue549cc1b-b.wav](audio/D05_initial__ue549cc1b-b.wav)  
  <audio controls src="audio/D05_initial__ue549cc1b-b.wav"></audio>
- **ASR of original (agent saw):** “Set a timer for 13 minutes.”  ·  blind SNR bucket est.: clean
- **Initial belief top-5:** set a timer for 13 minutes (0.872); set a timer for 14 minutes (0.063); set a timer for 30 minutes (0.063); set a timer for 20 minutes (0.001); set a timer for 10 minutes (0.001)  ·  max-prob 0.8723, entropy 0.4802
- **Expected utilities considered:** `{"abstain": 0.0, "answer": 0.7447, "ask_closed:duration_min": 0.6932, "ask_closed:intent": 0.5447, "ask_closed:lead_time_min": 0.5447, "ask_confirm": 0.7583, "ask_open": 0.5971}`
- **Initial decision:** **ask_confirm**
- **Clarification question (confirm):** “Do you want me to set a timer for 13 minutes?”
- **Intended simulated reply:** “Yes.” (type `yes`)  ·  relevance rule: simulator says Yes only if the proposal is acceptable, else No (bare or corrective)
- **Reply audio:** [audio/D05_reply__r4e4dc5146c.wav](audio/D05_reply__r4e4dc5146c.wav)  
  <audio controls src="audio/D05_reply__r4e4dc5146c.wav"></audio>
- **ASR of reply (agent saw):** “Yes.”  ·  parsed outcome: 0
- **Updated belief top-3:** set a timer for 13 minutes (0.992); set a timer for 14 minutes (0.004); set a timer for 30 minutes (0.004)
- **Final decision:** **answer** → “set a timer for 13 minutes”  ·  success=True, wrong=False, reward=+1.0, cost=0.10, **utility=+0.90**
- **Automatic flags:** F10 clarification asked although initial MAP task was already acceptable
- **Human assessment:** *(blank, see `review_checklist.csv`, row D05)*

---

## D06 · `p1_intent_03` · intent · 20 dB · nm_voi · split=train

- **Episode key:** `nm_voi|u83a16275-c` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “record this”
- **Intended meaning (hidden gold):** record audio only  ·  acceptable set: `[{"intent": "record_audio", "slots": {}}]`
- **Initial audio:** [audio/D06_initial__u83a16275-c.wav](audio/D06_initial__u83a16275-c.wav)  
  <audio controls src="audio/D06_initial__u83a16275-c.wav"></audio>
- **ASR of original (agent saw):** “Record this.”  ·  blind SNR bucket est.: 20
- **Initial belief top-5:** record video (0.209); record audio only (0.209); record the screen (0.209); email the report to my assistant (0.064); email the report to the team (0.064)  ·  max-prob 0.2086, entropy 2.7654
- **Expected utilities considered:** `{"abstain": 0.0, "answer": -0.5829, "ask_closed:attachment": -0.2, "ask_closed:intent": 0.358, "ask_closed:recipient": -0.0621, "ask_confirm": 0.081, "ask_open": 0.4286}`
- **Initial decision:** **ask_open**
- **Clarification question (open):** “Could you tell me more precisely what you want?”
- **Intended simulated reply:** “Audio only” (type `answer`)  ·  relevance rule: generic open question: simulator restates the unresolved information
- **Reply audio:** [audio/D06_reply__r0e695010e2.wav](audio/D06_reply__r0e695010e2.wav)  
  <audio controls src="audio/D06_reply__r0e695010e2.wav"></audio>
- **ASR of reply (agent saw):** “Audio only.”  ·  parsed outcome: 1
- **Updated belief top-3:** record audio only (0.970); record video (0.008); record the screen (0.008)
- **Final decision:** **answer** → “record audio only”  ·  success=True, wrong=False, reward=+1.0, cost=0.30, **utility=+0.70**
- **Automatic flags:** F6 initial near-tie between two top tasks (winner chosen by hash jitter)
- **Human assessment:** *(blank, see `review_checklist.csv`, row D06)*

---

## D07 · `p1_intent_02` · intent · 5 dB · never_ask · split=train

- **Episode key:** `never_ask|u80493a1a-d` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “dentist tomorrow at ten”
- **Intended meaning (hidden gold):** add a calendar event for dentist tomorrow at 10 am  ·  acceptable set: `[{"intent": "create_calendar_event", "slots": {"date": "tomorrow", "task": "dentist", "time": "10:00"}}]`
- **Initial audio:** [audio/D07_initial__u80493a1a-d.wav](audio/D07_initial__u80493a1a-d.wav)  
  <audio controls src="audio/D07_initial__u80493a1a-d.wav"></audio>
- **ASR of original (agent saw):** “Dentist, tomorrow at 10.”  ·  blind SNR bucket est.: 5
- **Initial belief top-5:** add a calendar event for dentist tomorrow at 10 am (0.333); save a note about dentist tomorrow at 10 am (0.333); set a reminder for dentist at 10 am tomorrow (0.333); get the weather in Denver for tomorrow (0.000); get the weather in Boston for tomorrow (0.000)  ·  max-prob 0.3333, entropy 1.0988
- **Initial decision:** **answer**
- **Final decision:** **answer** → “add a calendar event for dentist tomorrow at 10 am”  ·  success=True, wrong=False, reward=+1.0, cost=0.00, **utility=+1.00**
- **Automatic flags:** F6 initial near-tie between two top tasks (winner chosen by hash jitter)
- **Human assessment:** *(blank, see `review_checklist.csv`, row D07)*

---

## D08 · `p1_intent_01` · intent · 5 dB · noise_indep_voi · split=train

- **Episode key:** `noise_indep_voi|u9a21f58d-c` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “play Thriller”
- **Intended meaning (hidden gold):** play the song Thriller  ·  acceptable set: `[{"intent": "play_song", "slots": {"title": "Thriller"}}]`
- **Initial audio:** [audio/D08_initial__u9a21f58d-c.wav](audio/D08_initial__u9a21f58d-c.wav)  
  <audio controls src="audio/D08_initial__u9a21f58d-c.wav"></audio>
- **ASR of original (agent saw):** “Place Reelers.”  ·  blind SNR bucket est.: 5
- **Initial belief top-5:** text Jan saying I'm running late (0.015); text Jenna saying I'm running late (0.015); text Sam saying running late (0.015); text Jen saying running late (0.015); text Jen saying I'm running late (0.015)  ·  max-prob 0.0145, entropy 5.1054
- **Expected utilities considered:** `{"abstain": 0.0, "answer": -0.9711, "ask_closed:body": -0.2, "ask_closed:intent": -0.2, "ask_closed:recipient": -0.2, "ask_confirm": -0.1, "ask_open": -0.3}`
- **Initial decision:** **abstain**
- **Final decision:** **abstain**  ·  success=False, wrong=False, reward=+0.0, cost=0.00, **utility=+0.00**
- **Automatic flags:** F6 initial near-tie between two top tasks (winner chosen by hash jitter); F8 initial ASR transcript differs from reference text; F13 gold task not in agent's initial top-5
- **Human assessment:** *(blank, see `review_checklist.csv`, row D08)*

---

## D09 · `p1_intent_09` · intent · clean dB · confidence_threshold · split=val

- **Episode key:** `confidence_threshold|ua7eec175-c` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “Priya's number”
- **Intended meaning (hidden gold):** call Priya  ·  acceptable set: `[{"intent": "call_contact", "slots": {"contact": "Priya"}}]`
- **Initial audio:** [audio/D09_initial__ua7eec175-c.wav](audio/D09_initial__ua7eec175-c.wav)  
  <audio controls src="audio/D09_initial__ua7eec175-c.wav"></audio>
- **ASR of original (agent saw):** “Priya's number.”  ·  blind SNR bucket est.: clean
- **Initial belief top-5:** show Priya contact details (0.317); open a chat with Priya (0.317); call Priya (0.317); text Jan saying I'm outside (0.000); text Jan saying running late (0.000)  ·  max-prob 0.3171, entropy 1.4934
- **Initial decision:** **ask_closed**
- **Clarification question (closed):** “Which do you mean: call, show contact details, open a chat, or something else?”  (variable: intent; options: ['call', 'show contact details', 'open a chat'])
- **Intended simulated reply:** “Call” (type `select`)  ·  relevance rule: simulator answers with the offered option equal to its hidden value, else 'None of those.'
- **Reply audio:** [audio/D09_reply__rf4df748e2b.wav](audio/D09_reply__rf4df748e2b.wav)  
  <audio controls src="audio/D09_reply__rf4df748e2b.wav"></audio>
- **ASR of reply (agent saw):** “call”  ·  parsed outcome: 0
- **Updated belief top-3:** call Priya (0.993); call Tom (0.001); call Dan (0.001)
- **Final decision:** **answer** → “call Priya”  ·  success=True, wrong=False, reward=+1.0, cost=0.20, **utility=+0.80**
- **Automatic flags:** F6 initial near-tie between two top tasks (winner chosen by hash jitter)
- **Human assessment:** *(blank, see `review_checklist.csv`, row D09)*

---

## D10 · `p1_intent_07` · intent · clean dB · never_ask · split=train

- **Episode key:** `never_ask|uc41613fa-d` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “book Paris for June”
- **Intended meaning (hidden gold):** book a hotel in Paris for June  ·  acceptable set: `[{"intent": "book_hotel", "slots": {"destination": "Paris", "month": "June"}}]`
- **Initial audio:** [audio/D10_initial__uc41613fa-d.wav](audio/D10_initial__uc41613fa-d.wav)  
  <audio controls src="audio/D10_initial__uc41613fa-d.wav"></audio>
- **ASR of original (agent saw):** “Book Paris for June”  ·  blind SNR bucket est.: clean
- **Initial belief top-5:** book a flight to Paris for June (0.333); book a hotel in Paris for June (0.333); book a rental car in Paris for June (0.333); book 1 ticket to the jazz festival (0.000); book a table for 4 at 8 pm today (0.000)  ·  max-prob 0.3333, entropy 1.0987
- **Initial decision:** **answer**
- **Final decision:** **answer** → “book a flight to Paris for June”  ·  success=False, wrong=True, reward=-1.0, cost=0.00, **utility=-1.00**
- **Automatic flags:** F6 initial near-tie between two top tasks (winner chosen by hash jitter)
- **Human assessment:** *(blank, see `review_checklist.csv`, row D10)*

---

## D11 · `p1_none_07` · none · 10 dB · never_ask · split=train

- **Episode key:** `never_ask|u259d5ac3-b` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “call my sister”
- **Intended meaning (hidden gold):** call my sister  ·  acceptable set: `[{"intent": "call_contact", "slots": {"contact": "my sister"}}]`
- **Initial audio:** [audio/D11_initial__u259d5ac3-b.wav](audio/D11_initial__u259d5ac3-b.wav)  
  <audio controls src="audio/D11_initial__u259d5ac3-b.wav"></audio>
- **ASR of original (agent saw):** “Call my sister.”  ·  blind SNR bucket est.: 10
- **Initial belief top-5:** call my sister (0.986); text Sam saying call me (0.001); text Jen saying call me (0.001); text Jaden saying call me (0.001); text Jenna saying call me (0.001)  ·  max-prob 0.986, entropy 0.1143
- **Initial decision:** **answer**
- **Final decision:** **answer** → “call my sister”  ·  success=True, wrong=False, reward=+1.0, cost=0.00, **utility=+1.00**
- **Automatic flags:** none
- **Human assessment:** *(blank, see `review_checklist.csv`, row D11)*

---

## D12 · `p1_none_03` · none · 20 dB · always_ask · split=train

- **Episode key:** `always_ask|ucc9d9795-a` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “what's the weather in Boston tomorrow”
- **Intended meaning (hidden gold):** get the weather in Boston for tomorrow  ·  acceptable set: `[{"intent": "get_forecast", "slots": {"city": "Boston", "date": "tomorrow"}}]`
- **Initial audio:** [audio/D12_initial__ucc9d9795-a.wav](audio/D12_initial__ucc9d9795-a.wav)  
  <audio controls src="audio/D12_initial__ucc9d9795-a.wav"></audio>
- **ASR of original (agent saw):** “What's the weather in Boston tomorrow?”  ·  blind SNR bucket est.: 20
- **Initial belief top-5:** get the weather in Boston for tomorrow (0.997); get the weather in Austin for tomorrow (0.001); get the weather in Chicago for tomorrow (0.001); get the weather in Denver for tomorrow (0.001); get the weather in Seattle for tomorrow (0.001)  ·  max-prob 0.9973, entropy 0.0231
- **Initial decision:** **ask_closed**
- **Clarification question (closed):** “Which do you mean: Boston, Austin, Chicago, or something else?”  (variable: city; options: ['Boston', 'Austin', 'Chicago'])
- **Intended simulated reply:** “Boston” (type `select`)  ·  relevance rule: simulator answers with the offered option equal to its hidden value, else 'None of those.'
- **Reply audio:** [audio/D12_reply__rf9953a9794.wav](audio/D12_reply__rf9953a9794.wav)  
  <audio controls src="audio/D12_reply__rf9953a9794.wav"></audio>
- **ASR of reply (agent saw):** “Boston.”  ·  parsed outcome: 0
- **Updated belief top-3:** get the weather in Boston for tomorrow (1.000); get the weather in Boston for today (0.001); get the weather in Austin for tomorrow (0.000)
- **Final decision:** **answer** → “get the weather in Boston for tomorrow”  ·  success=True, wrong=False, reward=+1.0, cost=0.20, **utility=+0.80**
- **Automatic flags:** F10 clarification asked although initial MAP task was already acceptable
- **Human assessment:** *(blank, see `review_checklist.csv`, row D12)*

---

## D13 · `p1_none_04` · none · 20 dB · noise_indep_voi · split=train

- **Episode key:** `noise_indep_voi|ue5b18d4f-a` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “play the next song”
- **Intended meaning (hidden gold):** skip to the next track  ·  acceptable set: `[{"intent": "next_track", "slots": {}}]`
- **Initial audio:** [audio/D13_initial__ue5b18d4f-a.wav](audio/D13_initial__ue5b18d4f-a.wav)  
  <audio controls src="audio/D13_initial__ue5b18d4f-a.wav"></audio>
- **ASR of original (agent saw):** “Play the next song.”  ·  blind SNR bucket est.: 20
- **Initial belief top-5:** skip to the next track (0.653); play the song Thriller (0.337); go back to the previous track (0.001); play music by Beck (0.001); play music by Bach (0.001)  ·  max-prob 0.6526, entropy 0.7274
- **Expected utilities considered:** `{"abstain": 0.0, "answer": 0.3052, "ask_closed:intent": 0.699, "ask_closed:recipient": 0.1052, "ask_closed:title": 0.6354, "ask_confirm": 0.8331, "ask_open": 0.5591}`
- **Initial decision:** **ask_confirm**
- **Clarification question (confirm):** “Do you want me to skip to the next track?”
- **Intended simulated reply:** “Yes.” (type `yes`)  ·  relevance rule: simulator says Yes only if the proposal is acceptable, else No (bare or corrective)
- **Reply audio:** [audio/D13_reply__r8ed73455ca.wav](audio/D13_reply__r8ed73455ca.wav)  
  <audio controls src="audio/D13_reply__r8ed73455ca.wav"></audio>
- **ASR of reply (agent saw):** “Yes.”  ·  parsed outcome: 0
- **Updated belief top-3:** skip to the next track (0.972); play the song Thriller (0.027); go back to the previous track (0.000)
- **Final decision:** **answer** → “skip to the next track”  ·  success=True, wrong=False, reward=+1.0, cost=0.10, **utility=+0.90**
- **Automatic flags:** F10 clarification asked although initial MAP task was already acceptable
- **Human assessment:** *(blank, see `review_checklist.csv`, row D13)*

---

## D14 · `p1_none_06` · none · 5 dB · confidence_threshold · split=train

- **Episode key:** `confidence_threshold|u5a8e443a-d` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “add eggs to my grocery list”
- **Intended meaning (hidden gold):** add eggs to your grocery list  ·  acceptable set: `[{"intent": "add_item", "slots": {"item": "eggs", "list": "grocery"}}]`
- **Initial audio:** [audio/D14_initial__u5a8e443a-d.wav](audio/D14_initial__u5a8e443a-d.wav)  
  <audio controls src="audio/D14_initial__u5a8e443a-d.wav"></audio>
- **ASR of original (agent saw):** “Add eggs to my grocery list.”  ·  blind SNR bucket est.: 5
- **Initial belief top-5:** add eggs to your grocery list (0.996); add peas to your grocery list (0.001); add milk to your grocery list (0.001); add pears to your grocery list (0.001); add beers to your grocery list (0.001)  ·  max-prob 0.9961, entropy 0.0338
- **Initial decision:** **answer**
- **Final decision:** **answer** → “add eggs to your grocery list”  ·  success=True, wrong=False, reward=+1.0, cost=0.00, **utility=+1.00**
- **Automatic flags:** none
- **Human assessment:** *(blank, see `review_checklist.csv`, row D14)*

---

## D15 · `p1_none_10` · none · 5 dB · evpi · split=val

- **Episode key:** `evpi|u71fbfd32-d` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “remind me to take my medicine at nine PM”
- **Intended meaning (hidden gold):** set a reminder for take my medicine at 9 pm  ·  acceptable set: `[{"intent": "create_reminder", "slots": {"task": "take my medicine", "time": "21:00"}}]`
- **Initial audio:** [audio/D15_initial__u71fbfd32-d.wav](audio/D15_initial__u71fbfd32-d.wav)  
  <audio controls src="audio/D15_initial__u71fbfd32-d.wav"></audio>
- **ASR of original (agent saw):** “Remind me to take my medicine at 9 p.m.”  ·  blind SNR bucket est.: 5
- **Initial belief top-5:** set a reminder for take my medicine at 9 pm (0.500); set a reminder for take my medicine at 9 am (0.500); set a reminder for take my medicine at 6 pm (0.000); set a reminder for buy flowers at 9 pm (0.000); set a reminder for buy flax at 9 pm (0.000)  ·  max-prob 0.4998, entropy 0.6963
- **Expected utilities considered:** `{"abstain": 0.0, "answer": -0.0004, "ask_closed:intent": -0.2, "ask_closed:task": -0.2, "ask_closed:time": 0.8, "ask_confirm": 0.8992, "ask_open": 0.7}`
- **Initial decision:** **ask_confirm**
- **Clarification question (confirm):** “Do you want me to set a reminder for take my medicine at 9 pm?”
- **Intended simulated reply:** “Yes.” (type `yes`)  ·  relevance rule: simulator says Yes only if the proposal is acceptable, else No (bare or corrective)
- **Reply audio:** [audio/D15_reply__r0ed19644a7.wav](audio/D15_reply__r0ed19644a7.wav)  
  <audio controls src="audio/D15_reply__r0ed19644a7.wav"></audio>
- **ASR of reply (agent saw):** “Yes.”  ·  parsed outcome: 0
- **Updated belief top-3:** set a reminder for take my medicine at 9 pm (1.000); set a reminder for take my medicine at 9 am (0.000); set a reminder for take my medicine at 6 pm (0.000)
- **Final decision:** **answer** → “set a reminder for take my medicine at 9 pm”  ·  success=True, wrong=False, reward=+1.0, cost=0.10, **utility=+0.90**
- **Automatic flags:** F6 initial near-tie between two top tasks (winner chosen by hash jitter); F10 clarification asked although initial MAP task was already acceptable
- **Human assessment:** *(blank, see `review_checklist.csv`, row D15)*

---

## D16 · `p1_underspec_01` · underspecified · 10 dB · evpi · split=train

- **Episode key:** `evpi|u32a3480f-b` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “set an alarm”
- **Intended meaning (hidden gold):** set an alarm for 7 am  ·  acceptable set: `[{"intent": "set_alarm", "slots": {"time": "07:00"}}]`
- **Initial audio:** [audio/D16_initial__u32a3480f-b.wav](audio/D16_initial__u32a3480f-b.wav)  
  <audio controls src="audio/D16_initial__u32a3480f-b.wav"></audio>
- **ASR of original (agent saw):** “Set an alarm.”  ·  blind SNR bucket est.: 10
- **Initial belief top-5:** set an alarm for 8 am (0.196); set an alarm for 7:30 am (0.196); set an alarm for 7 am (0.196); set an alarm for 6:30 am (0.196); set an alarm for 6 am (0.196)  ·  max-prob 0.1964, entropy 1.7335
- **Expected utilities considered:** `{"abstain": 0.0, "answer": -0.6072, "ask_closed:intent": -0.1968, "ask_closed:task": -0.2, "ask_closed:time": 0.3892, "ask_confirm": 0.0964, "ask_open": 0.6887}`
- **Initial decision:** **ask_open**
- **Clarification question (open):** “Could you tell me more precisely what you want?”
- **Intended simulated reply:** “Seven in the morning” (type `answer`)  ·  relevance rule: generic open question: simulator restates the unresolved information
- **Reply audio:** [audio/D16_reply__r8c4a60d1b1.wav](audio/D16_reply__r8c4a60d1b1.wav)  
  <audio controls src="audio/D16_reply__r8c4a60d1b1.wav"></audio>
- **ASR of reply (agent saw):** “7 in the morning.”  ·  parsed outcome: None
- **Updated belief top-3:** set an alarm for 8 am (0.196); set an alarm for 7:30 am (0.196); set an alarm for 7 am (0.196)
- **Final decision:** **abstain**  ·  success=False, wrong=False, reward=+0.0, cost=0.30, **utility=-0.30**
- **Automatic flags:** F6 initial near-tie between two top tasks (winner chosen by hash jitter); F2 reply not usable by the agent (parsed outcome None; belief not updated)
- **Human assessment:** *(blank, see `review_checklist.csv`, row D16)*

---

## D17 · `p1_underspec_03` · underspecified · 10 dB · evpi · split=train

- **Episode key:** `evpi|u7bb0180c-d` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “play some music”
- **Intended meaning (hidden gold):** play some jazz  ·  acceptable set: `[{"intent": "play_genre", "slots": {"genre": "*"}}]`
- **Initial audio:** [audio/D17_initial__u7bb0180c-d.wav](audio/D17_initial__u7bb0180c-d.wav)  
  <audio controls src="audio/D17_initial__u7bb0180c-d.wav"></audio>
- **ASR of original (agent saw):** “Play some music.”  ·  blind SNR bucket est.: 10
- **Initial belief top-5:** play some rock (0.146); play some classical (0.146); play some jazz (0.146); play music by Beck (0.140); play the music video for Thriller (0.140)  ·  max-prob 0.1457, entropy 1.9637
- **Expected utilities considered:** `{"abstain": 0.0, "answer": -0.7087, "ask_closed:artist": 0.2203, "ask_closed:genre": 0.237, "ask_closed:intent": -0.0599, "ask_confirm": 0.0457, "ask_open": 0.6981}`
- **Initial decision:** **ask_open**
- **Clarification question (open):** “Could you tell me more precisely what you want?”
- **Intended simulated reply:** “Jazz” (type `answer`)  ·  relevance rule: generic open question: simulator restates the unresolved information
- **Reply audio:** [audio/D17_reply__r1341226200.wav](audio/D17_reply__r1341226200.wav)  
  <audio controls src="audio/D17_reply__r1341226200.wav"></audio>
- **ASR of reply (agent saw):** “Jazz.”  ·  parsed outcome: None
- **Updated belief top-3:** play some rock (0.146); play some classical (0.146); play some jazz (0.146)
- **Final decision:** **abstain**  ·  success=False, wrong=False, reward=+0.0, cost=0.30, **utility=-0.30**
- **Automatic flags:** F6 initial near-tie between two top tasks (winner chosen by hash jitter); F2 reply not usable by the agent (parsed outcome None; belief not updated); F10 clarification asked although initial MAP task was already acceptable
- **Human assessment:** *(blank, see `review_checklist.csv`, row D17)*

---

## D18 · `p1_underspec_08` · underspecified · 10 dB · noise_indep_voi · split=val

- **Episode key:** `noise_indep_voi|u89655f2b-c` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “cancel it”
- **Intended meaning (hidden gold):** cancel the dentist appointment  ·  acceptable set: `[{"intent": "cancel_appointment", "slots": {"appointment": "dentist appointment"}}]`
- **Initial audio:** [audio/D18_initial__u89655f2b-c.wav](audio/D18_initial__u89655f2b-c.wav)  
  <audio controls src="audio/D18_initial__u89655f2b-c.wav"></audio>
- **ASR of original (agent saw):** “cancel it”  ·  blind SNR bucket est.: 10
- **Initial belief top-5:** cancel the haircut (0.039); cancel the gym class (0.039); cancel your Friday restaurant reservation (0.039); cancel your Friday shopping order (0.039); cancel the dentist appointment (0.039)  ·  max-prob 0.0389, entropy 4.9381
- **Expected utilities considered:** `{"abstain": 0.0, "answer": -0.9222, "ask_closed:date": -0.2, "ask_closed:intent": -0.2, "ask_closed:recipient": -0.2, "ask_confirm": -0.1, "ask_open": -0.2044}`
- **Initial decision:** **abstain**
- **Final decision:** **abstain**  ·  success=False, wrong=False, reward=+0.0, cost=0.00, **utility=+0.00**
- **Automatic flags:** F6 initial near-tie between two top tasks (winner chosen by hash jitter)
- **Human assessment:** *(blank, see `review_checklist.csv`, row D18)*

---

## D19 · `p1_underspec_06` · underspecified · 5 dB · confidence_threshold · split=train

- **Episode key:** `confidence_threshold|u4136e82e-d` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “turn up the volume”
- **Intended meaning (hidden gold):** turn the volume up on the living room speaker  ·  acceptable set: `[{"intent": "adjust_volume", "slots": {"device": "living room speaker", "direction": "up"}}]`
- **Initial audio:** [audio/D19_initial__u4136e82e-d.wav](audio/D19_initial__u4136e82e-d.wav)  
  <audio controls src="audio/D19_initial__u4136e82e-d.wav"></audio>
- **ASR of original (agent saw):** “Turn up the volume.”  ·  blind SNR bucket est.: 5
- **Initial belief top-5:** turn the volume up on the TV (0.333); turn the volume up on the living room speaker (0.333); turn the volume up on the phone (0.333); set the volume to 15 percent (0.000); set the living room speaker volume to 70 percent (0.000)  ·  max-prob 0.3333, entropy 1.0988
- **Initial decision:** **ask_closed**
- **Clarification question (closed):** “Which do you mean: TV, living room speaker, phone, or something else?”  (variable: device; options: ['TV', 'living room speaker', 'phone'])
- **Intended simulated reply:** “Living room speaker” (type `select`)  ·  relevance rule: simulator answers with the offered option equal to its hidden value, else 'None of those.'
- **Reply audio:** [audio/D19_reply__r328b3ee848.wav](audio/D19_reply__r328b3ee848.wav)  
  <audio controls src="audio/D19_reply__r328b3ee848.wav"></audio>
- **ASR of reply (agent saw):** “Living Room Speaker.”  ·  parsed outcome: 1
- **Updated belief top-3:** turn the volume up on the living room speaker (1.000); turn the volume up on the TV (0.000); turn the volume up on the phone (0.000)
- **Final decision:** **answer** → “turn the volume up on the living room speaker”  ·  success=True, wrong=False, reward=+1.0, cost=0.20, **utility=+0.80**
- **Automatic flags:** F6 initial near-tie between two top tasks (winner chosen by hash jitter)
- **Human assessment:** *(blank, see `review_checklist.csv`, row D19)*

---

## D20 · `p1_underspec_09` · underspecified · clean dB · always_ask · split=val

- **Episode key:** `always_ask|u38f57003-b` (in `data/phase5/dev.v1/episodes.jsonl`)
- **Original utterance (reference text):** “get me a ticket”
- **Intended meaning (hidden gold):** book 1 ticket to the Coldplay concert  ·  acceptable set: `[{"intent": "book_ticket", "slots": {"event": "Coldplay concert", "quantity": 1}}]`
- **Initial audio:** [audio/D20_initial__u38f57003-b.wav](audio/D20_initial__u38f57003-b.wav)  
  <audio controls src="audio/D20_initial__u38f57003-b.wav"></audio>
- **ASR of original (agent saw):** “Get me a ticket.”  ·  blind SNR bucket est.: clean
- **Initial belief top-5:** book 1 ticket to the comedy show (0.127); book 1 ticket to the jazz festival (0.127); book 1 ticket to the Coldplay concert (0.127); buy tickets to the Giants game today (0.050); get the weather in Porto (0.026)  ·  max-prob 0.127, entropy 3.4474
- **Initial decision:** **ask_closed**
- **Clarification question (closed):** “Which do you mean: book ticket, get the weather, set a reminder, or something else?”  (variable: intent; options: ['book ticket', 'get the weather', 'set a reminder'])
- **Intended simulated reply:** “Book ticket” (type `select`)  ·  relevance rule: simulator answers with the offered option equal to its hidden value, else 'None of those.'
- **Reply audio:** [audio/D20_reply__r97129b188a.wav](audio/D20_reply__r97129b188a.wav)  
  <audio controls src="audio/D20_reply__r97129b188a.wav"></audio>
- **ASR of reply (agent saw):** “Book Ticket”  ·  parsed outcome: 0
- **Updated belief top-3:** book 1 ticket to the comedy show (0.333); book 1 ticket to the jazz festival (0.333); book 1 ticket to the Coldplay concert (0.333)
- **Final decision:** **answer** → “book 1 ticket to the comedy show”  ·  success=False, wrong=True, reward=-1.0, cost=0.20, **utility=-1.20**
- **Automatic flags:** F6 initial near-tie between two top tasks (winner chosen by hash jitter); F9 final answer wrong although the reply was parsed as intended
- **Human assessment:** *(blank, see `review_checklist.csv`, row D20)*
