"""Training data for the chatbot's intent classifier.

A task-oriented assistant must recognise *what* is being asked before it can
answer. We frame that as supervised text classification: (question -> intent).
This module generates a diverse, paraphrase-rich labelled dataset per intent,
plus a separate hand-written **hard** evaluation set (wordings deliberately
unlike the training templates) so generalisation can be measured honestly.

Intents map 1:1 onto the answer handlers in
:class:`assbi.chatbot.assistant.SurveillanceAssistant`.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

# intent -> diverse example phrasings. Deliberately varied vocabulary and
# sentence shapes so a model learns the *meaning*, not a few keywords.
_TEMPLATES: dict[str, list[str]] = {
    "greet": [
        "hi", "hello", "hey there", "good morning", "yo", "hello assistant",
        "hey, you there?", "hi bot", "greetings", "good evening", "howdy",
        "hiya", "morning", "hey buddy", "what's up",
    ],
    "help": [
        "help", "what can you do", "what can you tell me", "how do i use this",
        "what questions can i ask", "show me the options", "what do you know",
        "what can you answer", "guide me", "what are my options",
        "how does this work", "what info do you have", "list what you can report",
        "what are you capable of", "give me some example questions",
    ],
    "summary": [
        "give me a summary", "overview please", "summarise the session",
        "what's the overall report", "tell me everything", "full report",
        "session overview", "general summary", "what happened overall",
        "key numbers please", "what are the kpis", "brief me on this session",
        "recap the whole thing", "high level summary", "sum it up for me",
        "what's the big picture",
    ],
    "people_in": [
        "how many people came in", "how many entered", "count of people in",
        "how many people went in", "number of pedestrians entering",
        "how many came inward", "inbound people count", "people in",
        "how many walked in", "entries count", "how many folks arrived",
        "total entering the area", "how many headed inside",
        "number who walked inward", "how many came through inbound",
    ],
    "people_out": [
        "how many people went out", "how many left", "count of people out",
        "how many people exited", "number of pedestrians leaving",
        "how many went outward", "outbound people count", "people out",
        "how many walked out", "exits count", "how many folks departed",
        "tally of people who left", "headcount that exited",
        "how many headed away", "number who walked out the other side",
        "how many made their way out",
    ],
    "people_total": [
        "how many people crossed", "total pedestrians", "how many people in total",
        "overall footfall", "total crossings of people", "how many people overall",
        "total number of people", "how busy with people", "footfall total",
        "how many humans crossed the line", "combined in and out count",
        "total people both directions", "grand total of crossings",
        "how many crossed altogether", "sum of all pedestrian crossings",
    ],
    "busiest_hour": [
        "which hour was busiest", "what time was the peak", "busiest hour",
        "when were the most crossings", "what was the rush hour",
        "which period had the most people", "peak time of day",
        "what hour had most traffic", "when was it busiest",
        "which minute had most people out", "what time had the most exits",
        "busiest period", "when was the place most packed",
        "what was the busiest moment in time", "which slot saw the most movement",
        "what time of day was heaviest", "when did footfall peak",
    ],
    "hourly": [
        "footfall by hour", "people per hour", "breakdown by hour", "hourly counts",
        "show me each hour", "crossings hour by hour", "give me the hourly breakdown",
        "how many each hour", "per hour numbers", "hourly footfall",
        "numbers grouped by each hour", "split the counts by hour",
        "show the per-hour table", "counts for every hour",
        "how did each hour compare",
    ],
    "last_window": [
        "how many in the last 10 minutes", "people in the last hour",
        "crossings in the last 30 minutes", "how many in the past 15 minutes",
        "last 5 minutes count", "how busy in the last 2 hours",
        "people out in the last 20 minutes", "recent crossings last 10 min",
        "how many entered in the last 45 minutes", "last hour footfall",
        "in the previous twenty minutes how many walked through",
        "what happened in the final 15 minutes", "count over the last half hour",
        "how many crossed in the past hour", "footfall during the last 5 mins",
        "recently how many came through",
    ],
    "peak": [
        "what was the peak crowd", "most people at once", "maximum crowd",
        "busiest moment", "highest number of people in view", "peak density",
        "what's the max crowd", "largest crowd size", "peak number of people",
        "when was the crowd biggest", "how packed did it get at most",
        "biggest crowd recorded", "what was the crowd maximum",
        "most crowded it got", "top crowd size",
    ],
    "anomalies": [
        "were there anomalies", "any unusual activity", "anomaly count",
        "did anything abnormal happen", "were there any surges", "unusual moments",
        "any spikes detected", "how many anomalies", "anything strange",
        "were there abnormal crowd levels", "any weird spikes in the crowd",
        "did the crowd do anything odd", "were there sudden surges or drops",
        "anything out of the ordinary", "did anything unusual get flagged",
    ],
    "forecast": [
        "what's the forecast", "predict the crowd", "what's the trend",
        "is the crowd rising or falling", "future crowd estimate",
        "what comes next", "crowd prediction", "where is footfall heading",
        "trend of the crowd", "forecast the next intervals",
        "where is the crowd heading next", "is it getting busier or quieter",
        "project the crowd going forward", "what will the crowd do next",
        "expected footfall ahead",
    ],
    "confidence": [
        "what's the detection confidence", "how accurate is it",
        "average confidence", "model confidence", "how confident is the model",
        "detection accuracy", "what is the accuracy", "confidence score",
        "how reliable are the detections", "mean confidence",
        "how trustworthy are these detections", "how sure is the detector",
        "what's the average certainty", "how good is the detection quality",
        "confidence level of the model",
    ],
}

# Prefix/suffix fillers to multiply realistic phrasing variety.
_PREFIXES = ["", "", "", "can you tell me ", "i want to know ", "please ", "so ",
             "hey ", "could you tell me ", "just curious ", "quick question ", "tell me "]
_SUFFIXES = ["", "", "", "?", " please", " for this session", " right now",
             " exactly", " in this footage", " overall"]

# Hand-written HARD test set: wordings intentionally unlike the templates above.
# Never used for training — only to report honest generalisation accuracy.
_HARD_EVAL: list[tuple[str, str]] = [
    ("roughly how many bodies walked inside", "people_in"),
    ("give me the number of arrivals", "people_in"),
    ("how many souls left the scene", "people_out"),
    ("count everyone who walked off", "people_out"),
    ("what's the combined pedestrian tally", "people_total"),
    ("all crossings added together", "people_total"),
    ("which stretch of time was the most hectic", "busiest_hour"),
    ("at what point were the most folks leaving", "busiest_hour"),
    ("lay out the counts for each sixty-minute block", "hourly"),
    ("hand me the hour-by-hour figures", "hourly"),
    ("over the past quarter of an hour, how many", "last_window"),
    ("tally for the most recent ten minutes", "last_window"),
    ("what's the largest gathering you saw", "peak"),
    ("top simultaneous head count", "peak"),
    ("flag any odd behaviour in the crowd", "anomalies"),
    ("were there bizarre jumps in numbers", "anomalies"),
    ("how is the crowd expected to evolve", "forecast"),
    ("will it get more crowded soon", "forecast"),
    ("how dependable are the boxes drawn", "confidence"),
    ("what certainty does the detector report", "confidence"),
    ("just give me the whole rundown", "summary"),
    ("walk me through everything that happened", "summary"),
    ("what sorts of things can i ask you", "help"),
    ("good afternoon", "greet"),
]


def generate(seed: int = 0, augment: int = 5) -> list[tuple[str, str]]:
    """Return ``(text, intent)`` training pairs with paraphrase augmentation."""
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for intent, phrases in _TEMPLATES.items():
        for phrase in phrases:
            rows.append((phrase, intent))
            for _ in range(augment):
                pre = rng.choice(_PREFIXES)
                suf = rng.choice(_SUFFIXES)
                rows.append((f"{pre}{phrase}{suf}".strip(), intent))
    seen, unique = set(), []
    for text, intent in rows:
        if (text, intent) not in seen:
            seen.add((text, intent))
            unique.append((text, intent))
    rng.shuffle(unique)
    return unique


def hard_eval() -> list[tuple[str, str]]:
    """Held-out, hand-written generalisation test (never trained on)."""
    return list(_HARD_EVAL)


def intents() -> list[str]:
    return list(_TEMPLATES.keys())


def save_csv(rows: list[tuple[str, str]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "intent"])
        w.writerows(rows)
    return path
