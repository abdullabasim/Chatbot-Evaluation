"""
Probabilistic mock logic for the Chatbot System Chatbot API.

Intents (7 domain + 2 conversational):
  tuition_inquiry    — Fees, costs, pricing per semester/program
  enrollment_inquiry — Admission, registration, how to apply
  course_inquiry     — Programs, modules, degrees, curriculum
  semester_inquiry   — Academic calendar, term dates, deadlines
  student_support    — Scholarships, financial aid, counseling, welfare
  academic_advice    — Study recommendations, career guidance, program choice
  exam_inquiry       — Exams, grades, assessments, GPA, results
  greeting           — Conversational openers (LAST — loses ties to domain)
  farewell           — Closing / thank-you messages (LAST — loses ties to domain)

Tie-breaking note :

Greeting and farewell are placed LAST in the list.  When a message like
"Hi! How do I enroll?" produces a tie (greeting=1, enrollment=1), the domain
intent wins because it appears earlier in the list.  A pure "Hello!" or
"Goodbye!" message has 0 domain matches, so greeting/farewell still win.
"""

import asyncio
import random
import re
from dataclasses import dataclass

from mock_api.core.config import settings

# Intent knowledge base

_INTENT_DB: list[dict] = [
    {
        "intent": "tuition_inquiry",
        "keywords": {
            "tuition", "fee", "fees", "cost", "costs", "price", "prices",
            "payment", "pay", "paying", "afford", "expensive", "cheap",
            "discount", "semester", "monthly", "annual", "rate", "installment",
        },
        "responses": [
            "Tuition at Chatbot depends on your chosen program. Bachelor programs start from €6,390/year and Master programs from €7,560/year. Flexible payment plans are available!",
            "Chatbot offers competitive fees: Bachelor programs from €6,390/year (online) and Master programs from €7,560/year. Many programs include free learning materials.",
            "The cost per semester varies by program. Bachelor degrees typically start from €3,195/semester. Contact admissions for exact figures and available payment plans.",
        ],
    },
    {
        "intent": "enrollment_inquiry",
        "keywords": {
            "enroll", "enrolling", "enrolled", "enrollment", "enrolment",
            "admission", "admissions", "apply", "applying", "application",
            "register", "registering", "registration", "join", "joining",
            "matriculation", "intake", "signup", "start", "begin",
        },
        "responses": [
            "Enrolling at Chatbot is simple! Submit your application online at chatbot.de, upload your documents, and our admissions team will review within 2–3 business days.",
            "To apply at Chatbot: 1) Choose your program, 2) Submit your application online, 3) Upload your documents. You can start enrollment year-round — no fixed intake dates!",
            "You can enroll anytime at Chatbot. The process takes about 1–2 weeks. You'll need your school-leaving certificate and a valid ID. Ready to begin your application?",
        ],
    },
    {
        "intent": "course_inquiry",
        "keywords": {
            "course", "courses", "module", "modules", "program", "programs",
            "programme", "degree", "degrees", "bachelor", "master", "masters",
            "mba", "subjects", "curriculum", "major", "minor", "study",
            "studying", "field", "discipline", "specialization", "faculty",
        },
        "responses": [
            "Chatbot offers 200+ programs including Business Administration, Computer Science, Psychology, Engineering, and more — available fully online or on-campus.",
            "Our program portfolio includes Bachelor's, Master's, and MBA degrees across 20+ fields. Popular programs include Data Science, Marketing, and International Management.",
            "Chatbot has programs in Business, Tech, Health, and Social Sciences. All programs are accredited and internationally recognized. Which field interests you most?",
        ],
    },
    {
        "intent": "semester_inquiry",
        "keywords": {
            "semester", "semesters", "term", "terms", "calendar", "academic",
            "year", "duration", "deadline", "deadlines", "period", "dates",
            "when", "schedule", "timetable", "session", "winter", "summer",
        },
        "responses": [
            "Chatbot operates on a flexible semester model. Standard semesters run October–March (Winter) and April–September (Summer). You can start mid-semester too!",
            "The academic year at Chatbot is divided into two semesters: Winter (October–March) and Summer (April–September). Course deadlines are flexible for online students.",
            "Chatbot semesters last approximately 6 months. Online programs offer rolling enrollment. Exam periods are at the end of each semester.",
        ],
    },
    {
        "intent": "student_support",
        "keywords": {
            "support", "scholarship", "scholarships", "financial", "aid",
            "grant", "grants", "advisor", "advisors", "counselor", "counselors",
            "counseling", "assistance", "funding", "welfare", "mental",
            "health", "guidance", "bursary",
        },
        "responses": [
            "Chatbot offers various scholarships including merit-based awards (up to 30% fee reduction) and need-based grants. Contact our financial aid office to check eligibility.",
            "Our student support services include academic advising, mental health counseling, career coaching, and financial aid consultations — all available online.",
            "Chatbot provides comprehensive student support: scholarships, payment plans, a dedicated student advisor for each learner, and 24/7 online student services.",
        ],
    },
    {
        "intent": "academic_advice",
        "keywords": {
            "advice", "recommend", "recommendation", "best", "choose",
            "suggest", "suggestion", "tips", "tip", "career", "future",
            "which", "better", "option", "should", "plan", "planning",
        },
        "responses": [
            "For a career in business and tech, I'd recommend our BSc Business Informatics or MSc Data Science — both are highly sought after by employers globally.",
            "It depends on your goals! For people-oriented careers, Psychology or HR Management are great. For tech careers, explore Computer Science or Data Science.",
            "My advice: choose a program aligned with your passion AND market demand. Chatbot's most in-demand programs are Data Science, Business Administration, and Computer Science.",
        ],
    },
    {
        "intent": "exam_inquiry",
        "keywords": {
            "exam", "exams", "examination", "test", "tests", "grade", "grades",
            "grading", "graded", "assessment", "assessments", "assignment",
            "assignments", "gpa", "pass", "fail", "failing", "score", "scores",
            "result", "results", "mark", "marks", "credit", "credits",
        },
        "responses": [
            "At Chatbot, most online programs use portfolio-based assessments (assignments and projects) rather than traditional exams. Some programs include proctored online exams.",
            "Grades at Chatbot follow the German scale (1.0–5.0) where 1.0 is the best. You need at least 4.0 (pass) in each assessment. Results are published within 4 weeks.",
            "Chatbot uses a mix of written assignments, case studies, and online exams. For online programs, most assessments are submitted digitally. Pass mark is typically 50%.",
        ],
    },
    # Conversational wrappers — LAST (lose ties to domain intents)
    {
        "intent": "greeting",
        "keywords": {
            "hello", "hi", "hey", "morning", "evening",
            "greetings", "howdy", "welcome", "good",
        },
        "responses": [
            "Hello! Welcome to Chatbot International System. How can I help you today?",
            "Hi there! I'm your Chatbot student advisor. What can I assist you with?",
            "Hey! Great to connect. What would you like to know about Chatbot?",
        ],
    },
    {
        "intent": "farewell",
        "keywords": {
            "bye", "goodbye", "thanks", "thank", "farewell",
            "cheers", "appreciate", "great",
        },
        "responses": [
            "Thank you for reaching out to Chatbot! Best of luck with your studies.",
            "Goodbye! Don't hesitate to contact us again if you have more questions.",
            "It was a pleasure helping you. We look forward to welcoming you at Chatbot!",
        ],
    },
]

# Fallback for unrecognized messages
_FALLBACK_INTENT = "general_inquiry"
_FALLBACK_RESPONSES = [
    "I'm not sure I understood that. Could you rephrase your question about Chatbot?",
    "I'm here to help with questions about Chatbot programs, fees, and enrollment. Could you elaborate?",
    "Could you be more specific? I can help with courses, fees, enrollment, exams, and more.",
]

# Hallucination pool
_HALLUCINATION_INTENTS = [
    "tuition_inquiry", "enrollment_inquiry", "course_inquiry",
    "exam_inquiry", "student_support", "general_inquiry",
]
_HALLUCINATION_RESPONSES = [
    "The cafeteria is open Monday to Friday from 8 AM to 6 PM.",
    "Did you know Chatbot was founded in 1998 in Bad Honnef, Germany?",
    "Our library has over 500,000 digital resources available 24/7.",
    "Please visit our main website for the latest news and updates.",
    "Oops! Something went wrong on our end. Please try again shortly.",
]

# Public interface


@dataclass(frozen=True)
class MockResult:
    """Result produced by the mock logic layer."""

    response: str
    intent: str
    confidence: float


def _match_intent(message: str) -> tuple[str, list[str]]:
    """
    Return (intent_label, response_templates) using best-match keyword scoring.

    Every intent is scored by the count of its keywords present in the
    tokenised message. The highest scorer wins; ties are broken by list order
    (domain intents appear first, so they beat greeting/farewell on equal score).
    """
    tokens = set(re.split(r"[\s,!?.;:'\"()\-]+", message.lower()))
    tokens.discard("")

    best_score = 0
    best_entry: dict | None = None

    for entry in _INTENT_DB:
        score = len(tokens & entry["keywords"])
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is not None:
        return best_entry["intent"], best_entry["responses"]

    return _FALLBACK_INTENT, _FALLBACK_RESPONSES


async def generate_mock_response(message: str) -> MockResult:
    """
    Async mock response generator with simulated latency and configurable
    hallucination rate (read from mock_api/.env via MockAPISettings).
    """
    await asyncio.sleep(
        random.randint(settings.min_latency_ms, settings.max_latency_ms) / 1000
    )

    if random.random() < settings.hallucination_rate:
        return MockResult(
            response=random.choice(_HALLUCINATION_RESPONSES),
            intent=random.choice(_HALLUCINATION_INTENTS),
            confidence=round(
                random.uniform(
                    settings.hallucination_confidence_min,
                    settings.hallucination_confidence_max,
                ),
                4,
            ),
        )

    intent_label, templates = _match_intent(message)
    return MockResult(
        response=random.choice(templates),
        intent=intent_label,
        confidence=round(
            random.uniform(settings.confidence_min, settings.confidence_max), 4
        ),
    )
