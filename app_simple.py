from flask import Flask, render_template, request, jsonify, session
import uuid
import json
import os
from datetime import datetime
import random

app = Flask(__name__)
app.secret_key = os.urandom(24)

# --- Configuration ---
LOG_FILE = os.path.join(os.path.dirname(__file__), 'chat_logs.jsonl')
EMOTIONS = ['Happy', 'Sad', 'Angry', 'Surprised', 'Afraid', 'Disgusted', 'Neutral', 'Other']
CHARACTERS = ['Detective', 'Alien', 'Teacher', 'Robot', 'Pirate', 'Doctor', 'Wizard']

# ---- Emotion Opposites ---- #
emotion_opposites = {
    "admiration": "humility", "amusement": "seriousness", "anger": "calm",
    "annoyance": "ease", "approval": "detachment", "caring": "indifference",
    "confusion": "clarity", "curiosity": "certainty", "desire": "satisfaction",
    "disappointment": "hope", "disapproval": "acceptance", "embarrassment": "confidence",
    "excitement": "composure", "fear": "reassurance", "gratitude": "entitlement",
    "grief": "comfort", "joy": "reflection", "love": "detachment",
    "nervousness": "confidence", "neutral": "engagement", "optimism": "skepticism",
    "pride": "modesty", "realization": "uncertainty", "relief": "tension",
    "remorse": "forgiveness", "sadness": "hope", "surprise": "predictability"
}

# Simple emotion detection based on keywords
def simple_emotion_detection(text):
    """Simple emotion detection using keyword matching."""
    text_lower = text.lower()
    
    emotion_keywords = {
        "joy": ["happy", "joy", "excited", "great", "wonderful", "amazing", "love", "loved"],
        "anger": ["angry", "mad", "furious", "hate", "hated", "terrible", "awful", "horrible"],
        "sadness": ["sad", "depressed", "lonely", "miserable", "unhappy", "crying", "tears"],
        "fear": ["afraid", "scared", "fear", "terrified", "worried", "anxious", "nervous"],
        "surprise": ["surprised", "shocked", "amazed", "wow", "unexpected", "incredible"],
        "disgust": ["disgusting", "gross", "nasty", "revolting", "sickening"]
    }
    
    for emotion, keywords in emotion_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            return emotion, [{"label": emotion, "score": 0.8}]
    
    return "neutral", [{"label": "neutral", "score": 0.8}]

def get_opposite_emotion(emotion):
    return emotion_opposites.get(emotion.lower(), "neutral")

def get_fallback_response(character, target_emotion, user_message):
    """Get a simple fallback response based on character and target emotion."""
    fallback_responses = {
        "Detective": {
            "calm": "I see what's happening here. Let's approach this systematically.",
            "hope": "Every case has a solution. We'll figure this out together.",
            "confidence": "I've seen cases like this before. We can handle this.",
            "engagement": "Tell me more about what's going on. I'm listening."
        },
        "Teacher": {
            "calm": "I understand this is challenging. Let's take a deep breath and work through it.",
            "hope": "Every problem has a solution. We'll find the right approach.",
            "confidence": "You have the skills to handle this. Let's break it down.",
            "engagement": "I'm here to help you learn and grow through this."
        },
        "Robot": {
            "calm": "Processing your input. Analyzing optimal response patterns.",
            "hope": "Calculating positive outcomes. Probability of success: high.",
            "confidence": "My circuits are designed to help. Let's solve this together.",
            "engagement": "I am fully operational and ready to assist you."
        },
        "Pirate": {
            "calm": "Aye, matey! Let's navigate these rough waters together.",
            "hope": "There's treasure in every challenge, me hearty!",
            "confidence": "With the right crew, we can weather any storm!",
            "engagement": "Tell me your tale, and I'll help you find your way."
        },
        "Doctor": {
            "calm": "I understand this is stressful. Let's address this step by step.",
            "hope": "There's always a path to improvement. We'll find it together.",
            "confidence": "You're stronger than you think. Let's work through this.",
            "engagement": "I'm here to help you heal and grow through this."
        },
        "Wizard": {
            "calm": "The ancient spells of wisdom can guide us through this.",
            "hope": "Magic exists in every challenge. We'll find the right spell.",
            "confidence": "Your inner magic is powerful. Let's channel it together.",
            "engagement": "Share your story, and I'll help you find your magical path."
        },
        "Alien": {
            "calm": "From my observations, this situation requires careful analysis.",
            "hope": "My species believes in infinite possibilities. Solutions exist.",
            "confidence": "Your human resilience is remarkable. We can overcome this.",
            "engagement": "I am fascinated by your experience. Please continue."
        }
    }
    
    # Get character-specific responses
    char_responses = fallback_responses.get(character, fallback_responses["Detective"])
    
    # Get response for target emotion, or default to engagement
    response = char_responses.get(target_emotion, char_responses["engagement"])
    
    return response

def log_message(role, message, detected_emotion=None, self_reported_emotion=None, target_emotion=None, emotion_scores=None):
    """Appends a message to the log file with emotion information."""
    try:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'session_id': session.get('session_id'),
            'character': session.get('character'),
            'role': role,
            'message': message,
            'detected_emotion': detected_emotion,
            'self_reported_emotion': self_reported_emotion,
            'target_emotion': target_emotion,
            'emotion_scores': emotion_scores
        }
        
        print(f"DEBUG: Logging message - Role: {role}, Message: {message[:50]}...")
        print(f"DEBUG: Session ID: {session.get('session_id')}, Character: {session.get('character')}")
        
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"DEBUG: Successfully logged to {LOG_FILE}")
        
    except Exception as e:
        print(f"ERROR: Failed to log message: {e}")
        print(f"ERROR: Log file path: {LOG_FILE}")
        import traceback
        traceback.print_exc()

def agent_response(user_message, detected_emotion, self_reported_emotion, character):
    """Generate agent response using simple emotion detection and fallback responses."""
    # Get the opposite emotion to target
    target_emotion = get_opposite_emotion(detected_emotion)
    
    # Use fallback response
    fallback_response = get_fallback_response(character, target_emotion, user_message)
    
    return fallback_response, target_emotion

# --- Routes ---
@app.route('/')
def index():
    """Serves the main chat page."""
    # Initialize session if it's the user's first visit
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['history'] = []
    return render_template('index.html', characters=CHARACTERS + ['Random'])

@app.route('/start', methods=['POST'])
def start_chat():
    """Handles character selection and starts the chat session."""
    data = request.json
    selected_character = data.get('character')

    if selected_character == 'Random':
        session['character'] = random.choice(CHARACTERS)
    else:
        session['character'] = selected_character
    
    return jsonify({
        'status': 'success', 
        'character': session['character'],
        'message': 'Chat started.'
    })

@app.route('/send', methods=['POST'])
def send_message():
    data = request.json
    if data is None:
        return jsonify({'error': 'Invalid request data'}), 400
        
    user_message = data.get('message')
    self_reported_emotion = data.get('emotion') or "Neutral"

    if not user_message:
        return jsonify({'error': 'Message cannot be empty'}), 400

    # Initialize conversation buffer if not present
    if 'conversation' not in session:
        session['conversation'] = []

    # Detect emotion from user message using simple keyword detection
    detected_emotion, emotion_scores = simple_emotion_detection(user_message)
    
    # Append user message to conversation buffer
    session['conversation'].append({
        'role': 'user',
        'message': user_message,
        'detected_emotion': detected_emotion,
        'self_reported_emotion': self_reported_emotion,
        'timestamp': datetime.utcnow().isoformat()
    })

    # Generate agent response
    character = session.get('character', 'Assistant')
    agent_msg, target_emotion = agent_response(user_message, detected_emotion, self_reported_emotion, character)

    # Append agent message to conversation buffer
    session['conversation'].append({
        'role': 'agent',
        'message': agent_msg,
        'target_emotion': target_emotion,
        'timestamp': datetime.utcnow().isoformat()
    })

    session.modified = True

    # If user wants to end the session, write the whole conversation as one dict
    if user_message.strip().lower() in ['exit', 'quit']:
        log_entry = {
            'session_id': session.get('session_id'),
            'character': session.get('character'),
            'conversation': session['conversation']
        }
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
        session.pop('conversation', None)  # Clear conversation for new session

    return jsonify({
        'agent_message': agent_msg,
        'detected_emotion': detected_emotion,
        'target_emotion': target_emotion,
        'emotion_scores': emotion_scores
    })

if __name__ == '__main__':
    print("🚀 Starting simplified chatbot (no model loading required)")
    print("📝 Logging to:", LOG_FILE)
    app.run(debug=True, port=5001) 