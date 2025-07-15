from flask import Flask, render_template, request, jsonify, session
import uuid
import json
import os
import sys
from datetime import datetime
import random
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer as GenTokenizer
import torch

app = Flask(__name__)
# A secret key is needed for session management
app.secret_key = os.urandom(24)

# Global variables for models
emotion_classifier = None
llama_generator = None
memory_store = {}  # Global memory store for all sessions

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

# ---- Conversation Memory Tracker ---- #
class ConversationMemory:
    def __init__(self, session_id, max_turns=5):
        self.session_id = session_id
        self.history = []
        self.max_turns = max_turns

    def _update_history(self, user_input, assistant_response):
        self.history.append({"user": user_input, "assistant": assistant_response})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_history_prompt(self):
        prompt = ""
        for turn in self.history:
            prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
        return prompt

    def add_turn(self, user_input, assistant_response):
        self._update_history(user_input, assistant_response)

def get_opposite_emotion(emotion):
    return emotion_opposites.get(emotion.lower(), "neutral")

def load_emotion_classifier():
    """Load the emotion classification model."""
    global emotion_classifier
    if emotion_classifier is None:
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True,
            truncation=True
        )
    return emotion_classifier

def classify_emotion(text):
    """Classify emotion in text using the loaded model."""
    classifier = load_emotion_classifier()
    if classifier is None:
        return "neutral", []
    
    try:
        scores = classifier(text)[0]
        # Convert scores to a format that can be sorted
        score_list = [(score['label'], float(score['score'])) for score in scores]
        sorted_scores = sorted(score_list, key=lambda x: x[1], reverse=True)
        top_emotion = sorted_scores[0][0]
        return top_emotion, scores
    except Exception as e:
        print(f"Error classifying emotion: {e}")
        return "neutral", []

def load_llama(model_id="microsoft/DialoGPT-small"):  # Much lighter model
    """Load a lightweight model for text generation."""
    global llama_generator
    if llama_generator is None:
        try:
            print("Loading lightweight model for faster responses...")
            tokenizer = GenTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            llama_generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to a simpler model or mock response
            return None
    return llama_generator

def make_prompt(opposite_emotion, user_msg, memory_prompt="", character="Assistant"):
    """Create an emotion-driven prompt for the model."""
    return f"""User: {user_msg}
Assistant: I understand you're feeling this way. As a {character}, let me respond in a way that might help you feel more {opposite_emotion}."""

def generate_response(generator, prompt):
    """Generate response using the model with fallback."""
    if generator is None:
        # Fallback responses based on emotion
        return "I understand how you're feeling. Let me help you process this."
    
    try:
        output = generator(prompt, max_new_tokens=50, temperature=0.7, do_sample=True, return_full_text=False)
        response = output[0]["generated_text"].strip()
        
        # Clean up the response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response if response else "I hear you. Let's work through this together."
        
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm here to listen and help you through this."

# --- Helper Functions ---
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

def agent_response(user_message, detected_emotion, self_reported_emotion, character):
    """Generate agent response using emotion detection and model/fallback."""
    # Get the opposite emotion to target
    target_emotion = get_opposite_emotion(detected_emotion)
    
    # Get conversation memory
    session_id = session.get('session_id')
    memory = memory_store.get(session_id)

    if memory is None:
        memory = ConversationMemory(session_id)
        memory_store[session_id] = memory
    
    # Try to generate response with model first
    generator = load_llama()
    if generator is not None:
        try:
            # Create prompt with memory
            memory_prompt = memory.get_history_prompt()
            prompt = make_prompt(target_emotion, user_message, memory_prompt, character)
            
            # Generate response
            response = generate_response(generator, prompt)
            
            # Update memory
            memory.add_turn(user_message, response)
            memory_store[session_id] = memory
            
            return response, target_emotion
            
        except Exception as e:
            print(f"Model generation failed, using fallback: {e}")
    
    # Use fallback response if model fails or is not available
    fallback_response = get_fallback_response(character, target_emotion, user_message)
    
    # Update memory with fallback response
    memory.add_turn(user_message, fallback_response)
    memory_store[session_id] = memory
    
    return fallback_response, target_emotion

# --- Routes ---
@app.route('/')
def index():
    """Serves the main chat page."""
    # Initialize session if it's the user's first visit
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['history'] = []

    session_id = session['session_id']
    if session_id not in memory_store:
        memory_store[session_id] = ConversationMemory(session_id)
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
    
    # Reset memory for new chat
    session_id = session.get('session_id')
    memory_store[session_id] = ConversationMemory(session_id)
    
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

    # Detect emotion from user message
    detected_emotion, emotion_scores = classify_emotion(user_message)
    
    # Log user message with emotion information
    log_message('user', user_message, detected_emotion, self_reported_emotion, emotion_scores=emotion_scores)
    session['history'].append({
        'role': 'user', 
        'message': user_message, 
        'detected_emotion': detected_emotion,
        'self_reported_emotion': self_reported_emotion
    })

    # Generate agent response
    character = session.get('character', 'Assistant')
    agent_msg, target_emotion = agent_response(user_message, detected_emotion, self_reported_emotion, character)

    # Log agent response
    log_message('agent', agent_msg, target_emotion=target_emotion)
    session['history'].append({'role': 'agent', 'message': agent_msg})
    session.modified = True

    return jsonify({
        'agent_message': agent_msg,
        'detected_emotion': detected_emotion,
        'target_emotion': target_emotion,
        'emotion_scores': emotion_scores
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001) 