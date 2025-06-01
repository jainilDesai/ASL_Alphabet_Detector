""" import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from textblob import TextBlob
from fuzzywuzzy import fuzz
import time
from collections import deque

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
model = load_model('models/asl_model.h5')
labels = [chr(i) for i in range(65, 91)]  # A-Z

# English dictionary for fuzzy matching (you can use a larger list here)
# You can use words from a file or a larger word list to enhance the matching.
ENGLISH_WORDS = set([
    "hello", "world", "how", "are", "you", "good", "help", "please", "thanks",
    "yes", "no", "goodbye", "morning", "night", "food", "school", "work", "friend",
    "love", "computer", "keyboard", "happy", "sad", "beautiful", "book", "read"
])

# Webcam
cap = cv2.VideoCapture(0)

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.5   # Lowered for better capture
PREDICTION_HISTORY_LENGTH = 10  # How many frames to consider for smoothing
prediction_history = deque(maxlen=PREDICTION_HISTORY_LENGTH)

# Word building
current_word = ""
final_text = ""
last_hand_time = time.time()
last_letter_time = time.time()

# Function to find closest word from the dictionary using fuzzy matching
def correct_word_with_fuzzy(word):
    best_match = None
    best_score = 0
    for dictionary_word in ENGLISH_WORDS:
        score = fuzz.ratio(word.lower(), dictionary_word.lower())
        if score > best_score:
            best_score = score
            best_match = dictionary_word
    return best_match if best_score > 75 else None  # If score is high enough, accept correction

while True:
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        last_hand_time = time.time()

        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Predict
            pred = model.predict(np.array([landmarks]))[0]
            confidence = np.max(pred)
            predicted_index = np.argmax(pred)
            predicted_letter = labels[predicted_index]

            # Save prediction to history if confidence > 0.5
            if confidence > CONFIDENCE_THRESHOLD:
                prediction_history.append(predicted_letter)

            # Check if most of the history agrees
            if len(prediction_history) == PREDICTION_HISTORY_LENGTH:
                most_common = max(set(prediction_history), key=prediction_history.count)
                freq = prediction_history.count(most_common)

                if freq > PREDICTION_HISTORY_LENGTH // 2:
                    # Accept prediction only if majority matches
                    if len(current_word) == 0 or current_word[-1] != most_common:
                        current_word += most_common
                    prediction_history.clear()  # Reset after accepting letter

            # Show prediction
            text = f'{predicted_letter} ({confidence:.2f})'
            color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(frame, text, (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    else:
        elapsed = time.time() - last_hand_time

        if elapsed > 1.5 and current_word != "":  # Wait for a pause longer than 1.5 seconds to finalize the word
            # First, try fuzzy match for the word
            corrected_word = correct_word_with_fuzzy(current_word)

            if corrected_word:
                final_text += corrected_word + " "
            else:
                # If no good fuzzy match, use TextBlob for spelling correction
                corrected_word = str(TextBlob(current_word).correct()).capitalize()
                final_text += corrected_word + " "
            
            current_word = ""  # Reset current word after adding to final_text

    # Display texts (Shifted upward + little smaller)
    cv2.putText(frame, f'Current: {current_word}', (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Final: {final_text}', (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("ASL Word Speller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

 """
 
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from textblob import TextBlob
from fuzzywuzzy import fuzz
import time
from collections import deque
import nltk

# Download NLTK words if not already
# nltk.download('words')

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
model = load_model('models/asl_model.h5')

# A-Z labels for predictions
labels = [chr(i) for i in range(65, 91)]  # A-Z

# English dictionary for fuzzy matching (example words)
ENGLISH_WORDS = set([
    "hello", "world", "how", "are", "you", "good", "help", "please", "thanks",
    "yes", "no", "goodbye", "morning", "night", "food", "school", "work", "friend",
    "love", "computer", "keyboard", "happy", "sad", "beautiful", "book", "read"
])

# Webcam setup
cap = cv2.VideoCapture(0)

# Check if webcam opens successfully
if not cap.isOpened():
    print("❌ ERROR: Cannot open webcam")
    exit()
else:
    print("✅ Webcam opened successfully!")

# Parameters for prediction
CONFIDENCE_THRESHOLD = 0.5
PREDICTION_HISTORY_LENGTH = 10  # History for smoothing predictions
prediction_history = deque(maxlen=PREDICTION_HISTORY_LENGTH)

# Word construction variables
current_word = ""
final_text = ""
last_hand_time = time.time()

# Fuzzy matching function for correction
def correct_word_with_fuzzy(word):
    best_match = None
    best_score = 0
    for dictionary_word in ENGLISH_WORDS:
        score = fuzz.ratio(word.lower(), dictionary_word.lower())
        if score > best_score:
            best_score = score
            best_match = dictionary_word
    return best_match if best_score > 75 else None

# Main loop
while True:
    ret, frame = cap.read()
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        print("Hand detected!")  # Debugging: Check if hand is detected
        last_hand_time = time.time()

        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Predict the letter
            pred = model.predict(np.array([landmarks]))[0]
            confidence = np.max(pred)
            predicted_index = np.argmax(pred)
            predicted_letter = labels[predicted_index]

            # If confidence is high enough, add to history
            if confidence > CONFIDENCE_THRESHOLD:
                print(f"Prediction: {predicted_letter} with confidence {confidence:.2f}")  # Debugging
                prediction_history.append(predicted_letter)

            # Check for majority agreement in prediction history
            if len(prediction_history) == PREDICTION_HISTORY_LENGTH:
                most_common = max(set(prediction_history), key=prediction_history.count)
                freq = prediction_history.count(most_common)

                if freq > PREDICTION_HISTORY_LENGTH // 2:
                    if len(current_word) == 0 or current_word[-1] != most_common:
                        current_word += most_common
                    prediction_history.clear()

            # Display the predicted letter
            text = f'{predicted_letter} ({confidence:.2f})'
            color = (0, 255, 0) if confidence > CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    else:
        elapsed = time.time() - last_hand_time

        if elapsed > 1.5 and current_word != "":  # Pause longer than 1.5 seconds to finalize word
            # First, try fuzzy match for the word
            corrected_word = correct_word_with_fuzzy(current_word)

            if corrected_word:
                final_text += corrected_word + " "
            else:
                corrected_word = str(TextBlob(current_word).correct()).capitalize()
                final_text += corrected_word + " "
            
            current_word = ""  # Reset after adding to final text

    # Display current and final texts
    cv2.putText(frame, f'Current: {current_word}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f'Final: {final_text}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Show the frame with predictions
    cv2.imshow("ASL Word Speller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
