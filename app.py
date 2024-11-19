from flask import Flask, render_template, Response, request, jsonify
import cv2
import pickle
import mediapipe as mp
import numpy as np

app = Flask(__name__, template_folder='./templates', static_folder='./static')

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
labels_dict = {i: chr(65 + i) for i in range(26)}

# Global theme variable
current_theme = {'theme': 'main'}  # Default theme is 'main'


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if current_theme['theme'] == 'pink':
                    # Draw multi-colored landmarks and connections
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=4),
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2)
                    )
                    
                    # Calculate bounding box coordinates
                    h, w, _ = frame.shape
                    x_min = int(min(lm.x for lm in hand_landmarks.landmark) * w)
                    y_min = int(min(lm.y for lm in hand_landmarks.landmark) * h)
                    x_max = int(max(lm.x for lm in hand_landmarks.landmark) * w)
                    y_max = int(max(lm.y for lm in hand_landmarks.landmark) * h)

                    # Draw bounding box around the hand
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Collect data for predictions
            data_aux, x_, y_ = [], [], []
            for lm in results.multi_hand_landmarks[0].landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in results.multi_hand_landmarks[0].landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            # Predict the character
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Display predicted character
            fontScale = 3  # Adjust for size
            thickness = 5  # Adjust for boldness
            cv2.putText(
                frame, 
                predicted_character, 
                (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                fontScale, 
                (0, 0, 0),  # Black text color
                thickness
            )

        # Encode and yield the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/switch_theme', methods=['POST'])
def switch_theme():
    global current_theme
    new_theme = request.json.get('theme', 'main')  # Default to 'main' theme
    current_theme['theme'] = new_theme
    return jsonify({'message': f'Theme switched to {new_theme}'})


if __name__ == "__main__":
    app.run()
