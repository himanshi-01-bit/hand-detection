import cv2
import mediapipe as mp
import math

class HandDetector:
    def __init__(self, max_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        self.finger_bases = [3, 6, 10, 14, 18]  # Updated base points for more accurate detection

    def find_hands(self, img, draw=True):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        all_hands = []
        hand_sides = []

        if self.results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                hand_info = {}
                lm_list = []
                
                # Get hand side (Left/Right)
                hand_side = self.results.multi_handedness[hand_idx].classification[0].label
                hand_sides.append(hand_side)

                # Get landmarks
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])

                # Draw landmarks if requested
                if draw:
                    self.mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        self.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

                hand_info["landmarks"] = lm_list
                all_hands.append(hand_info)

        return img, all_hands, hand_sides

    def count_fingers(self, hand_landmarks, hand_side):
        if not hand_landmarks:
            return 0

        fingers = []
        landmarks = hand_landmarks["landmarks"]

        # Thumb (special case - different for left and right hands)
        if hand_side == "Right":
            # For right hand, thumb is open if it's to the left of the knuckle
            if landmarks[self.finger_tips[0]][1] < landmarks[self.finger_bases[0]][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        else:  # Left hand
            # For left hand, thumb is open if it's to the right of the knuckle
            if landmarks[self.finger_tips[0]][1] > landmarks[self.finger_bases[0]][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        # Other fingers - same for both hands
        # A finger is considered open if the tip is higher (lower y value) than the pip joint
        for id in range(1, 5):
            if landmarks[self.finger_tips[id]][2] < landmarks[self.finger_bases[id]][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return sum(fingers)

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        # Flip image horizontally for mirror effect
        img = cv2.flip(img, 1)

        # Find hands
        img, hands, hand_sides = detector.find_hands(img)

        # Process each detected hand
        for i, (hand, hand_side) in enumerate(zip(hands, hand_sides)):
            fingers = detector.count_fingers(hand, hand_side)
            
            # Display finger count and hand side
            cv2.putText(
                img,
                f"{hand_side}: {fingers} fingers",
                (10, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

        # Display the image
        cv2.imshow("Hand Tracking", img)

        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

