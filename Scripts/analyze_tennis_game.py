import cv2
import numpy as np
from collections import deque
import time
from ultralytics import YOLO
import sys
import os

class TennisGameAnalyzer:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            sys.exit()

        self.ball_trajectory = deque(maxlen=30)
        self.last_event = ""
        self.event_timestamp = 0
        self.event_display_duration = 2
        self.event_cooldown = 0.5
        self.last_event_time = 0
        self.ball_in_play = False

        self.table_poly = None
        self.net_line = None
        self.side1_poly = None
        self.side2_poly = None
        self.points = []
        self.setup_complete = False

        self.score_player1 = 0
        self.score_player2 = 0
        self.last_fault = ""
        self.last_bounce_side = None

        self.out = None

    def _select_points_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 6:
            self.points.append((x, y))

    def _interactive_setup(self):
        ret, first_frame = self.cap.read()
        if not ret:
            print("Error: Could not read the first frame.")
            return False

        clone = first_frame.copy()
        cv2.namedWindow("Setup: Define Play Area")
        cv2.setMouseCallback("Setup: Define Play Area", self._select_points_callback)

        instructions = [
            "Select 6 points in order:",
            "1-4: Table corners (top-left, top-right, bottom-right, bottom-left)",
            "5-6: Two ends of the net",
            "Press 'c' to confirm, 'r' to reset, 'q' to quit."
        ]

        while True:
            frame_copy = clone.copy()
            for i, p in enumerate(self.points):
                cv2.circle(frame_copy, p, 5, (0, 255, 0), -1)
                cv2.putText(frame_copy, str(i+1), (p[0]+10, p[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            for i, text in enumerate(instructions):
                cv2.putText(frame_copy, text, (10, 30 + i*30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            cv2.imshow("Setup: Define Play Area", frame_copy)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return False
            elif key == ord('r'):
                self.points = []
            elif key == ord('c'):
                if len(self.points) == 6:
                    table_pts = self.points[:4]
                    self.table_poly = np.array(table_pts, dtype=np.int32)
                    self.net_line = (self.points[4], self.points[5])

                    net_mid_x = int((self.net_line[0][0] + self.net_line[1][0]) / 2)
                    h_min = min(p[1] for p in table_pts)
                    h_max = max(p[1] for p in table_pts)

                    self.side1_poly = np.array([
                        (min(p[0] for p in table_pts), h_min),
                        (net_mid_x, h_min),
                        (net_mid_x, h_max),
                        (min(p[0] for p in table_pts), h_max)
                    ], dtype=np.int32)

                    self.side2_poly = np.array([
                        (net_mid_x, h_min),
                        (max(p[0] for p in table_pts), h_min),
                        (max(p[0] for p in table_pts), h_max),
                        (net_mid_x, h_max)
                    ], dtype=np.int32)

                    self.setup_complete = True
                    cv2.destroyAllWindows()
                    return True

    def _detect_bounce(self):
        if len(self.ball_trajectory) < 3:
            return
        p1, p2, p3 = self.ball_trajectory[-3], self.ball_trajectory[-2], self.ball_trajectory[-1]
        is_moving_down = p2[1] > p1[1]
        is_moving_up = p3[1] < p2[1]
        if is_moving_down and is_moving_up:
            if cv2.pointPolygonTest(self.table_poly, p2, False) >= 0:
                self.ball_in_play = True
                side = "Player 1 Side" if cv2.pointPolygonTest(self.side1_poly, p2, False) >= 0 else "Player 2 Side"
                self._log_event(f"BOUNCE on {side}")
            else:
                if self.ball_in_play:
                    self._log_event("OUT")
                    self.ball_in_play = False

    def _detect_net_hit(self):
        if len(self.ball_trajectory) < 2:
            return
        p1 = self.ball_trajectory[-2]
        p2 = self.ball_trajectory[-1]
        net_x_avg = (self.net_line[0][0] + self.net_line[1][0]) / 2
        net_y_min = min(self.net_line[0][1], self.net_line[1][1])
        net_y_max = max(self.net_line[0][1], self.net_line[1][1])
        if (p1[0] < net_x_avg < p2[0]) or (p2[0] < net_x_avg < p1[0]):
            if net_y_min <= p2[1] <= net_y_max:
                self._log_event("NET HIT")
                self.ball_in_play = False

    def _log_event(self, event_text):
        current_time = time.time()
        if current_time - self.last_event_time > self.event_cooldown:
            self.last_event = event_text
            self.event_timestamp = current_time
            self.last_event_time = current_time
            frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            print(f"EVENT: {event_text} at frame {frame_num}")

            if event_text.startswith("OUT"):
                if self.last_bounce_side == "Player 1 Side":
                    self.score_player2 += 1
                    self.last_fault = "FAULT: OUT by Player 1"
                elif self.last_bounce_side == "Player 2 Side":
                    self.score_player1 += 1
                    self.last_fault = "FAULT: OUT by Player 2"
                else:
                    self.last_fault = "FAULT: OUT (unknown player)"
                self.ball_in_play = False

            elif event_text.startswith("NET HIT"):
                if self.last_bounce_side == "Player 1 Side":
                    self.score_player2 += 1
                    self.last_fault = "FAULT: NET HIT by Player 1"
                elif self.last_bounce_side == "Player 2 Side":
                    self.score_player1 += 1
                    self.last_fault = "FAULT: NET HIT by Player 2"
                else:
                    self.last_fault = "FAULT: NET HIT (unknown player)"
                self.ball_in_play = False

            elif event_text.startswith("BOUNCE on"):
                self.last_bounce_side = event_text.split("on ")[1]
                self.last_fault = ""

    def _draw_elements(self, frame, fps):
        cv2.polylines(frame, [self.table_poly], True, (255, 255, 0), 2)
        cv2.line(frame, self.net_line[0], self.net_line[1], (0, 0, 255), 2)
        cv2.polylines(frame, [self.side1_poly], True, (0, 255, 0), 2)
        cv2.polylines(frame, [self.side2_poly], True, (255, 0, 0), 2)

        if len(self.ball_trajectory) > 1:
            pts = np.array(list(self.ball_trajectory), np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (0, 255, 0), 2)

        cv2.putText(frame, f"Player 1: {self.score_player1}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Player 2: {self.score_player2}", (frame.shape[1]-250, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if self.last_fault:
            cv2.putText(frame, self.last_fault, (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if self.last_event and (time.time() - self.event_timestamp < self.event_display_duration):
            overlay = frame.copy()
            cv2.rectangle(overlay, (20, 150), (600, 210), (0, 0, 0), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
            cv2.putText(frame, self.last_event, (30, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def run(self):
        if not self._interactive_setup():
            print("Setup was not completed. Exiting.")
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

        while self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()

            if len(results[0].boxes) > 0:
                box = results[0].boxes[0].xyxy[0]
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
                self.ball_trajectory.append((cx, cy))
            else:
                if len(self.ball_trajectory) > 0:
                    self.ball_trajectory.popleft()

            self._detect_bounce()
            self._detect_net_hit()

            fps_calc = 1 / (time.time() - start_time)
            final_frame = self._draw_elements(annotated_frame, fps_calc)

            self.out.write(final_frame)

            cv2.imshow("Table Tennis Analysis", final_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    MODEL_PATH = "runs/detect/ball_detector/weights/best.pt"
    VIDEO_PATH = "path_to_your_video.mp4"

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
    elif not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at '{VIDEO_PATH}'")
    else:
        analyzer = TennisGameAnalyzer(MODEL_PATH, VIDEO_PATH)
        analyzer.run()
