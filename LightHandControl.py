import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math

class LightHandControl:
    def __init__(self):
        self.wCam, self.hCam = 640, 480
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, self.wCam)
        self.cap.set(4, self.hCam)
        self.pTime = 0
        self.detector = htm.handDetector(detectionCon=0.7)
        self.minVol, self.maxVol = 0, 640
        self.Bar, self.Per = 100, 0
        self.start_time = None
        self.prev_length = None
        self.tolerance = 5
        self.wait_time = 2
        self.is_confirmed = False
        self.adjusting = True
        self.light = 0

    def get_light(self):
        success, img = self.cap.read()
        if not success:
            return None

        img = self.detector.findHands(img)
        lmList, _ = self.detector.findPosition(img, draw=False)

        if len(lmList) >= 21:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)

            if self.adjusting:
                self.light = np.interp(length, [50, 300], [self.minVol, self.maxVol])
                self.Bar = np.interp(length, [50, 300], [400, 150])
                self.Per = np.interp(length, [50, 300], [0, 640])

                if self.prev_length and abs(self.prev_length - length) < self.tolerance:
                    if self.start_time is None:
                        self.start_time = time.time()
                    else:
                        elapsed_time = time.time() - self.start_time
                        if elapsed_time >= self.wait_time:
                            self.is_confirmed = True
                            self.adjusting = False
                else:
                    self.start_time = None

                self.prev_length = length

            if length < 50:
                cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

            fist_detected = all([lmList[tip][2] > lmList[tip - 2][2] for tip in [8, 12, 16, 20]])
            if fist_detected:
                self.adjusting = True
                self.is_confirmed = False

        # Draw bar
        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
        cv2.rectangle(img, (50, int(self.Bar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(self.Per)} ', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        # Display lock/unlock status
        status = "Locked" if self.is_confirmed else "Adjusting"
        cv2.putText(img, f'Status: {status}', (200, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Volume Control", img)
        cv2.waitKey(1)

        return int(self.light)

    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()

