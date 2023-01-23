import cv2

class video():
    def __init__(self, resize=False, width=1920, height=1080, fps=30, name="received frame", display=True):
        self.resize = resize
        self.width = width
        self.height = height
        self.fps = fps
        self.name = name
        self.display = display
        self.cap = None
        if self.display: cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
    
    def open(self, device=0):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    
    def read(self):
        return self.cap.read()
    
    def resize_frame(self, frame, width, height, interpolation=cv2.INTER_NEAREST):
        if self.resize:
            return cv2.resize(frame, (height, width), interpolation=interpolation)
        else:
            return frame
    
    def imshow(self, frame):
        # Print in botom of frame the frame shape
        cv2.putText(frame, f"{self.name} shape: {frame.shape}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (179,147,97), 1)
        cv2.imshow(self.name, frame)
        if cv2.waitKey(1) == ord('q'):
            return False
        else:
            return True
    
    def encode_frame(self, frame, format='.jpg'):
        return cv2.imencode(format, frame)
    
    def decode_frame(self, frame, codec='mp4v', fps=30):
        return cv2.imdecode(self.frame, cv2.IMREAD_UNCHANGED)
    
    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()