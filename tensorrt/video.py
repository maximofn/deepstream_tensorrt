import cv2
from PIL import Image
import numpy as np

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
        self.navig_palette = [170, 170, 170, 0, 255, 0, 255, 0, 0, 0, 120, 255, 0, 0, 255, 255,255,153]
    
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
        cv2.putText(frame, f"{self.name} shape: {frame.shape}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, frame.shape[0]/1000, (10,147,255), 1)
        cv2.imshow(self.name, frame)
        if cv2.waitKey(1) == ord('q'):
            return False
        else:
            return True

    def colorized_mask(self, mask, img=None, overlay=0):
        # Colorize mask
        new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
        new_mask.putpalette(self.navig_palette)
        new_mask = np.array(new_mask.convert("RGB"))
        new_mask = new_mask[:, :, ::-1].copy()
        return new_mask
    
    def overlay_mask(self, mask, img, overlay=0):
        # Overlay mask on image
        if img.shape[:2] != mask.shape[:2]:
            img = cv2.resize(img, (mask.shape[1], mask.shape[0]))
        new_mask = cv2.addWeighted(img,overlay,mask,1-overlay,0)
        return new_mask
    
    def maskshow(self, mask, img=None, colorize=False, overlay=False):
        # Print in botom of mask the mask shape
        if colorize:
            mask_color = self.colorized_mask(mask, img, overlay)
        else:
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
        if overlay and img is not None:
            mask_color = self.overlay_mask(mask_color, img, overlay)
        cv2.putText(mask_color, f"{self.name} shape: {mask_color.shape}", (10, mask_color.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, mask.shape[0]/1000, (10,147,255), 1)
        cv2.imshow(self.name, mask_color)
        if cv2.waitKey(1) == ord('q'):
            return False
        else:
            return True
    
    def encode_frame(self, frame, format='.jpg'):
        return cv2.imencode(format, frame, [cv2.IMWRITE_JPEG_QUALITY,80])
    
    def decode_frame(self, buffer, codec='mp4v', fps=30):
        return cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    
    def close(self):
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()