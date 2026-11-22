"""
live_shim.py - lets the UNCHANGED hand_analysis_system_2.py run on a server without a screen.

The original script shows its tracking result with cv2.imshow() in a desktop window. A cloud server has
no screen, so this launcher replaces the window functions with versions that save every frame the script
"shows" to one JPEG file (HTA_LIVE_FRAME). The web page displays that file as the live tracking view.
Nothing else in hand_analysis_system_2.py is touched.
"""
import os
import runpy
import sys
import time

import cv2

_FRAME = os.environ["HTA_LIVE_FRAME"]
_SCRIPT = os.environ["HTA_LIVE_SCRIPT"]


def _imshow(name, img):
    try:
        h, w = img.shape[:2]
        s = min(1.0, 800.0 / max(w, 1))
        if s < 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        tmp = _FRAME + ".tmp.jpg"
        cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        os.replace(tmp, _FRAME)
    except Exception:
        pass


def _key(delay=1):
    time.sleep(0.005)          # no keyboard on a server
    return -1


cv2.imshow = _imshow
cv2.namedWindow = lambda *a, **k: None
cv2.resizeWindow = lambda *a, **k: None
cv2.setMouseCallback = lambda *a, **k: None
cv2.waitKey = _key
cv2.waitKeyEx = _key
cv2.destroyAllWindows = lambda *a, **k: None

sys.argv = [_SCRIPT]
runpy.run_path(_SCRIPT, run_name="__main__")
