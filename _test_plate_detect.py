"""Self-contained plate-detection + OCR test that writes results to a file."""
import os
import sys

OUT = "/tmp/plate_test_result.txt"


def log(msg):
    with open(OUT, "a") as f:
        f.write(str(msg) + "\n")
    print(msg, flush=True)


def main():
    if os.path.exists(OUT):
        os.remove(OUT)
    import cv2
    import numpy as np

    def make_plate_patch(text="HR26AB1234", w=360, h=110):
        img = np.full((h, w, 3), 235, dtype=np.uint8)
        cv2.rectangle(img, (2, 2), (w - 3, h - 3), (30, 30, 30), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, f"{text[:2]} {text[2:4]}", (w // 2 - 90, 34), font, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(img, text[4:6], (w // 2 - 60, 82), font, 1.3, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(img, text[6:], (w // 2 + 20, 82), font, 1.3, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.rectangle(img, (0, h - 12), (w, h), (40, 40, 40), -1)
        return img

    img = make_plate_patch()

    # 1) Dedicated plate model detection.
    from modules.plate_detector import get_plate_detector
    d = get_plate_detector()
    model = d.ensure_available()
    log("MODEL_LOADED=" + d.load_status)
    boxes = d.detect(img, conf=0.3)
    log("PLATE_MODEL_BOXES=" + repr(boxes))
    detected = len(boxes) > 0

    # 2) OCR via engine.
    from modules.anpr_video import VideoANPREngine
    eng = VideoANPREngine(debug=True)
    eng.prepare()
    read_text = ""
    for (x1, y1, x2, y2) in boxes:
        crop = img[y1:y2, x1:x2]
        if crop.size and crop.shape[0] >= 8:
            read = eng._read_plate(crop)
            if read:
                read_text = read.text
                log("OCR_TEXT=" + read.text + " conf=" + "%.2f" % read.confidence + " raw=" + repr(read.raw_text))

    ok = detected and bool(read_text)
    log("RESULT=" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
