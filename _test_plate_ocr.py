"""Cleaner single-line plate OCR test writing results to /tmp."""
import os
import sys

OUT = "/tmp/plate_ocr_result.txt"


def log(msg):
    with open(OUT, "a") as f:
        f.write(str(msg) + "\n")
    print(msg, flush=True)


def main():
    if os.path.exists(OUT):
        os.remove(OUT)
    import cv2
    import numpy as np

    # Clean single-line white plate with dark text and a thin border.
    def make_single_plate(text="HR26AB1234", w=400, h=110):
        img = np.full((h, w, 3), 245, dtype=np.uint8)
        cv2.rectangle(img, (3, 3), (w - 4, h - 4), (25, 25, 25), 2)  # thin border
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text[:2], (40, 75), font, 1.6, (15, 15, 15), 4, cv2.LINE_AA)   # HR
        cv2.putText(img, text[2:4], (140, 75), font, 1.6, (15, 15, 15), 4, cv2.LINE_AA) # 26
        cv2.putText(img, text[4:6], (230, 75), font, 1.6, (15, 15, 15), 4, cv2.LINE_AA) # AB
        cv2.putText(img, text[6:], (300, 75), font, 1.6, (15, 15, 15), 4, cv2.LINE_AA)  # 1234
        return img

    img = make_single_plate()
    cv2.imwrite("/tmp/plate_ocr.png", img)

    from modules.plate_detector import get_plate_detector
    d = get_plate_detector()
    d.ensure_available()
    log("MODEL_LOADED=" + d.load_status)
    boxes = d.detect(img, conf=0.3)
    log("PLATE_BOXES=" + repr(boxes))
    detected = len(boxes) > 0

    from modules.anpr_video import VideoANPREngine
    eng = VideoANPREngine(debug=True)
    eng.prepare()
    read_text = ""
    for (x1, y1, x2, y2) in boxes:
        crop = img[y1:y2, x1:x2]
        h, w = crop.shape[:2]
        if crop.size and h >= 8:
            log("CROP_SIZE=%dx%d" % (w, h))
            read = eng._read_plate(crop)
            if read:
                read_text = read.text
                log("OCR_TEXT=" + read.text + " conf=" + "%.2f" % read.confidence + " raw=" + repr(read.raw_text))
            else:
                log("OCR_NO_VALID")

    ok = detected and bool(read_text)
    log("RESULT=" + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
