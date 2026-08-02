import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import math
import numpy as np
from collections import deque, Counter
import os
import urllib.request

# ── Model auto-download ──────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_FACE_MODEL = os.path.join(_DIR, "face_landmarker.task")
_HAND_MODEL = os.path.join(_DIR, "hand_landmarker.task")

_MODELS = {
    _FACE_MODEL: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    _HAND_MODEL: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}

for _path, _url in _MODELS.items():
    if not os.path.exists(_path):
        print(f"Descargando {os.path.basename(_path)} ...")
        urllib.request.urlretrieve(_url, _path)
        print("  OK")

# ── Detectors (Tasks API) ────────────────────────────────────────────
face_landmarker = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_FACE_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.7,
        min_face_presence_confidence=0.7,
        min_tracking_confidence=0.7))

hand_landmarker = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7))

def d(a, b):
    return math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)

def esc(lm):
    return d(lm[152], lm[10]) + 1e-6

def px(pt, W, H):
    return (int(pt.x * W), int(pt.y * H))

def dedos_estado(lm, izq=False):
    tip = [8,12,16,20]
    mid_j = [6,10,14,18]
    out = [1 if (lm[4].x > lm[3].x if izq else lm[4].x < lm[3].x) else 0]
    for t, m in zip(tip, mid_j):
        out.append(1 if lm[t].y < lm[m].y else 0)
    return out

class Cal:
    N = 45

    def __init__(self):
        self.buf = {k: [] for k in ['ci','cd','cen','lap','llb','bi_y','bd_y','gap']}
        self.done = False
        self.thr = dict(
            ci=0.180, cd=0.180, cen_lo=0.185,
            lap=0.055, llb=0.145,
            bi_y_lo=0.30, bd_y_lo=0.30,
            gap_lo=0.10
        )

    def feed(self, lm):
        if self.done:
            return
        e = esc(lm)
        self.buf['ci'].append(d(lm[52], lm[159]) / e)
        self.buf['cd'].append(d(lm[282], lm[386]) / e)
        self.buf['cen'].append(d(lm[55], lm[285]) / e)
        self.buf['lap'].append(d(lm[13], lm[14]) / e)
        self.buf['llb'].append(d(lm[17], lm[152]) / e)
        self.buf['bi_y'].append(lm[55].y - lm[9].y)
        self.buf['bd_y'].append(lm[285].y - lm[9].y)
        self.buf['gap'].append(abs(lm[55].x - lm[285].x))
        if len(self.buf['ci']) >= self.N:
            self._calc()

    def _calc(self):
        m  = lambda k: float(np.median(self.buf[k]))
        s  = lambda k: float(np.std(self.buf[k]))
        mg_c = lambda k: max(1.5 * s(k), 0.015)
        mg_b = lambda k, mn: max(3 * s(k), mn)
        self.thr['ci']      = m('ci')  + mg_c('ci')
        self.thr['cd']      = m('cd')  + mg_c('cd')
        self.thr['cen_lo']  = m('cen') - mg_c('cen')
        self.thr['lap']     = m('lap') + mg_b('lap', 0.032)
        self.thr['llb']     = m('llb') - mg_b('llb', 0.018)
        self.thr['bi_y_lo'] = m('bi_y') + mg_c('bi_y')
        self.thr['bd_y_lo'] = m('bd_y') + mg_c('bd_y')
        self.thr['gap_lo']  = m('gap')  - mg_c('gap')
        self.done = True

    @property
    def progress(self):
        return min(len(self.buf['ci']) / self.N, 1.0)


def det_lengua(lm, cal):
    e = esc(lm)
    boca_abierta = d(lm[13], lm[14]) / e > cal.thr['lap']
    lengua_baja  = d(lm[17], lm[152]) / e < cal.thr['llb']
    punta_fuera  = lm[17].y > lm[14].y + 0.012
    return boca_abierta and lengua_baja and punta_fuera

def det_ceja(lm, cal):
    e    = esc(lm)
    ci   = d(lm[52],  lm[159]) / e
    cd   = d(lm[282], lm[386]) / e
    cen  = d(lm[55],  lm[285]) / e
    bi_y = lm[55].y  - lm[9].y
    bd_y = lm[285].y - lm[9].y
    gap  = abs(lm[55].x - lm[285].x)
    return (
        ci   > cal.thr['ci']      or
        cd   > cal.thr['cd']      or
        cen  < cal.thr['cen_lo']  or
        bi_y > cal.thr['bi_y_lo'] or
        bd_y > cal.thr['bd_y_lo'] or
        gap  < cal.thr['gap_lo']
    )

def det_cristiano(manos, lm_cara):
    boca = lm_cara[13]
    return any(d(lm[8], boca) < 0.09 or d(lm[12], boca) < 0.09
               for _, lm in manos)

def det_rata(ded):
    return ded == [0, 1, 1, 0, 0]

def det_sonic(manos, lm_cara):
    if len(manos) != 2:
        return False
    nariz_y = lm_cara[1].y
    return all(lm[9].y < nariz_y for _, lm in manos)

def det_cara(manos):
    if len(manos) != 2:
        return False
    for ded, lm in manos:
        if ded[1:] != [1, 1, 1, 1] or lm[0].y < 0.50:
            return False
    return abs(manos[0][1][0].x - manos[1][1][0].x) >= 0.20


FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,
             397,365,379,378,400,377,152,148,176,149,150,136,
             172,58,132,93,234,127,162,21,54,103,67,109,10]
EYE_L  = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7,33]
EYE_R  = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382,362]
BROW_L = [70,63,105,66,107,55,65,52,53,46]
BROW_R = [300,293,334,296,336,285,295,282,283,276]
LIPS_OUT = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61]
LIPS_IN  = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191,78]
NOSE = [168,6,197,195,5,4,1,19,94,2]

def draw_face_minimal(frame, lm, W, H, cal):
    e        = esc(lm)
    ci       = d(lm[52],  lm[159]) / e
    cd       = d(lm[282], lm[386]) / e
    cen      = d(lm[55],  lm[285]) / e
    boca_act = (d(lm[13], lm[14]) / e > cal.thr['lap'] and
                d(lm[17], lm[152]) / e < cal.thr['llb'])
    ceja_act = ci > cal.thr['ci'] or cd > cal.thr['cd'] or cen < cal.thr['cen_lo']

    COL_BASE = (140, 200, 140)
    COL_ACT  = (80,  240,  80)
    COL_CEJA = COL_ACT if ceja_act else COL_BASE
    COL_BOCA = COL_ACT if boca_act else COL_BASE

    def draw_path(indices, col, close=False):
        pts = [px(lm[i], W, H) for i in indices]
        for j in range(len(pts) - 1):
            cv2.line(frame, pts[j], pts[j+1], col, 1, cv2.LINE_AA)
        if close and len(pts) > 1:
            cv2.line(frame, pts[-1], pts[0], col, 1, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 1, col, -1, cv2.LINE_AA)

    draw_path(FACE_OVAL, COL_BASE, close=False)
    draw_path(EYE_L,     COL_BASE, close=True)
    draw_path(EYE_R,     COL_BASE, close=True)
    draw_path(BROW_L,    COL_CEJA)
    draw_path(BROW_R,    COL_CEJA)
    draw_path(NOSE,      COL_BASE)
    draw_path(LIPS_OUT,  COL_BOCA, close=True)
    draw_path(LIPS_IN,   COL_BOCA, close=True)


HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

def draw_hand_minimal(frame, lm, W, H, ded):
    COL = (140, 200, 140)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px(lm[a], W, H), px(lm[b], W, H), COL, 1, cv2.LINE_AA)
    for i in range(21):
        cv2.circle(frame, px(lm[i], W, H), 2, COL, -1, cv2.LINE_AA)
    for i, tip in enumerate([4, 8, 12, 16, 20]):
        if ded[i]:
            cv2.circle(frame, px(lm[tip], W, H), 3, (80, 240, 80), -1, cv2.LINE_AA)


def hud(frame, img_actual, manos_info, W, H):
    nombre = img_actual if img_actual else "neutral"
    col    = (80, 220, 80) if img_actual else (160, 160, 160)
    ov     = frame.copy()
    cv2.rectangle(ov, (8, 8), (min(W - 8, 14 + len(nombre) * 14 + 20), 36), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, nombre, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
    for i, (lado, ded) in enumerate(manos_info):
        cv2.putText(frame, f"{lado}: {ded}", (14, 58 + 24 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: no se pudo abrir la camara")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    for _ in range(5):
        cap.read()

    ret, frame0 = cap.read()
    if not ret:
        print("Error: no se pudo leer frame inicial")
        cap.release()
        return

    frame0 = cv2.flip(frame0, 1)
    H, W   = frame0.shape[:2]
    fondo  = np.full((H, W, 3), 30, dtype=np.uint8)

    cv2.namedWindow("Tu Camara",      cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("Meme Detectado", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("Tu Camara",      frame0)
    cv2.imshow("Meme Detectado", fondo)
    cv2.waitKey(1)

    cal        = Cal()
    buf        = deque(maxlen=10)
    img_actual = None
    MINVOTOS   = 6
    frame_ts   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        frame_ts += 1
        fr    = face_landmarker.detect_for_video(mp_img, frame_ts)
        hr    = hand_landmarker.detect_for_video(mp_img, frame_ts)

        det        = None
        lm_cara    = None
        manos      = []
        manos_info = []

        if not cal.done:
            pct = cal.progress
            ov  = frame.copy()
            cv2.rectangle(ov, (0, 0), (W, H), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cy  = H // 2
            cv2.putText(frame, "Mira al frente  cara neutral",
                        (W // 2 - 200, cy - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (200, 200, 200), 2, cv2.LINE_AA)
            bx1, bx2 = W // 2 - 140, W // 2 + 140
            cv2.rectangle(frame, (bx1, cy + 10), (bx2, cy + 28), (40, 40, 40), -1)
            cv2.rectangle(frame, (bx1, cy + 10),
                          (bx1 + int(280 * pct), cy + 28), (80, 220, 80), -1)
            cv2.rectangle(frame, (bx1, cy + 10), (bx2, cy + 28), (120, 120, 120), 1)
            cv2.putText(frame, f"{int(pct * 100)}%",
                        (W // 2 - 18, cy + 48), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (160, 160, 160), 1, cv2.LINE_AA)
            if fr.face_landmarks:
                cal.feed(fr.face_landmarks[0])
            cv2.imshow("Tu Camara",      frame)
            cv2.imshow("Meme Detectado", fondo)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        if fr.face_landmarks:
            lm_cara = fr.face_landmarks[0]

        if hr.hand_landmarks:
            for i, hl in enumerate(hr.hand_landmarks):
                lm  = hl
                izq = hr.handedness[i][0].category_name == "Left"
                ded = dedos_estado(lm, izq)
                draw_hand_minimal(frame, lm, W, H, ded)
                manos.append((ded, lm))
                manos_info.append(("I" if izq else "D", ded))

        if lm_cara and len(manos) == 2 and det_sonic(manos, lm_cara):
            det = "Sonic.jpeg"
        elif len(manos) == 2 and det_cara(manos):
            det = "cara.jpeg"
        elif lm_cara and manos and det_cristiano(manos, lm_cara):
            det = "cristiano.png"
        elif lm_cara and det_lengua(lm_cara, cal):
            det = "gato1.png"
        elif lm_cara and det_ceja(lm_cara, cal):
            det = "perro.jpeg"
        elif len(manos) == 1:
            ded_m, lm_m = manos[0]
            if det_rata(ded_m):
                det = "rata.jpeg"

        buf.append(det)
        conteo     = Counter(buf)
        top, votos = conteo.most_common(1)[0]
        if votos >= MINVOTOS:
            img_actual = top

        hud(frame, img_actual, manos_info, W, H)
        cv2.imshow("Tu Camara", frame)

        if img_actual:
            meme = cv2.imread(img_actual)
            if meme is not None and meme.size > 0:
                cv2.imshow("Meme Detectado", cv2.resize(meme, (W, H)))
            else:
                err = fondo.copy()
                cv2.putText(err, f"Falta: {img_actual}", (20, H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 220), 2)
                cv2.imshow("Meme Detectado", err)
        else:
            cv2.imshow("Meme Detectado", fondo)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    hand_landmarker.close()


if __name__ == "__main__":
    main()