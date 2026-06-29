import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque, Counter
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Deterministic mapping: each gesture ALWAYS shows the same meme. Where a meme
# has an iconic pose, the gesture matches it (both hands up = ABSOLUTE CINEMA,
# pointing = "is this a pigeon", finger on lips = shush, etc.).

# Face / two-hand gestures -> meme file
GESTURE_MEME = {
    "hands_up":   "absolute_cinema.jpg",   # both hands raised up   -> ABSOLUTE CINEMA
    "hands_face": "cara.jpeg",             # two open palms beside the face
    "shh":        "cristiano.png",         # a finger on the mouth
    "tongue":     "gato1.png",             # tongue out
    "eyebrows":   "perro.jpeg",            # raised / furrowed eyebrows
}

# Single-hand finger shapes [thumb, index, middle, ring, pinky] -> meme file
HAND_MEME = {
    (0, 1, 1, 0, 0): "rata.jpeg",                 # peace / victory   (index+middle)
    (1, 0, 0, 0, 0): "drake_hotline_bling.jpg",   # thumbs up
    (0, 1, 0, 0, 0): "is_this_a_pigeon.jpg",      # pointing (index only)
    (0, 0, 0, 0, 0): "buff_doge_vs_cheems.png",   # fist
    (1, 1, 1, 1, 1): "epic_handshake.jpg",        # open palm (all five)
    (0, 1, 1, 1, 0): "expanding_brain.jpg",       # three fingers
    (0, 1, 1, 1, 1): "change_my_mind.jpg",        # four fingers
    (0, 1, 0, 0, 1): "disaster_girl.jpg",         # rock horns (index+pinky)
    (1, 0, 0, 0, 1): "this_is_fine.jpg",          # shaka / call me (thumb+pinky)
    (1, 1, 0, 0, 0): "two_buttons.jpg",           # L / finger-gun (thumb+index)
    (0, 0, 0, 0, 1): "grumpy_cat.jpg",            # pinky only
    (1, 1, 1, 0, 0): "trade_offer.jpg",           # thumb+index+middle
}

# Drop any mapping whose image file is missing, so a gesture never shows a
# broken meme.
GESTURE_MEME = {g: f for g, f in GESTURE_MEME.items()
                if os.path.exists(os.path.join(HERE, f))}
HAND_MEME = {k: f for k, f in HAND_MEME.items()
             if os.path.exists(os.path.join(HERE, f))}

mp_face  = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.7, min_tracking_confidence=0.7)
hands_det = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7, min_tracking_confidence=0.7)

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


def draw_button(frame, rect, label, hot=False):
    x1, y1, x2, y2 = rect
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), (40, 40, 40), -1)
    cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
    border = (80, 240, 80) if hot else (120, 120, 120)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)
    cv2.putText(frame, label, (x1 + 12, y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = param["btn"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            param["recalibrate"] = True


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

    cal          = Cal()
    buf          = deque(maxlen=10)
    img_actual   = None
    MINVOTOS     = 6
    btn_w, btn_h = 150, 32
    ui           = {"recalibrate": False,
                    "btn": (W - btn_w - 10, 10, W - 10, 10 + btn_h)}
    cv2.setMouseCallback("Tu Camara", on_mouse, ui)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        H, W  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fr    = face_mesh.process(rgb)
        hr    = hands_det.process(rgb)

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
            if fr.multi_face_landmarks:
                cal.feed(fr.multi_face_landmarks[0].landmark)
            cv2.imshow("Tu Camara",      frame)
            cv2.imshow("Meme Detectado", fondo)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        if fr.multi_face_landmarks:
            lm_cara = fr.multi_face_landmarks[0].landmark

        if hr.multi_hand_landmarks:
            for i, hl in enumerate(hr.multi_hand_landmarks):
                lm  = hl.landmark
                izq = hr.multi_handedness[i].classification[0].label == "Left"
                ded = dedos_estado(lm, izq)
                draw_hand_minimal(frame, lm, W, H, ded)
                manos.append((ded, lm))
                manos_info.append(("I" if izq else "D", ded))

        if lm_cara and len(manos) == 2 and det_sonic(manos, lm_cara):
            det = GESTURE_MEME.get("hands_up")
        elif len(manos) == 2 and det_cara(manos):
            det = GESTURE_MEME.get("hands_face")
        elif lm_cara and manos and det_cristiano(manos, lm_cara):
            det = GESTURE_MEME.get("shh")
        elif lm_cara and det_lengua(lm_cara, cal):
            det = GESTURE_MEME.get("tongue")
        elif lm_cara and det_ceja(lm_cara, cal):
            det = GESTURE_MEME.get("eyebrows")
        elif len(manos) == 1:
            ded_m, lm_m = manos[0]
            det = HAND_MEME.get(tuple(ded_m))

        buf.append(det)
        conteo     = Counter(buf)
        top, votos = conteo.most_common(1)[0]
        if votos >= MINVOTOS:
            img_actual = top

        hud(frame, img_actual, manos_info, W, H)
        ui["btn"] = (W - btn_w - 10, 10, W - 10, 10 + btn_h)
        draw_button(frame, ui["btn"], "RECALIBRATE", hot=ui["recalibrate"])
        cv2.imshow("Tu Camara", frame)

        if img_actual:
            meme = cv2.imread(os.path.join(HERE, img_actual))
            if meme is not None and meme.size > 0:
                cv2.imshow("Meme Detectado", cv2.resize(meme, (W, H)))
            else:
                err = fondo.copy()
                cv2.putText(err, f"Falta: {img_actual}", (20, H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 220), 2)
                cv2.imshow("Meme Detectado", err)
        else:
            cv2.imshow("Meme Detectado", fondo)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord('r'), ord('R')) or ui["recalibrate"]:
            ui["recalibrate"] = False
            cal        = Cal()
            buf.clear()
            img_actual = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()