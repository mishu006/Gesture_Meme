# Gestos detectados — Hamster Meme

## Implementados (8)

| # | Gesto | Imagen | Lógica de detección | Prioridad |
|---|-------|--------|---------------------|-----------|
| 1 | **Fairy** | `fairy.jpeg` | Sonrisa (boca más ancha que el umbral calibrado) + mano en la mitad superior del frame (`lm[0].y < 0.45`, simula sostener una varita). Se recomienda sostener un lápiz o palito. | 🔺 1 |
| 2 | **Nerd** | `finger-glasses-nerd.jpeg` | Una mano con solo el índice levantado `[0,1,0,0,0]` + detección de gafas mediante Haar Cascade (`haarcascade_eye_tree_eyeglasses.xml`) en el frame actual. | 🔺 2 |
| 3 | **Fist on nose** | `fist-on-nose.jpeg` | Una mano con índice, medio y anular contraídos (`ded[1:4] == [0,0,0]`) + el nudillo central (`lm[9]`) cerca del puente nasal (`lm[168]`). | 🔺 3 |
| 4 | **Head scratching** | `head-scratching.jpeg` | Una mano sobre la línea de los ojos **y** próxima horizontalmente a la nariz (`|x_mano - x_nariz| < 0.25`). | 🔺 4 |
| 5 | **Silence** | `silence.jpeg` | Una mano con índice levantado y los demás dedos (excepto pulgar) contraídos (`ded[1]==1, ded[2:]==[0,0,0]`) + punta del índice (`lm[8]`) a menos de 0.12 del centro de la boca. | 🔺 5 |
| 6 | **Laugh** | `laugh.jpeg` | Boca muy abierta: el gap entre labios (`d(lm[13], lm[14]) / esc`) supera 1.5× el umbral calibrado de apertura bucal. | 🔻 6 |
| 7 | **Like** | `like.jpeg` | Una mano con solo el pulgar arriba `[1,0,0,0,0]`. | 🔻 7 |
| 8 | **Peace and love** | `peace and love .jpeg` | Una mano en V con índice y medio `[0,1,1,0,0]`. | 🔻 8 |

### Notas

- **Fairy**: La "varita" (lápiz, palito) no se detecta por visión — se asume que el usuario la sostiene. La detección es: sonrisa + mano en mitad superior del frame.
- **Nerd**: La detección de gafas se ejecuta frame a frame. Usa el clasificador Haar de OpenCV. Puede fallar si las gafas tienen monturas muy finas o reflejos.
- **Head scratching**: Se añadió filtro horizontal para evitar confundirlo con Fairy (mano levantada sonriendo). La mano debe estar cerca de la cara, no solo arriba.
- **Peace and love**: Usa el mismo patrón de dedos que `det_rata` en `main.py` original.

---

## Pendientes (7)

| # | Gesto | Imagen | Dificultad | Motivo |
|---|-------|--------|------------|--------|
| 1 | **Dislike** | `dislike.jpeg` | 🟡 Media | Requiere detectar pulgar hacia abajo (umbral en eje Y del dedo pulgar). Es factible pero no se ha implementado. |
| 2 | **Cry** | `cry.jpeg` | 🔴 Alta | Expresión facial compleja (cejas + boca fruncida). Se podría aproximar con cejas elevadas + boca pequeña, pero hay riesgo de falsos positivos. |
| 3 | **Anxious** | `Anxious .jpeg` | 🔴 Alta | Dos manos juntas al pecho. Ambigüedad con otras posturas de dos manos. |
| 4 | **Crossed arms** | `crosset arms.jpeg` | 🔴 Alta | Sin skeleton de brazos es impreciso. Se podría detectar manos en lados opuestos del torso, pero no hay landmarks de torso. |
| 5 | **Hands heart** | `hands-heart.jpeg` | 🔴 Alta | Requiere geometría específica de ambas manos formando un corazón. |
| 6 | **Muscle** | `muscle.jpeg` | 🔴 Alta | Una mano cerca del hombro (brazo flexionado). Sin tracking de brazo es difícil distinguir de otras posturas. |
| 7 | **No no no** | `no-no-no.jpeg` | 🔴 Muy alta | Requiere tracking temporal (movimiento oscilante del dedo índice). No es detectable en un solo frame. |

---

## Cómo ejecutar

```bash
python main_hamster.py
```

Requiere calibración inicial: mirar al frente con cara neutral ~3 segundos mientras se llena la barra de progreso.
