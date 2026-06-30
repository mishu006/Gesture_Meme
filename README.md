# Gesture Meme

Proyecto de visión por computadora que detecta gestos faciales y de manos en tiempo real usando la cámara web y muestra un meme correspondiente.

## Tecnologías

- **Python 3.10.0**
- **OpenCV**
- **MediaPipe**
- **NumPy**

## Requisitos

- **Python 3.8 - 3.11** (MediaPipe no es compatible con versiones superiores)

Instala las dependencias con:

    pip install opencv-python mediapipe numpy

O si tienes el archivo de requisitos:

    pip install -r requirements.txt

## Gestos detectados

| Gesto | Meme |
|-------|------|
| **Cejas levantadas o fruncidas** | `perro.jpeg` |
| **Lengua afuera** | `gato1.png` |
| **Dedo tocando la boca** | `cristiano.png` |
| **Dos manos a los lados de la cara** | `cara.jpeg` |
| **Dos manos por encima de la nariz** | `Sonic.jpeg` |
| **Índice y medio extendidos** | `rata.jpeg` |

## Estructura del proyecto

    gesture_meme/
    ├── main.py
    ├── requirements.txt
    ├── cara.jpeg
    ├── cristiano.png
    ├── gato1.png
    ├── perro.jpeg
    ├── rata.jpeg
    └── Sonic.jpeg

## Uso

1. Clona el repositorio
2. Instala las dependencias
3. Coloca las imágenes en la misma carpeta que `main.py`
4. Ejecuta:

       python main.py

5. Al iniciar, mira al frente con cara neutral durante la **calibración**
6. Una vez calibrado, prueba los gestos frente a la cámara
7. Presiona **ESC** para salir

## Notas

- La **calibración** toma unos segundos al inicio, es necesaria para que los gestos funcionen correctamente
- Las imágenes deben estar en la **misma carpeta** que `main.py`
- Funciona mejor con **buena iluminación**
- Compatible con **Windows** y **Linux** (usa `CAP_DSHOW` en Windows, `CAP_ANY` en Linux)
