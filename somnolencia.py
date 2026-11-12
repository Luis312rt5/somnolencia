#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detector de somnolencia v3.1  -  Raspberry Pi 4 + camara USB + buzzer

Que hace:
  * Rostro: malla de 478 puntos (MediaPipe Face Landmarker). Con los puntos de los
    parpados calcula el EAR (apertura del ojo), calibrado para cada persona.
  * Cabeza: calcula giro e inclinacion; avisa si la cabeza se desvia o cae.
  * Manos: 21 puntos por mano (MediaPipe Hand Landmarker). Si no hay manos dentro de
    la zona del volante, suena la alarma.
  * Alcance: busca el rostro por mosaicos y luego lo sigue con un recorte alrededor
    de la cara, asi funciona con la camara lejos.
  * FPS: la camara se lee en un hilo aparte (siempre el cuadro mas reciente, sin
    retraso) y las manos se procesan en otro hilo.
  * Solo el conductor: el rostro se busca unicamente dentro de la ZONA DEL CONDUCTOR,
    se descartan caras lejanas a la que se sigue, y se ignoran caras/manos pequenas
    (los pasajeros de atras se ven mas pequenos).

Teclas (con ventana):  q = salir | c = recalibrar ojos/cabeza
                       k = marcar zona del conductor (cabeza) | v = marcar zona del volante (manos)
Opciones:  --info-camara  (muestra zoom y resoluciones de la camara y sale)
           --sin-ventana  --sin-manos  --sin-calibrar  --camara N
           --guardar-img  (guarda /tmp/vista.jpg con los dibujos, para verla por SSH)
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import deque

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")   # evita el error del plugin "wayland"
os.environ.setdefault("GLOG_minloglevel", "2")    # menos mensajes de TensorFlow Lite

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    raise SystemExit("Falta mediapipe. Dentro del entorno virtual ejecuta:  pip install mediapipe")

# =============================== CONFIGURACION ===============================
CAMARA = 0                    # /dev/video0
ANCHO, ALTO = 1280, 720       # resolucion de captura (mas pixeles = mas alcance)
FPS_CAMARA = 30
ZOOM_MINIMO = True            # intenta poner el zoom de la camara al minimo (mas campo de vision)
MOSTRAR_VENTANA = True        # False si corres por SSH sin monitor
ANCHO_VENTANA = 800           # ancho de la imagen mostrada (mas pequeno = mas rapido)
PIN_BUZZER = 24               # GPIO24 = pin fisico 18

# --- Calibracion (mirar al frente con los ojos abiertos) ---
CALIBRAR_AL_INICIAR = True
SEGUNDOS_CALIBRACION = 4

# --- Ojos ---
RATIO_OJO_CERRADO = 0.70      # ojo "cerrado" si EAR < este % del EAR normal de la persona
TIEMPO_OJOS_CERRADOS = 0.6    # segundos seguidos con ojos cerrados para la alarma
USAR_PERCLOS = True           # tambien alarma si pasa mucho tiempo con ojos cerrados
VENTANA_PERCLOS = 10.0        # segundos que se miran para el PERCLOS
UMBRAL_PERCLOS = 0.30         # fraccion de tiempo con ojos cerrados que activa la alarma

# --- Cabeza (grados respecto a la posicion calibrada) ---
MAX_GIRO = 35                 # girar la cabeza a los lados
MAX_INCLINACION = 25          # cabeza caida/levantada
TIEMPO_CABEZA_DESVIADA = 2.0  # segundos seguidos fuera de limite para la alarma

# --- Rostro ---
TIEMPO_SIN_ROSTRO = 2.0       # segundos sin rostro para la alarma
MARGEN_RECORTE = 2.6          # el recorte alrededor de la cara mide esto x el ancho de la cara
LADO_MINIMO_RECORTE = 224     # tamano minimo del recorte, en pixeles
# Zona donde puede estar la cara del conductor: (x1, y1, x2, y2) como fraccion de la imagen.
# Tecla k para marcarla. Hazla lo mas estrecha posible alrededor de tu cabeza.
ZONA_CONDUCTOR = (0.0, 0.0, 1.0, 1.0)
LADO_MIN_ROSTRO = 0           # px minimos del rostro; 0 = sin filtro (ponlo ~70% del tamano de tu cara)
DISTANCIA_MAX_SEGUIMIENTO = 0.8   # una cara nueva debe estar a menos de (esto x ancho de cara) de la seguida

# --- Manos ---
USAR_MANOS = True
MANOS_REQUERIDAS = 1         # cuantas manos deben estar en el volante (1 o 2)
TIEMPO_SIN_MANOS = 1.5        # segundos sin las manos requeridas para la alarma
ZONA_VOLANTE = (0.0, 0.45, 1.0, 1.0)   # (x1, y1, x2, y2) como fraccion de la imagen; tecla v para marcarla
REQUERIR_AGARRE = False       # True: ademas exige dedos doblados (mano cerrada sobre el volante)
INTERVALO_MANOS = 0.10        # segundos minimos entre detecciones de manos
MIN_TAM_MANO = 0              # px minimos de la mano; 0 = sin filtro (ponlo ~70% del tamano de tu mano)
# ============================================================================

CARPETA_MODELOS = os.path.expanduser("~/modelos_somno")
ARCHIVO_CONFIG = os.path.expanduser("~/zona_somno.json")
URL_MODELOS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

# Puntos de la malla facial (MediaPipe) para el EAR: esquina, 2 arriba, esquina, 2 abajo
OJO_DER = [33, 160, 158, 133, 153, 144]
OJO_IZQ = [362, 385, 387, 263, 373, 380]
CONEXIONES_MANO = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
                   (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
                   (18, 19), (19, 20), (0, 17)]

PINCTRL = shutil.which("pinctrl") or "/usr/bin/pinctrl"


# ================================== BUZZER ===================================
def _pin(estado):  # "dh" = encendido, "dl" = apagado
    r = subprocess.run([PINCTRL, "set", str(PIN_BUZZER), "op", estado], capture_output=True, text=True)
    if r.returncode != 0:
        print("Error pinctrl:", r.stderr.strip() or r.stdout.strip())


class Alarma:
    def __init__(self):
        self.sonando = False

    def activar(self):
        if not self.sonando:
            _pin("dh")
            self.sonando = True

    def detener(self):
        if self.sonando:
            _pin("dl")
            self.sonando = False


def pitido(duracion=0.15):
    _pin("dh")
    time.sleep(duracion)
    _pin("dl")


# ============================ CAMARA: ZOOM E INFO ============================
def _v4l2(args):
    try:
        r = subprocess.run(["v4l2-ctl"] + args, capture_output=True, text=True, timeout=5)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 1, ""


def ajustar_zoom_minimo(indice):
    """Pone el zoom de la camara en su valor minimo (si la camara tiene zoom). Devuelve el valor o None."""
    dev = f"/dev/video{indice}"
    _, salida = _v4l2(["-d", dev, "--list-ctrls"])
    m = re.search(r"zoom_absolute\b[^\n]*?min=(-?\d+)", salida)
    if not m:
        return None
    minimo = int(m.group(1))
    _v4l2(["-d", dev, f"--set-ctrl=zoom_absolute={minimo}"])
    return minimo


def info_camara(indice):
    dev = f"/dev/video{indice}"
    print(f"=== Controles de {dev} (zoom y enfoque) ===")
    _, salida = _v4l2(["-d", dev, "--list-ctrls"])
    lineas = [l for l in salida.splitlines() if re.search(r"zoom|focus", l)]
    print("\n".join(lineas) if lineas else "Esta camara NO tiene zoom ni enfoque ajustables por software.")
    print(f"\n=== Resoluciones y formatos de {dev} ===")
    _, salida = _v4l2(["-d", dev, "--list-formats-ext"])
    print(salida.strip() or "No se pudo leer (instala v4l-utils: sudo apt install v4l-utils)")


class Camara:
    """Lee la camara en un hilo y guarda solo el cuadro mas reciente (sin acumular retraso)."""

    def __init__(self, indice):
        self.cap = cv2.VideoCapture(indice, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
        self.cap.set(cv2.CAP_PROP_FPS, FPS_CAMARA)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            self.cap.release()
            raise SystemExit("No se pudo abrir la camara")
        self.ancho = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.alto = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = 0.0
        self.error = False
        self._frame = None
        self._idx = 0
        self._lock = threading.Lock()
        self._activo = True
        threading.Thread(target=self._bucle, daemon=True).start()

    def _bucle(self):
        cuenta, t0 = 0, time.time()
        while self._activo:
            ok, f = self.cap.read()
            if not ok:
                self.error = True
                time.sleep(0.02)
                continue
            self.error = False
            with self._lock:
                self._frame, self._idx = f, self._idx + 1
            cuenta += 1
            if time.time() - t0 >= 1.0:
                self.fps = cuenta / (time.time() - t0)
                cuenta, t0 = 0, time.time()

    def leer(self, ultimo_idx, timeout=1.0):
        limite = time.time() + timeout
        while time.time() < limite:
            with self._lock:
                if self._frame is not None and self._idx != ultimo_idx:
                    return self._frame, self._idx
            time.sleep(0.002)
        return None, ultimo_idx

    def cerrar(self):
        self._activo = False
        time.sleep(0.05)
        self.cap.release()


# ================================ UTILIDADES =================================
def asegurar_modelos():
    os.makedirs(CARPETA_MODELOS, exist_ok=True)
    rutas = {}
    for nombre, url in URL_MODELOS.items():
        ruta = os.path.join(CARPETA_MODELOS, nombre)
        if not os.path.exists(ruta) or os.path.getsize(ruta) < 100000:
            print(f"Descargando {nombre} (solo la primera vez)...")
            try:
                urllib.request.urlretrieve(url, ruta + ".tmp")
                os.replace(ruta + ".tmp", ruta)
            except Exception as e:
                raise SystemExit(f"No se pudo descargar {nombre}: {e}\n"
                                 f"Descargalo a mano (con internet) y reinicia:\n  wget -O {ruta} {url}")
        rutas[nombre] = ruta
    return rutas


def cargar_zonas():
    """Devuelve (zona_volante, zona_conductor). Si no hay archivo, usa los valores de la configuracion."""
    try:
        with open(ARCHIVO_CONFIG) as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}

    def _z(clave, defecto):
        z = d.get(clave)
        try:
            if z and len(z) == 4:
                return tuple(float(v) for v in z)
        except (TypeError, ValueError):
            pass
        return defecto
    return _z("volante", ZONA_VOLANTE), _z("conductor", ZONA_CONDUCTOR)


def guardar_zonas(volante, conductor):
    try:
        with open(ARCHIVO_CONFIG, "w") as f:
            json.dump({"volante": list(volante), "conductor": list(conductor)}, f)
    except OSError as e:
        print("No se pudo guardar la zona:", e)


def zona_a_px(z, w, h):
    return int(z[0] * w), int(z[1] * h), int(z[2] * w), int(z[3] * h)


def expandir(zona_px, w, h, m=0.12):
    x1, y1, x2, y2 = zona_px
    dx, dy = (x2 - x1) * m, (y2 - y1) * m
    return (int(max(0, x1 - dx)), int(max(0, y1 - dy)), int(min(w, x2 + dx)), int(min(h, y2 + dy)))


def dif_ang(a, b):
    """a - b en grados, envuelto a [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def ear(pts, idx):
    p1, p2, p3, p4, p5, p6 = (pts[i][:2] for i in idx)
    horizontal = 2.0 * np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / horizontal)


def ear_medio(pts):
    return (ear(pts, OJO_DER) + ear(pts, OJO_IZQ)) / 2.0


def angulos_cabeza(matriz):
    """(inclinacion, giro) en grados a partir de la matriz de transformacion facial 4x4."""
    R = np.array(matriz, dtype=np.float64)[:3, :3]
    n = np.linalg.norm(R, axis=0)
    n[n < 1e-9] = 1.0
    R = R / n
    sy = math.hypot(R[2, 1], R[2, 2])
    pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    yaw = math.degrees(math.atan2(-R[2, 0], sy))
    return pitch, yaw


def mano_agarra(pts):
    """True si al menos 3 dedos estan doblados (la punta queda mas cerca de la muneca que la articulacion)."""
    muneca = pts[0][:2]
    dobl = 0
    for punta, art in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if np.linalg.norm(pts[punta][:2] - muneca) < np.linalg.norm(pts[art][:2] - muneca) * 1.05:
            dobl += 1
    return dobl >= 3


def contar_manos_en_volante(manos, zona_px):
    x1, y1, x2, y2 = zona_px
    n = 0
    for m in manos:
        c = m["pts"][[0, 5, 9, 13, 17]].mean(axis=0)   # centro de la palma
        dentro = x1 <= c[0] <= x2 and y1 <= c[1] <= y2
        m["en_volante"] = bool(dentro and (not REQUERIR_AGARRE or mano_agarra(m["pts"])))
        n += m["en_volante"]
    return n


# ============================ DETECTOR DE ROSTRO =============================
class DetectorRostro:
    # Subzonas (relativas a la zona del conductor) donde se busca el rostro mientras no se ha encontrado
    MOSAICOS = [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.62, 1.0), (0.38, 0.0, 1.0, 1.0), (0.19, 0.0, 0.81, 0.85)]

    def __init__(self, modelo):
        opciones = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=modelo),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opciones)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.caja = None      # (cx, cy, lado) del rostro seguido, en pixeles
        self.perdidos = 0
        self.fase = 0

    def _mejorar(self, crop):
        if cv2.mean(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))[0] >= 85:
            return crop
        l, a, b = cv2.split(cv2.cvtColor(crop, cv2.COLOR_BGR2LAB))
        return cv2.cvtColor(cv2.merge((self.clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

    def _inferir(self, frame, x1, y1, cw, ch):
        crop = self._mejorar(frame[y1:y1 + ch, x1:x1 + cw])
        rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        caras = []
        for i, lms in enumerate(res.face_landmarks):
            pts = np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * cw + x1
            pts[:, 1] = pts[:, 1] * ch + y1
            pts[:, 2] = pts[:, 2] * cw
            mats = res.facial_transformation_matrixes
            caras.append((pts, mats[i] if i < len(mats) else None))
        return caras

    @staticmethod
    def _centro(c):
        p = c[0][:, :2]
        return (p.min(0) + p.max(0)) / 2

    @staticmethod
    def _lado(c):
        p = c[0][:, :2]
        return float((p.max(0) - p.min(0)).max())

    def _dentro(self, c, zona_px):
        cx, cy = self._centro(c)
        x1, y1, x2, y2 = zona_px
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _elegir(self, caras):
        if self.caja is not None:
            return min(caras, key=lambda c: np.linalg.norm(self._centro(c) - np.array(self.caja[:2])))
        return max(caras, key=self._lado)

    def procesar(self, frame, zona_px):
        h, w = frame.shape[:2]
        zx1, zy1, zx2, zy2 = zona_px
        zw, zh = max(32, zx2 - zx1), max(32, zy2 - zy1)
        if self.caja is not None:   # seguimiento: recorte alrededor de la cara (mas preciso a distancia)
            cx, cy, lr = self.caja
            lado = int(min(max(lr * MARGEN_RECORTE, LADO_MINIMO_RECORTE), min(w, h)))
            x1 = int(np.clip(cx - lado / 2, 0, w - lado))
            y1 = int(np.clip(cy - lado / 2, 0, h - lado))
            caras = self._inferir(frame, x1, y1, lado, lado)
        else:                       # busqueda por mosaicos, solo dentro de la zona del conductor
            fx1, fy1, fx2, fy2 = self.MOSAICOS[self.fase % len(self.MOSAICOS)]
            self.fase += 1
            x1 = int(np.clip(zx1 + fx1 * zw, 0, w - 32))
            y1 = int(np.clip(zy1 + fy1 * zh, 0, h - 32))
            cw = int(min(max(32, (fx2 - fx1) * zw), w - x1))
            ch = int(min(max(32, (fy2 - fy1) * zh), h - y1))
            caras = self._inferir(frame, x1, y1, cw, ch)

        # Filtros: dentro de la zona del conductor, tamano minimo y cerca de la cara que se sigue
        caras = [c for c in caras if self._dentro(c, zona_px)]
        if LADO_MIN_ROSTRO:
            caras = [c for c in caras if self._lado(c) >= LADO_MIN_ROSTRO]
        if self.caja is not None:
            ref = np.array(self.caja[:2])
            maxd = self.caja[2] * DISTANCIA_MAX_SEGUIMIENTO
            caras = [c for c in caras if np.linalg.norm(self._centro(c) - ref) < maxd]

        if not caras:
            if self.caja is not None:
                self.perdidos += 1
                if self.perdidos >= 2:
                    self.caja, self.perdidos = None, 0
            return None

        self.perdidos = 0
        pts, matriz = self._elegir(caras)
        mn, mx = pts[:, :2].min(0), pts[:, :2].max(0)
        centro, lado = (mn + mx) / 2, float((mx - mn).max())
        if self.caja is None:
            self.caja = (float(centro[0]), float(centro[1]), lado)
        else:
            a = 0.6
            self.caja = tuple(a * n + (1 - a) * o for n, o in zip((centro[0], centro[1], lado), self.caja))
        return {"pts": pts, "matriz": matriz, "caja": (mn[0], mn[1], mx[0], mx[1]), "lado": lado}


# ============================= DETECTOR DE MANOS =============================
class DetectorManos(threading.Thread):
    """Detecta las manos en un hilo aparte, solo dentro de la zona del volante."""

    def __init__(self, modelo):
        super().__init__(daemon=True)
        opciones = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=modelo),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self.lm = vision.HandLandmarker.create_from_options(opciones)
        self.activo = True
        self.evento = threading.Event()
        self.lock = threading.Lock()
        self._entrada = None
        self._resultado = (0.0, [])
        self.error = None
        self.start()

    def enviar(self, frame, zona_px, t):
        with self.lock:
            self._entrada = (frame, zona_px, t)
        self.evento.set()

    def obtener(self):
        with self.lock:
            return self._resultado

    def detener(self):
        self.activo = False
        self.evento.set()

    def run(self):
        ultimo = 0.0
        while self.activo:
            self.evento.wait(0.3)
            self.evento.clear()
            with self.lock:
                entrada, self._entrada = self._entrada, None
            if entrada is None or time.time() - ultimo < INTERVALO_MANOS:
                continue
            ultimo = time.time()
            try:
                frame, zona_px, t = entrada
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = expandir(zona_px, w, h)
                crop = frame[y1:y2, x1:x2]
                rgb = np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                res = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                cw, ch = x2 - x1, y2 - y1
                manos = []
                for lms in res.hand_landmarks:
                    pts = np.array([(p.x * cw + x1, p.y * ch + y1) for p in lms], dtype=np.float32)
                    tam = float((pts.max(0) - pts.min(0)).max())
                    if tam < MIN_TAM_MANO:      # mano pequena = lejos = pasajero de atras
                        continue
                    manos.append({"pts": pts, "en_volante": False, "tam": tam})
                with self.lock:
                    self._resultado = (t, manos)
            except Exception as e:   # no dejar morir el hilo en silencio
                if str(e) != self.error:
                    print("Error en deteccion de manos:", e)
                self.error = str(e)


# ================================== MONITOR ==================================
class Monitor:
    """Decide cuando hay que sonar la alarma a partir de rostro, ojos, cabeza y manos."""

    def __init__(self):
        self.umbral = 0.19            # valores por defecto hasta calibrar
        self.base_pitch = 0.0
        self.base_yaw = 0.0
        self.calibrando = False
        self.t_cal = 0.0
        self.m_ear, self.m_pose = [], []
        self.hist_ear = deque(maxlen=3)
        self.hist_cierre = deque()
        self.t_cierre = None
        self.t_cabeza = None
        self.t_rostro = time.time()
        self.t_manos_ok = time.time()
        self.ear_base = None

    def iniciar_calibracion(self, t):
        self.calibrando = True
        self.t_cal = t
        self.m_ear, self.m_pose = [], []
        print("Calibrando: mira al frente con los ojos abiertos...")

    def _terminar_calibracion(self, t):
        if len(self.m_ear) >= 10:
            self.ear_base = float(np.median(self.m_ear))
            self.umbral = float(np.clip(self.ear_base * RATIO_OJO_CERRADO, 0.10, 0.25))
            p0, y0 = self.m_pose[0]
            self.base_pitch = p0 + float(np.median([dif_ang(p, p0) for p, _ in self.m_pose]))
            self.base_yaw = y0 + float(np.median([dif_ang(y, y0) for _, y in self.m_pose]))
            print(f"Calibrado: EAR normal {self.ear_base:.2f} -> umbral {self.umbral:.2f}")
            self.calibrando = False
            pitido(0.1)
            pitido(0.1)
        elif t - self.t_cal >= 3 * SEGUNDOS_CALIBRACION:
            print("No se vio el rostro para calibrar: uso valores por defecto (tecla c para reintentar).")
            self.calibrando = False

    def actualizar(self, t, cara, n_manos, manos_frescas):
        motivos = []
        info = {"ear": None, "umbral": self.umbral, "pitch": None, "yaw": None,
                "perclos": 0.0, "calibrando": self.calibrando}

        if cara is not None:
            self.t_rostro = t
            self.hist_ear.append(ear_medio(cara["pts"]))
            ear_s = float(np.median(self.hist_ear))   # mediana de los ultimos 3 cuadros (quita ruido)
            if cara["matriz"] is not None:
                pitch, yaw = angulos_cabeza(cara["matriz"])
            else:
                pitch, yaw = self.base_pitch, self.base_yaw
            info["ear"] = ear_s

            if self.calibrando:
                self.m_ear.append(ear_s)
                self.m_pose.append((pitch, yaw))
            else:
                d_pitch, d_yaw = dif_ang(pitch, self.base_pitch), dif_ang(yaw, self.base_yaw)
                info["pitch"], info["yaw"] = d_pitch, d_yaw

                desviada = abs(d_yaw) > MAX_GIRO or abs(d_pitch) > MAX_INCLINACION
                if desviada:
                    self.t_cabeza = self.t_cabeza or t
                    if t - self.t_cabeza >= TIEMPO_CABEZA_DESVIADA:
                        motivos.append("CABEZA DESVIADA")
                else:
                    self.t_cabeza = None

                cerrado = (not desviada) and ear_s < self.umbral
                info["cerrado"] = cerrado
                self.hist_cierre.append((t, cerrado))
                while self.hist_cierre and t - self.hist_cierre[0][0] > VENTANA_PERCLOS:
                    self.hist_cierre.popleft()

                if cerrado:
                    self.t_cierre = self.t_cierre or t
                    if t - self.t_cierre >= TIEMPO_OJOS_CERRADOS:
                        motivos.append("OJOS CERRADOS")
                else:
                    self.t_cierre = None

                if self.hist_cierre:
                    info["perclos"] = sum(1 for _, c in self.hist_cierre if c) / len(self.hist_cierre)
                    abarca = t - self.hist_cierre[0][0]
                    if USAR_PERCLOS and abarca >= VENTANA_PERCLOS * 0.5 and info["perclos"] >= UMBRAL_PERCLOS:
                        motivos.append("SOMNOLENCIA")
        else:
            self.t_cierre = None
            self.t_cabeza = None
            if not self.calibrando and t - self.t_rostro >= TIEMPO_SIN_ROSTRO:
                motivos.append("SIN ROSTRO")

        if USAR_MANOS:
            if manos_frescas and n_manos >= MANOS_REQUERIDAS:
                self.t_manos_ok = t
            if not self.calibrando and t - self.t_manos_ok >= TIEMPO_SIN_MANOS:
                motivos.append("MANOS FUERA DEL VOLANTE")
        else:
            self.t_manos_ok = t

        if self.calibrando:
            self.t_manos_ok = t
            if t - self.t_cal >= SEGUNDOS_CALIBRACION:
                self._terminar_calibracion(t)
            info["calibrando"] = self.calibrando
            return [], info
        return motivos, info


# ================================== DIBUJO ===================================
def dibujar(disp, s, cara, manos, zona_px, motivos, info, n_manos, fps_cam, fps_proc, calibrando, zona_cond_px=None):
    f = cv2.FONT_HERSHEY_SIMPLEX
    if zona_cond_px is not None:   # zona del conductor (azul)
        cx1, cy1, cx2, cy2 = (int(v * s) for v in zona_cond_px)
        cv2.rectangle(disp, (cx1, cy1), (cx2, cy2), (255, 120, 0), 1)
        cv2.putText(disp, "conductor", (cx1 + 4, cy2 - 6), f, 0.4, (255, 120, 0), 1, cv2.LINE_AA)

    if cara is not None:
        x1, y1, x2, y2 = (int(v * s) for v in cara["caja"])
        cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 255, 0), 1)
        col = (0, 0, 255) if info.get("cerrado") else (0, 255, 0)
        for idx in (OJO_DER, OJO_IZQ):
            cv2.polylines(disp, [(cara["pts"][idx][:, :2] * s).astype(np.int32)], True, col, 1)

    zx1, zy1, zx2, zy2 = (int(v * s) for v in zona_px)
    cv2.rectangle(disp, (zx1, zy1), (zx2, zy2), (0, 255, 0) if n_manos >= MANOS_REQUERIDAS else (0, 165, 255), 2)
    for m in manos:
        p = (m["pts"] * s).astype(np.int32)
        col = (0, 255, 0) if m["en_volante"] else (0, 0, 255)
        for a, b in CONEXIONES_MANO:
            cv2.line(disp, tuple(p[a]), tuple(p[b]), col, 2)
        for q in p:
            cv2.circle(disp, tuple(q), 3, (255, 255, 255), -1)
        cv2.putText(disp, f"{m.get('tam', 0):.0f}px", tuple(p[0]), f, 0.4, col, 1, cv2.LINE_AA)

    if calibrando:
        estado, color = "CALIBRANDO: mira al frente, ojos abiertos", (0, 255, 255)
    elif motivos:
        estado, color = " + ".join(motivos), (0, 0, 255)
    else:
        estado, color = "OK", (0, 255, 0)
    cv2.putText(disp, estado, (10, 30), f, 0.9, color, 2)
    lineas = [f"Camara {fps_cam:.0f} fps | Proceso {fps_proc:.0f} fps"]
    if cara is not None:
        lineas.append(f"Rostro {cara['lado']:.0f}px | EAR {info['ear']:.2f} (umbral {info['umbral']:.2f}) | PERCLOS {info['perclos']:.0%}")
        if info["pitch"] is not None:
            lineas.append(f"Cabeza: inclin {info['pitch']:+.0f} giro {info['yaw']:+.0f} grados")
    else:
        lineas.append("Rostro: no detectado")
    if USAR_MANOS:
        lineas.append(f"Manos en volante: {n_manos}/{MANOS_REQUERIDAS}")
    for i, txt in enumerate(lineas):
        cv2.putText(disp, txt, (10, 58 + 22 * i), f, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


# ============================= FABRICAS (pruebas) ============================
def crear_camara(indice):
    return Camara(indice)


def abrir_camara_con_reintentos(indice, intentos=30, espera=2.0):
    """Reintenta abrir la camara (al arrancar la Pi puede tardar en estar lista). Espera hasta ~1 minuto."""
    for n in range(1, intentos + 1):
        try:
            return crear_camara(indice)
        except SystemExit:
            print(f"Camara no disponible (intento {n}/{intentos}); reintento en {espera:.0f} s...")
            time.sleep(espera)
    raise SystemExit("No se pudo abrir la camara tras varios intentos")


def crear_detector_rostro(ruta):
    return DetectorRostro(ruta)


def crear_detector_manos(ruta):
    return DetectorManos(ruta)


# =================================== MAIN ====================================
def main(argv=None):
    global MOSTRAR_VENTANA, USAR_MANOS, CALIBRAR_AL_INICIAR, CAMARA
    ap = argparse.ArgumentParser(description="Detector de somnolencia v3.1")
    ap.add_argument("--camara", type=int, default=CAMARA)
    ap.add_argument("--info-camara", action="store_true", help="muestra zoom y resoluciones y sale")
    ap.add_argument("--sin-ventana", action="store_true")
    ap.add_argument("--sin-manos", action="store_true")
    ap.add_argument("--sin-calibrar", action="store_true")
    ap.add_argument("--guardar-img", action="store_true", help="guarda /tmp/vista.jpg para verla por SSH")
    ap.add_argument("--max-cuadros", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    CAMARA = args.camara
    if args.info_camara:
        info_camara(CAMARA)
        return
    MOSTRAR_VENTANA = MOSTRAR_VENTANA and not args.sin_ventana
    USAR_MANOS = USAR_MANOS and not args.sin_manos
    CALIBRAR_AL_INICIAR = CALIBRAR_AL_INICIAR and not args.sin_calibrar
    zona, zona_cond = cargar_zonas()

    rutas = asegurar_modelos()
    alarma = Alarma()
    _pin("dl")
    print("Probando buzzer...")
    pitido(0.3)

    if ZOOM_MINIMO:
        z = ajustar_zoom_minimo(CAMARA)
        print(f"Zoom de la camara al minimo ({z})." if z is not None else
              "La camara no tiene zoom por software (usa --info-camara para ver resoluciones).")

    cam = abrir_camara_con_reintentos(CAMARA)
    print(f"Resolucion real: {cam.ancho} x {cam.alto}")
    det_rostro = crear_detector_rostro(rutas["face_landmarker.task"])
    det_manos = crear_detector_manos(rutas["hand_landmarker.task"]) if USAR_MANOS else None

    monitor = Monitor()
    if CALIBRAR_AL_INICIAR:
        monitor.iniciar_calibracion(time.time())

    idx, cuadros, fps_proc, t_ant, estado_ant = 0, 0, 0.0, time.time(), ""
    t_img = 0.0
    fallos_lectura = 0
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))   # pkill/apagado: cierra y apaga el buzzer
    try:
        while True:
            frame, idx = cam.leer(idx, timeout=2.0)
            if frame is None:
                fallos_lectura += 1
                print(f"No se pudo leer la camara ({fallos_lectura})")
                if fallos_lectura >= 5:   # camara colgada: reabrirla en vez de cerrar el programa
                    print("Reabriendo la camara...")
                    cam.cerrar()
                    cam = abrir_camara_con_reintentos(CAMARA)
                    idx, fallos_lectura = 0, 0
                continue
            fallos_lectura = 0
            h, w = frame.shape[:2]
            t = time.time()
            fps_proc = 0.9 * fps_proc + 0.1 / max(t - t_ant, 1e-3)
            t_ant = t

            zona_cond_px = zona_a_px(zona_cond, w, h)
            cara = det_rostro.procesar(frame, zona_cond_px)

            zona_px = zona_a_px(zona, w, h)
            manos, n_manos, frescas = [], 0, False
            if det_manos is not None:
                det_manos.enviar(frame, zona_px, t)
                t_res, manos = det_manos.obtener()
                frescas = (t - t_res) < 0.8
                n_manos = contar_manos_en_volante(manos, zona_px) if frescas else 0
                if not frescas:
                    manos = []

            motivos, info = monitor.actualizar(t, cara, n_manos, frescas)
            if motivos:
                alarma.activar()
            else:
                alarma.detener()

            estado = " + ".join(motivos) if motivos else ("CALIBRANDO" if monitor.calibrando else "OK")
            if estado != estado_ant:
                print("Estado:", estado)
                estado_ant = estado

            if args.guardar_img and t - t_img >= 0.3:
                t_img = t
                s_img = ANCHO_VENTANA / w
                vista = cv2.resize(frame, (ANCHO_VENTANA, int(h * s_img)))
                dibujar(vista, s_img, cara, manos, zona_px, motivos, info, n_manos,
                        cam.fps, fps_proc, monitor.calibrando, zona_cond_px)
                cv2.imwrite("/tmp/vista_tmp.jpg", vista)
                os.replace("/tmp/vista_tmp.jpg", "/tmp/vista.jpg")

            if MOSTRAR_VENTANA:
                s = ANCHO_VENTANA / w
                disp = cv2.resize(frame, (ANCHO_VENTANA, int(h * s)), interpolation=cv2.INTER_LINEAR)
                dibujar(disp, s, cara, manos, zona_px, motivos, info, n_manos,
                        cam.fps, fps_proc, monitor.calibrando, zona_cond_px)
                cv2.imshow("Detector de somnolencia", disp)
                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == ord("c"):
                    monitor.iniciar_calibracion(time.time())
                if tecla == ord("v"):
                    nom = "Zona del volante: arrastra y pulsa ENTER (c = cancelar)"
                    r = cv2.selectROI(nom, disp, showCrosshair=False)
                    cv2.destroyWindow(nom)
                    if r[2] > 5 and r[3] > 5:
                        zona = (r[0] / disp.shape[1], r[1] / disp.shape[0],
                                (r[0] + r[2]) / disp.shape[1], (r[1] + r[3]) / disp.shape[0])
                        guardar_zonas(zona, zona_cond)
                        print("Zona del volante guardada:", tuple(round(v, 2) for v in zona))
                if tecla == ord("k"):
                    nom = "Zona del conductor: arrastra y pulsa ENTER (c = cancelar)"
                    r = cv2.selectROI(nom, disp, showCrosshair=False)
                    cv2.destroyWindow(nom)
                    if r[2] > 5 and r[3] > 5:
                        zona_cond = (r[0] / disp.shape[1], r[1] / disp.shape[0],
                                     (r[0] + r[2]) / disp.shape[1], (r[1] + r[3]) / disp.shape[0])
                        guardar_zonas(zona, zona_cond)
                        det_rostro.caja = None
                        print("Zona del conductor guardada:", tuple(round(v, 2) for v in zona_cond))

            cuadros += 1
            if args.max_cuadros and cuadros >= args.max_cuadros:
                break
    except KeyboardInterrupt:
        pass
    finally:
        alarma.detener()
        _pin("dl")
        if det_manos is not None:
            det_manos.detener()
        cam.cerrar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
