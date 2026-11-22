# ai_modules.py
# INTERCO — AI Görüntü İşleme Modül Yöneticisi
# Her modül bağımsız olarak açılıp kapatılabilir, hassasiyet ayarlanabilir,
# modeller talep üzerine indirilir (setup küçük kalır).

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent / "models"
CUSTOM_MODELS_DIR = MODELS_DIR / "custom"
CONFIG_FILE = Path(__file__).parent / "ai_modules_config.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# COCO sınıf isimleri (YOLOv8 varsayılan 80 sınıf)
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Poz iskeleti bağlantı çiftleri (COCO 17 keypoint)
POSE_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # baş
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # üst gövde & kollar
    (5, 11), (6, 12), (11, 12),              # gövde
    (11, 13), (13, 15), (12, 14), (14, 16)   # bacaklar
]

POSE_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 255), (255, 128, 0),
    (0, 128, 255), (128, 255, 0), (255, 0, 128), (0, 255, 128),
    (128, 128, 255), (255, 128, 128), (128, 255, 128), (200, 200, 0)
]


def _default_builtin_modules():
    """Fabrika yerleşik modül tanımları."""
    return [
        {
            "id": "pose",
            "name": "Poz Tahmini",
            "description": "İnsan iskelet noktaları (17 keypoint) tespiti",
            "model_file": "yolov8n-pose.pt",
            "model_type": "yolo_pose",
            "classes": None,
            "default_conf": 0.40,
            "builtin": True,
        },
        {
            "id": "person",
            "name": "İnsan Tanıma",
            "description": "Tek kişi tespiti ve takibi",
            "model_file": "yolov8n.pt",
            "model_type": "yolo",
            "classes": [0],
            "default_conf": 0.45,
            "builtin": True,
        },
        {
            "id": "multi_person",
            "name": "Çoklu İnsan Tanıma",
            "description": "Kalabalık ortamda çoklu insan tespiti",
            "model_file": "yolov8n.pt",
            "model_type": "yolo",
            "classes": [0],
            "default_conf": 0.30,
            "builtin": True,
        },
        {
            "id": "ocr",
            "name": "OCR Metin Tanıma",
            "description": "Görüntüdeki metin ve yazı tespiti (EasyOCR)",
            "model_file": "__easyocr__",
            "model_type": "easyocr",
            "classes": None,
            "default_conf": 0.35,
            "builtin": True,
        },
        {
            "id": "plate",
            "name": "Plaka Tanıma",
            "description": "Araç plakası tespiti ve okuma",
            "model_file": "plate_detect.pt",
            "model_type": "yolo",
            "classes": None,
            "default_conf": 0.40,
            "builtin": True,
        },
        {
            "id": "vehicle",
            "name": "Araç Tanıma",
            "description": "Otomobil, motosiklet, otobüs, kamyon tespiti",
            "model_file": "yolov8n.pt",
            "model_type": "yolo",
            "classes": [2, 3, 5, 7],
            "default_conf": 0.40,
            "builtin": True,
        },
        {
            "id": "aircraft",
            "name": "Hava Aracı Tanıma",
            "description": "Uçak ve hava araçları tespiti",
            "model_file": "yolov8n.pt",
            "model_type": "yolo",
            "classes": [4],
            "default_conf": 0.35,
            "builtin": True,
        },
        {
            "id": "animal",
            "name": "Hayvan Tanıma",
            "description": "Kuş, kedi, köpek, at, koyun, inek ve diğer hayvanlar",
            "model_file": "yolov8n.pt",
            "model_type": "yolo",
            "classes": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            "default_conf": 0.35,
            "builtin": True,
        },
        {
            "id": "object",
            "name": "Genel Nesne Tanıma",
            "description": "80 sınıf COCO nesne tespiti (tümü)",
            "model_file": "yolov8n.pt",
            "model_type": "yolo",
            "classes": None,
            "default_conf": 0.45,
            "builtin": True,
        },
    ]


class AIModule:
    """Tek bir AI işleme modülünü temsil eder."""

    def __init__(self, definition: dict, saved_state: dict | None = None):
        self.id: str = definition["id"]
        self.name: str = definition["name"]
        self.description: str = definition["description"]
        self.model_file: str = definition["model_file"]
        self.model_type: str = definition["model_type"]
        self.classes: list[int] | None = definition.get("classes")
        self.builtin: bool = definition.get("builtin", False)
        self.custom_url: str = definition.get("custom_url", "")

        self.confidence: float = definition.get("default_conf", 0.45)
        self.enabled: bool = False
        self.download_status: str = "not_downloaded"  # not_downloaded | downloading | downloaded | error | needs_install
        self.download_progress: float = 0.0
        self.download_error: str = ""
        self.model = None
        self._lock = threading.Lock()

        if saved_state:
            self.confidence = saved_state.get("confidence", self.confidence)
            self.enabled = saved_state.get("enabled", False)

        self._check_availability()

    def _check_availability(self):
        """Model dosyasının mevcut olup olmadığını kontrol et."""
        if self.model_type == "easyocr":
            try:
                import easyocr  # noqa: F401
                self.download_status = "downloaded"
            except ImportError:
                self.download_status = "needs_install"
            return

        if self.model_file.startswith("__"):
            return

        model_path = self._resolve_model_path()
        if model_path and os.path.isfile(model_path):
            self.download_status = "downloaded"

    def _resolve_model_path(self) -> str:
        """Model dosyasının tam yolunu döndürür."""
        if not self.builtin:
            custom_path = CUSTOM_MODELS_DIR / self.model_file
            if custom_path.exists():
                return str(custom_path)
            if os.path.isabs(self.model_file) and os.path.isfile(self.model_file):
                return self.model_file

        candidates = [
            MODELS_DIR / self.model_file,
            Path(__file__).parent / self.model_file,
        ]
        for c in candidates:
            if c.exists():
                return str(c)

        # Yerleşik YOLO modelleri ultralytics tarafından otomatik indirilebilir
        if self.builtin and self.model_file.startswith("yolov8"):
            return self.model_file

        return str(MODELS_DIR / self.model_file)

    def load_model(self):
        """Modeli belleğe yükler."""
        with self._lock:
            if self.model is not None:
                return True

            if self.download_status not in ("downloaded",):
                return False

            try:
                if self.model_type in ("yolo", "yolo_pose"):
                    from ultralytics import YOLO
                    path = self._resolve_model_path()
                    self.model = YOLO(path)
                    self.download_status = "downloaded"
                    return True

                elif self.model_type == "easyocr":
                    import easyocr
                    self.model = easyocr.Reader(["tr", "en"], gpu=_has_cuda())
                    return True

            except Exception as e:
                self.download_error = str(e)[:200]
                self.download_status = "error"
                return False

        return False

    def unload_model(self):
        """Modeli bellekten çıkarır."""
        with self._lock:
            self.model = None

    def process(self, frame: np.ndarray) -> dict:
        """Kareyi işler, tespit sonuçlarını döndürür."""
        result = {
            "module_id": self.id,
            "module_name": self.name,
            "detections": [],
        }

        if not self.enabled or self.model is None:
            return result

        try:
            if self.model_type == "yolo":
                result["detections"] = self._process_yolo(frame)
            elif self.model_type == "yolo_pose":
                result["detections"] = self._process_yolo_pose(frame)
            elif self.model_type == "easyocr":
                result["detections"] = self._process_ocr(frame)
        except Exception:
            pass

        return result

    def _process_yolo(self, frame: np.ndarray) -> list[dict]:
        detections = []
        results = self.model(
            frame,
            conf=self.confidence,
            classes=self.classes,
            verbose=False,
            imgsz=416,
        )
        if not results or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = COCO_NAMES[cls_id] if cls_id < len(COCO_NAMES) else f"class-{cls_id}"
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
            })
        return detections

    def _process_yolo_pose(self, frame: np.ndarray) -> list[dict]:
        detections = []
        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False,
            imgsz=416,
        )
        if not results:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        boxes = r.boxes
        keypoints_data = r.keypoints if hasattr(r, "keypoints") and r.keypoints is not None else None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            kp_list = []
            if keypoints_data is not None and i < len(keypoints_data.xy):
                kp_xy = keypoints_data.xy[i].cpu().numpy()
                kp_conf = keypoints_data.conf[i].cpu().numpy() if keypoints_data.conf is not None else np.ones(len(kp_xy))
                for ki in range(len(kp_xy)):
                    kp_list.append({
                        "x": int(kp_xy[ki][0]),
                        "y": int(kp_xy[ki][1]),
                        "conf": round(float(kp_conf[ki]), 3),
                    })

            detections.append({
                "label": "person",
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "class_id": 0,
                "extra": {"keypoints": kp_list, "skeleton": POSE_SKELETON},
            })
        return detections

    def _process_ocr(self, frame: np.ndarray) -> list[dict]:
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        results = self.model.readtext(gray, detail=1)
        for (bbox_pts, text, conf) in results:
            if conf < self.confidence:
                continue
            pts = np.array(bbox_pts, dtype=np.int32)
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            detections.append({
                "label": text,
                "confidence": round(float(conf), 3),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "extra": {"text": text},
            })
        return detections

    def to_state(self) -> dict:
        """UI ve kayıt için serileştirme."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model_file": self.model_file,
            "model_type": self.model_type,
            "builtin": self.builtin,
            "confidence": round(self.confidence, 2),
            "enabled": self.enabled,
            "download_status": self.download_status,
            "download_progress": round(self.download_progress, 1),
            "download_error": self.download_error,
            "model_loaded": self.model is not None,
            "custom_url": self.custom_url,
        }

    def to_save(self) -> dict:
        """Kalıcı ayarlar için serileştirme."""
        d = {
            "confidence": round(self.confidence, 2),
            "enabled": self.enabled,
        }
        if not self.builtin:
            d.update({
                "id": self.id,
                "name": self.name,
                "description": self.description,
                "model_file": self.model_file,
                "model_type": self.model_type,
                "custom_url": self.custom_url,
            })
        return d


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class AIModuleManager:
    """Tüm AI modüllerini yönetir."""

    def __init__(self):
        self._lock = threading.Lock()
        self.modules: dict[str, AIModule] = {}
        self._load_config()

    def _load_config(self):
        """Kayıtlı yapılandırmayı yükler ve yerleşik modüllerle birleştirir."""
        saved = {}
        custom_defs = []

        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                saved = data.get("module_states", {})
                custom_defs = data.get("custom_modules", [])
            except Exception:
                pass

        for defn in _default_builtin_modules():
            mod = AIModule(defn, saved_state=saved.get(defn["id"]))
            self.modules[mod.id] = mod

        for cdef in custom_defs:
            mid = cdef.get("id", "")
            if mid and mid not in self.modules:
                cdef["builtin"] = False
                cdef.setdefault("default_conf", cdef.get("confidence", 0.45))
                cdef.setdefault("model_type", "yolo")
                mod = AIModule(cdef, saved_state=saved.get(mid))
                self.modules[mid] = mod

    def _save_config(self):
        """Mevcut yapılandırmayı diske yazar."""
        states = {}
        customs = []
        for mod in self.modules.values():
            states[mod.id] = mod.to_save()
            if not mod.builtin:
                customs.append(mod.to_save())

        data = {"module_states": states, "custom_modules": customs}
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    # ── Genel API ──────────────────────────────────────────

    def get_modules(self) -> list[dict]:
        """Tüm modüllerin UI durumunu döndürür."""
        with self._lock:
            return [m.to_state() for m in self.modules.values()]

    def get_module(self, module_id: str) -> AIModule | None:
        return self.modules.get(module_id)

    def toggle_module(self, module_id: str, enabled: bool) -> bool:
        mod = self.modules.get(module_id)
        if not mod:
            return False
        mod.enabled = enabled
        if enabled and mod.download_status == "downloaded" and mod.model is None:
            threading.Thread(target=mod.load_model, daemon=True).start()
        elif not enabled:
            mod.unload_model()
        self._save_config()
        return True

    def set_confidence(self, module_id: str, value: float) -> bool:
        mod = self.modules.get(module_id)
        if not mod:
            return False
        mod.confidence = max(0.05, min(0.95, float(value)))
        self._save_config()
        return True

    def get_download_progress(self, module_id: str) -> float:
        mod = self.modules.get(module_id)
        if not mod:
            return 0.0
        return mod.download_progress

    # ── Model İndirme ──────────────────────────────────────

    def download_model(self, module_id: str) -> bool:
        """Modeli arka planda indirir / yükler."""
        mod = self.modules.get(module_id)
        if not mod:
            return False
        if mod.download_status == "downloading":
            return True

        threading.Thread(
            target=self._download_worker, args=(mod,), daemon=True
        ).start()
        return True

    def _download_worker(self, mod: AIModule):
        mod.download_status = "downloading"
        mod.download_progress = 0.0
        mod.download_error = ""

        try:
            if mod.model_type == "easyocr":
                self._install_easyocr(mod)
                return

            if mod.custom_url and mod.custom_url.startswith(("http://", "https://")):
                dest = CUSTOM_MODELS_DIR / mod.model_file
                self._download_file(mod.custom_url, str(dest), mod)
                mod.download_status = "downloaded"
                mod.download_progress = 100.0
                self._save_config()
                return

            # Yerleşik YOLO modelleri — ultralytics otomatik indirme
            if mod.model_file.startswith("yolov8"):
                from ultralytics import YOLO
                mod.download_progress = 10.0
                target_path = str(MODELS_DIR / mod.model_file)
                # ultralytics, dosya yoksa otomatik indirir
                _ = YOLO(mod.model_file)
                # İndirilen dosyayı models/ altına taşı (zaten yerinde olabilir)
                default_loc = Path(mod.model_file)
                if default_loc.exists() and not (MODELS_DIR / mod.model_file).exists():
                    default_loc.rename(MODELS_DIR / mod.model_file)
                mod.download_status = "downloaded"
                mod.download_progress = 100.0
                self._save_config()
                return

            # Özel model dosya yolu (yerel)
            if os.path.isfile(mod.model_file):
                mod.download_status = "downloaded"
                mod.download_progress = 100.0
                self._save_config()
                return

            mod.download_status = "error"
            mod.download_error = "Model dosyası bulunamadı ve indirme URL'si yok."

        except Exception as e:
            mod.download_status = "error"
            mod.download_error = str(e)[:200]

    def _download_file(self, url: str, dest: str, mod: AIModule):
        """URL'den dosya indirir, ilerleme takibi yapar."""
        req = urllib.request.Request(url, headers={"User-Agent": "INTERCO-GCS/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 65536

            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        mod.download_progress = round((downloaded / total) * 100.0, 1)

    def _install_easyocr(self, mod: AIModule):
        """EasyOCR paketini pip ile kurar."""
        mod.download_progress = 10.0
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "easyocr", "--quiet"],
                timeout=300,
            )
            mod.download_status = "downloaded"
            mod.download_progress = 100.0
            self._save_config()
        except Exception as e:
            mod.download_status = "error"
            mod.download_error = f"pip install easyocr başarısız: {str(e)[:150]}"

    # ── Özel Model Yönetimi ────────────────────────────────

    def add_custom_module(
        self, name: str, model_path_or_url: str, description: str = ""
    ) -> dict:
        """Kullanıcı tanımlı model ekler."""
        mid = "custom_" + uuid.uuid4().hex[:8]
        is_url = model_path_or_url.startswith(("http://", "https://"))

        if is_url:
            fname = model_path_or_url.split("/")[-1].split("?")[0]
            if not fname.endswith((".pt", ".onnx", ".engine")):
                fname = f"{mid}.pt"
            model_file = fname
            custom_url = model_path_or_url
            dl_status = "not_downloaded"
        else:
            if os.path.isfile(model_path_or_url):
                model_file = os.path.basename(model_path_or_url)
                dest = CUSTOM_MODELS_DIR / model_file
                if not dest.exists():
                    import shutil
                    shutil.copy2(model_path_or_url, str(dest))
                custom_url = ""
                dl_status = "downloaded"
            else:
                return {"ok": False, "error": f"Dosya bulunamadı: {model_path_or_url}"}

        defn = {
            "id": mid,
            "name": name or f"Özel Model ({model_file})",
            "description": description or "Kullanıcı tanımlı model",
            "model_file": model_file,
            "model_type": "yolo",
            "classes": None,
            "default_conf": 0.45,
            "builtin": False,
            "custom_url": custom_url,
        }
        mod = AIModule(defn)
        mod.download_status = dl_status

        with self._lock:
            self.modules[mid] = mod
        self._save_config()

        return {"ok": True, "module_id": mid, "module": mod.to_state()}

    def remove_custom_module(self, module_id: str) -> bool:
        """Özel modülü siler."""
        mod = self.modules.get(module_id)
        if not mod or mod.builtin:
            return False

        mod.unload_model()
        # Özel model dosyasını sil
        custom_path = CUSTOM_MODELS_DIR / mod.model_file
        if custom_path.exists():
            try:
                custom_path.unlink()
            except Exception:
                pass

        with self._lock:
            del self.modules[module_id]
        self._save_config()
        return True

    # ── Kare İşleme ───────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> list[dict]:
        """Aktif tüm modülleri kare üzerinde çalıştırır."""
        results = []
        for mod in list(self.modules.values()):
            if not mod.enabled:
                continue
            if mod.model is None:
                continue
            try:
                res = mod.process(frame)
                if res["detections"]:
                    results.append(res)
            except Exception:
                pass
        return results

    def draw_results_on_frame(self, frame: np.ndarray, results: list[dict]) -> np.ndarray:
        """Tüm modül sonuçlarını kare üzerine çizer."""
        overlay = frame.copy()

        # Modül başına renk
        palette = {
            "pose": (255, 100, 255),
            "person": (57, 255, 90),
            "multi_person": (57, 255, 90),
            "ocr": (56, 189, 248),
            "plate": (248, 189, 56),
            "vehicle": (100, 200, 255),
            "aircraft": (255, 90, 90),
            "animal": (100, 255, 200),
            "object": (200, 200, 200),
        }

        for module_result in results:
            mid = module_result["module_id"]
            mname = module_result["module_name"]
            color = palette.get(mid, (180, 180, 180))

            for det in module_result["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                label = det["label"]

                # Kutucuk
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

                # Etiket
                tag = f"[{mname[:6]}] {label} {int(conf*100)}%"
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(overlay, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
                cv2.putText(
                    overlay, tag, (x1 + 2, max(th + 2, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA
                )

                # Poz iskeleti çizimi
                extra = det.get("extra", {})
                keypoints = extra.get("keypoints", [])
                skeleton = extra.get("skeleton", POSE_SKELETON)
                if keypoints:
                    for ki, kp in enumerate(keypoints):
                        kx, ky, kc = kp["x"], kp["y"], kp["conf"]
                        if kc > 0.3 and kx > 0 and ky > 0:
                            kcolor = POSE_COLORS[ki % len(POSE_COLORS)]
                            cv2.circle(overlay, (kx, ky), 4, kcolor, -1)

                    for s_a, s_b in skeleton:
                        if s_a < len(keypoints) and s_b < len(keypoints):
                            ka, kb = keypoints[s_a], keypoints[s_b]
                            if ka["conf"] > 0.3 and kb["conf"] > 0.3 and ka["x"] > 0 and kb["x"] > 0:
                                cv2.line(
                                    overlay,
                                    (ka["x"], ka["y"]),
                                    (kb["x"], kb["y"]),
                                    color, 2
                                )

                # OCR metni
                if "text" in extra:
                    cv2.putText(
                        overlay, extra["text"],
                        (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (56, 189, 248), 1, cv2.LINE_AA
                    )

        return overlay


# Tekil yönetici örneği
ai_module_mgr = AIModuleManager()
