import os
# main.py
import sys
import signal
import time
import asyncio
import webbrowser
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

from config import cfg
from mavlink_handler import MavlinkController
from vision_engine import VisionSystem
from cpdd_gimbal import cpdd_ctrl
from ai_modules import ai_module_mgr

app = FastAPI(title="INTERCO CPDD075 + 3D Cesium GCS")

mav_ctrl = MavlinkController()
vision_sys = VisionSystem(mav_ctrl)

class LockRequest(BaseModel):
    target_id: int

class CameraRequest(BaseModel):
    mode: str

class FlightCmdRequest(BaseModel):
    cmd: str
    alt: float = 50.0

class ModeRequest(BaseModel):
    mode: str

class CpddCmdRequest(BaseModel):
    action: str

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

async def mjpeg_generator():
    while True:
        jpg = vision_sys.latest_annotated_jpeg
        if jpg:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
        await asyncio.sleep(0.033)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
    )

@app.post("/api/lock")
async def api_lock_target(req: LockRequest):
    vision_sys.lock_target(req.target_id)
    return {"status": "ENGAGED", "target_id": req.target_id}

@app.post("/api/abort")
async def api_abort():
    vision_sys.unlock_target()
    return {"status": "POSHOLD", "engaged": False, "locked_target_id": None}

@app.post("/api/set_mode")
async def api_set_mode(req: ModeRequest):
    vision_sys.locked_target_id = None
    mav_ctrl.set_flight_mode(req.mode)
    return {"status": "OK", "mode": req.mode.upper()}

@app.post("/api/flight_cmd")
async def api_flight_cmd(req: FlightCmdRequest):
    cmd = req.cmd.upper().strip()
    if cmd == "RTL":
        vision_sys.locked_target_id = None
        mav_ctrl.command_rtl()
    elif cmd == "TAKEOFF":
        mav_ctrl.command_takeoff(req.alt)
    elif cmd == "LAND":
        vision_sys.locked_target_id = None
        mav_ctrl.command_land()
    elif cmd == "POSHOLD":
        vision_sys.locked_target_id = None
        mav_ctrl.command_poshold()
    return {"status": "OK", "cmd": cmd}

@app.post("/api/camera")
async def api_set_camera(req: CameraRequest):
    vision_sys.set_camera_mode(req.mode)
    return {
        "status": "OK",
        "active_camera": vision_sys.active_cam_key,
        "cpdd_view": cpdd_ctrl.current_view
    }

@app.post("/api/cpdd/cmd")
async def api_cpdd_cmd(req: CpddCmdRequest):
    act = req.action.upper().strip()
    if act == "ZOOM_IN":
        cpdd_ctrl.zoom_step(1)
    elif act == "ZOOM_OUT":
        cpdd_ctrl.zoom_step(-1)
    elif act == "IR_PALETTE":
        cpdd_ctrl.switch_ir_palette()
    elif act == "CENTER":
        cpdd_ctrl.gimbal_center()
    elif act == "FUSION":
        vision_sys.set_camera_mode("FUSION")
    return {
        "status": "OK",
        "action": act,
        "zoom_level": cpdd_ctrl.zoom_level,
        "ir_palette": cpdd_ctrl.ir_palette_idx,
        "cpdd_view": cpdd_ctrl.current_view
    }


class CpddRawRequest(BaseModel):
    mode: str = "TEXT"   # "TEXT" veya "HEX"
    text_cmd: str = ""
    type_b: str = "2F"
    type_cd: str = "01 00 00 00"

@app.post("/api/cpdd/raw")
async def api_cpdd_raw(req: CpddRawRequest):
    if req.mode.upper() == "HEX":
        ok = await asyncio.to_thread(cpdd_ctrl.send_custom_hex, req.type_b, req.type_cd)
        return {"ok": ok, "reply": f"HEX GÖNDERİLDİ -> TypeB:{req.type_b} TypeCD:{req.type_cd}"}
    else:
        reply = await asyncio.to_thread(cpdd_ctrl.send_text_cmd, req.text_cmd, True)
        return {"ok": True, "reply": reply if reply else "(UDP gönderildi, metin yanıtı dönmedi)"}

@app.get("/api/scan_webcams")
async def api_scan_webcams():
    cams = await asyncio.to_thread(vision_sys.scan_local_webcams)
    return {"webcams": cams}

@app.get("/api/dataset_folders")
async def api_dataset_folders():
    folders = await asyncio.to_thread(vision_sys.scan_dataset_folders)
    return {"folders": folders}

# ── AI Modül API ──────────────────────────────────────────

@app.get("/api/ai_modules")
async def get_ai_modules():
    return {"modules": ai_module_mgr.get_modules()}

@app.post("/api/ai_modules/{module_id}/toggle")
async def toggle_ai_module(module_id: str, payload: dict):
    ok = ai_module_mgr.toggle_module(module_id, bool(payload.get("enabled", False)))
    return {"ok": ok}

@app.post("/api/ai_modules/{module_id}/config")
async def set_ai_module_config(module_id: str, payload: dict):
    if "confidence" in payload:
        ai_module_mgr.set_confidence(module_id, float(payload["confidence"]))
    return {"ok": True}

@app.post("/api/ai_modules/{module_id}/download")
async def download_ai_model(module_id: str):
    ok = ai_module_mgr.download_model(module_id)
    return {"ok": ok}

@app.get("/api/ai_modules/{module_id}/download_status")
async def get_download_status(module_id: str):
    mod = ai_module_mgr.get_module(module_id)
    if not mod:
        return {"status": "error", "progress_pct": 0}
    return {
        "status": mod.download_status,
        "progress_pct": round(mod.download_progress, 1),
        "error": mod.download_error,
    }

@app.post("/api/ai_modules/{module_id}/delete_model")
async def delete_ai_model(module_id: str):
    mod = ai_module_mgr.get_module(module_id)
    if not mod:
        return {"ok": False}
    mod.unload_model()
    mod.enabled = False
    from pathlib import Path
    from ai_modules import MODELS_DIR, CUSTOM_MODELS_DIR
    for d in [CUSTOM_MODELS_DIR, MODELS_DIR]:
        p = d / mod.model_file
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    mod.download_status = "not_downloaded"
    mod.download_progress = 0.0
    ai_module_mgr._save_config()
    return {"ok": True}

@app.post("/api/ai_modules/custom")
async def add_custom_module(payload: dict):
    result = ai_module_mgr.add_custom_module(
        name=payload.get("name", ""),
        model_path_or_url=payload.get("model_url", ""),
        description=payload.get("description", ""),
    )
    return result

@app.delete("/api/ai_modules/custom/{module_id}")
async def remove_custom_module(module_id: str):
    ok = ai_module_mgr.remove_custom_module(module_id)
    return {"ok": ok}

# ── Klasör Yönetimi API ──────────────────────────────────

@app.get("/api/folders")
async def get_folders():
    folders = cfg.data.get("folders", {})
    return {"folders": folders}

@app.post("/api/folders")
async def create_folder(payload: dict):
    name = payload.get("name", "").strip()
    if not name:
        return {"ok": False, "error": "Klasör adı gerekli"}
    folders = cfg.data.setdefault("folders", {})
    folder_id = "folder_" + name.lower().replace(" ", "_").replace("/", "_")[:20]
    if folder_id in folders:
        return {"ok": False, "error": "Bu isimde klasör zaten var"}
    folders[folder_id] = {"name": name, "cameras": []}
    cfg.save(cfg.data)
    return {"ok": True, "folder_id": folder_id}

@app.delete("/api/folders/{folder_id}")
async def delete_folder(folder_id: str):
    folders = cfg.data.get("folders", {})
    if folder_id not in folders:
        return {"ok": False, "error": "Klasör bulunamadı"}
    del folders[folder_id]
    cfg.save(cfg.data)
    return {"ok": True}

@app.post("/api/folders/{folder_id}/cameras")
async def add_camera_to_folder(folder_id: str, payload: dict):
    folders = cfg.data.get("folders", {})
    if folder_id not in folders:
        return {"ok": False, "error": "Klasör bulunamadı"}
    cam_key = payload.get("camera_key", "")
    if cam_key and cam_key not in folders[folder_id]["cameras"]:
        folders[folder_id]["cameras"].append(cam_key)
        cfg.save(cfg.data)
    return {"ok": True}

@app.delete("/api/folders/{folder_id}/cameras/{camera_key}")
async def remove_camera_from_folder(folder_id: str, camera_key: str):
    folders = cfg.data.get("folders", {})
    if folder_id not in folders:
        return {"ok": False}
    if camera_key in folders[folder_id]["cameras"]:
        folders[folder_id]["cameras"].remove(camera_key)
        cfg.save(cfg.data)
    return {"ok": True}

# ── AI Profil Yönetimi API ────────────────────────────────

PROFILES_DIR = os.path.join(os.path.dirname(__file__), "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

@app.get("/api/ai_profiles")
async def list_ai_profiles():
    profiles = []
    for f in sorted(os.listdir(PROFILES_DIR)):
        if f.endswith(".json"):
            profiles.append(f[:-5])
    return {"profiles": profiles}

@app.post("/api/ai_profiles/save")
async def save_ai_profile(payload: dict):
    import json as _json
    name = payload.get("name", "").strip()
    if not name:
        return {"ok": False, "error": "Profil adı gerekli"}
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "" for c in name).strip()
    if not safe_name:
        return {"ok": False, "error": "Geçersiz profil adı"}
    modules_state = ai_module_mgr.get_modules()
    profile_data = {
        "name": safe_name,
        "modules": {m["id"]: {"confidence": m["confidence"], "enabled": m["enabled"]} for m in modules_state},
    }
    path = os.path.join(PROFILES_DIR, f"{safe_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(profile_data, f, indent=2, ensure_ascii=False)
    return {"ok": True, "profile": safe_name}

@app.post("/api/ai_profiles/load")
async def load_ai_profile(payload: dict):
    import json as _json
    name = payload.get("name", "").strip()
    path = os.path.join(PROFILES_DIR, f"{name}.json")
    if not os.path.isfile(path):
        return {"ok": False, "error": "Profil bulunamadı"}
    with open(path, "r", encoding="utf-8") as f:
        profile = _json.load(f)
    for mid, state in profile.get("modules", {}).items():
        if "confidence" in state:
            ai_module_mgr.set_confidence(mid, state["confidence"])
        if "enabled" in state:
            ai_module_mgr.toggle_module(mid, state["enabled"])
    return {"ok": True, "applied": name}

@app.delete("/api/ai_profiles/{profile_name}")
async def delete_ai_profile(profile_name: str):
    path = os.path.join(PROFILES_DIR, f"{profile_name}.json")
    if os.path.isfile(path):
        os.remove(path)
        return {"ok": True}
    return {"ok": False, "error": "Profil bulunamadı"}

# ── Ayarlar API ───────────────────────────────────────────

## ── AI Modülleri API ────────────────────────────────────────

@app.get("/api/ai_modules")
async def api_ai_modules():
    return {"modules": ai_module_mgr.get_modules()}

class AiToggleReq(BaseModel):
    enabled: bool

@app.post("/api/ai_modules/{module_id}/toggle")
async def api_ai_toggle(module_id: str, req: AiToggleReq):
    ok = ai_module_mgr.toggle_module(module_id, req.enabled)
    return {"ok": ok}

class AiConfReq(BaseModel):
    confidence: float

@app.post("/api/ai_modules/{module_id}/config")
async def api_ai_config(module_id: str, req: AiConfReq):
    ok = ai_module_mgr.set_confidence(module_id, req.confidence)
    return {"ok": ok}

@app.post("/api/ai_modules/{module_id}/download")
async def api_ai_download(module_id: str):
    ok = ai_module_mgr.download_model(module_id)
    return {"ok": ok}

@app.get("/api/ai_modules/{module_id}/download_status")
async def api_ai_dl_status(module_id: str):
    mod = ai_module_mgr.get_module(module_id)
    if not mod:
        return {"status": "unknown", "progress_pct": 0}
    return {
        "status": mod.download_status,
        "progress_pct": round(mod.download_progress, 1),
        "error": mod.download_error,
    }

@app.post("/api/ai_modules/{module_id}/delete_model")
async def api_ai_delete_model(module_id: str):
    mod = ai_module_mgr.get_module(module_id)
    if not mod:
        return {"ok": False}
    mod.unload_model()
    mod.enabled = False
    path = mod._resolve_model_path()
    if path and os.path.isfile(path) and "models" in path:
        try:
            os.remove(path)
        except Exception:
            pass
    mod.download_status = "not_downloaded"
    mod.download_progress = 0.0
    return {"ok": True}

class AiCustomReq(BaseModel):
    name: str
    model_url: str
    model_type: str = "yolov8_detect"
    default_confidence: float = 0.5

@app.post("/api/ai_modules/custom")
async def api_ai_add_custom(req: AiCustomReq):
    result = ai_module_mgr.add_custom_module(req.name, req.model_url, "")
    return result

@app.delete("/api/ai_modules/custom/{module_id}")
async def api_ai_remove_custom(module_id: str):
    ok = ai_module_mgr.remove_custom_module(module_id)
    return {"ok": ok}

## ── Ayarlar API ─────────────────────────────────────────────

@app.get("/api/settings")
async def get_settings():
    return cfg.data

@app.post("/api/settings/save_and_test")
async def save_and_test_settings(payload: dict):
    cfg.save(payload)
    mav_ctrl.reconnect()
    vision_sys.reload_all()
    await asyncio.sleep(1.2)
    conn_str = cfg.get_mavlink_connection_string()
    mav_test = mav_ctrl.test_connection(conn_str)
    vis_test = vision_sys.test_cameras_and_model()
    return {
        "saved": True,
        "mavlink": mav_test,
        "vision": vis_test
    }

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            stream = vision_sys._get_active_stream()
            active_cam_online = bool(stream and stream.ret)
            active_cam_status = stream.status if stream else "NO VIDEO"
            active_cam_error = stream.error_detail if stream else ""

            payload = {
                "telemetry": mav_ctrl.telemetry,
                "active_camera": vision_sys.active_cam_key,
                "active_cam_online": active_cam_online,
                "active_cam_status": active_cam_status,
                "active_cam_error": active_cam_error,
                "auto_camera": vision_sys.auto_switch_enabled,
                "locked_target_id": vision_sys.locked_target_id,
                "targets": vision_sys.detected_targets,
                "model_status": "READY" if any(m["enabled"] for m in ai_module_mgr.get_modules()) else "STANDBY",
                "perf": {
                    "cam_fps": stream.stream_fps if stream else 0.0,
                    "out_fps": vision_sys.pipeline_fps,
                    "infer_ms": vision_sys.infer_ms,
                    "device": vision_sys.compute_device,
                    "decoder": stream.decoder_type if stream else "NONE",
                    "res": f"{stream.frame.shape[1]}x{stream.frame.shape[0]}" if (stream and stream.frame is not None) else "-"
                },
                "ai_modules": {
                    "active": [m["id"] for m in ai_module_mgr.get_modules() if m["enabled"]],
                    "results": getattr(vision_sys, "ai_module_results", {}),
                    "downloads": {
                        m["id"]: {"status": m["download_status"], "pct": m["download_progress"]}
                        for m in ai_module_mgr.get_modules()
                        if m["download_status"] == "downloading"
                    },
                },
                "logs": sorted(
                    list(vision_sys.logs) + list(mav_ctrl.logs),
                    key=lambda x: x["t"] if isinstance(x, dict) else 0
                )[-80:],
                "cpdd": {
                    "view": cpdd_ctrl.current_view,
                    "zoom": cpdd_ctrl.zoom_level,
                    "palette": cpdd_ctrl.ir_palette_idx,
                    "status": cpdd_ctrl.last_status
                }
            }
            await websocket.send_json(payload)
            await asyncio.sleep(0.04)
    except WebSocketDisconnect:
        pass

def _shutdown(signum=None, frame=None):
    """CTRL+C ile temiz kapanış."""
    print("\n[INTERCO] Kapatılıyor...")
    vision_sys.running = False
    vision_sys.main_stream.running = False
    vision_sys.test_stream.running = False
    mav_ctrl.running = False
    # Kısa süre bekle ki thread'ler kapansın
    time.sleep(0.3)
    os._exit(0)

if __name__ == "__main__":
    # CTRL+C handler — Windows ve Linux'ta çalışır
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    port = int(cfg.data.get("web_port", 8005))
    host = cfg.data.get("web_host", "0.0.0.0")
    print(f"INTERCO CPDD075 GCS AKTIF -> http://{host}:{port}")
    print(f"Kapatmak için CTRL+C basın")
    if os.environ.get("INTERCO_HEADLESS", "0") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()
    try:
        uvicorn.run("main:app", host=host, port=port, reload=False, log_level="warning")
    except (KeyboardInterrupt, SystemExit):
        _shutdown()
