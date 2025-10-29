# face_whitelist.py (face_recognition optional; ORB fallback)
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import os, json
import cv2
import numpy as np

try:
    import face_recognition
    _HAS_FR = True
except Exception:
    _HAS_FR = False

DB_DIR = Path("whitelist_db")
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_META = DB_DIR / "faces.json"

class FaceWhitelist:
    """Simple whitelist: add images with names; match detections by face_recognition or ORB."""
    def __init__(self):
        self.db: Dict[str, Dict[str, str]] = {}  # name -> {"image": path, "thumb": path}
        if DB_META.exists():
            try:
                self.db = json.loads(DB_META.read_text(encoding="utf-8"))
            except Exception:
                self.db = {}

    def save(self):
        try:
            DB_META.write_text(json.dumps(self.db, indent=2), encoding="utf-8")
        except Exception:
            pass

    def get_whitelisted_names(self) -> List[str]:
        return sorted(list(self.db.keys()))

    def get_thumbnail_path(self, name: str) -> Optional[str]:
        meta = self.db.get(name)
        if not meta:
            return None
        t = meta.get("thumb")
        return t if t and Path(t).exists() else None

    def add_face(self, image_path: str, name: str) -> bool:
        p = Path(image_path)
        if not p.exists():
            return False
        name = name.strip()
        if not name:
            return False
        # create thumbnail
        thumb = DB_DIR / f"{name}_thumb.jpg"
        try:
            img = cv2.imread(str(p))
            if img is None: return False
            th = cv2.resize(img, (96,96), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(thumb), th)
            self.db[name] = {"image": str(p), "thumb": str(thumb)}
            self.save()
            return True
        except Exception:
            return False

    def remove_face(self, name: str):
        if name in self.db:
            # do not delete originals; thumbnail can be removed optionally
            self.db.pop(name, None)
            self.save()

    # ---------- matching ----------

    def _encode_face_fr(self, bgr: np.ndarray, bbox) -> Optional[np.ndarray]:
        if not _HAS_FR: return None
        (x, y, w, h) = bbox
        rgb = bgr[:, :, ::-1]
        loc = [(y, x+w, y+h, x)]  # top, right, bottom, left
        try:
            encs = face_recognition.face_encodings(rgb, known_face_locations=loc, model="small")
            return encs[0] if encs else None
        except Exception:
            return None

    def _cos_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None: return 0.0
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _orb_score(self, patchA: np.ndarray, patchB: np.ndarray) -> float:
        try:
            orb = cv2.ORB_create(500)
            kp1, des1 = orb.detectAndCompute(patchA, None)
            kp2, des2 = orb.detectAndCompute(patchB, None)
            if des1 is None or des2 is None: return 0.0
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if not matches: return 0.0
            # invert avg distance -> similarity in [0,1] approx
            avg = sum(m.distance for m in matches) / len(matches)
            return max(0.0, min(1.0, 1.0 - avg/128.0))
        except Exception:
            return 0.0

    def match_detected_faces(self, bgr_img: np.ndarray, bboxes: List[tuple], threshold: float = 0.60):
        """Return list of dict: {'bbox':(x,y,w,h), 'is_whitelisted':bool, 'matched_name':str|None, 'confidence':float}"""
        results = []
        if not bboxes:
            return results
        names = list(self.db.keys())
        if not names:
            for bb in bboxes:
                results.append({'bbox': bb, 'is_whitelisted': False, 'matched_name': None, 'confidence': 0.0})
            return results

        # Prepare whitelist encodings or patches
        whitelist_encs = {}
        whitelist_patches = {}
        for name in names:
            meta = self.db.get(name, {})
            img_path = meta.get("image")
            if not img_path or not Path(img_path).exists():
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            if _HAS_FR:
                # naive: detect biggest face and encode once
                rgb = img[:, :, ::-1]
                locs = face_recognition.face_locations(rgb, model="hog")
                if locs:
                    encs = face_recognition.face_encodings(rgb, known_face_locations=[locs[0]], model="small")
                    if encs:
                        whitelist_encs[name] = encs[0]
                else:
                    # fallback: center crop thumbnail
                    h, w = img.shape[:2]
                    sz = min(h, w) // 2
                    cy, cx = h//2, w//2
                    patch = img[cy-sz:cy+sz, cx-sz:cx+sz]
                    whitelist_patches[name] = patch
            else:
                whitelist_patches[name] = img

        for bb in bboxes:
            x, y, w, h = bb
            face_patch = bgr_img[y:y+h, x:x+w]
            best_name = None
            best_conf = 0.0

            if _HAS_FR and whitelist_encs:
                det_enc = self._encode_face_fr(bgr_img, bb)
                if det_enc is not None:
                    for name, enc in whitelist_encs.items():
                        sim = self._cos_sim(det_enc, enc)  # cosine in [0,1] approx (face_rec encs aren't unit-norm by default)
                        if sim > best_conf:
                            best_conf, best_name = sim, name
            else:
                # ORB fallback on grayscale patches
                det_gray = cv2.cvtColor(face_patch, cv2.COLOR_BGR2GRAY)
                for name, patch in whitelist_patches.items():
                    base_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    score = self._orb_score(det_gray, base_gray)
                    if score > best_conf:
                        best_conf, best_name = score, name

            is_white = best_conf >= threshold
            results.append({'bbox': bb, 'is_whitelisted': is_white, 'matched_name': (best_name if is_white else None), 'confidence': best_conf})
        return results
