# detector_ensemble.py (kept for completeness; not used in YuNet-only mode)
from typing import List, Tuple
BBox = Tuple[int,int,int,int]

def iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw<=0 or ah<=0 or bw<=0 or bh<=0: return 0.0
    ax2, ay2 = ax+aw, ay+ah
    bx2, by2 = bx+bw, by+bh
    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    if inter <= 0: return 0.0
    union = aw*ah + bw*bh - inter
    return inter / float(union) if union>0 else 0.0

def fuse_detections_iou(a: List[BBox], b: List[BBox], iou_merge_thresh: float = 0.30) -> List[BBox]:
    fused: List[BBox] = list(a)
    for bb in b:
        merged = False
        for i, fb in enumerate(list(fused)):
            if iou(bb, fb) >= iou_merge_thresh:
                ba, fa = bb[2]*bb[3], fb[2]*fb[3]
                fused[i] = bb if ba > fa else fb
                merged = True
                break
        if not merged:
            fused.append(bb)
    return fused
