# tree_clearance.py
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import laspy
except Exception as e:
    print("请先安装 laspy: pip install laspy\n错误:", e); sys.exit(1)

# 尝试用 SciPy 的 cKDTree；如果不可用，退化为块状暴力最近邻（慢一些）
try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

def nearest_blocked(A, B, block=100000, step=200000):
    # 纯 numpy 的分块最近邻（内存友好，但较慢）
    m = A.shape[0]
    best = np.full(m, np.inf, dtype=np.float64)
    for i in range(0, m, block):
        a = A[i:i+block]
        sub_best = np.full(a.shape[0], np.inf, dtype=np.float64)
        for j in range(0, B.shape[0], step):
            b = B[j:j+step]
            a2 = (a**2).sum(axis=1)[:, None]
            b2 = (b**2).sum(axis=1)[None, :]
            d2 = a2 + b2 - 2 * (a @ b.T)
            sub_best = np.minimum(sub_best, d2.min(axis=1))
        best[i:i+block] = np.sqrt(sub_best)
    return best

CLASS_NAMES = {
    0:"Created, never classified",1:"Unclassified",2:"Ground",
    3:"Low Vegetation",4:"Medium Vegetation",5:"High Vegetation",
    6:"Building",7:"Low Noise",8:"Key-point",9:"Water",
    10:"Rail",11:"Road Surface",12:"Reserved",
    13:"Wire-Guard",14:"Wire-Conductor",15:"Transmission Tower",16:"Wire-Structure Connector"
}

def main(in_path, clearance=4.0, out_prefix=None):
    in_path = Path(in_path)
    if out_prefix is None:
        out_prefix = in_path.stem

    las = laspy.read(in_path)
    cls = np.array(las.classification)

    # —— 类码统计
    uniq, cnt = np.unique(cls, return_counts=True)
    print("分类直方表：")
    for u, c in zip(uniq, cnt):
        print(f"  class {u:2d} ({CLASS_NAMES.get(int(u),'Unknown'):24s}): {c}")

    # —— 构造掩码
    veg_mask  = np.isin(cls, [3,4,5])       # 植被
    wire_mask = np.isin(cls, [14,13])       # 优先 14，其次 13

    veg_n, wire_n = int(veg_mask.sum()), int(wire_mask.sum())
    if veg_n == 0:
        print("❌ 文件中没有植被类 (3/4/5) 点，无法做树障分析。"); sys.exit(2)
    if wire_n == 0:
        print("❌ 文件中没有导线类 (14/13) 点；请先用分割/检测工具提取导线后再运行。"); sys.exit(3)

    V = np.vstack([las.x[veg_mask],  las.y[veg_mask],  las.z[veg_mask]]).T
    W = np.vstack([las.x[wire_mask], las.y[wire_mask], las.z[wire_mask]]).T
    print(f"\n植被点: {len(V):,}  导线点: {len(W):,}  净空阈值: {clearance} m")

    # —— 最近邻求距
    if HAVE_SCIPY:
        tree = cKDTree(W)
        dist, idx = tree.query(V, k=1, workers=-1)   # 最近导线点距离（米）
    else:
        print("未检测到 SciPy；使用 numpy 分块最近邻（较慢）")
        dist = nearest_blocked(V, W)

    enc_mask = dist < clearance
    enc_V, enc_D = V[enc_mask], dist[enc_mask]
    print(f"超限点数量: {len(enc_V):,}")

    # —— 输出 CSV
    csv_path = Path(f"{out_prefix}_encroachments.csv")
    pd.DataFrame({"x":enc_V[:,0],"y":enc_V[:,1],"z":enc_V[:,2],"clearance_m":enc_D}).to_csv(csv_path, index=False)
    print("已写出:", csv_path)

    # —— 写仅含超限点的 LAS（附加 clearance_m 额外维度）
    sub = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    sub.X, sub.Y, sub.Z = las.X[veg_mask][enc_mask], las.Y[veg_mask][enc_mask], las.Z[veg_mask][enc_mask]
    for attr in ["intensity","return_number","number_of_returns","scan_direction_flag","edge_of_flight_line",
                 "classification","scan_angle_rank","user_data","point_source_id","gps_time","red","green","blue"]:
        if hasattr(las, attr):
            arr = getattr(las, attr)
            try: setattr(sub, attr, arr[veg_mask][enc_mask])
            except: pass
    try:
        sub.add_extra_dim(laspy.ExtraBytesParams(name="clearance_m", type=np.float32))
        sub.clearance_m = enc_D.astype(np.float32)
    except Exception as e:
        print("写入 clearance_m 额外维度失败（可忽略）：", e)
    enc_las_path = Path(f"{out_prefix}_encroachment_points.las")
    sub.write(enc_las_path)
    print("已写出:", enc_las_path)

    # —— 可选：在原始 LAS 上为植被写回 clearance_m 字段（非超限也写，便于着色）
    try:
        full = las.copy()
        full.add_extra_dim(laspy.ExtraBytesParams(name="clearance_m", type=np.float32))
        tmp = np.full(len(full.x), np.nan, dtype=np.float32)
        tmp[np.where(veg_mask)[0]] = dist.astype(np.float32)
        full.clearance_m = tmp
        out_full = Path(f"{out_prefix}_with_clearance.las")
        full.write(out_full)
        print("已写出:", out_full)
    except Exception as e:
        print("回写 clearance_m 到完整 LAS 失败（可忽略）：", e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python tree_clearance.py <input.las|laz> [clearance_m] [out_prefix]")
        sys.exit(64)
    in_file = sys.argv[1]
    clearance = float(sys.argv[2]) if len(sys.argv) >= 3 else 4.0
    out_prefix = sys.argv[3] if len(sys.argv) >= 4 else None
    main(in_file, clearance, out_prefix)
