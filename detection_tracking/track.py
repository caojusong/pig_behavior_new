from ultralytics import YOLO
import cv2, numpy as np, os, pandas as pd
import time
from tqdm import tqdm  # 添加 tqdm 库用于进度条显示
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# —— 轨迹存储 —— 
trajectory_data = {}

def smooth_trajectory(traj, w=5):
    if len(traj) < w: return traj
    sm = []
    for i in range(len(traj)):
        s, e = max(0, i-w//2), min(len(traj), i+w//2+1)
        xs = [p[0] for p in traj[s:e]]; ys = [p[1] for p in traj[s:e]]
        sm.append((int(np.mean(xs)), int(np.mean(ys))))
    return sm

def update_traj_and_draw(frame, box, traj_len=20, smooth_w=5):
    tid = int(box.id.item())
    # 中心点
    cx, cy = map(int, box.xywh[0].cpu().numpy()[:2])
    traj = trajectory_data.setdefault(tid, [])
    traj.append((cx, cy))
    if len(traj) > traj_len: traj.pop(0)
    pts = smooth_trajectory(traj, smooth_w)
    # 画轨迹线
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], (0,255,0), 2)
    # 画 ID
    x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
    cv2.putText(frame, f"ID:{tid}", (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

def main():
    # ——— 配置区 ———
    weights     = '/home/caojs/ultralytics/runs/detect/train3/weights/best.pt'
    tracker_cfg = '/home/caojs/ultralytics-main/ultralytics/cfg/trackers/bytetrack.yaml'
    video_in    = '/home/caojs/ultralytics-main/trackout/MOT17-train/D06_20210908140809_49.5-54.5_fps30.mp4'
    txt_out     = '/home/caojs/ultralytics-main/trackout/MOT17-train/D06_20210908140809_49.5-54.5_fps30.txt'
    csv_out     = '/home/caojs/ultralytics-main/trackout/MOT17-train/D06_20210908140809_49.5-54.5_fps30_traj.csv'
    video_out   = '/home/caojs/ultralytics-main/trackout/MOT17-train/D06_20210908140809_49.5-54.5_fps30_out.mp4'
    os.makedirs(os.path.dirname(txt_out), exist_ok=True)

    # 初始化模型、视频、输出
    model = YOLO(weights)
    cap   = cv2.VideoCapture(video_in)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

    # 获取视频总帧数用于进度条
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建 tqdm 进度条
    progress_bar = tqdm(
        total=total_frames, 
        desc="处理视频", 
        unit="帧", 
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    # 打开 TXT 文件
    f_txt = open(txt_out, 'w')

    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 跟踪推理
        res = model.track(
            source=frame,
            tracker=tracker_cfg,
            imgsz=960,
            persist=True,
            conf=0.3,
            verbose=False,
            device=0
        )[0]

        # 遍历每个 box
        for box in res.boxes:
            # 如果没有 tracker id，就跳过
            if box.id is None:
                continue
            # 坐标
            x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
            cx, cy, w_, h_ = map(int, box.xywh[0].cpu().numpy())
            tid, conf = int(box.id.item()), float(box.conf.item())

            # -- 画检测框 --
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 写 MOT2D 格式到 TXT
            f_txt.write(f"{frame_idx},{tid},{x1},{y1},{w_},{h_},{conf:.2f},1,1,1\n")

            # 更新并画轨迹
            update_traj_and_draw(frame, box)

        # 写帧到输出视频
        writer.write(frame)
        
        # 更新进度条
        progress_bar.update(1)
        
        # 计算并显示估计剩余时间
        if frame_idx % 10 == 0:  # 每10帧更新一次估计
            elapsed_time = time.time() - start_time
            fps_current = frame_idx / elapsed_time
            remaining_frames = total_frames - frame_idx
            estimated_remaining = remaining_frames / fps_current if fps_current > 0 else 0
            progress_bar.set_postfix({
                'FPS': f'{fps_current:.2f}',
                '预计剩余时间': f'{estimated_remaining/60:.2f}分钟'
            })

    # 关闭进度条
    progress_bar.close()

    # 释放资源
    cap.release()
    writer.release()
    f_txt.close()

    # 保存轨迹到 CSV
    rows = []
    for tid, traj in trajectory_data.items():
        for idx, (x, y) in enumerate(traj):
            rows.append({'track_id': tid, 'frame': idx, 'x': x, 'y': y})
    pd.DataFrame(rows).to_csv(csv_out, index=False)

    # 计算总处理时间
    total_time = time.time() - start_time
    print(f"✔ 处理完成！总耗时: {total_time/60:.2f} 分钟")
    print(f"  • MOT TXT : {txt_out}")
    print(f"  • VIDEO   : {video_out}")
    print(f"  • CSV traj: {csv_out}")

if __name__ == "__main__":
    main()