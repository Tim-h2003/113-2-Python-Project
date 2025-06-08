import sys
import os
import glob
import time
import threading
import cv2
import numpy as np

from PyQt6 import QtCore, QtWidgets
from PyQt6.QtWidgets import (
    QApplication, QTextEdit, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt, QTimer, QRectF

from ultralytics import YOLO  

from carUI import Ui_Form  


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def clear_folder(folder):
    if os.path.exists(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.is_running = False

        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "best.pt")
        print(f"Looking for model at: {model_path}")

        self.model = YOLO(model_path)

        self.timer = None          
        self.cap = None            
        self.scene = None          
        self.image_item = None     

        self.scene = QGraphicsScene()
        self.ui.playimage.setScene(self.scene)

        self.image_pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_pixmap_item)

        self.is_drawing_line = False
        self.line_pts = []
        self.recorded_lines = []  

        self.last_displayed_image = None

        self.ui.yessit.clicked.connect(self.capture_image)       
        self.ui.drawline.clicked.connect(self.enable_draw_line)  
        self.ui.start.clicked.connect(self.start_process)        
        self.ui.yesline.clicked.connect(self.confirm_lines)      
        
        self.ui.replay.setObjectName("replay")
        self.ui.replay.clicked.connect(self.replay_images)

        self.ui.rawimage.isChecked()
        self.ui.annoimage.isChecked()

        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)

        self.site_url_map = {
            "國道10號 0K+746 鼎金系統交流道到左營端": "https://cctvs.freeway.gov.tw/live-view/mjpg/video.cgi?camera=110&t=0.399633067787975",
            "國道6號 4K+205 舊正交流道到東草屯交流道": "https://cctvc.freeway.gov.tw/abs2mjpg/bmjpg?camera=374",
            "國道1號 22K+900 內湖交流道到圓山交流道": "https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=12290&t=0.01836966182358668",
        }



    def start_process(self):
        if len(self.line_pts) < 2:
            self.ui.working.append("[請先畫好至少一組線段再開始偵測]")
            return

        self.is_running = True
        self.ui.working.append("[開始擷取與偵測流程]")

        self.capture_images2()
        if not self.is_running:  
            self.ui.working.append("[流程中斷]")
            return

        self.detect_and_track()
        self.ui.working.append("擷取與偵測流程執行完畢")

    def stop_process(self):
        self.is_running = False
        self.ui.working.append("已停止流程")


    def capture_image(self):
        location_text = self.ui.sits.currentText()

        stream_url = self.site_url_map.get(location_text, None)

        if stream_url is None:
            self.ui.working.setPlainText("[無法取得串流網址，請確認選項是否正確]")
            return

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            self.ui.working.setPlainText(f"[無法開啟串流：{stream_url}]")
            return

        ret, frame = cap.read()
        cap.release()

        if ret:
            self.ui.working.setPlainText("[成功擷取影像]")
            self.last_displayed_image = frame.copy()  
            self.show_image_in_view(frame)
        else:
            self.ui.working.setPlainText("[擷取影像失敗]")

    def show_image_in_view(self, img):
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_pixmap_item.setPixmap(pixmap)
        self.scene.update()

    def enable_draw_line(self):
        if self.last_displayed_image is None:
            self.ui.working.setPlainText("[請先擷取影像]")
            return

        self.line_pts.clear()
        self.is_drawing_line = True
        self.ui.working.setPlainText("請點擊畫線點（共4點，2條線），按下 Enter 鍵完成")

        self.ui.playimage.viewport().setMouseTracking(True)
        self.ui.playimage.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if self.is_drawing_line and event.type() == QtCore.QEvent.Type.MouseButtonPress:
            pos = event.position()
            scene_pos = self.ui.playimage.mapToScene(int(pos.x()), int(pos.y()))
            self.line_pts.append((int(scene_pos.x()), int(scene_pos.y())))
            self.update_lines_on_image()

            if len(self.line_pts) == 4:
                self.is_drawing_line = False
                self.ui.playimage.viewport().removeEventFilter(self)
                self.ui.working.setPlainText(f"畫線完成，共點選 {len(self.line_pts)} 點")

            return True  

        if self.is_drawing_line and event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
                if len(self.line_pts) == 4:
                    self.is_drawing_line = False
                    self.ui.playimage.viewport().removeEventFilter(self)
                    self.ui.working.setPlainText(f"畫線完成，共點選 {len(self.line_pts)} 點")
                else:
                    self.ui.working.setPlainText("[請點選完整4點再結束]")
                return True

        return super().eventFilter(obj, event)

    def update_lines_on_image(self):
        if self.last_displayed_image is None:
            return

        image = self.last_displayed_image.copy()

        for pt in self.line_pts:
            cv2.circle(image, pt, 2, (0, 0, 255), -1)

        if len(self.line_pts) >= 2:
            cv2.line(image, self.line_pts[0], self.line_pts[1], (0, 255, 0), 1)
        if len(self.line_pts) == 4:
            cv2.line(image, self.line_pts[2], self.line_pts[3], (255, 0, 0), 1)

        self.show_image_in_view(image)

    def confirm_lines(self):
        
        if len(self.line_pts) == 4:
            self.recorded_lines = self.line_pts.copy()

            self.ui.working.setPlainText(f"畫線完成，共點選 {len(self.line_pts)} 點\n線條座標：{self.recorded_lines}")

            self.is_drawing_line = False
            self.ui.playimage.viewport().removeEventFilter(self)

        else:
            self.ui.working.setPlainText("線條未完成，請點選完整4點")

    
    def capture_images2(self):
        location_text = self.ui.sits.currentText()
        stream_url = self.site_url_map.get(location_text, None)
        self.ui.working.append(f"串流來源：{stream_url}")

        self.save_folder_raw = "raw_frames"
        self.save_folder_annotated = "annotated_frames"
        self.total_frames = 300
        self.frame_interval = 0.05  

        ensure_folder(self.save_folder_raw)
        ensure_folder(self.save_folder_annotated)
        clear_folder(self.save_folder_raw)
        clear_folder(self.save_folder_annotated)

        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            self.ui.working.append("[無法開啟串流，請檢查來源。]")
            return

        frame_count = 0
        last_capture_time = 0
        self.ui.working.setPlainText("[開始擷取影像...]\n")
        QApplication.processEvents()

        while frame_count < self.total_frames and self.is_running:
            ret, frame = cap.read()
            if not ret:
                self.ui.working.append("[串流錯誤，重新連線中...]")
                QApplication.processEvents()
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(stream_url)
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(
                self.ui.playimage.width(),
                self.ui.playimage.height(),
                Qt.AspectRatioMode.KeepAspectRatio
            )
            self.image_pixmap_item.setPixmap(pixmap)

            now = time.time()
            if now - last_capture_time >= self.frame_interval:
                filename = os.path.join(self.save_folder_raw, f"frame_{frame_count:03d}.png")
                cv2.imwrite(filename, frame)
                self.ui.working.append(f"[擷取] {filename}")
                frame_count += 1
                last_capture_time = now

            QApplication.processEvents()
            cv2.waitKey(1)

        cap.release()
        self.ui.working.append("[圖片擷取完成]\n")
        QApplication.processEvents()

        self.ui.working.append("[正在將圖片合成影片...]")
        QApplication.processEvents()

        image_folder = self.save_folder_raw
        video_output_path = os.path.join(image_folder, "output_video.mp4")
        self.fps =  10                                       

        image_files = sorted([         
            f for f in os.listdir(image_folder)
            if f.endswith('.png')
        ])

        if not image_files:
            self.ui.working.append("[沒有圖片可以合成影片。]")
            return

        first_image_path = os.path.join(image_folder, image_files[0])
        frame = cv2.imread(first_image_path)
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_output_path, fourcc, self.fps, (width, height))

        for img_file in image_files:
            img_path = os.path.join(image_folder, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                video.write(img)

        video.release()
        self.ui.working.append(f"[影片已儲存為：{video_output_path}]")
        QApplication.processEvents()

    def detect_and_track(self):
        video_path = os.path.join(self.save_folder_raw, "output_video.mp4")

        if not os.path.exists(video_path):
            self.ui.working.append("[找不到影片檔案，請先合成影片]")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.ui.working.append(f"[無法開啟影片：{video_path}]")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_video_path = os.path.join(self.save_folder_annotated, "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def is_crossing(p1, p2, l1, l2):
            return ccw(p1, l1, l2) != ccw(p2, l1, l2) and ccw(p1, p2, l1) != ccw(p1, p2, l2)

        self.ui.working.append("[開始偵測與追蹤...]\n")



        track_history = {}
        counted_ids = [set() for _ in range(len(self.line_pts) // 2)]
        cross_counts = [0] * (len(self.line_pts) // 2)
        frame_idx = 0

        while True:
            ret, img = cap.read()
            if not ret:
                break

            img_for_annotate = img.copy()
            

            current_dir = os.path.dirname(os.path.abspath(__file__))
            tracker_path = os.path.join(current_dir, "botsort.yaml")
            img = cv2.resize(img, (352, 240))
            results = self.model.track(
                source=img,
                persist=True,
                conf=0.1,
                iou=0.7,
                tracker=tracker_path,
                device=0 
            )

            log_text = results[0].verbose()  
            self.ui.working.append(log_text)

            if results and results[0].boxes.id is not None:
                for box, track_id in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.id.cpu().tolist()):
                    x1, y1, x2, y2 = map(int, box[:4])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append((cx, cy))

                    for i in range(len(self.line_pts) // 2):
                        if len(track_history[track_id]) >= 2:
                            prev = track_history[track_id][-2]
                            curr = track_history[track_id][-1]
                            l1, l2 = self.line_pts[i * 2], self.line_pts[i * 2 + 1]
                            if is_crossing(prev, curr, l1, l2) and track_id not in counted_ids[i]:
                                cross_counts[i] += 1
                                counted_ids[i].add(track_id)
                                msg = f"車輛 {track_id} 跨越線 {i+1}！目前總數：{cross_counts[i]}"
                                self.ui.working.append(msg)

                    cv2.rectangle(img_for_annotate, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_for_annotate, f"ID: {track_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            for tid, points in track_history.items():
                for j in range(1, len(points)):
                    cv2.line(img_for_annotate, points[j - 1], points[j], (255, 0, 0), 1)

            for i in range(0, len(self.line_pts), 2):
                cv2.line(img_for_annotate, self.line_pts[i], self.line_pts[i + 1], (0, 255, 0), 1)

            avg = sum(cross_counts) / max(1, len(self.line_pts) // 2)
            duration_sec = frame_idx / self.fps
            duration_hr = duration_sec / 3600
            traffic_flow = avg / duration_hr if duration_hr > 0 else 0


            text_output = ""

            if len(cross_counts) > 0:
                text_output += f"Line1: {cross_counts[0]}\n"

            if len(cross_counts) > 1:
                text_output += f"Line2: {cross_counts[1]}\n"

            text_output += f"Average: {avg:.1f}\n"
            text_output += f"Flow: {traffic_flow:.1f} veh/hr"

            self.ui.output.setText(text_output)

            out.write(img_for_annotate)
            frame_idx += 1
            QApplication.processEvents()

        cap.release()
        out.release()

        summary = (
            f"最終計數結果：\n"
            f"線1 總車輛數：{cross_counts[0]}\n"
            f"線2 總車輛數：{cross_counts[1] if len(cross_counts) > 1 else 0}\n"
            f"平均數量：{avg:.1f}\n"
            f"車流量：{traffic_flow:.1f} veh/hr\n"
            f"標註影片已儲存至：{out_video_path}"
        )
        self.ui.working.append(summary)


    def replay_images(self):
        print("replay_images triggered")
        self.ui.working.append("開始播放影片")

        if self.timer is not None:
            self.timer.stop()
            self.timer.deleteLater()
            self.timer = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.ui.rawimage.isChecked():
            video_path = os.path.join("raw_frames", "output_video.mp4")
        elif self.ui.annoimage.isChecked():
            video_path = os.path.join("annotated_frames", "annotated_output.mp4")
        else:
            self.ui.working.append("請先選擇要播放的影片類型（原始或標註）")
            return

        if not os.path.exists(video_path):
            self.ui.working.append(f"找不到影片檔案：{video_path}")
            return

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.ui.working.append(f"無法開啟影片：{video_path}")
            self.cap = None
            return

        if self.scene is None or self.image_item is None:
            self.scene = QGraphicsScene(self.ui.playimage)
            self.image_item = QGraphicsPixmapItem()
            self.scene.addItem(self.image_item)
            self.ui.playimage.setScene(self.scene)

        interval = int(1000 / self.fps)  
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.show_next_frame)
        self.timer.start(interval)
        print(f"播放間隔 {interval} ms, FPS: {self.fps}")


    def show_next_frame(self):
        print("show_next_frame called")
        if self.cap is None or not self.cap.isOpened():
            print("cap invalid")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.ui.working.append("影片播放完畢")
            print("video ended")
            self.cap.release()
            self.cap = None

            if self.timer is not None:
                self.timer.stop()
                self.timer.deleteLater()
                self.timer = None

            if self.image_item is not None:
                self.image_item.setPixmap(QPixmap())  

            return


        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        self.image_item.setPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        self.ui.playimage.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("車流計算")
    window.show()
    sys.exit(app.exec())
