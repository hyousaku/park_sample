# 必要なライブラリのインポート
import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'  # PyTorchの警告を回避

import cv2  # OpenCV - カメラ映像処理用
import pygame  # 画面表示と音声再生用
import numpy as np  # 数値計算用
import torch  # PyTorch - YOLOモデル用
from ultralytics import YOLO  # YOLOv8モデル
import random  # ランダム音階配置用
import time  # 時間管理用

# 画面設定
SCREEN_WIDTH = 1920  # Full HD 幅
SCREEN_HEIGHT = 1080  # Full HD 高さ
GRID_COLS = 16  # グリッド列数（横16分割）
GRID_ROWS = 9  # グリッド行数（縦9分割）
CELL_WIDTH = SCREEN_WIDTH // GRID_COLS  # 各セルの幅
CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS  # 各セルの高さ
SOUND_CHANGE_INTERVAL = 12  # 音階配置変更間隔（秒）

# Dペンタトニックスケールの周波数（Hz）
D_PENTATONIC = [293.66, 329.63, 369.99, 440.00, 493.88, 587.33, 659.25, 739.99]

class PianoSound:
    """ピアノ音を生成・再生するクラス"""
    def __init__(self):
        # 音声ミキサーの初期化
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.sounds = {}  # 周波数ごとの音声データを保存
        # Dペンタトニックの各音を事前生成
        for freq in D_PENTATONIC:
            self.sounds[freq] = self.generate_piano_tone(freq, 0.3)
    
    def generate_piano_tone(self, frequency, duration):
        """ピアノ音を生成する"""
        sample_rate = 22050  # サンプリングレート
        n_samples = int(sample_rate * duration)  # サンプル数
        
        # 時間軸の生成
        t = np.linspace(0, duration, n_samples, False)
        
        # 基音と倍音を合成してピアノらしい音色を作る
        wave = np.sin(2 * np.pi * frequency * t)  # 基音
        wave += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)  # 2倍音
        wave += 0.25 * np.sin(2 * np.pi * frequency * 3 * t)  # 3倍音
        
        # エンベロープ（減衰）を適用
        envelope = np.exp(-3 * t / duration)
        wave = wave * envelope
        
        # 音量調整と整数化
        wave = wave * 0.3
        wave = np.int16(wave * 32767)
        
        # ステレオ音声に変換
        stereo_wave = np.zeros((n_samples, 2), dtype=np.int16)
        stereo_wave[:, 0] = wave  # 左チャンネル
        stereo_wave[:, 1] = wave  # 右チャンネル
        
        sound = pygame.sndarray.make_sound(stereo_wave)
        return sound
    
    def play(self, frequency):
        """指定周波数の音を再生"""
        if frequency in self.sounds:
            self.sounds[frequency].play()

class GridCell:
    """グリッドセル（16x9の各マス）を表すクラス"""
    def __init__(self, col, row):
        self.rect = pygame.Rect(col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)  # セルの矩形領域
        self.frequency = None  # このセルに割り当てられた音階
        self.color = None  # このセルの色
        self.active = False  # 頭部が検出されているか
        self.alpha = 50  # 透明度（50=半透明、255=不透明）

class PersonDetectionApp:
    """人検出アプリケーションのメインクラス"""
    def __init__(self):
        # Pygameの初期化
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Person Detection with YOLO")
        
        # YOLOv8-poseモデルの読み込み（頭部検出用）
        self.model = YOLO('yolov8n-pose.pt')
        
        # WEBカメラの初期化
        camera_index = int(os.environ.get("CAMERA_INDEX", 1))
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            raise RuntimeError("Failed to open camera device")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
        
        # ピアノ音の初期化
        self.piano = PianoSound()
        
        # 16x9のグリッドセルを生成（合計144セル）
        self.grid_cells = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                self.grid_cells.append(GridCell(col, row))
        
        # 各セルの音階をランダムに割り当て
        self.cell_triggered = {}  # 各セルの音が鳴っているかを記録
        self.assign_sounds_to_grid()
        self.last_sound_change = time.time()  # 最後に音階を変更した時刻
        
        # フレームレート管理用
        self.clock = pygame.time.Clock()
        self.running = True
    
    def assign_sounds_to_grid(self):
        """各グリッドセルにランダムな音階と色を割り当てる（12秒ごとに実行）"""
        self.cell_triggered = {}  # トリガー状態をリセット
        
        # 6色のカラーパレット
        colors = [
            (255, 100, 100),  # 赤系
            (100, 255, 100),  # 緑系
            (100, 100, 255),  # 青系
            (255, 255, 100),  # 黄系
            (255, 100, 255),  # マゼンタ系
            (100, 255, 255),  # シアン系
        ]
        
        # 各セルに音階と色を割り当て
        for i, cell in enumerate(self.grid_cells):
            cell.frequency = random.choice(D_PENTATONIC)  # Dペンタトニックからランダムに選択
            cell.color = colors[i % len(colors)]  # 色を順番に割り当て
            self.cell_triggered[i] = False  # 音が鳴っていない状態に初期化
    
    def detect_persons(self, frame):
        """YOLOv8-poseで人の頭部位置を検出"""
        results = self.model(frame, verbose=False)  # YOLOで推論実行
        
        head_positions = []  # 検出された頭部の位置リスト
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints  # キーポイント（骨格）データ取得
            for person_kp in keypoints:
                kp_data = person_kp.xy.cpu().numpy()  # キーポイント座標を取得
                if len(kp_data) > 0 and len(kp_data[0]) > 0:
                    nose = kp_data[0][0]  # 鼻の位置（頭部の中心として使用）
                    if nose[0] > 0 and nose[1] > 0:  # 有効な座標か確認
                        head_size = 40  # 頭部領域のサイズ（80x80ピクセル）
                        x1 = int(nose[0] - head_size)
                        y1 = int(nose[1] - head_size)
                        x2 = int(nose[0] + head_size)
                        y2 = int(nose[1] + head_size)
                        head_positions.append((x1, y1, x2, y2))
        
        return head_positions
    
    def check_grid_collision(self, head_positions):
        """頭部とグリッドセルの衝突判定を行い、音を鳴らす"""
        for i, cell in enumerate(self.grid_cells):
            cell.active = False  # まず非アクティブ状態にリセット
            
            # 各検出された頭部について
            for (x1, y1, x2, y2) in head_positions:
                head_center_x = (x1 + x2) // 2  # 頭部の中心X座標
                head_center_y = (y1 + y2) // 2  # 頭部の中心Y座標
                
                # 頭部の中心がこのセル内にあるか判定
                if cell.rect.collidepoint(head_center_x, head_center_y):
                    cell.active = True  # セルをアクティブ化
                    
                    # まだ音が鳴っていなければ音を鳴らす
                    if not self.cell_triggered[i]:
                        self.piano.play(cell.frequency)  # このセルの音階で演奏
                        self.cell_triggered[i] = True  # トリガー状態にする
                        cell.alpha = 150  # 透明度を上げる（半透明を維持）
                    break
            
            # 頭部が離れた場合
            if not cell.active:
                self.cell_triggered[i] = False  # トリガー解除
                # 透明度を徐々に元に戻す（フェードアウト）
                if cell.alpha > 50:
                    cell.alpha = max(50, cell.alpha - 10)
    
    def run(self):
        """メインループ"""
        while self.running:
            # イベント処理（ウィンドウ閉じる、ESCキーで終了）
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # カメラからフレームを取得
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # フレームをFull HDにリサイズ
            frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            
            # 人の頭部を検出
            head_positions = self.detect_persons(frame)
            
            # 頭部とグリッドの衝突判定・音再生
            self.check_grid_collision(head_positions)
            
            # 12秒ごとに音階配置を変更
            current_time = time.time()
            if current_time - self.last_sound_change >= SOUND_CHANGE_INTERVAL:
                self.assign_sounds_to_grid()
                self.last_sound_change = current_time
            
            # カメラ映像をpygame用に変換して描画
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.screen.blit(pygame.surfarray.make_surface(np.rot90(frame_rgb)), (0, 0))
            
            # アクティブなセルに半透明カラーを描画（左右反転）
            for cell in self.grid_cells:
                if cell.active:
                    cell_surface = pygame.Surface((cell.rect.width, cell.rect.height), pygame.SRCALPHA)
                    cell_surface.fill((*cell.color, int(cell.alpha)))
                    self.screen.blit(cell_surface, (SCREEN_WIDTH - cell.rect.x - cell.rect.width, cell.rect.y))
            
            # 画面更新
            pygame.display.flip()
            self.clock.tick(30)  # 30FPS
        
        self.cleanup()
    
    def cleanup(self):
        """終了処理"""
        self.cap.release()  # カメラを解放
        pygame.quit()  # Pygameを終了

# メインエントリーポイント
if __name__ == "__main__":
    app = PersonDetectionApp()
    app.run()
