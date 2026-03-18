# 必要なライブラリのインポート
import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'  # PyTorchの警告を回避

import cv2  # OpenCV - カメラ映像処理用
import pygame  # 画面表示と音声再生用
import numpy as np  # 数値計算用
import torch  # PyTorch - YOLOモデル用
from ultralytics import YOLO  # YOLOv8モデル
import random  # ランダム配置用
import math  # 数学関数用

# ========== チューニング可能なパラメータ ==========
# 画面設定
SCREEN_WIDTH = 1920  # Full HD 幅
SCREEN_HEIGHT = 1080  # Full HD 高さ

# 円の設定
CIRCLE_MIN_RADIUS = 80  # 円の最小半径（px）
CIRCLE_MAX_RADIUS = 120  # 円の最大半径（px）
CIRCLE_COUNT = 5  # 画面上に常時表示する円の数
CIRCLE_SPEED_MIN = 0.5  # 円の最小移動速度（px/frame）
CIRCLE_SPEED_MAX = 4.0  # 円の最大移動速度（px/frame）
CIRCLE_ALPHA = 120  # 円の透明度（0-255、低いほど透明）

# 当たり判定設定
HEAD_HIT_RADIUS = 100  # 頭の中心からの当たり判定半径（px）

# 打楽器音設定
PERCUSSION_FREQUENCIES = [
    200, 250, 300, 350, 400, 450, 500, 550  # 打楽器風の低めの周波数（Hz）
]
PERCUSSION_DURATION = 0.15  # 打楽器音の長さ（秒）

# カラーパレット
CIRCLE_COLORS = [
    (255, 100, 100),  # 赤系
    (100, 255, 100),  # 緑系
    (100, 100, 255),  # 青系
    (255, 255, 100),  # 黄系
    (255, 100, 255),  # マゼンタ系
    (100, 255, 255),  # シアン系
    (255, 150, 100),  # オレンジ系
    (150, 100, 255),  # 紫系
]
# ================================================

class PercussionSound:
    """打楽器音を生成・再生するクラス"""
    def __init__(self):
        # 音声ミキサーの初期化
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.sounds = {}  # 周波数ごとの音声データを保存
        # 各打楽器音を事前生成
        for freq in PERCUSSION_FREQUENCIES:
            self.sounds[freq] = self.generate_percussion_tone(freq, PERCUSSION_DURATION)
    
    def generate_percussion_tone(self, frequency, duration):
        """打楽器音を生成する"""
        sample_rate = 22050  # サンプリングレート
        n_samples = int(sample_rate * duration)  # サンプル数
        
        # 時間軸の生成
        t = np.linspace(0, duration, n_samples, False)
        
        # ノイズを混ぜた打楽器風の音を生成
        wave = np.sin(2 * np.pi * frequency * t)  # 基音
        noise = np.random.uniform(-0.3, 0.3, n_samples)  # ノイズ成分
        wave = wave * 0.7 + noise * 0.3
        
        # 急激な減衰エンベロープ（打楽器らしさ）
        envelope = np.exp(-8 * t / duration)
        wave = wave * envelope
        
        # 音量調整と整数化
        wave = wave * 0.4
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

class FloatingCircle:
    """ふわふわ移動する円を表すクラス"""
    def __init__(self):
        # ランダムな位置から開始
        self.x = random.uniform(CIRCLE_MAX_RADIUS, SCREEN_WIDTH - CIRCLE_MAX_RADIUS)
        self.y = random.uniform(CIRCLE_MAX_RADIUS, SCREEN_HEIGHT - CIRCLE_MAX_RADIUS)
        
        # ランダムな半径
        self.radius = random.uniform(CIRCLE_MIN_RADIUS, CIRCLE_MAX_RADIUS)
        
        # ランダムな移動速度と方向
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(CIRCLE_SPEED_MIN, CIRCLE_SPEED_MAX)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        
        # ランダムな色
        self.color = random.choice(CIRCLE_COLORS)
        
        # ランダムな周波数
        self.frequency = random.choice(PERCUSSION_FREQUENCIES)
        
        # 生存フラグ
        self.alive = True
    
    def update(self):
        """円の位置を更新"""
        # 位置を更新
        self.x += self.vx
        self.y += self.vy
        
        # 画面端で反射
        if self.x - self.radius <= 0 or self.x + self.radius >= SCREEN_WIDTH:
            self.vx = -self.vx
            self.x = max(self.radius, min(SCREEN_WIDTH - self.radius, self.x))
        
        if self.y - self.radius <= 0 or self.y + self.radius >= SCREEN_HEIGHT:
            self.vy = -self.vy
            self.y = max(self.radius, min(SCREEN_HEIGHT - self.radius, self.y))
    
    def draw(self, screen):
        """円を描画"""
        # 半透明の円を描画
        circle_surface = pygame.Surface((int(self.radius * 2), int(self.radius * 2)), pygame.SRCALPHA)
        pygame.draw.circle(
            circle_surface,
            (*self.color, CIRCLE_ALPHA),
            (int(self.radius), int(self.radius)),
            int(self.radius)
        )
        screen.blit(circle_surface, (int(self.x - self.radius), int(self.y - self.radius)))
    
    def check_collision(self, head_x, head_y):
        """頭部との衝突判定"""
        distance = math.sqrt((self.x - head_x) ** 2 + (self.y - head_y) ** 2)
        return distance <= (self.radius + HEAD_HIT_RADIUS)

class BubbleModeApp:
    """ふわふわ円モードのメインクラス"""
    def __init__(self):
        # Pygameの初期化
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Bubble Mode - Floating Circles")
        
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
        
        # 打楽器音の初期化
        self.percussion = PercussionSound()
        
        # 円のリスト
        self.circles = []
        for _ in range(CIRCLE_COUNT):
            self.circles.append(FloatingCircle())
        
        # フレームレート管理用
        self.clock = pygame.time.Clock()
        self.running = True
    
    def detect_head(self, frame):
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
                        head_positions.append((int(nose[0]), int(nose[1])))
        
        return head_positions
    
    def check_collisions(self, head_positions):
        """頭部と円の衝突判定"""
        for head_x, head_y in head_positions:
            for circle in self.circles:
                if circle.alive and circle.check_collision(head_x, head_y):
                    # 衝突した円を消して音を鳴らす
                    circle.alive = False
                    self.percussion.play(circle.frequency)
    
    def manage_circles(self):
        """円の管理（削除と生成）"""
        # 死んだ円を削除
        self.circles = [c for c in self.circles if c.alive]
        
        # 円が足りなければ新しく生成
        while len(self.circles) < CIRCLE_COUNT:
            self.circles.append(FloatingCircle())
    
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
            head_positions = self.detect_head(frame)
            
            # 衝突判定
            self.check_collisions(head_positions)
            
            # 円の管理
            self.manage_circles()
            
            # 円の位置を更新
            for circle in self.circles:
                circle.update()
            
            # カメラ映像をpygame用に変換（ミラーリング）
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            frame_surface = pygame.transform.flip(frame_surface, True, False)  # 左右反転
            
            # 背景として映像を描画
            self.screen.blit(frame_surface, (0, 0))
            
            # 円を描画
            for circle in self.circles:
                circle.draw(self.screen)
            
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
    app = BubbleModeApp()
    app.run()
