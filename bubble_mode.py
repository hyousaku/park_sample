# このプログラムで使う「道具箱」を読み込んでいるよ
import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'  # AIが警告を出さないようにするおまじない

import cv2  # カメラの映像を使うための道具
import pygame  # 画面に絵を描いたり、音を鳴らしたりする道具
import numpy as np  # 数字をたくさん計算するための道具
import torch  # AIを動かすための道具
from ultralytics import YOLO  # 人を見つけるAI（YOLO）を使うための道具
import random  # ランダムに何かを選ぶときに使う道具
import math  # sin・cos・sqrtなどの数学の計算をする道具

# ========== 自由に変えられる設定 ==========
# 画面の大きさ（単位はピクセル＝画面の点の数）
SCREEN_WIDTH = 1920   # 画面の横の広さ
SCREEN_HEIGHT = 1080  # 画面のたての広さ

# 円の大きさや動きの設定
CIRCLE_MIN_RADIUS = 80   # 円の一番小さい半径（ピクセル）
CIRCLE_MAX_RADIUS = 120  # 円の一番大きい半径（ピクセル）
CIRCLE_COUNT = 5         # 画面に同時に出す円の数
CIRCLE_SPEED_MIN = 0.5   # 円の一番遅い速さ（1フレームに動くピクセル数）
CIRCLE_SPEED_MAX = 4.0   # 円の一番速い速さ（1フレームに動くピクセル数）
CIRCLE_ALPHA = 120       # 円のすき通り具合（0=完全に透明、255=完全にこい）

# 頭との当たり判定の広さ
HEAD_HIT_RADIUS = 100  # 頭の中心からこのピクセル数以内に円が来たら当たりとみなす

# 打楽器音の音の高さ（Hz＝1秒間に何回ふるえるか）
PERCUSSION_FREQUENCIES = [
    200, 250, 300, 350, 400, 450, 500, 550  # 低めの音でドラムっぽく聞こえる
]
PERCUSSION_DURATION = 0.15  # 打楽器音を鳴らす長さ（秒）

# 円に使う色のリスト（赤・緑・青・黄・ピンク・水色・オレンジ・紫）
CIRCLE_COLORS = [
    (255, 100, 100),  # 赤っぽい色
    (100, 255, 100),  # 緑っぽい色
    (100, 100, 255),  # 青っぽい色
    (255, 255, 100),  # 黄色っぽい色
    (255, 100, 255),  # ピンクっぽい色
    (100, 255, 255),  # 水色っぽい色
    (255, 150, 100),  # オレンジっぽい色
    (150, 100, 255),  # 紫っぽい色
]
# ==========================================

class PercussionSound:
    """打楽器の音を作って鳴らすクラス（設計図）だよ"""
    def __init__(self):
        # 音を鳴らす準備をする（スピーカーの設定）
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.sounds = {}  # 音の高さごとに音データを保存しておく箱
        # あらかじめ全部の音を作っておく（鳴らすときに速くできるように）
        for freq in PERCUSSION_FREQUENCIES:
            self.sounds[freq] = self.generate_percussion_tone(freq, PERCUSSION_DURATION)
    
    def generate_percussion_tone(self, frequency, duration):
        """打楽器っぽい音を数学で作るよ"""
        sample_rate = 22050  # 1秒間に何個の音のデータを作るか
        n_samples = int(sample_rate * duration)  # 必要なデータの個数
        
        # 0から終わりまでの時間を細かく刻んだリストを作る
        t = np.linspace(0, duration, n_samples, False)
        
        # 波にランダムなザラザラ（ノイズ）を混ぜてドラムっぽい音にする
        wave = np.sin(2 * np.pi * frequency * t)      # 基本の波
        noise = np.random.uniform(-0.3, 0.3, n_samples)  # ランダムなザラザラ成分
        wave = wave * 0.7 + noise * 0.3  # 波7割・ザラザラ3割で混ぜる
        
        # ピアノより速く消える（打楽器らしくパッと鳴ってすぐ消える）
        envelope = np.exp(-8 * t / duration)  # 急激に小さくなる数を作る
        wave = wave * envelope  # 波に掛けてすぐ消えるようにする
        
        # 音が大きすぎないように調整して、コンピュータが扱える整数に変換する
        wave = wave * 0.4
        wave = np.int16(wave * 32767)
        
        # 左右のスピーカー（ステレオ）で同じ音が出るようにする
        stereo_wave = np.zeros((n_samples, 2), dtype=np.int16)
        stereo_wave[:, 0] = wave  # 左のスピーカー用
        stereo_wave[:, 1] = wave  # 右のスピーカー用
        
        sound = pygame.sndarray.make_sound(stereo_wave)
        return sound
    
    def play(self, frequency):
        """指定した高さの音を鳴らすよ"""
        if frequency in self.sounds:
            self.sounds[frequency].play()

class FloatingCircle:
    """画面の中をふわふわ動く円1つを表すクラス（設計図）だよ"""
    def __init__(self):
        # 画面のどこかランダムな場所からスタート（端っこすぎないようにする）
        self.x = random.uniform(CIRCLE_MAX_RADIUS, SCREEN_WIDTH - CIRCLE_MAX_RADIUS)
        self.y = random.uniform(CIRCLE_MAX_RADIUS, SCREEN_HEIGHT - CIRCLE_MAX_RADIUS)
        
        # 円の大きさをランダムに決める
        self.radius = random.uniform(CIRCLE_MIN_RADIUS, CIRCLE_MAX_RADIUS)
        
        # 動く方向と速さをランダムに決める
        # angle（角度）をランダムに選んで、cos・sinで横・たての速さに変換する
        angle = random.uniform(0, 2 * math.pi)  # 0°〜360°のランダムな角度
        speed = random.uniform(CIRCLE_SPEED_MIN, CIRCLE_SPEED_MAX)  # ランダムな速さ
        self.vx = math.cos(angle) * speed  # 横方向の速さ（1フレームに動くピクセル数）
        self.vy = math.sin(angle) * speed  # たて方向の速さ（1フレームに動くピクセル数）
        
        # 色をランダムに選ぶ
        self.color = random.choice(CIRCLE_COLORS)
        
        # 当たったときに鳴らす音の高さをランダムに選ぶ
        self.frequency = random.choice(PERCUSSION_FREQUENCIES)
        
        # 生きているか（True=まだ画面にいる、False=当たって消えた）
        self.alive = True
    
    def update(self):
        """毎フレーム円の場所を動かすよ"""
        # 速さの分だけ位置を動かす
        self.x += self.vx
        self.y += self.vy
        
        # 画面の左端か右端にぶつかったら横方向の速さを逆にする（跳ね返る）
        if self.x - self.radius <= 0 or self.x + self.radius >= SCREEN_WIDTH:
            self.vx = -self.vx
            self.x = max(self.radius, min(SCREEN_WIDTH - self.radius, self.x))  # 画面の外に出ないように直す
        
        # 画面の上端か下端にぶつかったらたて方向の速さを逆にする（跳ね返る）
        if self.y - self.radius <= 0 or self.y + self.radius >= SCREEN_HEIGHT:
            self.vy = -self.vy
            self.y = max(self.radius, min(SCREEN_HEIGHT - self.radius, self.y))  # 画面の外に出ないように直す
    
    def draw(self, screen):
        """円を画面に描くよ"""
        # 半透明（すき通って見える）の円を描くための小さな画面を作る
        circle_surface = pygame.Surface((int(self.radius * 2), int(self.radius * 2)), pygame.SRCALPHA)
        pygame.draw.circle(
            circle_surface,
            (*self.color, CIRCLE_ALPHA),  # 色＋透明度を指定
            (int(self.radius), int(self.radius)),  # 小さな画面の真ん中に描く
            int(self.radius)
        )
        screen.blit(circle_surface, (int(self.x - self.radius), int(self.y - self.radius)))
    
    def check_collision(self, head_x, head_y):
        """頭がこの円に当たっているか調べるよ（距離で判定）"""
        # 三平方の定理（ピタゴラスの定理）で円の中心と頭の中心の距離を計算する
        distance = math.sqrt((self.x - head_x) ** 2 + (self.y - head_y) ** 2)
        # 距離が「円の半径＋頭の当たり判定の広さ」より小さければ当たり！
        return distance <= (self.radius + HEAD_HIT_RADIUS)

class BubbleModeApp:
    """バブルモードのアプリ全体を動かすクラス（設計図）だよ"""
    def __init__(self):
        # 画面を表示する準備をする
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Bubble Mode - Floating Circles")
        
        # 人を見つけるAI（YOLO）を読み込む
        self.model = YOLO('yolov8n-pose.pt')
        
        # ウェブカメラを使えるように準備する
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            raise RuntimeError("Failed to open camera device")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
        
        # 打楽器の音の準備をする
        self.percussion = PercussionSound()
        
        # 最初に円を5個作ってリストに入れる
        self.circles = []
        for _ in range(CIRCLE_COUNT):
            self.circles.append(FloatingCircle())
        
        # 画面を1秒間に何回更新するか管理する時計
        self.clock = pygame.time.Clock()
        self.running = True  # Trueのあいだアプリが動き続ける
    
    def detect_head(self, frame):
        """AIを使ってカメラ映像から人の頭の場所を探すよ"""
        results = self.model(frame, verbose=False)  # AIに画像を渡して人を探してもらう
        
        head_positions = []  # 見つかった頭の場所（x, y）を入れるリスト
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints  # AIが見つけた体のポイント（骨格）を取り出す
            for person_kp in keypoints:
                kp_data = person_kp.xy.cpu().numpy()  # 各ポイントの座標（x, y）を取り出す
                if len(kp_data) > 0 and len(kp_data[0]) > 0:
                    nose = kp_data[0][0]  # 鼻の位置を頭の中心として使う
                    if nose[0] > 0 and nose[1] > 0:  # 鼻が画面の中に映っているか確認
                        head_positions.append((int(nose[0]), int(nose[1])))
        
        return head_positions
    
    def check_collisions(self, head_positions):
        """頭が円に当たっているか全部の円について調べるよ"""
        for head_x, head_y in head_positions:
            for circle in self.circles:
                if circle.alive and circle.check_collision(head_x, head_y):
                    # 当たった円を消して音を鳴らす
                    circle.alive = False
                    self.percussion.play(circle.frequency)
    
    def manage_circles(self):
        """消えた円を取り除いて、足りない分だけ新しい円を作るよ"""
        # alive=False（消えた）円をリストから取り除く
        self.circles = [c for c in self.circles if c.alive]
        
        # 円が5個より少なくなったら新しい円を追加して5個に補充する
        while len(self.circles) < CIRCLE_COUNT:
            self.circles.append(FloatingCircle())
    
    def run(self):
        """アプリのメインループ（ずっとくり返す処理）"""
        while self.running:
            # キーボードやウィンドウの×ボタンなどの操作を受け取る
            for event in pygame.event.get():
                if event.type == pygame.QUIT:      # ウィンドウの×を押したら終了
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:  # ESCキーを押したら終了
                        self.running = False
            
            # カメラから今の映像を1枚取り出す
            ret, frame = self.cap.read()
            if not ret:
                continue  # うまく取れなかった場合はやり直す
            
            # 映像をちょうど画面と同じ大きさに調整する
            frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            
            # AIを使って映像の中の人の頭を探す
            head_positions = self.detect_head(frame)
            
            # 頭が円に当たっているか調べる
            self.check_collisions(head_positions)
            
            # 消えた円を取り除いて新しい円を補充する
            self.manage_circles()
            
            # 全部の円の場所を1フレーム分動かす
            for circle in self.circles:
                circle.update()
            
            # カメラ映像を鏡のように左右反転して表示する
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
            frame_surface = pygame.transform.flip(frame_surface, True, False)  # 左右反転
            
            # 反転した映像を背景として画面に描く
            self.screen.blit(frame_surface, (0, 0))
            
            # 映像の上に円を重ねて描く
            for circle in self.circles:
                circle.draw(self.screen)
            
            # 描いた内容を実際の画面に反映させる
            pygame.display.flip()
            self.clock.tick(30)  # 1秒間に30回くり返す（30FPS）
        
        self.cleanup()
    
    def cleanup(self):
        """アプリを終了するときの後片付け"""
        self.cap.release()  # カメラを使い終わったので解放する
        pygame.quit()  # pygameを終了する

# このファイルを直接実行したときだけ動くようにする
if __name__ == "__main__":
    app = BubbleModeApp()
    app.run()
