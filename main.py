# このプログラムで使う「道具箱」を読み込んでいるよ
import os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'  # AIが警告を出さないようにするおまじない

import cv2  # カメラの映像を使うための道具
import pygame  # 画面に絵を描いたり、音を鳴らしたりする道具
import numpy as np  # 数字をたくさん計算するための道具
import torch  # AIを動かすための道具
from ultralytics import YOLO  # 人を見つけるAI（YOLO）を使うための道具
import random  # ランダムに何かを選ぶときに使う道具
import time  # 時間を計るときに使う道具

# 画面の大きさを決めるよ（単位はピクセル＝画面の点の数）
SCREEN_WIDTH = 1920   # 画面の横の広さ
SCREEN_HEIGHT = 1080  # 画面のたての広さ
GRID_COLS = 16  # 画面を横に16マスに分ける
GRID_ROWS = 9   # 画面をたてに9マスに分ける
CELL_WIDTH = SCREEN_WIDTH // GRID_COLS    # 1マスの横の広さ（自動で計算）
CELL_HEIGHT = SCREEN_HEIGHT // GRID_ROWS  # 1マスのたての広さ（自動で計算）
SOUND_CHANGE_INTERVAL = 12  # 何秒ごとに音の配置を変えるか（12秒）
YOLO_SKIP_FRAMES = 3   # 何フレームに1回AIを動かすか（3=1秒に10回）
YOLO_INPUT_WIDTH = 640  # AIに渡す映像の横幅（小さいほど速い）
YOLO_INPUT_HEIGHT = 360  # AIに渡す映像のたての幅（小さいほど速い）

# ドレミの音の高さ（Hz＝1秒間に何回ふるえるか）を並べたリスト
# これは「Dペンタトニック」という5音のスケールだよ
D_PENTATONIC = [293.66, 329.63, 369.99, 440.00, 493.88, 587.33, 659.25, 739.99]

class PianoSound:
    """ピアノの音を作って鳴らすクラス（設計図）だよ"""
    def __init__(self):
        # 音を鳴らす準備をする（スピーカーの設定）
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.sounds = {}  # 音の高さごとに音データを保存しておく箱
        # あらかじめ全部の音を作っておく（鳴らすときに速くできるように）
        for freq in D_PENTATONIC:
            self.sounds[freq] = self.generate_piano_tone(freq, 0.3)
    
    def generate_piano_tone(self, frequency, duration):
        """ピアノの音を数学で作るよ（波を合わせて音を作る）"""
        sample_rate = 22050  # 1秒間に何個の音のデータを作るか
        n_samples = int(sample_rate * duration)  # 必要なデータの個数
        
        # 0から終わりまでの時間を細かく刻んだリストを作る
        t = np.linspace(0, duration, n_samples, False)
        
        # 波を重ね合わせてピアノっぽい音色を作る
        # （ギターの弦が震えるのと同じ仕組みで、倍の速さの波も混ざっているよ）
        wave = np.sin(2 * np.pi * frequency * t)        # 基本の波（1番目）
        wave += 0.5 * np.sin(2 * np.pi * frequency * 2 * t)   # 2倍の速さの波（少し混ぜる）
        wave += 0.25 * np.sin(2 * np.pi * frequency * 3 * t)  # 3倍の速さの波（もう少し混ぜる）
        
        # ピアノみたいに最初が大きくて、だんだん小さくなるようにする
        envelope = np.exp(-3 * t / duration)  # 時間が経つほど小さくなる数を作る
        wave = wave * envelope  # 波に掛けて音を小さくしていく
        
        # 音が大きすぎないように調整して、コンピュータが扱える整数に変換する
        wave = wave * 0.3
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

class GridCell:
    """画面を分けた1つ1つのマスを表すクラス（設計図）だよ"""
    def __init__(self, col, row):
        self.rect = pygame.Rect(col * CELL_WIDTH, row * CELL_HEIGHT, CELL_WIDTH, CELL_HEIGHT)  # このマスの場所と大きさ
        self.frequency = None  # このマスに割り当てる音の高さ
        self.color = None      # このマスの色
        self.active = False    # 今このマスに頭があるか（True=ある、False=ない）
        self.alpha = 50        # マスの色の濃さ（50=うすい、255=こい）
        # 毎フレーム作り直さないように Surface をあらかじめ用意しておく
        self.surface = pygame.Surface((CELL_WIDTH, CELL_HEIGHT), pygame.SRCALPHA)

    def draw(self, screen, flip_x):
        """マスを画面に描く（flip_x=True なら左右反転した位置に描く）"""
        self.surface.fill((*self.color, int(self.alpha)))
        screen.blit(self.surface, (flip_x, self.rect.y))

class PersonDetectionApp:
    """このアプリ全体を動かすクラス（設計図）だよ"""
    def __init__(self):
        # 画面を表示する準備をする
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Person Detection with YOLO")
        
        # 人を見つけるAI（YOLO）を読み込む
        self.model = YOLO('yolov8n-pose.pt')
        
        # ウェブカメラを使えるように準備する
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            raise RuntimeError("Failed to open camera device")
        # カメラはカメラが対応している解像度で取得し、あとで拡大する（バッファが小さく速い）
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # フレームレートを明示的に指定
        
        # ピアノの音の準備をする
        self.piano = PianoSound()
        
        # 画面を16×9＝144マスに分けて、全部のマスを作る
        self.grid_cells = []
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                self.grid_cells.append(GridCell(col, row))
        
        # 各マスに音の高さをランダムに割り当てる
        self.cell_triggered = {}  # どのマスで音が鳴っているかを覚えておく辞書
        self.assign_sounds_to_grid()
        self.last_sound_change = time.time()  # 最後に音の配置を変えた時刻を記録
        
        # フレームスキップ用カウンター（毎フレームAIを動かさないようにする）
        self.frame_count = 0
        self.last_head_positions = []  # 前回のAI結果を使い回す
        
        # 映像表示用のSurfaceをあらかじめ用意（毎フレーム作り直さない）
        self.frame_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        
        # 画面を1秒間に何回更新するか管理する時計
        self.clock = pygame.time.Clock()
        self.running = True  # Trueのあいだアプリが動き続ける
    
    def assign_sounds_to_grid(self):
        """全マスに音の高さと色をランダムに割り当てる（12秒ごとに呼ばれる）"""
        self.cell_triggered = {}  # 音が鳴っているかの記録をリセットする
        
        # 使う6色を決める（赤・緑・青・黄・ピンク・水色）
        colors = [
            (255, 100, 100),  # 赤っぽい色
            (100, 255, 100),  # 緑っぽい色
            (100, 100, 255),  # 青っぽい色
            (255, 255, 100),  # 黄色っぽい色
            (255, 100, 255),  # ピンクっぽい色
            (100, 255, 255),  # 水色っぽい色
        ]
        
        # 全マスに1つずつ音と色を割り当てる
        for i, cell in enumerate(self.grid_cells):
            cell.frequency = random.choice(D_PENTATONIC)  # 音の高さをランダムに選ぶ
            cell.color = colors[i % len(colors)]  # 色を順番に割り当てる（6色くり返し）
            self.cell_triggered[i] = False  # 最初は音が鳴っていない状態にする
    
    def detect_persons(self, frame):
        """AIを使ってカメラ映像から人の頭の位置を探す"""
        # AIには小さい画像を渡す（速さ優先）。座標は後で画面サイズに拡大する
        small = cv2.resize(frame, (YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT))
        results = self.model(small, verbose=False)  # 小さい画像でAI推論
        
        # 小さい画像→画面サイズへの拡大比率
        scale_x = SCREEN_WIDTH / YOLO_INPUT_WIDTH
        scale_y = SCREEN_HEIGHT / YOLO_INPUT_HEIGHT
        
        head_positions = []  # 見つかった頭の場所を入れるリスト
        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints  # AIが見つけた体のポイント（骨格）を取り出す
            for person_kp in keypoints:
                kp_data = person_kp.xy.cpu().numpy()  # 各ポイントの座標（x, y）を取り出す
                if len(kp_data) > 0 and len(kp_data[0]) > 0:
                    nose = kp_data[0][0]  # 鼻の位置を頭の中心として使う
                    if nose[0] > 0 and nose[1] > 0:  # 鼻が画面の中に映っているか確認
                        # 小さい画像の座標を画面サイズに戻す
                        nx = nose[0] * scale_x
                        ny = nose[1] * scale_y
                        head_size = 40  # 頭の大きさ（鼻を中心に±40ピクセル＝80×80の四角）
                        x1 = int(nx - head_size)
                        y1 = int(ny - head_size)
                        x2 = int(nx + head_size)
                        y2 = int(ny + head_size)
                        head_positions.append((x1, y1, x2, y2))
        
        return head_positions
    
    def check_grid_collision(self, head_positions):
        """頭がどのマスに入っているか調べて、入っていたら音を鳴らす"""
        for i, cell in enumerate(self.grid_cells):
            cell.active = False  # いったん「頭がいない」状態にリセット
            
            # 検出された頭1つ1つについて調べる
            for (x1, y1, x2, y2) in head_positions:
                head_center_x = (x1 + x2) // 2  # 頭の真ん中のX座標
                head_center_y = (y1 + y2) // 2  # 頭の真ん中のY座標
                
                # 頭の真ん中がこのマスの中に入っているか調べる
                if cell.rect.collidepoint(head_center_x, head_center_y):
                    cell.active = True  # 「頭がいる！」とマークする
                    
                    # まだ音が鳴っていないなら音を鳴らす（入った瞬間だけ鳴らす）
                    if not self.cell_triggered[i]:
                        self.piano.play(cell.frequency)  # このマスの音を鳴らす
                        self.cell_triggered[i] = True  # 「もう音を鳴らした」と記録する
                        cell.alpha = 150  # マスの色を少し濃くする
                    break  # 1つの頭がマスに入っていれば十分なので次の頭は調べない
            
            # 頭がマスから出た場合
            if not cell.active:
                self.cell_triggered[i] = False  # 「まだ音を鳴らしていない」状態に戻す
                # マスの色をゆっくり薄くしていく（フェードアウト）
                if cell.alpha > 50:
                    cell.alpha = max(50, cell.alpha - 10)
    
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
            
            # 映像を画面サイズに拡大（表示用）
            frame_full = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))
            
            # AIはフレームスキップして実行（毎フレーム動かさず負荷を下げる）
            self.frame_count += 1
            if self.frame_count % YOLO_SKIP_FRAMES == 0:
                # detect_persons 内で YOLO_INPUT サイズに縮小してから推論する
                self.last_head_positions = self.detect_persons(frame_full)
            
            # 頭がどのマスに入っているか調べて音を鳴らす
            self.check_grid_collision(self.last_head_positions)
            
            # 12秒経ったら音の配置をシャッフルする
            current_time = time.time()
            if current_time - self.last_sound_change >= SOUND_CHANGE_INTERVAL:
                self.assign_sounds_to_grid()
                self.last_sound_change = current_time
            
            # カメラ映像をpygameが使える形に変換して画面に表示する
            # rot90の代わりにtransposeで回転（メモリコピーが少なく速い）
            frame_rgb = cv2.cvtColor(frame_full, cv2.COLOR_BGR2RGB)
            frame_transposed = frame_rgb.transpose(1, 0, 2)  # HWC → WHC（rot90相当）
            pygame.surfarray.blit_array(self.frame_surface, frame_transposed)
            # 左右反転して鏡のように表示
            flipped = pygame.transform.flip(self.frame_surface, True, False)
            self.screen.blit(flipped, (0, 0))
            
            # 頭が入っているマスに色をつけて表示する（鏡のように左右反転）
            for cell in self.grid_cells:
                if cell.active:
                    flip_x = SCREEN_WIDTH - cell.rect.x - cell.rect.width
                    cell.draw(self.screen, flip_x)
            
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
    app = PersonDetectionApp()
    app.run()
