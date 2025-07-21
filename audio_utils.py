import pygame
import time
import threading
import os


class AudioPlayer:
    def __init__(self):
        pygame.mixer.init()
        self.playing = False

    def play_music(self, path):
        if not self.playing:
            self.playing = True
            threading.Thread(target=self._play, args=(path,)).start()

    def _play(self, path):
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
        self.playing = False

    def stop_music(self):
        try:
            if pygame.mixer.get_init():  # 检查mixer是否初始化
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.fadeout(500)  # 添加淡出效果
                    pygame.mixer.music.stop()
                    print("音乐已平滑停止")
        except Exception as e:
            print(f"停止音乐时出错: {str(e)}")
        finally:
            self.playing = False


class VoiceAlert:
    def __init__(self):
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.sound = None

    def play_alert(self, path):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"音频文件不存在: {path}")

            if pygame.mixer.get_busy():
                pygame.mixer.stop()

            self.sound = pygame.mixer.Sound(path)
            self.sound.play()
        except Exception as e:
            print(f"音频播放失败: {str(e)}")


class TimedAudioPlayer(AudioPlayer):
    def __init__(self):
        super().__init__()
        self.last_trigger = {'music': 0, 'voice': 0}

    def should_play(self, alert_type, cooldown=2):
        current_time = time.time()
        if current_time - self.last_trigger[alert_type] > cooldown:
            self.last_trigger[alert_type] = current_time
            return True
        return False