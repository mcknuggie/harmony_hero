# shows how aubio can be used for real-time pitch detection
# docs:
# - https://aubio.org/manual/latest/index.html
# - https://aubio.org/doc/latest/pitch_8h.html

import sys, os

sys.path.insert(0, os.path.abspath(".."))

from imslib.core import BaseWidget, run, lookup
from imslib.audio import Audio
from imslib.gfxutil import topleft_label, CLabelRect, CEllipse, Line, CRectangle
from imslib.mixer import Mixer
from kivy.clock import Clock as kivyClock
from kivy.core.window import Window
from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.core.image import Image
from imslib.kivyparticle import ParticleSystem
from imslib.wavegen import WaveGenerator
from imslib.wavesrc import WaveBuffer, WaveFile
from kivy.graphics.transformation import Matrix
from kivy.graphics import PushMatrix, PopMatrix, Scale, Rotate, Translate
from librosa import feature
from imslib.screen import Screen, ScreenManager
from kivy.uix.button import Button

import math


import numpy as np
import aubio

# etablish relative positioning factors
now_height = 0.2282
now_cent = 0.3402
now_width = 0.002
record_x = 0.3975
record_y = 0.5800
ind_size = 0.05


class AudioBuffer(object):
    """converts a variable buffer size to a fixed buffer size. Audio data is received
    at insert(self, data). Data can be any size. When enough data has accumulated, it
    calls the receive_audio_cb function with the fixed-length data buffer of size
    output_size. Every call to insert() may generate 0, 1 or more calls to receive_audio_cb
    """

    def __init__(self, output_size, receive_audio_cb):
        super(AudioBuffer, self).__init__()
        self.output_size = output_size
        self.receive_audio_cb = receive_audio_cb

        # holds onto remaining chunk that is too small to send out now
        self.buffer = np.zeros(0, dtype=np.float32)

    def insert(self, data):
        """receive audio of some non-fixed size. Perhaps call receive_audio_cb for as many
        complete buffers (of size output_size) as are available
        """

        # create a complete buffer of previous data and new data
        buf = np.concatenate((self.buffer, data), dtype=np.float32)

        # ship out full buffers:
        ptr = 0
        while ptr + self.output_size < len(buf):
            self.receive_audio_cb(buf[ptr : ptr + self.output_size])
            ptr += self.output_size

        # retain remainder for next time
        self.buffer = buf[ptr:]


class PitchDetector(object):
    """pitch detection based on inputted audio. Input audio data can be of any length
    When enough input data has come in to produce a reading, that reading is stored in
    self.pitch and a confidence stored in self.conf"""

    def __init__(self):
        super(PitchDetector, self).__init__()

        self.buf_size = 4096  # the algorithm's window size
        self.hop_size = (
            1024  # this is the amount we feed into aubio.pitch at each time step
        )

        self.buffer = AudioBuffer(self.hop_size, self.process)

        self.pitch_o = aubio.pitch(
            "yin", self.buf_size, self.hop_size, Audio.sample_rate
        )
        self.pitch_o.set_tolerance(0.5)
        self.pitch_o.set_unit("midi")

        self.pitch = 0
        self.conf = 0

        self.volume = 0

    def insert(self, data):
        self.buffer.insert(data)

    def process(self, data):
        assert len(data) == self.hop_size
        self.pitch = self.pitch_o(data)[0]
        self.conf = self.pitch_o.get_confidence()
        self.volume = feature.rms(y=data)[0][0] * 500


class PitchIndicator(InstructionGroup):
    def __init__(self, min_pitch, max_pitch):
        super(PitchIndicator, self).__init__()

        self.pitch = 60
        self.confidence = 0
        self.base_pitch = 48
        self.recent_sung = []

        w = Window.width
        h = Window.height

        self.ind_size = (ind_size * w, ind_size * h)
        self.ind_cent = (record_x * w, now_cent * h - now_height / 2 * h)
        self.texture = Image("failarrow.png").texture

        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.divisions = now_height * h / (self.max_pitch - self.min_pitch + 1)

        self.indicator = CRectangle(
            texture=self.texture,
            csize=self.ind_size,
            cpos=(self.ind_cent[0], self.ind_cent[1]),
        )
        self.add(self.indicator)

    def on_update(self, pitch, confidence, sung_volume):
        self.pitch = pitch
        self.confidence = confidence

        if self.confidence >= 0.7 and sung_volume > 5:
            ypos = self.ind_cent[1] + self.divisions * (self.pitch - self.min_pitch + 1)
            if self.pitch < self.min_pitch:
                ypos = self.ind_cent[1] + self.min_pitch
            elif self.pitch > self.max_pitch:
                ypos = self.ind_cent[1] + self.divisions * (
                    self.max_pitch - self.min_pitch + 1
                )

            self.indicator.cpos = (self.ind_cent[0], ypos)

    def on_resize(self, win_size):
        w = win_size[0]
        h = win_size[1]
        updated_size = min(ind_size * w, ind_size * h)
        self.ind_size = (updated_size, updated_size)
        self.divisions = now_height * 0.7307 * w / (self.max_pitch - self.min_pitch + 1)
        self.indicator.csize = self.ind_size

        if h > 0.7307 * w:
            ind_x = record_x * w
            ind_y = (
                now_cent * (0.7307 * w)
                - (now_height * 0.7307 * w) / 2
                + (h - 0.7307 * w) / 2
            )
            self.divisions = (
                now_height * 0.7307 * w / (self.max_pitch - self.min_pitch + 1)
            )
        else:
            ind_x = record_x * (h / 0.7307) + (w - h / 0.7307) / 2
            ind_y = now_cent * h - now_height / 2 * h
            self.divisions = now_height * h / (self.max_pitch - self.min_pitch + 1)
        self.ind_cent = (ind_x, ind_y)


class LineDisplay(InstructionGroup):
    """
    Draws a single Line (rectangle) for pitch matching and moves it across the screen
    """

    def __init__(self, pitch, start_time, duration, min_pitch, max_pitch):
        super(LineDisplay, self).__init__()

        self.window_width = Window.width
        self.window_height = Window.height

        # will dictate the ypos of the rectangle on the screen
        self.pitch = pitch

        # start and end times will dictate width of rectangle (line) and xpos of it
        self.start_time = start_time
        self.duration = duration

        # default initialization of rectangle (can ignore)
        self.line = Rectangle(
            size=(50, 50), pos=(record_x * self.window_width, self.window_height / 2)
        )

        self.start_xpos = 0
        self.end_xpos = 0
        self.ypos = 0

        self.add(Color(1, 1, 1))
        self.add(self.line)

        # True if right of the nowbar, False if left of it
        self.active = True

        self.ind_cent = (
            record_x * self.window_width,
            now_cent * self.window_height - now_height / 2 * self.window_height,
        )
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.divisions = (
            now_height * self.window_height / (self.max_pitch - self.min_pitch + 1)
        )

    def time_to_xpos(self, time):
        speed = -300
        return (self.window_width * record_x) - (time * speed)

    def pitch_to_ypos(self, pitch):
        return self.ind_cent[1] + self.divisions * (pitch - self.min_pitch + 1)

    def on_update(self, now_time):
        self.start_xpos = self.time_to_xpos(self.start_time - now_time)
        self.end_xpos = self.time_to_xpos(self.start_time - now_time + self.duration)
        self.ypos = self.pitch_to_ypos(self.pitch)

        # if within bounds of the screen, update position and return true
        if self.start_xpos <= self.window_width and self.end_xpos >= 0:
            self.line.pos = (self.start_xpos, self.ypos)
            self.line.size = (self.end_xpos - self.start_xpos, 10)
            if self.end_xpos < self.ind_cent[0]:
                self.active = False
            return True

        if self.end_xpos < 0:
            return False

    def on_resize(self, win_size):
        w = win_size[0]
        h = win_size[1]
        self.window_width = w
        self.window_height = h
        self.divisions = now_height * h / (self.max_pitch - self.min_pitch + 1)
        ind_x = record_x * w
        ind_y = now_cent * h - now_height / 2 * h
        if h > 0.7307 * w:
            ind_y = (
                now_cent * (0.7307 * w)
                - (now_height * 0.7307 * w) / 2
                + (h - 0.7307 * w) / 2
            )
            self.divisions = (
                now_height * 0.7307 * w / (self.max_pitch - self.min_pitch + 1)
            )
        else:
            ind_x = record_x * (h / 0.7307) + (w - h / 0.7307) / 2
        self.ind_cent = (ind_x, ind_y)


class RecordDisplay(InstructionGroup):
    def __init__(self, song_choice):
        super(RecordDisplay, self).__init__()
        self.win_size = Window.width, Window.height
        w, h = self.win_size
        self.img_w = w * 1
        self.img_h = 0.7307 * self.img_w

        self.xcent = record_x * w
        self.ycent = record_y * h
        self.angle = 0
        kivyClock.schedule_interval(self.on_update, 1.0 / 60.0)

        if song_choice == "valerie":
            self.bgtexture = Image("bg_valerie.png").texture
            self.record_texture = Image("vinyl_valerie.png").texture
        elif song_choice == "allstar":
            self.bgtexture = Image("bg_allstar.png").texture
            self.record_texture = Image("vinyl_allstar.png").texture
        elif song_choice == "bohemian":
            self.bgtexture = Image("bg_bohemian.png").texture
            self.record_texture = Image("vinyl_bohemian.png").texture
        self.armtexture = Image("tonearm.png").texture

        self.bg = CRectangle(
            texture=self.bgtexture,
            csize=(self.img_w, self.img_h),
            cpos=(self.xcent, self.ycent),
        )
        self.record = CRectangle(
            texture=self.record_texture,
            csize=(self.img_w, self.img_h),
            cpos=(self.xcent, self.ycent),
        )
        self.arm = CRectangle(
            texture=self.armtexture,
            csize=(self.img_w, self.img_h),
            cpos=(self.xcent, self.ycent),
        )

        # Adding to canvas
        self.add(self.bg)
        self.add(PushMatrix())
        self.rotation = Rotate(angle=self.angle, origin=(self.xcent, self.ycent))
        self.add(self.rotation)
        self.add(self.record)

        self.add(PopMatrix())
        self.add(self.arm)

        self.nowbar = CRectangle(
            csize=(5, now_height * self.img_h),
            cpos=(record_x * self.img_w, now_cent * self.img_h),
        )
        self.add(self.nowbar)

    def on_resize(self, win_size):
        # Calculate scale factors
        w, h = win_size
        if w / self.img_w < h / self.img_h:
            scale = w / self.img_w
        else:
            scale = h / self.img_h

        csize = (self.img_w * scale, self.img_h * scale)
        cpos = (w / 2, h / 2)
        self.bg.cpos = cpos
        self.bg.csize = csize
        self.record.cpos = cpos
        self.record.csize = csize
        self.arm.csize = csize
        self.arm.cpos = cpos
        self.nowbar.csize = (now_width * csize[0], now_height * csize[1])
        self.nowbar.cpos = (
            record_x * csize[0] + (w - csize[0]) / 2,
            now_cent * csize[1] + (h - csize[1]) / 2,
        )
        self.xcent = record_x * csize[0] + (w - csize[0]) / 2
        self.ycent = record_y * csize[1] + (h - csize[1]) / 2
        self.rotation.origin = (self.xcent, self.ycent)

        # update self.win_size to the new win_size
        self.win_size = win_size

    def on_update(self, dt):
        self.angle += dt * 30  # Adjust rotation speed as needed
        self.rotation.angle = self.angle % 360


class InputVolumeDisplay(InstructionGroup):
    def __init__(self):
        super(InputVolumeDisplay, self).__init__()

        self.window_width = Window.width
        self.window_height = Window.height

        self.add(Color((1, 1, 1)))
        self.outline_bar = Rectangle(
            pos=(9 * self.window_width / 10, 8.5 * self.window_height / 10),
            size=(25, 150),
        )
        self.add(self.outline_bar)

        self.add(Color(hsv=(0.39, 1, 1)))
        self.loudness_bar = Rectangle(
            pos=(9 * self.window_width / 10, 8.5 * self.window_height / 10),
            size=(25, 0),
        )
        self.add(self.loudness_bar)

        self.add(Color(1, 0, 0))
        self.threshold_line = Rectangle(
            pos=(9 * self.window_width / 10, (8.5 * self.window_height / 10) + 5),
            size=(25, 2),
        )
        self.add(self.threshold_line)

    def on_update(self, sung_volume):
        self.loudness_bar.size = (25, sung_volume)

    def on_resize(self, win_size):
        self.window_width = win_size[0]
        self.window_height = win_size[1]
        self.remove(self.outline_bar)
        self.outline_bar = Rectangle(
            pos=(9 * self.window_width / 10, 8.5 * self.window_height / 10),
            size=(25, 150),
        )
        self.add(Color((1, 1, 1)))
        self.add(self.outline_bar)
        self.remove(self.loudness_bar)
        self.add(Color(hsv=(0.39, 1, 1)))
        self.loudness_bar = Rectangle(
            pos=(9 * self.window_width / 10, 8.5 * self.window_height / 10),
            size=(25, 0),
        )
        self.add(self.loudness_bar)

        self.remove(self.threshold_line)
        self.add(Color(1, 0, 0))
        self.threshold_line = Rectangle(
            pos=(9 * self.window_width / 10, (8.5 * self.window_height / 10) + 5),
            size=(25, 2),
        )
        self.add(self.threshold_line)


class GameDisplay(InstructionGroup):
    def __init__(self, song_choice):
        super(GameDisplay, self).__init__()

        valerie_file = "valerie_notes.txt"
        valerie_lines = open(valerie_file).readlines()
        allstar_file = "allstar_notes.txt"
        allstar_lines = open(allstar_file).readlines()
        bohemian_file = "bohemian_notes.txt"
        bohemian_lines = open(bohemian_file).readlines()
        self.max_points = 1

        if song_choice == "bohemian":
            song_lines = bohemian_lines
            self.max_points = 1450300
            self.gold = Image("vinyl_bohemian_gold.png").texture
            self.plat = Image("vinyl_bohemian_platinum.png").texture
        elif song_choice == "allstar":
            song_lines = allstar_lines
            self.max_points = 1074506
            self.gold = Image("vinyl_allstar_gold.png").texture
            self.plat = Image("vinyl_allstar_platinum.png").texture
        elif song_choice == "valerie":
            song_lines = valerie_lines
            self.max_points = 902500
            self.gold = Image("vinyl_valerie_gold.png").texture
            self.plat = Image("vinyl_valerie_platinum.png").texture

        def notes_from_line(line):
            if song_choice == "allstar":
                pitch, start, duration = (
                    line.strip().split("\t")[1],
                    line.strip().split("\t")[0],
                    line.strip().split("\t")[2],
                )
                return (
                    69 + 12 * math.log2(float(pitch) / 440) - 12,
                    float(start),
                    float(duration),
                )
            else:
                pitch, start, duration = (
                    line.strip().split("\t")[1],
                    line.strip().split("\t")[0],
                    line.strip().split("\t")[2],
                )
                return (
                    69 + 12 * math.log2(float(pitch) / 440),
                    float(start),
                    float(duration),
                )

        self.line_data = [notes_from_line(l1) for l1 in song_lines]
        first_column_values = [item[0] for item in self.line_data]

        # Getting the minimum and maximum values
        w = Window.width
        h = Window.height
        self.min_value = min(first_column_values)
        self.max_value = max(first_column_values)
        self.divisions = (now_height * h) / (self.max_value - self.min_value - 1)

        self.success = False

        self.ind_cent = (record_x * w, now_cent * h - now_height / 2 * h)

        self.record_display = RecordDisplay(song_choice)

        self.add(self.record_display)

        self.add(Color(1, 1, 1, 0.5))
        self.backdrop = Rectangle(
            pos=(self.ind_cent[0], self.ind_cent[1]),
            size=((w * (1 - record_x)), now_height * h),
        )
        self.add(self.backdrop)
        self.add(Color(1, 1, 1))

        self.ps = ParticleSystem("particle/particle.pex")
        self.ps.emitter_x = self.ind_cent[0]
        self.ps.emitter_y = self.ind_cent[1]

        self.pitch_indicator = PitchIndicator(self.min_value, self.max_value)
        self.add(self.pitch_indicator)

        self.recent_pitches = []

        self.vocal_range = "tenor"
        self.base_pitch = 48

        self.score = 0

        self.add(Color(0, 0, 0))
        self.scoreboard = CLabelRect(
            (4 * Window.width / 5, 9 * Window.height / 10),
            "",
            font_size=20,
            font_name="Arial",
        )
        self.add(self.scoreboard)

        self.return_home = CLabelRect(
            (8 * Window.width / 10, 1 * Window.height / 10),
            "Press R to Exit",
            font_size=15,
            font_name="Arial",
        )
        self.add(self.return_home)

        self.current_line = None

        # data format: (pitch, start_time, duration)
        self.lines = []
        for line in self.line_data:
            self.lines.append(
                LineDisplay(line[0], line[1], line[2], self.min_value, self.max_value)
            )

        for line in self.lines:
            self.add(line)

        self.input_volume_display = InputVolumeDisplay()
        self.add(self.input_volume_display)
        self.arrow_color = Color(1, 0, 0)

    def light_up_arrow(self, color):
        self.success = True

        if color == "gold":
            color = Color(0.83, 0.69, 0.22)
            self.arrow_color = color
            self.pitch_indicator.indicator.texture = Image("arrow.png").texture
            self.ps.start_color[0] = 0.83
            self.ps.start_color[1] = 0.69
            self.ps.start_color[2] = 0.22
            self.ps.end_color[0] = 0.83
            self.ps.end_color[1] = 0.69
            self.ps.end_color[2] = 0.22
        if color == "plat":
            color = Color(0.9, 0.9, 0.95)
            self.arrow_color = color
            self.pitch_indicator.indicator.texture = Image("arrow.png").texture
            self.ps.end_color[0] = 0.9
            self.ps.start_color[1] = 0.9
            self.ps.start_color[2] = 0.95
            self.ps.end_color[0] = 0.9
            self.ps.end_color[1] = 0.9
            self.ps.end_color[2] = 0.95
        self.ps.start()
        self.add(self.arrow_color)
        self.remove(self.pitch_indicator)
        self.add(self.pitch_indicator)

    def darken_arrow(self):
        self.success = False
        color = Color(1, 0, 0)
        self.arrow_color = color
        self.ps.stop()
        self.pitch_indicator.indicator.texture = Image("failarrow.png").texture
        self.add(self.arrow_color)
        self.remove(self.pitch_indicator)
        self.add(self.pitch_indicator)

    def get_current_line(self, lines):
        """
        returns the line currently intersecting the nowbar (if one exists)
        """

        for line in lines:
            if (
                line.start_xpos <= self.ind_cent[0]
                and line.end_xpos >= self.ind_cent[0]
            ):
                return line
        return None

    def add_to_score(self, new_points):
        self.score += new_points

    def amount_off_to_points_earned(self, amount_off, sung_volume):  # slop window

        if amount_off > 0.1 and amount_off <= 0.3 and sung_volume > 5:
            return int(100 - (amount_off - 0.1) / 0.2 * 100)
        elif amount_off <= 0.1:
            return 100
        else:
            return 0

    def determine_pitch_type(self, current_pitch, average_pitch, reference_pitch):
        """
        Takes in the current pitch from this frame and the average pitch from recent frames
        and determines which is closer to the reference pitch
        """
        current_off = abs(current_pitch - reference_pitch)
        average_off = abs(average_pitch - reference_pitch)

        if average_off < current_off:
            return average_off, "avg"
        else:
            return current_off, "curr"

    def on_update(self, sung_pitch, confidence, now_time, paused, sung_volume):
        self.ps.emitter_y = self.pitch_indicator.indicator.cpos[1]
        self.ps.emitter_x = self.pitch_indicator.indicator.cpos[0]

        self.recent_pitches.append(sung_pitch)
        if len(self.recent_pitches) > 30:
            self.recent_pitches.pop(0)

        average_pitch = sum(self.recent_pitches) / len(self.recent_pitches)

        pitch_type = "curr"

        self.current_line = self.get_current_line(self.lines)
        if self.current_line:
            current_reference_pitch = self.current_line.pitch

            amount_off, pitch_type = self.determine_pitch_type(
                sung_pitch, average_pitch, current_reference_pitch
            )

            points_earned = self.amount_off_to_points_earned(amount_off, sung_volume)

            if not paused:
                self.add_to_score(points_earned)
                if points_earned == 100:
                    self.light_up_arrow("plat")
                elif points_earned > 0:
                    self.light_up_arrow("gold")
                else:
                    self.darken_arrow()
        else:
            self.current_line = None
            self.darken_arrow()

        if pitch_type == "avg":
            self.pitch_indicator.on_update(average_pitch, confidence, sung_volume)
        else:
            self.pitch_indicator.on_update(sung_pitch, confidence, sung_volume)

        if self.score > 0.1 * self.max_points:
            self.record_display.record.texture = self.plat
        elif self.score > 0.05 * self.max_points:
            self.record_display.record.texture = self.gold

        self.scoreboard.set_text("Score: " + str(self.score))

        for line in self.lines:
            visible = line.on_update(now_time)
            if visible:
                if line not in self.children:
                    self.add(line)
            else:
                self.remove(line)

        self.input_volume_display.on_update(sung_volume)

    def on_resize(self, win_size):
        self.record_display.on_resize(win_size)
        self.pitch_indicator.on_resize(win_size)
        self.ind_cent = self.pitch_indicator.ind_cent

        self.remove(self.backdrop)
        self.add(Color(0.3, 0.3, 0.3, 0.9))
        self.backdrop = Rectangle(
            pos=(self.ind_cent[0], self.ind_cent[1]),
            size=((win_size[0] * (1 - record_x)), now_height * win_size[1]),
        )
        self.add(self.backdrop)
        self.add(Color(1, 1, 1))

        for line in self.lines:
            line.on_resize(win_size)

        self.remove(self.pitch_indicator)
        self.add(self.arrow_color)
        self.add(self.pitch_indicator)
        self.input_volume_display.on_resize(win_size)
        self.ps.emitter_x = self.pitch_indicator.ind_cent[0]

        self.remove(self.scoreboard)
        self.add(Color(0, 0, 0))
        self.scoreboard = CLabelRect(
            (4 * win_size[0] / 5, 9 * win_size[1] / 10),
            "",
            font_size=20,
            font_name="Arial",
        )
        self.add(self.scoreboard)

        self.remove(self.return_home)
        self.return_home = CLabelRect(
            (8 * win_size[0] / 10, 1 * win_size[1] / 10),
            "Press R to Exit",
            font_size=15,
            font_name="Arial",
        )
        self.add(self.return_home)


class AudioController(object):
    def __init__(self, song_path, song_path2):
        super(AudioController, self).__init__()

        self.audio = Audio(2)
        self.mixer = Mixer()
        self.audio.set_generator(self.mixer)

        self.backing_track = WaveGenerator(WaveFile(song_path))
        self.music = WaveGenerator(WaveFile(song_path2))
        self.paused = self.backing_track.paused
        self.paused2 = self.music.paused

        self.mixer.add(self.backing_track)
        self.mixer.add(self.music)

        # intialize with everything paused
        self.backing_track.pause()
        self.music.pause()

    def get_total_duration(self):
        return self.total_duration

    # start / stop both the backing and solo tracks
    def toggle(self):
        self.backing_track.play_toggle()
        self.music.play_toggle()

    # return current time (in seconds) of song
    def get_time(self):
        return self.backing_track.frame / self.audio.sample_rate

    # needed to update audio
    def on_update(self):
        self.audio.on_update()
        self.paused = self.backing_track.paused


class SongSelectScreen(Screen):
    def __init__(self, **kwargs):
        super(SongSelectScreen, self).__init__(**kwargs)

        self.info = topleft_label()
        self.info.text = "Home Screen\n"

        self.song_select_display = SongSelectDisplay()
        self.canvas.add(self.song_select_display)

    def on_key_down(self, keycode, modifiers):
        if keycode[1] == "1":
            self.switch_to("allstar")
        if keycode[1] == "2":
            self.switch_to("valerie")
        if keycode[1] == "3":
            self.switch_to("bohemian")

    def on_update(self):
        """
        self.info.text = "Home Screen\n"
        self.info.text += "1: All Star by Smash Mouth\n"
        self.info.text += "2: Valerie by Amy Winehouse\n"
        self.info.text += "3: Bohemian Rhapsody by Queen\n"
        """

    def on_resize(self, win_size):
        self.song_select_display.on_resize(win_size)


class SongSelectDisplay(InstructionGroup):
    def __init__(self):
        super(SongSelectDisplay, self).__init__()

        self.window_width = Window.width
        self.window_height = Window.height

        self.add(Color(0, 0, 0, 0))
        self.all_star_button = CRectangle(
            cpos=((self.window_width / 2) - 500, self.window_height / 2),
            csize=(1000, 100),
        )
        self.add(self.all_star_button)
        self.add(Color(1, 1, 1))
        self.all_star_text = CLabelRect(
            cpos=((self.window_width / 2) - 600, (self.window_height / 2)),
            text='"1": All Star by Smash Mouth',
            font_size="14",
        )
        self.add(self.all_star_text)

        self.add(Color(0, 0, 0, 0))
        self.valerie_button = CRectangle(
            cpos=(self.window_width / 2, self.window_height / 2), csize=(1000, 100)
        )
        self.add(self.valerie_button)
        self.add(Color(1, 1, 1))
        self.valerie_text = CLabelRect(
            cpos=((self.window_width / 2), (self.window_height / 2)),
            text='"2": Valerie by Amy Winehouse',
            font_size="14",
        )
        self.add(self.valerie_text)

        self.add(Color(0, 0, 0, 0))
        self.bohemian_button = CRectangle(
            cpos=((self.window_width / 2) + 500, self.window_height / 2),
            csize=(1000, 100),
        )
        self.add(self.bohemian_button)
        self.add(Color(1, 1, 1))
        self.bohemian_text = CLabelRect(
            cpos=((self.window_width / 2) + 600, (self.window_height / 2)),
            text='"3": Bohemian Rhapsody by Queen',
            font_size="14",
        )
        self.add(self.bohemian_text)

    def on_resize(self, win_size):
        self.window_width = win_size[0]
        self.window_height = win_size[1]
        self.bohemian_text.cpos = (self.window_width / 2 + 600, self.window_height / 2)
        self.valerie_text.cpos = (self.window_width / 2, self.window_height / 2)
        self.all_star_text.cpos = (self.window_width / 2 - 600, self.window_height / 2)


class GameScreen(Screen):
    def __init__(self, title, **kwargs):
        super(GameScreen, self).__init__(**kwargs)

        song_name = title + ".wav"
        song_name2 = title + "_music.wav"
        self.song_choice = title

        # Create Audio for stereo output AND mono input. Incoming audio data from microphone
        # gets sent to the callback self.receive_audio()

        self.audio_controller = AudioController(song_name, song_name2)
        self.audio = Audio(2, input_func=self.receive_audio, num_input_channels=1)
        self.mixer = Mixer()
        self.audio.set_generator(self.mixer)
        self.pitch_detector = PitchDetector()
        self.volume = self.pitch_detector.volume

        self.game_display = GameDisplay(self.song_choice)
        self.particle_sys = self.game_display.ps

        self.canvas.add(self.game_display)
        self.add_widget(self.particle_sys)

    def on_update(self):
        self.audio.on_update()

        pitch = self.pitch_detector.pitch
        conf = self.pitch_detector.conf

        self.audio_controller.on_update()
        now_time = self.audio_controller.get_time()
        paused = self.audio_controller.paused

        self.volume = self.pitch_detector.volume
        self.game_display.on_update(pitch, conf, now_time, paused, self.volume)

        """
        self.info.text = f'fps:{kivyClock.get_fps():.1f}\n'
        self.info.text += f'audio load: {self.audio.get_cpu_load():.2f}\n'
        self.info.text += f'pitch: {pitch:.2f}\n'
        self.info.text += f'conf: {conf:.2f}\n'
        self.info.text+=f'success: {self.game_display.success}\n'
        self.info.text+=f'vocal range: {self.game_display.vocal_range}\n'
        self.info.text += f"Press P to play/pause!"
        """

    def receive_audio(self, frames, num_channels):
        # this just handles one channel. If you want to support stereo input,
        # mix down stereo to mono before proceeding
        assert num_channels == 1
        self.pitch_detector.insert(frames)

    def on_key_down(self, keycode, modifiers):
        # play / pause toggle
        if keycode[1] == "p":
            self.audio_controller.toggle()
        elif keycode[1] == "r":
            self.switch_to("song_select_screen")
            self.game_display.score = 0

    def on_resize(self, win_size):
        self.game_display.on_resize(win_size)


# Screen Manager Setup
sm = ScreenManager()

sm.add_screen(SongSelectScreen(name="song_select_screen"))
sm.add_screen(GameScreen(title="allstar", name="allstar"))
sm.add_screen(GameScreen(title="valerie", name="valerie"))
sm.add_screen(GameScreen(title="bohemian", name="bohemian"))

run(sm)
