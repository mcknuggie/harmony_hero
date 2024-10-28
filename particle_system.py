#####################################################################
#
# This software is to be used for MIT's class Interactive Music Systems only.
# Since this file may contain answers to homework problems, you MAY NOT release it publicly.
#
#####################################################################
#
# example_particle_system.py
# Example code of how to use the particle system
# for more information on creating and editing particles, see: ../imslib/kivyparticle/README.md
#
#####################################################################

import sys, os
sys.path.insert(0, os.path.abspath('..'))

from imslib.core import BaseWidget, run, lookup
from imslib.gfxutil import topleft_label
from imslib.kivyparticle import ParticleSystem

from kivy.clock import Clock as kivyClock

from random import random



class MainWidget(BaseWidget):
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        self.info = topleft_label()
        self.add_widget(self.info)

        # load up the particle system, set initial emitter position and start it.
        self.ps = ParticleSystem('particle/particle.pex')
        self.ps.emitter_x = 300.0
        self.ps.emitter_y = 300.0
        self.ps.start()

        # particle system is a widget, not a canvas instruction.
        self.add_widget(self.ps)

    def on_key_down(self, keycode, modifiers):
        # These are just a few of the particle parameters you can customize.
        # You can view and edit all of these parameters using editor.py in imslib/kivyparticle
        # The readme in imslib/kivyparticle explains how to use the editor
        if keycode[1] == '1':
            self.ps.start_color = [random(), random(), random(), random()] # RGBA
            self.ps.end_color = [random(), random(), random(), random()]   # RGBA

        elif keycode[1] == '2':
            self.ps.speed = min(self.ps.speed * 1.2, 300)

        elif keycode[1] == '3':
            self.ps.speed = max(30, self.ps.speed / 1.2)

        elif keycode[1] == '4':
            self.ps.max_num_particles = min(self.ps.max_num_particles + 100, 1000)

        elif keycode[1] == '5':
            self.ps.max_num_particles = max(10, self.ps.max_num_particles - 100)

        elif keycode[1] == '6':
            if self.ps.emission_time == 0:
                self.ps.start()
            else:
                self.ps.stop()


    def on_touch_down(self, touch):
        # set emitter location based on mouse position
        self.ps.emitter_x = touch.pos[0]
        self.ps.emitter_y = touch.pos[1]

    def on_touch_up(self, touch):
        self.ps.emitter_x = touch.pos[0]
        self.ps.emitter_y = touch.pos[1]

    def on_touch_move(self, touch):
        self.ps.emitter_x = touch.pos[0]
        self.ps.emitter_y = touch.pos[1]

    def on_update(self):

        # update info
        self.info.text = f'fps:{kivyClock.get_fps():.0f}\n'
        self.info.text += f'num_particles:{self.ps.num_particles}  '
        self.info.text += f'speed:{self.ps.speed:.2f}\n\n'

        self.info.text += 'Click or drag to move emitter location\n'
        self.info.text += '1: randomize colors\n'
        self.info.text += '2: increase particle speed\n'
        self.info.text += '3: decrease particle speed\n'
        self.info.text += '4: increase number of particles\n'
        self.info.text += '5: decrease number of particles\n'
        self.info.text += '6: start/stop toggle\n'

run(MainWidget())
