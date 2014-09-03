#!/usr/bin/env python3

# Pong
# Written in 2013 by Julian Marchant <onpon4@riseup.net>
#
# To the extent possible under law, the author(s) have dedicated all
# copyright and related and neighboring rights to this software to the
# public domain worldwide. This software is distributed without any
# warranty.
#
# You should have received a copy of the CC0 Public Domain Dedication
# along with this software. If not, see
# <http://creativecommons.org/publicdomain/zero/1.0/>.

import sge
import Queue
import threading
import math
import sys
import numpy.random as random
import pipereader

PADDLE_SPEED = 2
COMPUTER_PADDLE_SPEED = 2
PADDLE_VERTICAL_FORCE = 1 / 12
BALL_START_SPEED = 2
BALL_ACCELERATION = 0.1
BALL_MAX_SPEED = 15
FPS = 4
FRAMES_PER_STATE = int(math.ceil(0.05 * FPS))



class glob:
    # This class is for global variables.  While not necessary, using a
    # container class like this is less potentially confusing than using
    # actual global variables.

    players = [None, None]
    ball = None

    hits = [0, 0]
    misses = [0, 0]

class Game(sge.Game):
    def __init__(self, *args, **kwargs):
        self.reader = pipereader.PipeReader(sys.stdin)

        super(Game, self).__init__(*args, **kwargs)

    def event_key_press(self, key, char):
        if key == 'f8':
            sge.Sprite.from_screenshot().save('screenshot.jpg')
        elif key == 'escape':
            self.event_close()
        elif key in ('p', 'enter'):
            self.pause()

    def event_close(self):
        self.end()

    def event_paused_key_press(self, key, char):
        if key == 'escape':
            # This allows the player to still exit while the game is
            # paused, rather than having to unpause first.
            self.event_close()
        else:
            self.unpause()

    def event_paused_close(self):
        # This allows the player to still exit while the game is paused,
        # rather than having to unpause first.
        self.event_close()

    def event_step(self, t):
        input = self.reader.readline()
        while input is not None:
            data = input.split()
            if data[0] == "move":
                glob.players[int(data[1])].command = float(data[2])
            else:
                print "unrecognized input (%s)" % input

            input = self.reader.readline()

class ComputerPlayer(sge.StellarClass):
    lock = None
    queue = None

    def __init__(self, player_num):
        x = 32 if player_num == 0 else sge.game.width - 32
        y = sge.game.height / 2
        self.player_num = player_num
        self.hit_direction = 1 if player_num == 0 else -1
        self.command = 0

        super(ComputerPlayer, self).__init__(x, y, sprite="paddle_pc")

    def event_step(self, time_passed):
#        move_direction = 0
#        self.lock.acquire()
#        while not self.queue.empty():
#            move_direction = self.queue.get()
#        self.lock.release()
        self.yvelocity = self.command * COMPUTER_PADDLE_SPEED

        # Keep the paddle inside the window
        if self.bbox_top < 0:
            self.bbox_top = 0
        elif self.bbox_bottom > sge.game.height:
            self.bbox_bottom = sge.game.height

class Player(sge.StellarClass):

    def __init__(self, player_num):
        if player_num == 0:
            self.up_key = "w"
            self.down_key = "s"
            x = 32
            self.hit_direction = 1
        else:
            self.up_key = "up"
            self.down_key = "down"
            x = sge.game.width - 32
            self.hit_direction = -1

        self.player_num = player_num
        y = sge.game.height / 2

        super(Player, self).__init__(x, y, 0, sprite="paddle")

    def event_step(self, time_passed):
        # Movement
        key_motion = (sge.get_key_pressed(self.down_key) -
                      sge.get_key_pressed(self.up_key))

        self.yvelocity = key_motion * PADDLE_SPEED

        # Keep the paddle inside the window
        if self.bbox_top < 0:
            self.bbox_top = 0
        elif self.bbox_bottom > sge.game.height:
            self.bbox_bottom = sge.game.height


class Ball(sge.StellarClass):

    def __init__(self):
        x = sge.game.width / 2
        y = sge.game.height / 2

        super(Ball, self).__init__(x, y, 1, sprite="ball")

    def event_create(self):
        self.serve()
        self.set_alarm(0, FRAMES_PER_STATE)

    def event_alarm(self, alarm_id):
        print "state %f %f %f %f" % (glob.ball.x, glob.ball.y, glob.players[0].y, glob.players[1].y)
        self.set_alarm(alarm_id, FRAMES_PER_STATE)

    def event_step(self, time_passed):
        # Scoring
        loser = None
        if self.bbox_right < 0:
            loser = 0
        elif self.bbox_left > sge.game.width:
            loser = 1

        if loser is not None:
            glob.misses[loser] += 1

            self.serve(1 if loser == 0 else -1)

            print "reward %d %f" % (loser, -1)
            
        print "stats %d %d %d %d" % (glob.hits[0], glob.misses[0],
                                     glob.hits[1], glob.misses[1])


        # Bouncing off of the edges
        if self.bbox_bottom > sge.game.height:
            self.bbox_bottom = sge.game.height
            self.yvelocity = -abs(self.yvelocity)
        elif self.bbox_top < 0:
            self.bbox_top = 0
            self.yvelocity = abs(self.yvelocity)


    def event_collision(self, other):
        if isinstance(other, (ComputerPlayer, Player)):
            if other.player_num == 0:
                self.bbox_left = other.bbox_right + 1
                self.xvelocity = min(abs(self.xvelocity) + BALL_ACCELERATION, BALL_MAX_SPEED)
                hitter = 0
            else:
                self.bbox_right = other.bbox_left - 1
                self.xvelocity = max(-abs(self.xvelocity) - BALL_ACCELERATION, -BALL_MAX_SPEED)
                hitter = 1
            self.yvelocity += (self.y - other.y) * (PADDLE_VERTICAL_FORCE + 0.01)

            glob.hits[hitter] += 1

            print "reward %d %f" % (hitter, 1)

    def serve(self, direction=1):
        self.x = sge.game.width / 2 + (200 if direction == -1 else -200)
        self.y = random.randint(0, sge.game.height)

        # Next round
        self.xvelocity = BALL_START_SPEED * direction
        self.yvelocity = 0


def main(players, seed=None):
    random.seed(seed)

    # Create Game object
    Game(640, 480, fps=FPS)

    # Load sprites
    paddle_sprite = sge.Sprite(ID="paddle", width=8, height=80, origin_x=4,
                               origin_y=40)
    paddle_sprite.draw_rectangle(0, 0, paddle_sprite.width,
                                 paddle_sprite.height, fill="white")

    paddle_sprite_pc = sge.Sprite(ID="paddle_pc", width=8, height=80, origin_x=4,
                                  origin_y=40)
    paddle_sprite_pc.draw_rectangle(0, 0, paddle_sprite.width,
                                 paddle_sprite.height, fill="white")


    ball_sprite = sge.Sprite(ID="ball", width=32, height=32, origin_x=16,
                             origin_y=16)
    ball_sprite.draw_rectangle(0, 0, ball_sprite.width, ball_sprite.height,
                               fill="white")

#    glob.hud_sprite = sge.Sprite(width=320, height=160, origin_x=160,
#                                 origin_y=0)
#    hud = sge.StellarClass(sge.game.width / 2, 0, -10, sprite=glob.hud_sprite,
#                           detects_collisions=False)

    # Load backgrounds
    layers = (sge.BackgroundLayer("ball", sge.game.width / 2, 0, -10000,
                                  xrepeat=False),)
    background = sge.Background (layers, "black")

#    # Load fonts
#    sge.Font('Liberation Mono', ID="hud", size=24)

    # Create objects
    for i in range(2):
        glob.players[i] = Player(i) if players[i] == "human" else \
                          ComputerPlayer(i)
    glob.ball = Ball()

    objects = glob.players + [glob.ball]

    # Create rooms
    sge.Room(objects, background=background)

    sge.game.start()

main(["computer", "computer"])
