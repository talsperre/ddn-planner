#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#          Mithun (mithun.babu@research.iiit.ac.in)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
    Example of automatic vehicle control using frenet frames from client side.

    Use CTRL+c to quit (will take 5-10 sec to disable syncronous mode) 

    Adjust simulation parameters on line 106
"""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import queue
import weakref
import pdb
import time
import xlwt
import csv
import skvideo.io
import numpy as np
sys.path.append('../')
import fplan.cubic_spline_planner
import fplan.frenet_optimal_trajectory as frenet_optimal_trajectory
import matplotlib.pyplot as plt
from fplan.frenet_optimal_trajectory import frenet_optimal_planning
from functools import wraps
from xlwt import Workbook

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla')[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.basic_agent import BasicAgent
from agents.tools.misc import is_within_distance

# ==============================================================================
# -- Simulation Parameters -----------------------------------------------------
# ==============================================================================

FPS = 10.0
DTFPS = 1.0/(FPS)
INTERSECTION_START = True
DEBUG_PATH = True
SAVE_VIDEO = True
SAVE_DATA = True
SPAWN_PLANNED = True
RECORD_DATA = True

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def simple_time_tracker(log_fun):
    def _simple_time_tracker(fn):
        @wraps(fn)
        def wrapped_fn(*args, **kwargs):
            start_time = time.time()

            try:
                result = fn(*args, **kwargs)
            finally:
                elapsed_time = time.time() - start_time

                # log the result
                log_fun({
                    'function_name': fn.__name__,
                    'total_time': elapsed_time,
                })
                
            return result

        return wrapped_fn
    return _simple_time_tracker

def _log(message):
    print('[SimpleTimeTracker] {function_name} {total_time:.3f}'.format(**message))

def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    return array

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    def __init__(self, carla_world, hud, actor_filter, ego_start):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart(ego_start)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, ego_start):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        
        # Get a blue ford mustang
        blueprint = self.world.get_blueprint_library().find(self._actor_filter)
        blueprint.set_attribute('role_name', 'hero')
        blueprint.set_attribute('color', '0,0,0')
        # Spawn the player.
        if INTERSECTION_START:
            spawn_points = self.map.get_spawn_points()
            spawn_point = spawn_points[ego_start]
        else:
            # point1
            # spawn_point = carla.Transform(carla.Location(x=4.5, y=-55, z=0.05), \
            #                             carla.Rotation(yaw=-90))
            # point 2
            # spawn_point = carla.Transform(carla.Location(x=94.0, y=-4.5, z=0.05), \
            #                             carla.Rotation(yaw=180))
            # point 3
            spawn_point = carla.Transform(carla.Location(x=-148.5, y=59, z = 0.05), \
                                          carla.Rotation(yaw=90))
        if self.player is not None:
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # spawn at random spawn_point or at (0,0,0)
        while self.player is None:
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()

# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    currentIndex = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(currentIndex)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' % (
                            'Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not (pygame.key.get_mods() & KMOD_CTRL):
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                keys = pygame.key.get_pressed()
                if sum(keys) > 0:
                    self._parse_vehicle_keys(keys, clock.get_time())
                    self._control.reverse = self._control.gear < 0
                    world.player.apply_control(self._control)
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
                world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

            def distance(l): return math.sqrt(
                (l.x - t.location.x) ** 2 + (l.y - t.location.y) ** 2 + (l.z - t.location.z) ** 2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame_number, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame_number)


# ==============================================================================
# -- A*_waypoints() ---------------------------------------------------------
# ==============================================================================


def get_points(world, agent):
    tops = agent._local_planner._waypoints_queue
    wrs = world.world
    wx = []
    wy = []
    dl = []
    dr = []
    for i in range(len(tops)-1):
        w1 = tops[i][0]
        w2 = tops[i+1][0]
        lane_width = w1.lane_width

        lcolor = w1.left_lane_marking.color
        ltype = w1.left_lane_marking.type
        rcolor = w1.right_lane_marking.color
        rtype = w1.right_lane_marking.type

        lchange = w1.lane_change

        if lchange == carla.libcarla.LaneChange.Both:
            dleft = -lane_width
            dright = lane_width
        elif lchange == carla.libcarla.LaneChange.Right:
            dleft = -0.01
            dright = lane_width
        elif lchange == carla.libcarla.LaneChange.Left:
            dleft = -lane_width
            dright = 0.01
        elif lchange == carla.libcarla.LaneChange.NONE:
            dleft = -0.01
            dright = 0.01
        else:
            raise ValueError('LaneChange decision error')
        # dleft = 0
        # dright = 0.001

        if i > 0:
            temp_1x = wx[-1]
            temp_1y = wy[-1]
            temp_2x = w1.transform.location.x
            temp_2y = w1.transform.location.y
            if ((temp_1x-temp_2x)**2+ (temp_1y-temp_2y)**2) <= 1e-5:
                continue

        wx.append(w1.transform.location.x)
        wy.append(w1.transform.location.y)
        dl.append(dleft)
        dr.append(dright)
        # if DEBUG_PATH:
        #     wrs.debug.draw_line(w1.transform.location, w2.transform.location, 
        #                         color = carla.Color(r=0, g=0, b=255), thickness=0.3, \
        #                         life_time=50, persistent_lines=True)
    return wx, wy, dl, dr



def get_obstacles(world, agent):
    ob = []
    actor_list = world.world.get_actors()
    vehicle_list = actor_list.filter("*vehicle*")
    ego_location = agent._vehicle.get_location()
    ego_location.x = ego_location.x
    ego_location.y = ego_location.y
    ego_id = agent._vehicle.id
    for target_vehicle in vehicle_list:
        # do not account for the ego vehicle
        if target_vehicle.id == ego_id:
            continue
        target_location = target_vehicle.get_location()
        if is_within_distance(target_location, ego_location, 40):
            temp_vel = target_vehicle.get_velocity()
            temp_vel = (np.hypot(temp_vel.x, temp_vel.y))
            #   print(temp_vel)
            temp_omega = target_vehicle.get_angular_velocity()
            temp_omega = np.hypot(temp_omega.z, temp_omega.y)
            temp_ob = {}
            temp_ob['x0'] = target_location.x
            temp_ob['y0'] = target_location.y
            temp_ob['th0'] = (target_vehicle.get_transform().rotation.yaw)*(math.pi/180) 
            temp_ob['v'] = temp_vel
            temp_ob['w'] = temp_omega*(math.pi/180) #not documented if deg/s or rad/s
            ob.append(temp_ob)
    return ob


def hazard_check(world, agent, allowed_cut, DEBUG_PATH):
        # code sourced from basic_agent.py

        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = world.world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        # check possible obstacles
        vehicle_state, vehicle, target_dist_1 = agent._is_vehicle_hazard(vehicle_list)
        if vehicle_state and not(allowed_cut):
            # if DEBUG_PATH:
            #     print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            hazard_detected = True
            target_dist_1 = target_dist_1 - 6.0

        # check for the state of the traffic lights
        # light_state, traffic_light, target_dist_2 = agent._is_light_red(lights_list, DEBUG_PATH)
        # if light_state:
        #     hazard_detected = True
        #     if target_dist_2 >= 27:
        #         target_dist_2 = target_dist_2 - 26.0
        #     else:
        #         target_dist_2 = 27.0/target_dist_2
        target_dist_2 = -1000.0
        target_dist = max(target_dist_1, target_dist_2)
        if target_dist == target_dist_1:
            harzard_type = 'VEHICLE'
        else:
            harzard_type = 'LIGHT'

        return hazard_detected, target_dist, harzard_type



# ==============================================================================
# -- game_loop() ---------------------------------------------------------
# ==============================================================================

def game_loop(args):
    all_pts = {}
    town_no = 3
    
    if town_no == 5:
        ## town 5
        #### intersection 01
        temp = {}
        temp['n'] = 3
        temp[0] = [104, 115, 134, 135, 105, 106]
        temp[1] = [26, 27, 156, 157, 82, 88]
        temp[2] = [73, 75]
        temp['g'] = [21, 132, 208]
        all_pts[1] = temp

        #### intersection 02
        temp = {}
        temp['n'] = 3
        temp[0] = [163, 164, 165]
        temp[1] = [132, 133, 127, 138]
        temp[2] = [170, 177, 178]
        temp['g'] = [135, 171, 170]
        all_pts[2] = temp

        #### intersection 03
        temp = {}
        temp['n'] = 4
        temp[0] = [92, 91]
        temp[1] = [73, 75]
        temp[2] = [149, 150, 111, 113]
        temp[3] = [32, 33, 0, 2]
        temp['g'] = [185, 185, 18, 65]
        all_pts[3] = temp


        #### intersection 04
        temp = {}
        temp['n'] = 3
        temp[0] = [74, 79, 20, 21, 147, 148]
        temp[1] = [160, 161, 28, 29, 128, 129]
        temp[2] = [107, 108]
        temp['g'] = [8, 157, 27]
        all_pts[4] = temp


        #### intersection 05
        temp = {}
        temp['n'] = 4
        temp[0] = [93, 94]
        temp[1] = [18, 19, 54, 64]
        temp[2] = [24, 25, 187, 188, 47, 49]
        temp[3] = [17, 16]
        temp['g'] = [0, 126, 194, 162]
        all_pts[5] = temp

        
        #### intersection 06
        temp = {}
        temp['n'] = 3
        temp[0] = [1, 6, 189, 190, 46, 48]
        temp[1] = [201, 206]
        temp[2] = [120, 121]
        temp['g'] = [140, 192, 22]
        all_pts[6] = temp

    elif town_no == 3:
        #town 03
        ### intersection 01
        temp = {}
        temp['n'] = 3
        temp[0] = [74, 73]
        temp[1] = [172, 173]
        temp[2] = [62, 63]
        temp['g'] = [61, 72, 72]
        all_pts[1] = temp

        ### intersection 02
        temp = {}
        temp['n'] = 4
        temp[0] = [100, 36, 92, 35]
        temp[1] = [130, 120, 103, 102]
        temp[2] = [32, 33, 34, 90, 101, 108, 43, 111]
        temp[3] = [28, 29, 30, 31, 193, 194]
        temp['g'] = [25, 25, 132, 132]
        all_pts[2] = temp

        ### intersection 03
        temp = {}
        temp['n'] = 2
        temp[0] = [163]
        temp[1] = [93, 94, 26, 27, 16, 25, 14, 15, 12, 13, 10, 11]
        temp[2] = [191]
        temp[3] = [61, 199]
        temp['g'] = [125, 9, 195, 30]
        all_pts[3] = temp

        ### intersection 04
        temp = {}
        temp['n'] = 4
        temp[0] = [183, 182, 56, 55, 84, 81, 119, 116]
        temp[1] = [146, 145, 48, 47, 52, 49, 54, 53, 97, 98]
        temp[2] = [192, 8, 9, 164]
        temp[3] = [175, 174, 0, 7, 57, 58, 153, 154]
        temp['g'] = [42, 44, 42, 1]
        all_pts[4] = temp

        ### intersection 05
        temp = {}
        temp['n'] = 2
        temp[0] = [50, 51]
        temp[1] = [158, 151, 60, 59]
        temp['g'] = [188, 188]
        all_pts[5] = temp
    else:
        print("Not selected any town")
        return


    sel_intersection = random.randint(1, len(all_pts))
    intersection_list = all_pts[sel_intersection]
    ego_direction = random.randint(0, intersection_list['n']-1)
    ego_start = random.choice(intersection_list[ego_direction])
    ego_end = intersection_list['g'][ego_direction]
    other_directions = list(range(intersection_list['n']))
    if ego_direction in intersection_list: 
        other_directions.remove(ego_direction)

    ilist = []
    sel_cars = random.randint(1, len(other_directions))
    for car in range(sel_cars):
        cur_direction = random.choice(other_directions)
        ilist.append(random.choice(intersection_list[cur_direction]))
        other_directions.remove(cur_direction)

    pygame.init()
    pygame.font.init()
    world = None

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)

        if RECORD_DATA:
            client.start_recorder(args.file_name + '.log')
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args.filter, ego_start)
        controller = KeyboardControl(world, False)
        blueprint_library = world.world.get_blueprint_library()

        print('enabling synchronous mode.')
        settings = world.world.get_settings()
        settings.synchronous_mode = True
        world.world.apply_settings(settings)

        if args.agent == "Roaming":
            agent = RoamingAgent(world.player)
        else:
            agent = BasicAgent(world.player)
            spawn_point = world.map.get_spawn_points()[ego_end]
            if INTERSECTION_START:
                agent.set_destination((spawn_point.location.x,
                                       spawn_point.location.y,
                                       spawn_point.location.z))
            else:
                # point 1
                # agent.set_destination((-61.0, -139.0, 0.0))
                # point 2
                # agent.set_destination((-58.0, -3.0, 0.0))
                # point 3
                agent.set_destination((25.8, -206, 0.0))

        # waypoints to destination
        wx, wy, dl, dr = get_points(world, agent)
        ego_loc = agent._vehicle.get_location()

        # camera to record video
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
        semantic_segmentation_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        depth_bp = blueprint_library.find('sensor.camera.depth')

        camera_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15))
        camera_rgb_tf = carla.Transform(carla.Location(x=0.8, z=1.7))
        semantic_segmentation_tf = carla.Transform(carla.Location(x=0.8, z=1.7))
        depth_tf = carla.Transform(carla.Location(x=0.8, z=1.7))

        camera = world.world.spawn_actor(camera_bp, camera_transform, attach_to=agent._vehicle)
        camera_rgb = world.world.spawn_actor(camera_rgb_bp, camera_rgb_tf, attach_to=agent._vehicle)
        semantic_segmentation = world.world.spawn_actor(semantic_segmentation_bp, semantic_segmentation_tf, attach_to=agent._vehicle)
        depth = world.world.spawn_actor(depth_bp, depth_tf, attach_to=agent._vehicle)

        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        camera_rgb_queue = queue.Queue()
        camera_rgb.listen(camera_rgb_queue.put)

        semantic_segmentation_queue = queue.Queue()
        semantic_segmentation.listen(semantic_segmentation_queue.put)

        depth_queue = queue.Queue()
        depth.listen(depth_queue.put)

        # do orthogonal projection of initial location
        p0 = np.array([[wx[0], wy[0]]])
        v = np.array([[(wx[1]-wx[0]), (wy[1]-wy[0])]])
        p = np.array([[ego_loc.x, ego_loc.y]])
        temp_p = np.dot(v.T, v)/(np.dot(v, v.T)).item()
        fp = np.dot(temp_p, p.T) + np.dot((np.identity(2)-temp_p),p0.T)
        wx.pop(0)
        wy.pop(0)
        wx.insert(0,fp[0].item())
        wy.insert(0,fp[1].item())

        tx, ty, tyaw, tc, csp = frenet_optimal_trajectory.generate_target_course(wx, wy)
        dsp = csp.s
        clock = pygame.time.Clock()
        ego_location = agent._vehicle.get_location()
        c_speed = 10.0*np.random.random()  # current speed [m/s]
        c_s_dd = 0
        # check outer product
        mx = (ego_location.x-tx[0])*(ty[1]-ty[0]) - (ego_location.y-ty[0])*(tx[1]-tx[0])
        if mx >= 0:
            c_d = -np.hypot(ego_location.x - tx[0], ego_location.y - ty[0])  # current lateral position [m]
        else:
            c_d = np.hypot(ego_location.x - tx[0], ego_location.y - ty[0])

        c_d_d = 0.0  # current lateral speed [m/s]
        c_d_dd = 0.0  # current latral acceleration [m/s]
        s0 = 0.0  # current course position
        agent._vehicle.set_simulate_physics(False)
        frame = None
        timestr = time.strftime("%Y%m%d-%H%M%S")
        save_folder = '/home/sraone/mithun/temp2/set8'
        # save_folder = '/media/sraone/06705C17056FE5FC/carla_intersection_data/set1'
        if SAVE_VIDEO:
            # cur_date = datetime.datetime.today().strftime('%Y_%m_%d')
            os.makedirs(save_folder, exist_ok=True)            
            vid_path = os.path.join(save_folder, str(args.file_name) + ".mp4")
            writer = skvideo.io.FFmpegWriter(vid_path, inputdict={'-r':'10'}, \
                                             outputdict={'-r':'10'})

        allowed_cut = True

        iters = 0
        start_time = time.time()
        spawn_distance = 60.0 # meters
        s_spawn = -1.0
        flag = 1
        actor_list = [camera, camera_rgb, semantic_segmentation, depth]
        ego_loc_list = [['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'frame_num']]
        obstacle_loc_list = [['id', 'x', 'y', 'z', 'yaw', 'pitch', 'roll', 'frame_num']]
        skip_time = True
        skip_counter = 0
        
        while skip_time:
            if SPAWN_PLANNED and flag:
                spawn_points = world.map.get_spawn_points()
                vlist = [spawn_points[j] for j in ilist]
                bps = blueprint_library.filter('vehicle.nissan.micra')
                spawned_vehicle_counter = 0
                for i in range(len(vlist)):
                    vehicle = world.world.try_spawn_actor(bps[0], vlist[i])
                    if vehicle is not None:
                        vehicle.set_autopilot(enabled=True)
                        actor_list.append(vehicle)
                        print('spawned vehicle ', str(i))
                        spawned_vehicle_counter = spawned_vehicle_counter + 1

                if spawned_vehicle_counter < 2:
                    raise ValueError("unable to spawn enough vehicles")
                flag = 0
            world.tick(clock)
            world.world.tick()

            if not world.world.wait_for_tick(10.0):
                continue
            world.render(display)
            pygame.display.flip()
            image = image_queue.get()
            if SAVE_VIDEO:
                image_arr = to_bgra_array(image)
                # Convert BGRA to RGB.
                image_arr = image_arr[:, :, :3]
                image_arr = image_arr[:, :, ::-1]
                writer.writeFrame(image_arr)

            if SAVE_DATA:
                camera_rgb_dir = os.path.join(save_folder, 'rgb', args.file_name)
                semantic_segmentation_dir = os.path.join(save_folder, 'semantics', args.file_name)
                depth_dir = os.path.join(save_folder, 'depth', args.file_name)

                os.makedirs(camera_rgb_dir, exist_ok=True)
                os.makedirs(semantic_segmentation_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)

                camera_path = os.path.join(camera_rgb_dir, 'img_' + str(iters).zfill(4) + '.png')
                semantic_segmentation_path = os.path.join(semantic_segmentation_dir, 'img_' + str(iters).zfill(4) + '.png')
                depth_path = os.path.join(depth_dir, 'img_' + str(iters).zfill(4) + '.png')

                camera_rgb_img = camera_rgb_queue.get()
                camera_rgb_img.save_to_disk(camera_path)

                semantic_segmentation_img = semantic_segmentation_queue.get()
                semantic_segmentation_img.save_to_disk(semantic_segmentation_path)

                depth_img = depth_queue.get()
                depth_img.save_to_disk(depth_path)

                ego_tf = agent._vehicle.get_transform()

                ego_loc_list.append([
                    ego_tf.location.x, 
                    ego_tf.location.y, 
                    ego_tf.location.z, 
                    ego_tf.rotation.yaw,
                    ego_tf.rotation.pitch,
                    ego_tf.rotation.roll,
                    iters
                ])

                actors_all = world.world.get_actors().filter('vehicle.*')

                for actor in actors_all:
                    if actor.id != agent._vehicle.id:
                        actor_tf = actor.get_transform()
                        obstacle_loc_list.append([
                            actor.id,
                            actor_tf.location.x,
                            actor_tf.location.y,
                            actor_tf.location.z,
                            actor_tf.rotation.yaw,
                            actor_tf.rotation.pitch,
                            actor_tf.rotation.roll,
                            iters
                        ])

            skip_counter = skip_counter + 1
            print('SKIP COUNTER : ', skip_counter)
            if skip_counter >= 50:
                skip_time = False

        while True:
        
            if controller.parse_events(client, world, clock):
                return
            ob = get_obstacles(world, agent)

            print("-"*100)
            print("Actors Position")
            for i, actor in enumerate(actor_list):
                print(i, actor.get_location().x, actor.get_location().y)

            world.tick(clock)
            world.world.tick()

            if not world.world.wait_for_tick(10.0):
                continue
            world.render(display)
            pygame.display.flip()

            hazard_detected, target_dist, harzard_type = hazard_check(world, agent, allowed_cut, DEBUG_PATH)
            OPTION = []
            print('target_dist : ', target_dist)
            if target_dist >= 0.5 or target_dist <= -999.0:
                if hazard_detected:
                    OPTION.append('STOP')
                    OPTION.append(target_dist)
                    OPTION.append(harzard_type)
                    print('STOP')
                else:
                    OPTION.append('OVERTAKE')
                    print('OVERTAKE')

                SETS = {'MAX_SPEED':12.5, 'TARGET_SPEED':8.33, 'D_T_S':1.0, 'N_S_SAMPLE':8}
                path, selected_paths, allowed_cut, ob, ob_cv = frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, \
                                           c_s_dd, ob, dl, dr, dsp, DTFPS, OPTION, SETS, -100, -100, -100, "cv", -100, -100, -100)
            else:
                pass


            for j in range(len(ob)):
                xs = ob[j]['x']
                ys = ob[j]['y']
                for i in range(len(xs)-1):
                    w1 = carla.Location(x=xs[i], y=ys[i], z=0)
                    wp1 = world.map.get_waypoint(w1)
                    w1 = carla.Location(x=xs[i], y=ys[i], z=wp1.transform.location.z)
                    w2 = carla.Location(x=xs[i+1], y=ys[i+1], z=0)
                    wp2 = world.map.get_waypoint(w2)
                    w2 = carla.Location(x=xs[i+1], y=ys[i+1], z=wp2.transform.location.z)
                    world.world.debug.draw_line(w1, w2, \
                                          color = carla.Color(r=0, g=0, b=255), thickness=0.1, \
                                          life_time=0.1, persistent_lines=True)

            # needs to be changed if physics is on
            s0 = path.s[1]
            c_d = path.d[1]
            c_d_d = path.d_d[1]
            c_d_dd = path.d_dd[1]
            c_speed = path.s_d[1]
            c_s_dd = path.s_dd[1]
            print('path cost : ', path.cf)
            print('current yaw : ', (path.yaw[1])*(180/math.pi))
            print('current s : ', s0)
            print('time selected : ', path.t[-1])
            print('length of path : ', path.s[-1]-path.s[0])
            print('vehicle velocity: ', path.vx[1])

            if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 0.5:
                print("Goal reached")
                break
            elif len(path.x) < 1:
                print("End")
                break
            image = image_queue.get()
            # print('distance_travelled : ', s0)
            if SAVE_VIDEO:
                image_arr = to_bgra_array(image)
                # Convert BGRA to RGB.
                image_arr = image_arr[:, :, :3]
                image_arr = image_arr[:, :, ::-1]
                writer.writeFrame(image_arr)

            if SAVE_DATA:
                camera_rgb_dir = os.path.join(save_folder, 'rgb', args.file_name)
                semantic_segmentation_dir = os.path.join(save_folder, 'semantics', args.file_name)
                depth_dir = os.path.join(save_folder, 'depth', args.file_name)

                os.makedirs(camera_rgb_dir, exist_ok=True)
                os.makedirs(semantic_segmentation_dir, exist_ok=True)
                os.makedirs(depth_dir, exist_ok=True)

                camera_path = os.path.join(camera_rgb_dir, 'img_' + str(iters).zfill(4) + '.png')
                semantic_segmentation_path = os.path.join(semantic_segmentation_dir, 'img_' + str(iters).zfill(4) + '.png')
                depth_path = os.path.join(depth_dir, 'img_' + str(iters).zfill(4) + '.png')

                camera_rgb_img = camera_rgb_queue.get()
                camera_rgb_img.save_to_disk(camera_path)

                semantic_segmentation_img = semantic_segmentation_queue.get()
                semantic_segmentation_img.save_to_disk(semantic_segmentation_path)

                depth_img = depth_queue.get()
                depth_img.save_to_disk(depth_path)

                ego_tf = agent._vehicle.get_transform()

                ego_loc_list.append([
                    ego_tf.location.x, 
                    ego_tf.location.y, 
                    ego_tf.location.z, 
                    ego_tf.rotation.yaw,
                    ego_tf.rotation.pitch,
                    ego_tf.rotation.roll,
                    iters
                ])

                actors_all = world.world.get_actors().filter('vehicle.*')

                for actor in actors_all:
                    if actor.id != agent._vehicle.id:
                        actor_tf = actor.get_transform()
                        obstacle_loc_list.append([
                            actor.id,
                            actor_tf.location.x,
                            actor_tf.location.y,
                            actor_tf.location.z,
                            actor_tf.rotation.yaw,
                            actor_tf.rotation.pitch,
                            actor_tf.rotation.roll,
                            iters
                        ])

            temp_w = world.world.get_map().get_waypoint(carla.Location(x=path.x[1], y=path.y[1], z=0))
                
            if DEBUG_PATH:
                for i in range(len(path.x)-1):
                    w1 = carla.Location(x=path.x[i], y=path.y[i], z=0)
                    wp1 = world.map.get_waypoint(w1)
                    w1 = carla.Location(x=path.x[i], y=path.y[i], z=wp1.transform.location.z)
                    w2 = carla.Location(x=path.x[i+1], y=path.y[i+1], z=0)
                    wp2 = world.map.get_waypoint(w2)
                    w2 = carla.Location(x=path.x[i+1], y=path.y[i+1], z=wp2.transform.location.z)
                    world.world.debug.draw_line(w1, w2, \
                                          color = carla.Color(r=255, g=0, b=0), thickness=0.1, \
                                          life_time=0.1, persistent_lines=True)
                    if i == 0:
                        world.world.debug.draw_line(w1, w2, \
                                                    color = carla.Color(r=0, g=255, b=0), thickness=0.3, \
                                                    life_time=10, persistent_lines=True)
#                for j in selected_paths:
#                    for i in range(len(j.x)-1):
#                        w1 = carla.Location(x=j.x[i], y=j.y[i], z=0)
#                        wp1 = world.map.get_waypoint(w1)
#                        w1 = carla.Location(x=j.x[i], y=j.y[i], z=wp1.transform.location.z)
#                        w2 = carla.Location(x=j.x[i+1], y=j.y[i+1], z=0)
#                        wp2 = world.map.get_waypoint(w2)
#                        w2 = carla.Location(x=j.x[i+1], y=j.y[i+1], z=wp2.transform.location.z)
#                        world.world.debug.draw_line(w1, w2, \
#                                              color = carla.Color(r=255, g=0, b=0), thickness=0.1, \
#                                              life_time=0.1, persistent_lines=True)
            # if abs(path.yaw[1] - path.yaw[0])*(180/math.pi) > 160 and abs(path.yaw[1] - path.yaw[0])*(180/math.pi) < 340:
            #     temp_yaw = path.yaw[0]
            # else:
            #     temp_yaw = path.yaw[1]
            agent._vehicle.set_transform(carla.Transform(carla.Location(x=path.x[1], y=path.y[1], \
                                        z=temp_w.transform.location.z), carla.Rotation(yaw=(path.yaw[1])*(180/math.pi), \
                                                                                       pitch=temp_w.transform.rotation.pitch)))
            
            # control = agent.run_step()
            # print(control.throttle)
            # control.manual_gear_shift = False
            # world.player.apply_control(control)
            print('simulation time :', iters*DTFPS)
            if iters*DTFPS >= 180.0:
                break
            iters = iters + 1
            print('--------------------------------------------------------------------')

    finally:
        if RECORD_DATA:
            client.stop_recorder()
        if SAVE_DATA:
            ego_loc_dir = os.path.join(save_folder, 'ego_loc')
            obstacle_loc_dir = os.path.join(save_folder, 'obstacle_loc')

            os.makedirs(ego_loc_dir, exist_ok=True)
            os.makedirs(obstacle_loc_dir, exist_ok=True)

            actor_path = os.path.join(ego_loc_dir, args.file_name + ".csv")
            with open(actor_path, "w") as f:
                csv_writer = csv.writer(f, delimiter=",")
                csv_writer.writerows(ego_loc_list)

            obstacle_path = os.path.join(obstacle_loc_dir, args.file_name + ".csv")
            with open(obstacle_path, "w") as f:
                csv_writer = csv.writer(f, delimiter=",")
                csv_writer.writerows(obstacle_loc_list)

        if world is not None:
            world.destroy()
        for actor in actor_list:
            actor.destroy()
        print('\ndisabling synchronous mode.')
        settings = world.world.get_settings()
        settings.synchronous_mode = False
        world.world.apply_settings(settings)
        if SAVE_VIDEO:
            writer.close()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='640x480',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.nissan.micra',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Roaming", "Basic"],
                           help="select which agent to run",
                           default="Basic")
    argparser.add_argument(
        '--file_name',
        default='1',
        help='File name')    
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()