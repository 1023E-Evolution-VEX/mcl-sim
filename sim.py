from dataclasses import dataclass
import math
import random
import time
import tkinter as tk

HALF_FIELD = 72.0
HALF_ROBOT = 9.0
PARTICLE_COUNT = 500
INITIAL_SPREAD_IN = 7.0
RESAMPLE_JITTER_IN = 0.22
RESAMPLE_JITTER_RAD = 0.003


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def clamp(value, low, high):
    return max(low, min(high, value))


@dataclass
class Pose:
    x: float
    y: float
    theta: float


def margin(theta):
    return HALF_ROBOT * (abs(math.cos(theta)) + abs(math.sin(theta)))


def valid(pose):
    limit = HALF_FIELD - margin(pose.theta)
    return abs(pose.x) <= limit and abs(pose.y) <= limit


def ranges(pose):
    # simulate distance sensor readings in four directions
    result = []
    for index in range(4):
        angle = pose.theta + index * math.pi / 2
        dx, dy = math.cos(angle), math.sin(angle)
        sx, sy = pose.x + HALF_ROBOT * dx, pose.y + HALF_ROBOT * dy
        tx = (math.copysign(HALF_FIELD, dx) - sx) / dx if abs(dx) > 1e-10 else math.inf
        ty = (math.copysign(HALF_FIELD, dy) - sy) / dy if abs(dy) > 1e-10 else math.inf
        result.append(max(0, min(tx, ty)))
    return result


class Robot:
    def __init__(self):
        self.pose = Pose(-36, -24, 0)
        self.speed = 0.0

    def move(self, target, dt):
        dx, dy = target[0] - self.pose.x, target[1] - self.pose.y
        distance = math.hypot(dx, dy)
        error = wrap(math.atan2(dy, dx) - self.pose.theta) if distance > 0.5 else 0
        turn = clamp(4 * error, -2.3, 2.3) * dt
        wanted = min(28, 2 * distance) * max(0, math.cos(error)) ** 4 if distance > 0.5 else 0
        self.speed += clamp(wanted - self.speed, -65 * dt, 45 * dt)
        travel = min(self.speed * dt, distance)
        # Integrate small steps so the full rotated square remains in bounds.
        steps = max(1, math.ceil(max(travel / 0.25, abs(turn) / 0.02)))
        moved = rotated = 0.0
        for _ in range(steps):
            angle = wrap(self.pose.theta + turn / steps)
            candidate = Pose(self.pose.x, self.pose.y, angle)
            if valid(candidate):
                self.pose = candidate
                rotated += turn / steps
            candidate = Pose(self.pose.x + travel / steps * math.cos(self.pose.theta),
                             self.pose.y + travel / steps * math.sin(self.pose.theta), self.pose.theta)
            if valid(candidate):
                self.pose = candidate
                moved += travel / steps
            else:
                self.speed = 0.0
        return moved, rotated


class ParticleFilter:
    def __init__(self, start, seed=None, count=PARTICLE_COUNT):
        self.rng = random.Random(seed)
        self.count = count
        self.diversity = 1.0
        self.effective_count = float(count)  # Before the latest resampling.
        # Known approximate starting pose, like the reference constructor.
        self.particles = []
        while len(self.particles) < count:
            p = Pose(self.rng.gauss(start.x, INITIAL_SPREAD_IN), self.rng.gauss(start.y, INITIAL_SPREAD_IN),
                     wrap(self.rng.gauss(start.theta, 0.09)))
            if valid(p):
                self.particles.append(p)
        self.weights = [1 / count] * count
        self.estimate = Pose(start.x, start.y, start.theta)

    def spread(self):
        mx = sum(p.x * w for p, w in zip(self.particles, self.weights))
        my = sum(p.y * w for p, w in zip(self.particles, self.weights))
        return math.sqrt(sum(w * ((p.x-mx)**2 + (p.y-my)**2)
                             for p, w in zip(self.particles, self.weights)))

    def scatter(self):
        # resetting the filter to a random pose
        diversity = self.diversity
        self.__init__(self.estimate, self.rng.randrange(2**32), self.count)
        self.diversity = diversity

    def predict(self, travel, rotation):
        for p in self.particles:
            d = travel + self.rng.gauss(0, 0.025 * abs(travel) + 0.008)
            a = rotation + self.rng.gauss(0, 0.01 * abs(rotation) + 0.0008)
            mid = p.theta + a / 2
            p.x += d * math.cos(mid) + self.rng.gauss(0, 0.008)
            p.y += d * math.sin(mid) + self.rng.gauss(0, 0.008)
            p.theta = wrap(p.theta + a)

    def update(self, measurements):
        # Gaussian range likelihoods
        logs = []
        for p, weight in zip(self.particles, self.weights):
            score = math.log(max(weight, 1e-300))
            if not valid(p):
                logs.append(-math.inf)
                continue
            for observed, expected in zip(measurements, ranges(p)):
                sigma = 0.7 + 0.01 * observed
                score -= 0.5 * ((observed - expected) / sigma) ** 2
            logs.append(score)
        peak = max(logs)
        if not math.isfinite(peak):
            self.scatter()
            return
        unscaled = [math.exp(value - peak) for value in logs]
        total = sum(unscaled)
        self.weights = [value / total for value in unscaled]
        self.estimate = Pose(
            sum(p.x * w for p, w in zip(self.particles, self.weights)),
            sum(p.y * w for p, w in zip(self.particles, self.weights)),
            math.atan2(sum(math.sin(p.theta) * w for p, w in zip(self.particles, self.weights)),
                       sum(math.cos(p.theta) * w for p, w in zip(self.particles, self.weights))))
        effective_count = 1 / sum(w * w for w in self.weights)
        self.effective_count = effective_count
        if effective_count < self.count * 0.6:
            self.resample()

    def resample(self):
        start = self.rng.random() / self.count
        index, cumulative = 0, self.weights[0]
        new = []
        for i in range(self.count):
            threshold = start + i / self.count
            while index < self.count - 1 and threshold > cumulative:
                index += 1
                cumulative += self.weights[index]
            parent = self.particles[index]
            child = Pose(parent.x + self.rng.gauss(0, RESAMPLE_JITTER_IN * self.diversity),
                         parent.y + self.rng.gauss(0, RESAMPLE_JITTER_IN * self.diversity),
                         wrap(parent.theta + self.rng.gauss(0, RESAMPLE_JITTER_RAD * self.diversity)))
            if not valid(child):
                child = Pose(parent.x, parent.y, parent.theta)
            new.append(child)
        self.particles = new
        self.weights = [1 / self.count] * self.count


class App:
    def __init__(self, root):
        self.root = root
        root.title('Monte Carlo localization')
        root.resizable(False, False)
        self.canvas = tk.Canvas(root, width=680, height=765, bg='#101419', highlightthickness=0)
        self.canvas.pack()
        self.robot = Robot()
        self.filter = ParticleFilter(Pose(-36, -24, 0))
        self.sensor_rng = random.Random()
        self.target = (-36, -24)
        self.canvas.bind('<Motion>', self.mouse)
        root.bind('<Escape>', lambda event: root.destroy())
        root.bind('<bracketleft>', lambda event: self.adjust_diversity(-0.5))
        root.bind('<bracketright>', lambda event: self.adjust_diversity(0.5))
        root.bind('<space>', lambda event: self.filter.scatter())
        self.last = time.perf_counter()
        self.sensor_elapsed = 0.0
        self.draw()
        self.tick()

    def adjust_diversity(self, amount):
        self.filter.diversity = clamp(self.filter.diversity + amount, 0.5, 8.0)

    def mouse(self, event):
        limit = HALF_FIELD - HALF_ROBOT * math.sqrt(2) - 0.5 #don't let robot go outside the field when turning
        self.target = (clamp((event.x - 340) / 4.2, -limit, limit),
                       clamp((340 - event.y) / 4.2, -limit, limit))

    @staticmethod
    def screen(x, y):
        return 340 + x * 4.2, 340 - y * 4.2

    def step(self, dt):
        travel, rotation = self.robot.move(self.target, dt)
        # add variation to readings to simulate wheel slip and IMU noise, but not enough to break the filter
        self.filter.predict(travel * 1.01, rotation)
        self.sensor_elapsed += dt
        if self.sensor_elapsed >= 0.05:
            measurements = [max(0, d + self.sensor_rng.gauss(0, 0.35))
                            for d in ranges(self.robot.pose)]
            self.filter.update(measurements)
            self.sensor_elapsed %= 0.05

    def draw(self):
        c = self.canvas
        c.delete('all')
        c.create_rectangle(*self.screen(-72, 72), *self.screen(72, -72), outline='#ced6df', width=2)
        for p in self.filter.particles:
            x, y = self.screen(p.x, p.y)
            c.create_oval(x-1.4, y-1.4, x+1.4, y+1.4, fill='#63c4a5', outline='')
        p = self.robot.pose
        co, si = math.cos(p.theta), math.sin(p.theta)
        points = []
        for x, y in [(-9, -9), (9, -9), (9, 9), (-9, 9)]:
            points.extend(self.screen(p.x + x*co - y*si, p.y + x*si + y*co))
        c.create_polygon(points, fill='', outline='#e5eaf0', width=2)
        c.create_line(*self.screen(p.x, p.y), *self.screen(p.x+8*co, p.y+8*si),
                      fill='#e5eaf0', arrow=tk.LAST, width=2)
        estimate = self.filter.estimate
        c.create_text(340, 681, text=f'X: {estimate.x:.2f} in     Y: {estimate.y:.2f} in',
                      fill='#e5eaf0', font=('Consolas', 15))
        c.create_text(340, 710,
                      text=f'Particles: {self.filter.count}   ESS: {self.filter.effective_count:.0f}'
                           f'   Spread: {self.filter.spread():.2f} in',
                      fill='#a8b4c1', font=('Consolas', 11))
        c.create_text(340, 737,
                      text=f'[ / ] jitter: {self.filter.diversity:.1f}x   Space: scatter',
                      fill='#a8b4c1', font=('Consolas', 11))

    def tick(self):
        now = time.perf_counter()
        dt = min(now - self.last, 0.05)
        self.last = now
        self.step(dt)
        self.draw()
        self.root.after(16, self.tick)


if __name__ == '__main__':
    root = tk.Tk()
    App(root)
    root.mainloop()
