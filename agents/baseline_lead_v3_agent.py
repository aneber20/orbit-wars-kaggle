import math


def fleet_speed(num_ships, max_speed=6.0):
    num_ships = max(1, int(num_ships))
    ratio = math.log(num_ships) / math.log(1000)
    ratio = max(0.0, ratio)
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


def angle_diff(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def distance(a, b):
    return math.hypot(a[2] - b[2], a[3] - b[3])


def split_planets(obs):
    player = obs["player"]
    planets = obs["planets"]
    my_planets = [p for p in planets if p[1] == player]
    targets = [p for p in planets if p[1] != player]
    return my_planets, targets


def get_angle(source, target):
    return math.atan2(target[3] - source[3], target[2] - source[2])


def get_nearest_target(source, targets):
    return min(targets, key=lambda t: distance(source, t))


def planet_xy_at(planet, obs, turns_ahead=0.0):
    if turns_ahead <= 0:
        return planet[2], planet[3]

    pid = planet[0]
    for group in obs.get("comets", []):
        if pid in group.get("planet_ids", []):
            idx = group.get("planet_ids", []).index(pid)
            path = group.get("paths", [[]])[idx]
            path_index = group.get("path_index", -1)
            future_index = max(
                0,
                min(len(path) - 1, path_index + int(math.ceil(turns_ahead))),
            )
            if path:
                return path[future_index][0], path[future_index][1]

    initial_by_id = {p[0]: p for p in obs.get("initial_planets", [])}
    initial = initial_by_id.get(pid)
    angular_velocity = obs.get("angular_velocity")
    if initial is None or angular_velocity is None:
        return planet[2], planet[3]

    dx = initial[2] - 50.0
    dy = initial[3] - 50.0
    orbital_radius = math.hypot(dx, dy)
    if orbital_radius + planet[4] >= 50.0:
        return planet[2], planet[3]

    initial_angle = math.atan2(dy, dx)
    future_step = obs.get("step", 0) + turns_ahead
    future_angle = initial_angle + angular_velocity * future_step
    return (
        50.0 + orbital_radius * math.cos(future_angle),
        50.0 + orbital_radius * math.sin(future_angle),
    )


def get_lead_angle(source, target, ships_to_send, obs, lead_scale=1.0, iterations=5):
    speed = fleet_speed(max(1, ships_to_send))
    turns = distance(source, target) / speed

    for _ in range(iterations):
        tx, ty = planet_xy_at(target, obs, turns * lead_scale)
        turns = math.hypot(tx - source[2], ty - source[3]) / speed

    tx, ty = planet_xy_at(target, obs, turns * lead_scale)
    return math.atan2(ty - source[3], tx - source[2])


def path_hits_sun(source, angle, sun_center=(50.0, 50.0), sun_radius=10.0, buffer=0.5):
    sx, sy = source[2], source[3]
    cx, cy = sun_center
    dx, dy = math.cos(angle), math.sin(angle)
    vx, vy = cx - sx, cy - sy

    t = vx * dx + vy * dy
    if t <= 0:
        return False

    closest_x = sx + t * dx
    closest_y = sy + t * dy
    return math.hypot(closest_x - cx, closest_y - cy) <= sun_radius + buffer


def infer_fleet_target(fleet, planets, angle_threshold=0.12, obs=None):
    fx, fy = fleet[2], fleet[3]
    fleet_angle = fleet[4]
    origin_id = fleet[5]

    best_planet = None
    best_diff = float("inf")

    for planet in planets:
        if planet[0] == origin_id:
            continue

        if obs is None:
            target_angle = math.atan2(planet[3] - fy, planet[2] - fx)
        else:
            pseudo_source = [-1, fleet[1], fx, fy, 0, fleet[6], 0]
            target_angle = get_lead_angle(pseudo_source, planet, fleet[6], obs)
        diff = angle_diff(target_angle, fleet_angle)

        if diff < best_diff:
            best_diff = diff
            best_planet = planet

    if best_planet is None or best_diff >= angle_threshold:
        return None

    return best_planet


def get_reserved_targets(obs, angle_threshold=0.1, use_capture_filter=False):
    player = obs["player"]
    planets = obs["planets"]
    reserved = set()

    for fleet in obs["fleets"]:
        if fleet[1] != player:
            continue

        target = infer_fleet_target(fleet, planets, angle_threshold=angle_threshold, obs=obs)
        if target is None:
            continue

        if use_capture_filter and fleet[6] <= target[5]:
            continue

        reserved.add(target[0])

    return reserved


def incoming_ships_by_planet(obs, owner=None, enemy_only=False, angle_threshold=0.12):
    player = obs["player"]
    planets = obs["planets"]
    incoming = {p[0]: 0 for p in planets}

    for fleet in obs["fleets"]:
        if enemy_only and fleet[1] == player:
            continue
        if owner is not None and fleet[1] != owner:
            continue

        target = infer_fleet_target(fleet, planets, angle_threshold=angle_threshold, obs=obs)
        if target is not None:
            incoming[target[0]] = incoming.get(target[0], 0) + fleet[6]

    return incoming


def estimate_target_defense(source, target, ships_to_send):
    arrival_turns = distance(source, target) / fleet_speed(max(1, ships_to_send))
    return math.ceil(target[5] + target[6] * arrival_turns)


def get_best_value_target(source, targets, score_type="prod_dist", eps=1e-6):
    def score(target):
        dist = distance(source, target)
        ships = target[5]
        prod = target[6]

        if score_type == "prod_dist":
            return prod / (dist + eps)
        if score_type == "prod_ship_dist":
            return prod / ((ships + 1) * (dist + eps))
        if score_type == "prod_over_ships":
            return prod / (ships + 1)
        if score_type == "nearest":
            return -dist

        raise ValueError(f"Unknown score_type: {score_type}")

    return max(targets, key=score)


def filter_targets_by_enemy_radar(obs, targets, angle_threshold=0.15):
    enemy_incoming = incoming_ships_by_planet(
        obs,
        enemy_only=True,
        angle_threshold=angle_threshold,
    )
    filtered = []

    for target in targets:
        if target[1] != -1:
            filtered.append(target)
            continue

        if enemy_incoming.get(target[0], 0) >= target[5] + 1:
            continue

        filtered.append(target)

    return filtered


def agent(obs):
    moves = []
    step = obs["step"]
    player = obs["player"]
    my_planets, targets = split_planets(obs)

    if not my_planets or not targets:
        return moves

    if step < 50:
        reserved_targets = set()
    else:
        reserved_targets = get_reserved_targets(
            obs,
            angle_threshold=0.1,
            use_capture_filter=False,
        )

    for source in my_planets:
        available_targets = [t for t in targets if t[0] not in reserved_targets]
        if not available_targets:
            continue

        available_targets = filter_targets_by_enemy_radar(
            obs,
            available_targets,
            angle_threshold=0.15,
        )
        if not available_targets:
            continue

        target = get_best_value_target(source, available_targets, score_type="prod_dist")

        ships_needed = target[5] + 1
        if target[1] not in (-1, player):
            ships_needed += 5
            estimated = estimate_target_defense(source, target, ships_needed)
            ships_needed = max(ships_needed, math.ceil(estimated * 0.8) + 1)

        if source[5] >= ships_needed:
            angle = get_lead_angle(source, target, ships_needed, obs)
            if not path_hits_sun(source, angle, buffer=0.5):
                moves.append([source[0], angle, ships_needed])
                reserved_targets.add(target[0])

    return moves
