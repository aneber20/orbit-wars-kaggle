import math


CONFIG = {
    "opening_until": 90,
    "enemy_attack_after": 80,
    "reservation_after": 45,
    "reserved_angle": 0.10,
    "radar_angle": 0.14,
    "sun_buffer": 0.5,
    "opening_reserve": 0,
    "mid_reserve": 1,
    "late_reserve": 2,
    "production_reserve_scale": 0.10,
    "opening_max_send_fraction": 0.95,
    "mid_max_send_fraction": 0.90,
    "late_max_send_fraction": 0.82,
    "max_targets_per_source": 8,
    "max_moves_per_turn": 999,
    "allow_duplicate_targets": False,
    "score_mode": "cost_aware",
    "neutral_bonus": 1.20,
    "enemy_bonus_mid": 0.85,
    "enemy_bonus_late": 1.10,
    "cost_weight": 0.25,
    "pressure_weight": 0.30,
    "distance_scale": 24.0,
    "production_offset": 0.25,
    "neutral_overkill": 1,
    "enemy_margin": 5,
    "estimate_scale": 0.8,
    "defend_after": 85,
    "defense_extra": 1,
    "max_defense_moves": 2,
}


def fleet_speed(num_ships, max_speed=6.0):
    num_ships = max(1, int(num_ships))
    ratio = math.log(num_ships) / math.log(1000)
    ratio = max(0.0, ratio)
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


def distance(a, b):
    return math.hypot(a[2] - b[2], a[3] - b[3])


def angle_diff(a, b):
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def split_planets(obs):
    player = obs["player"]
    planets = obs["planets"]
    my_planets = [p for p in planets if p[1] == player]
    targets = [p for p in planets if p[1] != player]
    return my_planets, targets


def get_angle(source, target):
    return math.atan2(target[3] - source[3], target[2] - source[2])


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


def infer_fleet_target(fleet, planets, angle_threshold=0.12):
    fx, fy = fleet[2], fleet[3]
    fleet_angle = fleet[4]
    origin_id = fleet[5]
    best_planet = None
    best_diff = float("inf")

    for planet in planets:
        if planet[0] == origin_id:
            continue

        target_angle = math.atan2(planet[3] - fy, planet[2] - fx)
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

        target = infer_fleet_target(fleet, planets, angle_threshold=angle_threshold)
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

        target = infer_fleet_target(fleet, planets, angle_threshold=angle_threshold)
        if target is not None:
            incoming[target[0]] = incoming.get(target[0], 0) + fleet[6]

    return incoming


def estimate_target_defense(source, target, ships_to_send):
    arrival_turns = distance(source, target) / fleet_speed(max(1, ships_to_send))
    return math.ceil(target[5] + target[6] * arrival_turns)


def v2_phase_reserve(source, step, config):
    if step < config["opening_until"]:
        base = config["opening_reserve"]
    elif step < 140:
        base = config["mid_reserve"]
    else:
        base = config["late_reserve"]

    return int(math.ceil(base + source[6] * config["production_reserve_scale"]))


def v2_max_send_fraction(step, config):
    if step < config["opening_until"]:
        return config["opening_max_send_fraction"]
    if step < 140:
        return config["mid_max_send_fraction"]
    return config["late_max_send_fraction"]


def v2_sendable_ships(source, step, already_launched, config):
    remaining = max(0, source[5] - already_launched)
    reserve = v2_phase_reserve(source, step, config)
    fraction_cap = int(math.floor(remaining * v2_max_send_fraction(step, config)))
    return max(0, min(remaining - reserve, fraction_cap))


def v2_ships_needed(source, target, player, config):
    if target[1] == -1:
        return target[5] + config["neutral_overkill"]

    base = target[5] + 1 + config["enemy_margin"]
    estimated = estimate_target_defense(source, target, base)
    return max(base, math.ceil(estimated * config["estimate_scale"]) + 1)


def v2_target_phase_allowed(target, step, any_neutrals, player, config):
    if target[1] == -1:
        return True
    if not any_neutrals:
        return True
    return step >= config["enemy_attack_after"]


def v2_candidate_score(source, target, obs, ships_needed, enemy_incoming, config):
    step = obs["step"]
    dist = max(distance(source, target), 1e-6)
    prod = target[6]
    owner = target[1]
    pressure = enemy_incoming.get(target[0], 0)

    if owner == -1 and pressure >= target[5] + config["neutral_overkill"] + 3:
        return -1e12

    if owner == -1:
        ownership_bonus = config["neutral_bonus"] if step < config["opening_until"] else 1.0
    else:
        ownership_bonus = config["enemy_bonus_late"] if step >= 140 else config["enemy_bonus_mid"]

    baseline_score = (prod + config["production_offset"]) * ownership_bonus / dist
    pressure_cost = config["pressure_weight"] * pressure
    cost_term = max(1.0, ships_needed + pressure_cost)
    return baseline_score / (cost_term ** config["cost_weight"])


def v2_build_candidates(obs, my_planets, targets, reserved_targets, launched_by_source, config):
    player = obs["player"]
    step = obs["step"]
    any_neutrals = any(t[1] == -1 for t in targets)
    enemy_incoming = incoming_ships_by_planet(
        obs, enemy_only=True, angle_threshold=config["radar_angle"]
    )
    candidates = []

    for source in my_planets:
        already = launched_by_source.get(source[0], 0)
        sendable = v2_sendable_ships(source, step, already, config)
        if sendable <= 0:
            continue

        plausible_targets = []
        for target in targets:
            if not config["allow_duplicate_targets"] and target[0] in reserved_targets:
                continue
            if not v2_target_phase_allowed(target, step, any_neutrals, player, config):
                continue

            ships_needed = v2_ships_needed(source, target, player, config)
            if ships_needed > sendable:
                continue

            angle = get_angle(source, target)
            if path_hits_sun(source, angle, buffer=config["sun_buffer"]):
                continue

            score = v2_candidate_score(source, target, obs, ships_needed, enemy_incoming, config)
            if score <= -1e11:
                continue

            plausible_targets.append((score, source, target, angle, ships_needed))

        plausible_targets.sort(reverse=True, key=lambda item: item[0])
        candidates.extend(plausible_targets[:config["max_targets_per_source"]])

    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates


def v2_add_late_defense(obs, my_planets, moves, launched_by_source, config):
    if obs["step"] < config["defend_after"]:
        return

    enemy_incoming = incoming_ships_by_planet(
        obs, enemy_only=True, angle_threshold=config["radar_angle"]
    )
    threatened = []

    for planet in my_planets:
        incoming = enemy_incoming.get(planet[0], 0)
        deficit = incoming + config["defense_extra"] - planet[5]
        if deficit > 0:
            threatened.append((deficit, planet))

    threatened.sort(reverse=True, key=lambda item: item[0])
    defense_moves = 0

    for deficit, target in threatened:
        donors = sorted(
            [p for p in my_planets if p[0] != target[0]],
            key=lambda p: distance(p, target),
        )

        for source in donors:
            already = launched_by_source.get(source[0], 0)
            sendable = v2_sendable_ships(source, obs["step"], already, config)
            ships = min(deficit, sendable)
            if ships <= 0:
                continue

            angle = get_angle(source, target)
            if path_hits_sun(source, angle, buffer=config["sun_buffer"]):
                continue

            moves.append([source[0], angle, ships])
            launched_by_source[source[0]] = launched_by_source.get(source[0], 0) + ships
            defense_moves += 1
            break

        if defense_moves >= config["max_defense_moves"]:
            break


def agent(obs):
    moves = []
    step = obs["step"]
    my_planets, targets = split_planets(obs)

    if not my_planets or not targets:
        return moves

    launched_by_source = {}

    if step >= CONFIG["reservation_after"]:
        reserved_targets = get_reserved_targets(
            obs,
            angle_threshold=CONFIG["reserved_angle"],
            use_capture_filter=False,
        )
    else:
        reserved_targets = set()

    candidates = v2_build_candidates(
        obs=obs,
        my_planets=my_planets,
        targets=targets,
        reserved_targets=reserved_targets,
        launched_by_source=launched_by_source,
        config=CONFIG,
    )

    used_sources = set()
    used_targets = set(reserved_targets)

    for score, source, target, angle, ships_needed in candidates:
        if len(moves) >= CONFIG["max_moves_per_turn"]:
            break
        if source[0] in used_sources:
            continue
        if not CONFIG["allow_duplicate_targets"] and target[0] in used_targets:
            continue

        already = launched_by_source.get(source[0], 0)
        if ships_needed > v2_sendable_ships(source, step, already, CONFIG):
            continue

        moves.append([source[0], angle, ships_needed])
        used_sources.add(source[0])
        used_targets.add(target[0])
        launched_by_source[source[0]] = already + ships_needed

    v2_add_late_defense(obs, my_planets, moves, launched_by_source, CONFIG)
    return moves
