"""
Qatalyst Pizza Engine - Scoring module

Ports the exact scoring logic from Pizza Race play.html (evalVanRoute, evalSolution)
to Python. Matching this is critical: if our API disagrees with the game on what
a route scores, the "Qatalyst engine" panel looks broken.

Reference: /mnt/user-data/uploads/play.html lines 3325-3443
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ========= Problem model =========

@dataclass
class Customer:
    id: int
    x: float
    y: float
    hot_by: Optional[float] = None  # minutes; None means no time window
    vip: bool = False
    maybe: bool = False


@dataclass
class Stage:
    """Matches a STAGE entry from the game's STAGES dict."""
    stage_id: int
    shop_x: float
    shop_y: float
    customers: list[Customer]
    multi_van: bool = False
    constraints: bool = False          # False for stage 0 (no constraints)
    fuel_tank: Optional[float] = None  # per-van fuel capacity
    shared_charger: bool = False
    has_vip: bool = False
    has_maybe: bool = False
    boss: bool = False

    def shop(self) -> tuple[float, float]:
        return (self.shop_x, self.shop_y)


# ========= Distance =========

def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Match JS dist(): Math.hypot(a.x-b.x, a.y-b.y)/10"""
    return math.hypot(a[0] - b[0], a[1] - b[1]) / 10


# ========= Van-route evaluation =========

@dataclass
class VanResult:
    d: float = 0.0           # total distance
    fuel: float = 0.0        # fuel used
    cold: int = 0            # cold deliveries
    hot: int = 0             # hot deliveries
    vip_cold: int = 0        # cold VIP deliveries
    over_fuel: bool = False
    charged_at: int = -1     # index in route where charging happened
    charge_time: float = 0   # minutes spent charging
    has_charged: bool = False
    skipped: int = 0         # maybe-customers who didn't show
    arr: list = field(default_factory=list)  # per-stop arrival info


def _pt(c_or_shop) -> tuple[float, float]:
    """Coerce customer or shop into (x, y) tuple."""
    if isinstance(c_or_shop, tuple):
        return c_or_shop
    return (c_or_shop.x, c_or_shop.y)


def eval_van_route(
    route: list[int],
    stage: Stage,
    realised_set: Optional[set[int]] = None,
) -> VanResult:
    """
    Port of JS evalVanRoute().

    Walks a single van's route from shop, through customers in order, back to shop.
    Computes distance, fuel, cold/hot deliveries, VIP cold, shared-charger events.
    """
    vr = VanResult()
    if not route:
        return vr

    t = 0.0
    prev = stage.shop()
    for idx, ci in enumerate(route):
        c = stage.customers[ci]

        # STAGE 8: maybe-customer that didn't show
        if stage.has_maybe and c.maybe and realised_set is not None and ci not in realised_set:
            vr.skipped += 1
            vr.arr.append({"cust": ci, "arrMin": t, "cold": False, "cancelled": True})
            continue

        leg = dist(prev, _pt(c))

        # Shared charger check (Stage 7): do we need to go back and charge?
        if stage.shared_charger and not vr.has_charged:
            leg_back = dist(_pt(c), stage.shop())
            has_remaining = idx + 1 < len(route)
            fuel_ceiling = (stage.fuel_tank or 0) * 0.85
            if has_remaining and vr.fuel + leg + leg_back > fuel_ceiling:
                # Detour to shop to charge
                back_to_shop = dist(prev, stage.shop())
                vr.d += back_to_shop
                vr.fuel = 0.0
                t += back_to_shop
                vr.charge_time = 4.0
                t += vr.charge_time
                vr.has_charged = True
                vr.charged_at = idx
                prev = stage.shop()
                new_leg = dist(stage.shop(), _pt(c))
                vr.d += new_leg
                vr.fuel += new_leg
                t += new_leg
            else:
                vr.d += leg
                vr.fuel += leg
                t += leg
        else:
            vr.d += leg
            vr.fuel += leg
            t += leg

        # Check hot/cold
        if stage.constraints:
            is_cold = c.hot_by is not None and t > c.hot_by
            vr.arr.append({"cust": ci, "arrMin": t, "cold": is_cold})
            if is_cold:
                vr.cold += 1
                if c.vip:
                    vr.vip_cold += 1
        else:
            vr.arr.append({"cust": ci, "arrMin": t, "cold": False})

        prev = _pt(c)

    # Return to shop
    return_leg = dist(prev, stage.shop())
    vr.d += return_leg
    vr.fuel += return_leg

    delivered = len(route) - vr.skipped
    vr.hot = delivered - vr.cold
    vr.over_fuel = (
        stage.constraints
        and stage.fuel_tank is not None
        and vr.fuel > stage.fuel_tank
        and not vr.has_charged
    )

    # Round to match JS display
    vr.d = round(vr.d * 10) / 10
    vr.fuel = round(vr.fuel * 10) / 10
    return vr


# ========= Solution evaluation (1 or 2 vans) =========

@dataclass
class SolutionScore:
    vans: list[VanResult]
    d: float
    cold: int
    hot: int
    vip_cold: int
    over_fuel: bool
    all_hot: bool
    valid: bool
    score: int
    charger_conflict: bool


def eval_solution(
    solution,
    stage: Stage,
    realised_set: Optional[set[int]] = None,
) -> SolutionScore:
    """
    Port of JS evalSolution().

    solution: either a list[int] (single van) or list[list[int]] (multi van)
    """
    # Normalise to list of routes
    if solution and isinstance(solution[0], list):
        routes = solution
    else:
        routes = [solution]

    van_results = [eval_van_route(r, stage, realised_set) for r in routes]

    # Shared-charger conflict penalty (Stage 7 multi-van only)
    charger_conflict = False
    if stage.shared_charger and len(van_results) == 2:
        v1, v2 = van_results
        if v1.has_charged and v2.has_charged:
            charger_conflict = True
            # Push van 2's post-charge arrivals forward by 4 minutes.
            # Recompute cold flags for affected stops.
            for i in range(v2.charged_at, len(v2.arr)):
                stop = v2.arr[i]
                c = stage.customers[stop["cust"]]
                new_arr = stop["arrMin"] + 4
                was_hot = not stop["cold"]
                is_cold_now = c.hot_by is not None and new_arr > c.hot_by
                if was_hot and is_cold_now:
                    stop["cold"] = True
                    v2.cold += 1
                    if c.vip:
                        v2.vip_cold += 1
                    v2.hot -= 1

    total_d = sum(r.d for r in van_results)
    total_cold = sum(r.cold for r in van_results)
    total_hot = sum(r.hot for r in van_results)
    total_vip_cold = sum(r.vip_cold for r in van_results)
    any_over_fuel = any(r.over_fuel for r in van_results)
    total_customers = sum(len(r) for r in routes)
    valid = total_customers == len(stage.customers)

    if stage.constraints:
        score = total_hot * 100 - total_cold * 50 - total_vip_cold * 150
        for r in van_results:
            if r.over_fuel and stage.fuel_tank is not None:
                score -= (r.fuel - stage.fuel_tank) * 5
        if charger_conflict:
            score -= 30
    else:
        score = -round(total_d * 10)

    return SolutionScore(
        vans=van_results,
        d=round(total_d * 10) / 10,
        cold=total_cold,
        hot=total_hot,
        vip_cold=total_vip_cold,
        over_fuel=any_over_fuel,
        all_hot=(total_cold == 0 and valid),
        valid=valid,
        score=round(score),
        charger_conflict=charger_conflict,
    )
