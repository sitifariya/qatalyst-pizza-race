"""
Qatalyst engine - Pizza Race stage definitions

Mirrors the STAGES dict in play.html exactly. If the game updates, update this too.

Reference: /mnt/user-data/uploads/play.html lines 2806-3260
"""
from scoring import Customer, Stage


def _make_stages() -> dict[int, Stage]:
    """Return all 9 stages keyed by stage_id."""

    stages: dict[int, Stage] = {}

    # ========= STAGE 0: First delivery =========
    stages[0] = Stage(
        stage_id=0,
        shop_x=350, shop_y=200,
        multi_van=False,
        constraints=False,
        customers=[
            Customer(id=0, x=160, y=120),
            Customer(id=1, x=560, y=160),
            Customer(id=2, x=340, y=310),
        ],
    )

    # ========= STAGE 1: The rush (5 customers, time windows, fuel 40) =========
    stages[1] = Stage(
        stage_id=1,
        shop_x=350, shop_y=200,
        multi_van=False, constraints=True, fuel_tank=40,
        customers=[
            Customer(id=0, x=140, y=115, hot_by=24),
            Customer(id=1, x=560, y=105, hot_by=28),
            Customer(id=2, x=100, y=295, hot_by=40),
            Customer(id=3, x=580, y=300, hot_by=32),
            Customer(id=4, x=350, y=340, hot_by=20),
        ],
    )

    # ========= STAGE 2: The split (6 customers, 2 vans, fuel 28) =========
    stages[2] = Stage(
        stage_id=2,
        shop_x=350, shop_y=200,
        multi_van=True, constraints=True, fuel_tank=28,
        customers=[
            Customer(id=0, x=120, y=105, hot_by=28),
            Customer(id=1, x=590, y=110, hot_by=32),
            Customer(id=2, x=90,  y=300, hot_by=42),
            Customer(id=3, x=600, y=300, hot_by=34),
            Customer(id=4, x=260, y=330, hot_by=22),
            Customer(id=5, x=460, y=335, hot_by=24),
        ],
    )

    # ========= STAGE 3: The rush (re-teach, same as stage 1 structure) =========
    # play.html re-uses The rush with identical setup for teaching depth
    stages[3] = Stage(
        stage_id=3,
        shop_x=350, shop_y=200,
        multi_van=False, constraints=True, fuel_tank=40,
        customers=[
            Customer(id=0, x=140, y=115, hot_by=24),
            Customer(id=1, x=560, y=105, hot_by=28),
            Customer(id=2, x=100, y=295, hot_by=40),
            Customer(id=3, x=580, y=300, hot_by=32),
            Customer(id=4, x=350, y=340, hot_by=20),
        ],
    )

    # ========= STAGE 4: The split (re-teach, same structure as stage 2) =========
    stages[4] = Stage(
        stage_id=4,
        shop_x=350, shop_y=200,
        multi_van=True, constraints=True, fuel_tank=28,
        customers=[
            Customer(id=0, x=120, y=105, hot_by=28),
            Customer(id=1, x=590, y=110, hot_by=32),
            Customer(id=2, x=90,  y=300, hot_by=42),
            Customer(id=3, x=600, y=300, hot_by=34),
            Customer(id=4, x=260, y=330, hot_by=22),
            Customer(id=5, x=460, y=335, hot_by=24),
        ],
    )

    # ========= STAGE 5: VIP rush (7 customers, 2 VIPs) =========
    stages[5] = Stage(
        stage_id=5,
        shop_x=350, shop_y=200,
        multi_van=False, constraints=True, fuel_tank=50, has_vip=True,
        customers=[
            Customer(id=0, x=120, y=105, hot_by=34, vip=True),
            Customer(id=1, x=580, y=110, hot_by=32),
            Customer(id=2, x=100, y=295, hot_by=38),
            Customer(id=3, x=600, y=295, hot_by=38, vip=True),
            Customer(id=4, x=230, y=340, hot_by=24),
            Customer(id=5, x=470, y=340, hot_by=24),
            Customer(id=6, x=350, y=345, hot_by=20),
        ],
    )

    # ========= STAGE 6: Friday night madness [BOSS] =========
    stages[6] = Stage(
        stage_id=6,
        shop_x=350, shop_y=200,
        multi_van=True, constraints=True, fuel_tank=28,
        has_vip=True, boss=True,
        customers=[
            Customer(id=0, x=110, y=95,  hot_by=36, vip=True),
            Customer(id=1, x=590, y=100, hot_by=34),
            Customer(id=2, x=85,  y=295, hot_by=42),
            Customer(id=3, x=605, y=295, hot_by=40, vip=True),
            Customer(id=4, x=215, y=340, hot_by=26),
            Customer(id=5, x=480, y=335, hot_by=24),
            Customer(id=6, x=235, y=85,  hot_by=22),
            Customer(id=7, x=475, y=85,  hot_by=24),
        ],
    )

    # ========= STAGE 7: Charger shuffle (8 customers, 2 vans, shared charger) =========
    stages[7] = Stage(
        stage_id=7,
        shop_x=350, shop_y=200,
        multi_van=True, constraints=True, fuel_tank=22,
        shared_charger=True,
        customers=[
            Customer(id=0, x=110, y=95,  hot_by=38),
            Customer(id=1, x=590, y=100, hot_by=34),
            Customer(id=2, x=85,  y=295, hot_by=42),
            Customer(id=3, x=605, y=295, hot_by=40),
            Customer(id=4, x=215, y=340, hot_by=26),
            Customer(id=5, x=480, y=335, hot_by=24),
            Customer(id=6, x=235, y=85,  hot_by=20),
            Customer(id=7, x=475, y=85,  hot_by=22),
        ],
    )

    # ========= STAGE 8: The maybe list (8 customers, 3 maybes) =========
    stages[8] = Stage(
        stage_id=8,
        shop_x=350, shop_y=200,
        multi_van=False, constraints=True, fuel_tank=55,
        has_maybe=True,
        customers=[
            Customer(id=0, x=120, y=105, hot_by=34),
            Customer(id=1, x=585, y=110, hot_by=34),
            Customer(id=2, x=95,  y=300, hot_by=40, maybe=True),
            Customer(id=3, x=600, y=290, hot_by=38),
            Customer(id=4, x=225, y=335, hot_by=22, maybe=True),
            Customer(id=5, x=475, y=335, hot_by=24),
            Customer(id=6, x=240, y=90,  hot_by=20, maybe=True),
            Customer(id=7, x=465, y=90,  hot_by=22),
        ],
    )

    return stages


STAGES = _make_stages()
