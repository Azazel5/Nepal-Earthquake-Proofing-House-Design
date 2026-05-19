"""Field meanings for DrivenData Richter's Predictor (Nepal earthquake damage) features."""

FEATURE_GLOSSARY = {
    "building_id": "Unique identifier for each building record.",
    "geo_level_1_id": "Geographic region ID (coarse administrative level in Nepal).",
    "geo_level_2_id": "Geographic region ID (mid-level administrative division).",
    "geo_level_3_id": "Geographic region ID (finest administrative / locality level).",
    "count_floors_pre_eq": "Number of floors before the earthquake.",
    "age": "Age of the building in years (995 is used as an unknown / sentinel bucket).",
    "area_percentage": "Relative footprint area of the building (ordinal scale, higher = larger).",
    "height_percentage": "Relative height of the building (ordinal scale, higher = taller).",
    "land_surface_condition": "Surface condition around the building site.",
    "foundation_type": "Type of foundation supporting the structure.",
    "roof_type": "Roof construction category.",
    "ground_floor_type": "Ground-floor construction / material type.",
    "other_floor_type": "Upper-floor construction / material type.",
    "position": "How the building sits relative to neighbors and streets.",
    "plan_configuration": "Overall floor-plan shape / layout category.",
    "has_superstructure_adobe_mud": "Binary: adobe/mud superstructure present.",
    "has_superstructure_mud_mortar_stone": "Binary: mud-mortar stone superstructure present.",
    "has_superstructure_stone_flag": "Binary: dry stone / flagstone superstructure present.",
    "has_superstructure_cement_mortar_stone": "Binary: cement-mortar stone superstructure present.",
    "has_superstructure_mud_mortar_brick": "Binary: mud-mortar brick superstructure present.",
    "has_superstructure_cement_mortar_brick": "Binary: cement-mortar brick superstructure present.",
    "has_superstructure_timber": "Binary: timber superstructure present.",
    "has_superstructure_bamboo": "Binary: bamboo superstructure present.",
    "has_superstructure_rc_non_engineered": "Binary: non-engineered reinforced concrete present.",
    "has_superstructure_rc_engineered": "Binary: engineered reinforced concrete present.",
    "has_superstructure_other": "Binary: other / unspecified superstructure material present.",
    "legal_ownership_status": "Legal ownership or use-right category for the building.",
    "count_families": "Number of families residing in the building.",
    "has_secondary_use": "Binary: building has any secondary (non-residential) use.",
    "has_secondary_use_agriculture": "Binary: secondary use — agriculture.",
    "has_secondary_use_hotel": "Binary: secondary use — hotel / lodging.",
    "has_secondary_use_rental": "Binary: secondary use — rental income.",
    "has_secondary_use_institution": "Binary: secondary use — institution.",
    "has_secondary_use_school": "Binary: secondary use — school.",
    "has_secondary_use_industry": "Binary: secondary use — industry.",
    "has_secondary_use_health_post": "Binary: secondary use — health post.",
    "has_secondary_use_gov_office": "Binary: secondary use — government office.",
    "has_secondary_use_use_police": "Binary: secondary use — police.",
    "has_secondary_use_other": "Binary: secondary use — other.",
    "damage_grade": "Target: post-earthquake damage (1 = low, 2 = medium, 3 = high / severe).",
}

CATEGORICAL_DECODINGS = {
    "land_surface_condition": {
        "n": "normal",
        "o": "obstacle (trees, rocks, etc.)",
        "t": "tilted / sloped terrain",
    },
    "foundation_type": {
        "h": "hard rock",
        "i": "rocky or sandy soil",
        "r": "reinforced mat / slab",
        "u": "stiff mat / slab",
        "w": "weak rock",
    },
    "roof_type": {
        "n": "normal",
        "q": "quality / improved",
        "x": "special design",
    },
    "ground_floor_type": {
        "f": "floor",
        "m": "masonry",
        "v": "vault",
        "x": "wood",
        "z": "stone / mud",
    },
    "other_floor_type": {
        "j": "joists / bamboo",
        "q": "mud and stone",
        "s": "stone / rubble",
        "x": "wood",
    },
    "position": {
        "j": "adjacent to street",
        "o": "detached",
        "s": "within settlement cluster",
        "t": "touching adjacent buildings",
    },
    "plan_configuration": {
        "a": "arc-shaped",
        "c": "covered balconies",
        "d": "regular",
        "f": "flat-shaped roof plan",
        "m": "multiple interior extensions",
        "n": "non-straight walls",
        "o": "in/out corner",
        "q": "U-shaped",
        "s": "square-shaped",
        "u": "U-shaped with courtyard",
    },
    "legal_ownership_status": {
        "a": "administrative",
        "r": "rental",
        "v": "owner-occupied",
        "w": "worship / religious",
    },
}

NUMERIC_FEATURES = [
    "geo_level_1_id",
    "geo_level_2_id",
    "geo_level_3_id",
    "count_floors_pre_eq",
    "age",
    "area_percentage",
    "height_percentage",
    "count_families",
]

CATEGORICAL_FEATURES = [
    "land_surface_condition",
    "foundation_type",
    "roof_type",
    "ground_floor_type",
    "other_floor_type",
    "position",
    "plan_configuration",
    "legal_ownership_status",
]

BINARY_PREFIXES = ("has_superstructure_", "has_secondary_use")

DAMAGE_LABELS = {1: "low damage", 2: "medium damage", 3: "high damage"}
