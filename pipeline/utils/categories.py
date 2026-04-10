"""Category mapping and brand/anchor detection for Overture Maps POI data.

Key distinction:
  - "Destination anchors" = retailers that DRAW people to an area (Whole Foods, Target, Apple)
  - "Infrastructure" = gas stations, ATMs, banks - brand doesn't matter, count does
  - "Quality signals" = DTC/experiential brands that indicate a premium retail environment

For scoring:
  - Destination anchors boost score
  - Quality signal brands boost score more
  - Infrastructure is neutral at low counts, NEGATIVE at high counts (too many = strip mall)
  - Grocery proximity is a strong positive signal
"""

# ──────────────────────────────────────────────────────────────
# Super-category mapping (for Overture basic_category field)
# ──────────────────────────────────────────────────────────────

CATEGORY_MAP = {
    # Food & Dining
    "restaurant": "Food & Dining",
    "fast_food_restaurant": "Food & Dining",
    "cafe": "Food & Dining",
    "coffee_shop": "Food & Dining",
    "bakery": "Food & Dining",
    "bar": "Food & Dining",
    "pub": "Food & Dining",
    "ice_cream_shop": "Food & Dining",
    "pizza_restaurant": "Food & Dining",
    "mexican_restaurant": "Food & Dining",
    "chinese_restaurant": "Food & Dining",
    "japanese_restaurant": "Food & Dining",
    "italian_restaurant": "Food & Dining",
    "indian_restaurant": "Food & Dining",
    "thai_restaurant": "Food & Dining",
    "vietnamese_restaurant": "Food & Dining",
    "korean_restaurant": "Food & Dining",
    "american_restaurant": "Food & Dining",
    "seafood_restaurant": "Food & Dining",
    "steakhouse": "Food & Dining",
    "barbecue_restaurant": "Food & Dining",
    "sushi_restaurant": "Food & Dining",
    "deli": "Food & Dining",
    "food_court": "Food & Dining",
    "buffet_restaurant": "Food & Dining",
    "diner": "Food & Dining",
    "breakfast_restaurant": "Food & Dining",
    "brunch_restaurant": "Food & Dining",
    "juice_bar": "Food & Dining",
    "tea_house": "Food & Dining",
    "dessert_shop": "Food & Dining",
    "donut_shop": "Food & Dining",
    "bubble_tea_shop": "Food & Dining",
    "food_truck": "Food & Dining",
    "wine_bar": "Food & Dining",
    "cocktail_bar": "Food & Dining",
    "brewery": "Food & Dining",
    "nightclub": "Food & Dining",
    "lounge": "Food & Dining",
    "sandwich_shop": "Food & Dining",
    "burrito_restaurant": "Food & Dining",
    "chicken_restaurant": "Food & Dining",
    "noodle_shop": "Food & Dining",
    "dumpling_restaurant": "Food & Dining",
    "mediterranean_restaurant": "Food & Dining",
    "greek_restaurant": "Food & Dining",
    "middle_eastern_restaurant": "Food & Dining",
    "african_restaurant": "Food & Dining",
    "caribbean_restaurant": "Food & Dining",
    "french_restaurant": "Food & Dining",
    "german_restaurant": "Food & Dining",
    "spanish_restaurant": "Food & Dining",
    "tapas_bar": "Food & Dining",

    # Shopping
    "clothing_store": "Shopping",
    "shoe_store": "Shopping",
    "department_store": "Shopping",
    "shopping_mall": "Shopping",
    "electronics_store": "Shopping",
    "furniture_store": "Shopping",
    "home_improvement_store": "Shopping",
    "hardware_store": "Shopping",
    "bookstore": "Shopping",
    "gift_shop": "Shopping",
    "jewelry_store": "Shopping",
    "toy_store": "Shopping",
    "pet_store": "Shopping",
    "sporting_goods_store": "Shopping",
    "bicycle_shop": "Shopping",
    "outdoor_store": "Shopping",
    "thrift_store": "Shopping",
    "antique_store": "Shopping",
    "art_gallery": "Shopping",
    "florist": "Shopping",
    "music_store": "Shopping",
    "mobile_phone_store": "Shopping",
    "optician": "Shopping",
    "cosmetics_store": "Shopping",
    "beauty_supply_store": "Shopping",
    "discount_store": "Shopping",
    "dollar_store": "Shopping",
    "variety_store": "Shopping",
    "outlet_store": "Shopping",
    "warehouse_store": "Shopping",
    "baby_store": "Shopping",
    "bridal_shop": "Shopping",
    "fabric_store": "Shopping",
    "craft_store": "Shopping",
    "hobby_shop": "Shopping",
    "garden_center": "Shopping",
    "nursery": "Shopping",
    "lighting_store": "Shopping",
    "mattress_store": "Shopping",
    "appliance_store": "Shopping",
    "auto_parts_store": "Shopping",
    "tire_shop": "Shopping",
    "pawn_shop": "Shopping",
    "smoke_shop": "Shopping",
    "vape_shop": "Shopping",
    "wine_shop": "Shopping",
    "liquor_store": "Shopping",

    # Entertainment & Fitness
    "movie_theater": "Entertainment & Fitness",
    "amusement_park": "Entertainment & Fitness",
    "bowling_alley": "Entertainment & Fitness",
    "arcade": "Entertainment & Fitness",
    "museum": "Entertainment & Fitness",
    "zoo": "Entertainment & Fitness",
    "aquarium": "Entertainment & Fitness",
    "theater": "Entertainment & Fitness",
    "concert_hall": "Entertainment & Fitness",
    "comedy_club": "Entertainment & Fitness",
    "karaoke": "Entertainment & Fitness",
    "escape_room": "Entertainment & Fitness",
    "mini_golf": "Entertainment & Fitness",
    "go_kart_track": "Entertainment & Fitness",
    "trampoline_park": "Entertainment & Fitness",
    "water_park": "Entertainment & Fitness",
    "laser_tag": "Entertainment & Fitness",
    "skating_rink": "Entertainment & Fitness",
    "sports_complex": "Entertainment & Fitness",
    "gym": "Entertainment & Fitness",
    "fitness_center": "Entertainment & Fitness",
    "yoga_studio": "Entertainment & Fitness",
    "pilates_studio": "Entertainment & Fitness",
    "dance_studio": "Entertainment & Fitness",
    "martial_arts_school": "Entertainment & Fitness",
    "swimming_pool": "Entertainment & Fitness",
    "golf_course": "Entertainment & Fitness",
    "tennis_court": "Entertainment & Fitness",
    "park": "Entertainment & Fitness",
    "playground": "Entertainment & Fitness",
    "recreation_center": "Entertainment & Fitness",
    "spa": "Entertainment & Fitness",
    "event_venue": "Entertainment & Fitness",
    "stadium": "Entertainment & Fitness",
    "arena": "Entertainment & Fitness",
    "casino": "Entertainment & Fitness",

    # Services
    "bank": "Services",
    "insurance_agency": "Services",
    "real_estate_agency": "Services",
    "law_office": "Services",
    "accounting_firm": "Services",
    "consulting_firm": "Services",
    "marketing_agency": "Services",
    "it_services": "Services",
    "staffing_agency": "Services",
    "print_shop": "Services",
    "shipping_store": "Services",
    "post_office": "Services",
    "dry_cleaner": "Services",
    "laundromat": "Services",
    "tailor": "Services",
    "barber_shop": "Services",
    "hair_salon": "Services",
    "nail_salon": "Services",
    "beauty_salon": "Services",
    "tattoo_parlor": "Services",
    "auto_repair_shop": "Services",
    "car_wash": "Services",
    "car_dealer": "Services",
    "storage_facility": "Services",
    "veterinarian": "Services",
    "dentist": "Services",
    "doctor": "Services",
    "hospital": "Services",
    "urgent_care": "Services",
    "clinic": "Services",
    "optometrist": "Services",
    "chiropractor": "Services",
    "physical_therapist": "Services",
    "mental_health_clinic": "Services",
    "daycare": "Services",
    "school": "Services",
    "tutoring_center": "Services",
    "driving_school": "Services",
    "church": "Services",
    "mosque": "Services",
    "synagogue": "Services",
    "temple": "Services",
    "funeral_home": "Services",
    "hotel": "Services",
    "motel": "Services",
    "office": "Services",
    "coworking_space": "Services",

    # Convenience (infrastructure)
    "convenience_store": "Convenience",
    "gas_station": "Convenience",
    "pharmacy": "Convenience",
    "atm": "Convenience",
    "currency_exchange": "Convenience",
    "check_cashing": "Convenience",
    "ev_charging_station": "Convenience",
    "parking_lot": "Convenience",
    "parking_garage": "Convenience",

    # Grocery
    "grocery_store": "Grocery",
    "supermarket": "Grocery",
    "farmers_market": "Grocery",
    "butcher_shop": "Grocery",
    "fish_market": "Grocery",
    "organic_store": "Grocery",
    "health_food_store": "Grocery",
    "specialty_food_store": "Grocery",
}

CATEGORY_KEYWORD_FALLBACKS = {
    "restaurant": "Food & Dining", "food": "Food & Dining",
    "eat": "Food & Dining", "cafe": "Food & Dining",
    "coffee": "Food & Dining", "bar": "Food & Dining",
    "shop": "Shopping", "store": "Shopping",
    "mall": "Shopping", "retail": "Shopping",
    "theater": "Entertainment & Fitness", "entertainment": "Entertainment & Fitness",
    "sport": "Entertainment & Fitness", "fitness": "Entertainment & Fitness",
    "gym": "Entertainment & Fitness", "recreation": "Entertainment & Fitness",
    "bank": "Services", "office": "Services",
    "clinic": "Services", "salon": "Services",
    "service": "Services", "school": "Services",
    "church": "Services", "worship": "Services",
    "gas": "Convenience", "pharmacy": "Convenience",
    "convenience": "Convenience",
    "grocery": "Grocery", "supermarket": "Grocery",
}

# ──────────────────────────────────────────────────────────────
# DESTINATION ANCHORS: retailers that draw foot traffic
# These are what matter for retail health. NOT gas stations/ATMs.
# ──────────────────────────────────────────────────────────────

DESTINATION_ANCHORS = {
    # Grocery anchors (very high signal - people need groceries)
    "whole foods", "trader joe's", "trader joes", "central market",
    "sprouts", "kroger", "h-e-b", "heb", "tom thumb", "albertsons",
    "aldi", "publix",

    # Big box destination (people drive to these)
    "target", "costco", "sam's club", "ikea",
    "best buy", "home depot", "lowe's", "lowes",

    # Department / fashion destination
    "nordstrom", "macy's", "macys", "neiman marcus",
    "saks fifth avenue", "bloomingdale",

    # Specialty destination
    "apple store", "apple ",
    "tesla", "peloton",
    "sephora", "ulta",
    "total wine", "spec's",

    # Entertainment destination
    "topgolf", "top golf", "dave & buster's", "dave and busters",
    "main event", "movie tavern", "alamo drafthouse",
    "amc", "cinemark", "regal",

    # Fitness destination
    "equinox", "lifetime fitness", "life time",
}

# ──────────────────────────────────────────────────────────────
# QUALITY SIGNAL brands: DTC/experiential/premium
# Their presence indicates a curated, high-quality retail area
# ──────────────────────────────────────────────────────────────

QUALITY_SIGNAL_BRANDS = {
    # ── Internet-native / DTC with brick-and-mortar ──
    "warby parker", "allbirds", "glossier", "bonobos", "everlane",
    "casper", "away", "outdoor voices", "reformation", "vuori",
    "alo yoga", "fabletics", "rothy's", "rothys",
    "buck mason", "marine layer", "faherty", "rhone",
    "on running", "hoka", "tracksmith",
    "brilliant earth", "mejuri", "gorjana", "kendra scott",
    "untuckit", "suitsupply", "indochino",

    # ── Major brands that signal premium retail areas ──
    "apple store", "apple ",  # Apple retail
    "tesla", "peloton", "dyson", "microsoft store", "samsung experience",
    "lululemon", "athleta", "free people", "anthropologie",
    "madewell", "aritzia", "bluemercury", "drybar",
    "nike ", "nike store", "adidas originals",
    "patagonia", "rei ", "north face",
    "uniqlo", "zara", "cos ",

    # ── Premium food / coffee (location-selective) ──
    "whole foods", "trader joe's", "trader joes", "central market",
    "sweetgreen", "cava", "shake shack", "north italia",
    "flower child", "true food kitchen", "eataly",
    "starbucks", "blue bottle", "la colombe", "verve coffee", "philz coffee",

    # ── Premium home / lifestyle ──
    "pottery barn", "west elm", "restoration hardware", "rh ",
    "crate & barrel", "crate and barrel", "cb2", "williams sonoma",
    "arhaus", "container store", "sur la table",

    # ── Premium beauty / wellness ──
    "sephora", "ulta", "lush ", "kiehl's", "kiehls", "aesop",
    "jo malone", "le labo", "bluemercury",

    # ── Luxury (very strong signal) ──
    "tiffany", "louis vuitton", "gucci", "hermes", "hermès",
    "chanel", "prada", "burberry", "cartier", "rolex",
    "david yurman", "tory burch", "kate spade",
    "nordstrom", "neiman marcus", "saks fifth avenue",

    # ── Premium fitness ──
    "equinox", "soulcycle", "soul cycle", "barry's", "barrys",
    "orangetheory", "orange theory", "lifetime fitness", "life time",
}

# ──────────────────────────────────────────────────────────────
# INFRASTRUCTURE: brand doesn't matter, only presence/count
# These should NOT be "anchors" - they're utilities
# ──────────────────────────────────────────────────────────────

INFRASTRUCTURE_CATEGORIES = {
    "atm", "gas_station", "parking", "parking_lot", "parking_garage",
    "ev_charging_station", "currency_exchange", "check_cashing",
}

# Brands that are infrastructure (not destination anchors)
INFRASTRUCTURE_BRANDS = {
    # Gas stations
    "shell", "exxon", "chevron", "bp", "valero", "murphy usa",
    "quiktrip", "racetrac", "circle k", "7-eleven", "7 eleven",
    "buc-ee's", "buc-ees", "wawa", "sheetz",
    # ATMs / basic banking
    "atm",
    # Auto services
    "jiffy lube", "valvoline", "firestone", "goodyear", "discount tire",
    "autozone", "o'reilly auto", "advance auto parts",
}

# Brand archetype lists (for cluster classification)
DTC_EXPERIENTIAL_BRANDS = QUALITY_SIGNAL_BRANDS  # alias

FAST_CASUAL_TRENDY_BRANDS = {
    "chipotle", "sweetgreen", "cava", "shake shack", "wingstop",
    "raising cane's", "raising canes", "mod pizza",
    "blaze pizza", "first watch", "north italia", "flower child",
    "true food kitchen", "crumbl", "nothing bundt cakes",
    "starbucks", "blue bottle", "la colombe", "dutch bros", "7 brew",
    "philz coffee",
}

VALUE_BULK_BRANDS = {
    "walmart", "sam's club", "costco",
    "dollar general", "dollar tree", "family dollar", "five below",
    "big lots", "ollie's", "ollies", "aldi", "lidl",
    "ross", "dd's discounts", "burlington", "marshalls", "tj maxx",
    "t.j. maxx", "home goods", "homegoods",
}

FAMILY_ESSENTIALS_BRANDS = {
    "target", "kroger", "h-e-b", "heb", "tom thumb", "albertsons",
    "whole foods", "sprouts", "trader joe's", "trader joes", "central market",
    "petsmart", "petco",
    "hobby lobby", "michaels", "joann",
    "academy sports", "dick's sporting goods", "dicks sporting goods",
    "chick-fil-a", "chick fil a", "whataburger", "panera",
}

NIGHTLIFE_SOCIAL_BRANDS = {
    "topgolf", "top golf", "dave & buster's", "dave and busters",
    "main event", "punch bowl social", "chicken n pickle",
    "pinstripes", "lucky strike",
}

FITNESS_WELLNESS_BRANDS = {
    "equinox", "soulcycle", "soul cycle", "orangetheory", "orange theory",
    "f45", "barry's", "barrys", "crossfit",
    "pure barre", "barre3", "club pilates",
    "planet fitness", "la fitness", "lifetime fitness", "life time",
    "24 hour fitness", "crunch fitness", "gold's gym", "golds gym",
    "anytime fitness", "snap fitness",
    "massage envy", "hand & stone", "european wax center",
}

HOME_LIFESTYLE_BRANDS = {
    "pottery barn", "west elm", "crate & barrel", "crate and barrel",
    "restoration hardware", "rh ", "cb2", "williams sonoma",
    "arhaus", "ethan allen", "container store",
    "home depot", "lowe's", "lowes", "ace hardware",
    "floor & decor", "floor and decor",
    "ikea", "ashley furniture", "rooms to go",
}


def map_category(category: str) -> str:
    if not category:
        return "Other"
    cat_lower = category.lower().strip()
    if cat_lower in CATEGORY_MAP:
        return CATEGORY_MAP[cat_lower]
    for keyword, super_cat in CATEGORY_KEYWORD_FALLBACKS.items():
        if keyword in cat_lower:
            return super_cat
    return "Other"


def is_destination_anchor(name: str, brand_name: str = "") -> bool:
    """True retail anchors that draw foot traffic. NOT gas stations/ATMs."""
    for text in [name, brand_name]:
        if not text:
            continue
        text_lower = text.lower().strip()
        if any(a in text_lower for a in DESTINATION_ANCHORS):
            return True
    return False


def is_quality_signal(name: str, brand_name: str = "") -> bool:
    """DTC/experiential/premium brands that signal a quality retail area."""
    for text in [name, brand_name]:
        if not text:
            continue
        text_lower = text.lower().strip()
        if any(b in text_lower for b in QUALITY_SIGNAL_BRANDS):
            return True
    return False


def is_infrastructure(name: str, basic_category: str = "") -> bool:
    """Gas stations, ATMs, parking - brand doesn't matter."""
    if basic_category and basic_category.lower() in INFRASTRUCTURE_CATEGORIES:
        return True
    if name:
        name_lower = name.lower().strip()
        if any(b in name_lower for b in INFRASTRUCTURE_BRANDS):
            return True
    return False


def classify_brand_archetype(name: str, brand_name: str = "") -> str:
    if not name and not brand_name:
        return ""
    for text in [name, brand_name]:
        if not text:
            continue
        text_lower = text.lower().strip()
        if any(b in text_lower for b in DTC_EXPERIENTIAL_BRANDS):
            return "dtc_experiential"
        if any(b in text_lower for b in FAST_CASUAL_TRENDY_BRANDS):
            return "fast_casual_trendy"
        if any(b in text_lower for b in VALUE_BULK_BRANDS):
            return "value_bulk"
        if any(b in text_lower for b in FAMILY_ESSENTIALS_BRANDS):
            return "family_essentials"
        if any(b in text_lower for b in NIGHTLIFE_SOCIAL_BRANDS):
            return "nightlife_social"
        if any(b in text_lower for b in FITNESS_WELLNESS_BRANDS):
            return "fitness_wellness"
        if any(b in text_lower for b in HOME_LIFESTYLE_BRANDS):
            return "home_lifestyle"
    return ""


# Backward compat
def is_anchor(name: str) -> bool:
    return is_destination_anchor(name)

def is_family_anchor(name: str) -> bool:
    if not name:
        return False
    return any(c in name.lower() for c in FAMILY_ESSENTIALS_BRANDS)
