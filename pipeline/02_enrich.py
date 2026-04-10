"""Step 2: Enrich DFW POI data with categories, anchors, and H3 indices."""
import struct
import h3
import pandas as pd
import numpy as np
from config import H3_RESOLUTION, CHECKPOINTS
from utils.checkpoints import checkpoint_exists, load_checkpoint, save_checkpoint
from utils.categories import (
    map_category, is_destination_anchor, is_quality_signal,
    is_infrastructure, is_family_anchor, classify_brand_archetype,
)

# Minimum Overture confidence score to include a POI.
# Below this threshold, data quality is poor (mis-categorized, phantom locations).
CONFIDENCE_THRESHOLD = 0.6

# Categories that are NOT consumer-facing retail and should be excluded entirely.
# These add noise without contributing to retail health signal.
NON_RETAIL_CATEGORIES = {
    # Residential / housing
    "apartment_building", "housing", "senior_living_facility",
    "residential_building",
    # Agriculture / industrial
    "farm", "oil_or_gas_facility", "warehouse", "mining_site",
    # Government / civic (not retail-relevant)
    "government_office", "fire_station", "police_station", "courthouse",
    "military_site", "political_organization", "prison",
    # B2B / wholesale (not consumer-facing)
    "b2b_supplier_distributor", "b2b_service", "b2b_transportation_and_storage_service",
    "b2b_industrial_and_machine_service", "b2b_energy_and_utility_service",
    "b2b_office_and_professional_service", "wholesaler", "manufacturer",
    # Infrastructure / transport (not retail destinations)
    "airport", "train_station", "transportation_location",
    "ground_transport_facility_or_service", "air_transport_facility_or_service",
    # Misc non-retail
    "campus_building", "research_institute", "radio_station",
    "public_utility", "garbage_collection_service",
    "environmental_or_ecological_service", "software_development",
    "psychic_advising",
    "recreational_trail_or_path",
    "water_utility_service",
}

# Map Overture basic_category values to our super-categories
BASIC_CATEGORY_MAP = {
    # ── Food & Dining ──
    "restaurant": "Food & Dining",
    "fast_food_restaurant": "Food & Dining",
    "casual_eatery": "Food & Dining",
    "cafe": "Food & Dining",
    "coffee_shop": "Food & Dining",
    "bar": "Food & Dining",
    "pub": "Food & Dining",
    "bakery": "Food & Dining",
    "ice_cream_parlor": "Food & Dining",
    "dessert_shop": "Food & Dining",
    "juice_or_smoothie_bar": "Food & Dining",
    "pizza_place": "Food & Dining",
    "pizzaria": "Food & Dining",
    "food_truck": "Food & Dining",
    "food_truck_stand": "Food & Dining",
    "brewery_or_winery": "Food & Dining",
    "brewery": "Food & Dining",
    "winery": "Food & Dining",
    "nightlife_spot": "Food & Dining",
    "nightclub": "Food & Dining",
    "dance_club": "Food & Dining",
    "lounge": "Food & Dining",
    "sandwich_shop": "Food & Dining",
    "chicken_restaurant": "Food & Dining",
    "smoothie_juice_bar": "Food & Dining",
    "juice_bar": "Food & Dining",
    "breakfast_restaurant": "Food & Dining",
    "non_alcoholic_beverage_venue": "Food & Dining",
    "beverage_shop": "Food & Dining",

    # ── Shopping ──
    "fashion_and_apparel_store": "Shopping",
    "fashion_or_apparel_store": "Shopping",
    "clothing_store": "Shopping",
    "shoe_store": "Shopping",
    "specialty_store": "Shopping",
    "electronics_store": "Shopping",
    "furniture_store": "Shopping",
    "hardware_home_and_garden_store": "Shopping",
    "hardware_store": "Shopping",
    "home_improvement_center": "Shopping",
    "department_store": "Shopping",
    "shopping_center": "Shopping",
    "shopping_mall": "Shopping",
    "bookstore": "Shopping",
    "books_music_and_video_store": "Shopping",
    "gift_shop": "Shopping",
    "flowers_and_gifts_store": "Shopping",
    "flower_shop": "Shopping",
    "jewelry_store": "Shopping",
    "toy_store": "Shopping",
    "toys_and_games_store": "Shopping",
    "pet_supply_store": "Shopping",
    "animal_and_pet_store": "Shopping",
    "pet_store": "Shopping",
    "sporting_goods_store": "Shopping",
    "discount_store": "Shopping",
    "thrift_or_vintage_store": "Shopping",
    "second_hand_store": "Shopping",
    "second_hand_shop": "Shopping",
    "antique_shop": "Shopping",
    "art_gallery": "Shopping",
    "florist": "Shopping",
    "mobile_phone_store": "Shopping",
    "optical_store": "Shopping",
    "eyewear_store": "Shopping",
    "cosmetics_store": "Shopping",
    "personal_care_and_beauty_store": "Shopping",
    "outdoor_recreation_store": "Shopping",
    "craft_or_hobby_store": "Shopping",
    "arts_crafts_and_hobby_store": "Shopping",
    "art_craft_hobby_store": "Shopping",
    "music_store": "Shopping",
    "musical_instrument_and_pro_audio_store": "Shopping",
    "general_merchandise_store": "Shopping",
    "retail_location": "Shopping",
    "mattress_store": "Shopping",
    "appliance_store": "Shopping",
    "smoke_or_vape_shop": "Shopping",
    "liquor_store": "Shopping",
    "wine_store": "Shopping",
    "warehouse_store": "Shopping",
    "warehouse_club_store": "Shopping",
    "pawn_shop": "Shopping",
    "auto_parts_store": "Shopping",
    "vehicle_parts_store": "Shopping",
    "office_supply_store": "Shopping",

    # ── Entertainment & Fitness ──
    "movie_theater": "Entertainment & Fitness",
    "amusement_park": "Entertainment & Fitness",
    "bowling_alley": "Entertainment & Fitness",
    "arcade": "Entertainment & Fitness",
    "museum": "Entertainment & Fitness",
    "historic_site": "Entertainment & Fitness",
    "zoo": "Entertainment & Fitness",
    "aquarium": "Entertainment & Fitness",
    "performing_arts_venue": "Entertainment & Fitness",
    "theatre_venue": "Entertainment & Fitness",
    "music_venue": "Entertainment & Fitness",
    "comedy_club": "Entertainment & Fitness",
    "escape_room": "Entertainment & Fitness",
    "trampoline_park": "Entertainment & Fitness",
    "skating_rink": "Entertainment & Fitness",
    "sports_complex": "Entertainment & Fitness",
    "sport_or_fitness_facility": "Entertainment & Fitness",
    "sport_fitness_facility": "Entertainment & Fitness",
    "sport_or_recreation_club": "Entertainment & Fitness",
    "sport_field": "Entertainment & Fitness",
    "sport_court": "Entertainment & Fitness",
    "stadium_arena": "Entertainment & Fitness",
    "gym": "Entertainment & Fitness",
    "fitness_center": "Entertainment & Fitness",
    "fitness_studio": "Entertainment & Fitness",
    "yoga_studio": "Entertainment & Fitness",
    "dance_studio": "Entertainment & Fitness",
    "martial_arts_studio": "Entertainment & Fitness",
    "martial_arts_club": "Entertainment & Fitness",
    "swimming_pool": "Entertainment & Fitness",
    "golf_course": "Entertainment & Fitness",
    "tennis": "Entertainment & Fitness",
    "park": "Entertainment & Fitness",
    "dog_park": "Entertainment & Fitness",
    "playground": "Entertainment & Fitness",
    "recreation_center": "Entertainment & Fitness",
    "community_center": "Entertainment & Fitness",
    "spa": "Entertainment & Fitness",
    "massage_therapy": "Entertainment & Fitness",
    "event_venue": "Entertainment & Fitness",
    "event_space": "Entertainment & Fitness",
    "stadium": "Entertainment & Fitness",
    "arena": "Entertainment & Fitness",
    "casino": "Entertainment & Fitness",
    "entertainment_venue": "Entertainment & Fitness",
    "entertainment_location": "Entertainment & Fitness",
    "sports_or_recreation": "Entertainment & Fitness",
    "karaoke": "Entertainment & Fitness",
    "fairgrounds": "Entertainment & Fitness",
    "lake": "Entertainment & Fitness",
    "campground": "Entertainment & Fitness",
    "rv_park": "Entertainment & Fitness",
    "library": "Entertainment & Fitness",
    "resort": "Entertainment & Fitness",
    "gaming_venue": "Entertainment & Fitness",

    # ── Services ──
    "financial_service": "Services",
    "bank": "Services",
    "bank_or_credit_union": "Services",
    "credit_union": "Services",
    "loan_provider": "Services",
    "insurance_service": "Services",
    "insurance_agency": "Services",
    "real_estate_service": "Services",
    "real_estate_agency": "Services",
    "real_estate_developer": "Services",
    "attorney_or_law_firm": "Services",
    "legal_service": "Services",
    "professional_service": "Services",
    "home_service": "Services",
    "building_or_construction_service": "Services",
    "building_contractor_service": "Services",
    "plumbing_service": "Services",
    "hvac_service": "Services",
    "electrical_service": "Services",
    "roofing_service": "Services",
    "painting_service": "Services",
    "remodeling_service": "Services",
    "landscaping_gardening_service": "Services",
    "locksmith_service": "Services",
    "pest_control_service": "Services",
    "carpet_cleaning_service": "Services",
    "house_cleaning_service": "Services",
    "personal_or_beauty_service": "Services",
    "hair_salon": "Services",
    "beauty_salon": "Services",
    "nail_salon": "Services",
    "barber_shop": "Services",
    "tanning_salon": "Services",
    "tattoo_or_piercing_salon": "Services",
    "personal_service": "Services",
    "wellness_service": "Services",
    "complementary_and_alternative_medicine": "Services",
    "alternative_medicine": "Services",
    "physical_therapy": "Services",
    "physical_medicine_and_rehabilitation": "Services",
    "healthcare_location": "Services",
    "dental_clinic": "Services",
    "general_dentistry": "Services",
    "dental_care": "Services",
    "orthodontic_care": "Services",
    "doctors_office": "Services",
    "primary_care_or_general_clinic": "Services",
    "clinic_or_treatment_center": "Services",
    "outpatient_care_facility": "Services",
    "specialized_health_care": "Services",
    "specialized_medical_facility": "Services",
    "internal_medicine": "Services",
    "surgery": "Services",
    "cardiology": "Services",
    "pediatrics": "Services",
    "pediatric_clinic": "Services",
    "behavioral_or_mental_health_clinic": "Services",
    "mental_health": "Services",
    "psychiatry": "Services",
    "reproductive_perinatal_and_womens_care": "Services",
    "plastic_reconstructive_and_aesthetic_surgery": "Services",
    "vision_or_eye_care_clinic": "Services",
    "eye_care": "Services",
    "diagnostics_imaging_or_lab_service": "Services",
    "diagnostic_service": "Services",
    "medical_service": "Services",
    "hospital": "Services",
    "emergency_department": "Services",
    "emergency_room": "Services",
    "veterinary_clinic": "Services",
    "veterinarian": "Services",
    "animal_or_pet_service": "Services",
    "animal_service": "Services",
    "pet_service": "Services",
    "pet_grooming": "Services",
    "laundry_or_dry_cleaning": "Services",
    "laundry_service": "Services",
    "dry_cleaning_service": "Services",
    "auto_dealer": "Services",
    "car_dealer": "Services",
    "vehicle_dealer": "Services",
    "automotive_service": "Services",
    "vehicle_service": "Services",
    "auto_repair_service": "Services",
    "auto_body_shop": "Services",
    "auto_glass_service": "Services",
    "auto_rental_service": "Services",
    "truck_rental_service": "Services",
    "towing_service": "Services",
    "shipping_or_postal_service": "Services",
    "shipping_delivery_service": "Services",
    "shipping_or_delivery_service": "Services",
    "courier_service": "Services",
    "post_office": "Services",
    "storage_facility": "Services",
    "self_storage_facility": "Services",
    "moving_or_storage_service": "Services",
    "moving_service": "Services",
    "daycare": "Services",
    "child_care_or_day_care": "Services",
    "preschool": "Services",
    "school": "Services",
    "elementary_school": "Services",
    "middle_school": "Services",
    "high_school": "Services",
    "specialty_school": "Services",
    "college_university": "Services",
    "place_of_learning": "Services",
    "education_or_training": "Services",
    "educational_service": "Services",
    "tutoring_center": "Services",
    "tutoring_service": "Services",
    "christian_place_of_worship": "Services",
    "muslim_place_of_worship": "Services",
    "place_of_worship": "Services",
    "religious_organization": "Services",
    "funeral_service": "Services",
    "hotel": "Services",
    "motel": "Services",
    "lodging": "Services",
    "bed_and_breakfast": "Services",
    "corporate_or_business_office": "Services",
    "business_location": "Services",
    "social_or_community_service": "Services",
    "family_service": "Services",
    "youth_organization": "Services",
    "civic_organization": "Services",
    "civic_organization_office": "Services",
    "tax_service": "Services",
    "tax_preparation_service": "Services",
    "accountant_or_bookkeeper": "Services",
    "event_or_party_service": "Services",
    "cleaning_service": "Services",
    "business_cleaning_service": "Services",
    "photography_service": "Services",
    "print_or_copy_service": "Services",
    "printing_service": "Services",
    "staffing_or_recruiting_service": "Services",
    "human_resource_service": "Services",
    "technical_service": "Services",
    "media_service": "Services",
    "design_service": "Services",
    "web_design_service": "Services",
    "interior_design_service": "Services",
    "business_advertising_marketing": "Services",
    "business_management_service": "Services",
    "information_technology_service": "Services",
    "telecommunications_service": "Services",
    "travel_service": "Services",
    "travel_agent": "Services",
    "rental_service": "Services",
    "property_management_service": "Services",
    "housing_or_property_service": "Services",
    "notary_public": "Services",
    "engineering_service": "Services",
    "agricultural_service": "Services",
    "applicance_repair_service": "Services",
    "electronic_repiar_service": "Services",
    "print_media_service": "Services",
    "service_location": "Services",
    "supplier_or_distributor": "Services",
    "taxi_or_ride_share_service": "Services",
    "catering_service": "Services",
    "food_delivery_service": "Services",
    "food_service": "Services",
    "animal_boarding": "Services",
    "passport_visa_service": "Services",
    "vocational_technical_school": "Services",
    "sports_medicine": "Services",
    "motorcycle_dealer": "Services",
    "private_lodging": "Services",

    # ── Convenience ──
    "convenience_store": "Convenience",
    "gas_station": "Convenience",
    "pharmacy": "Convenience",
    "pharmacy_and_drug_store": "Convenience",
    "atm": "Convenience",
    "ev_charging_station": "Convenience",
    "parking": "Convenience",

    # ── Grocery ──
    "grocery_store": "Grocery",
    "supermarket": "Grocery",
    "farmers_market": "Grocery",
    "butcher": "Grocery",
    "food_and_beverage_store": "Grocery",
    "food_or_beverage_store": "Grocery",
    "organic_store": "Grocery",
    "health_food_store": "Grocery",
    "specialty_food_store": "Grocery",
    "superstore": "Grocery",
    "eating_drinking_location": "Food & Dining",
    "shipping_center": "Shopping",
}


def decode_wkb_point(geom_bytes):
    """Decode WKB point geometry to (lat, lng)."""
    if geom_bytes is None or len(geom_bytes) < 21:
        return None, None
    try:
        lng, lat = struct.unpack_from("<dd", geom_bytes, 5)
        return lat, lng
    except Exception:
        return None, None


def extract_primary_name(names):
    """Extract primary name from Overture names struct."""
    if names is None:
        return None
    if isinstance(names, dict):
        return names.get("primary")
    return None


def map_basic_category(cat):
    """Map basic_category to super-category using our mapping."""
    if not cat or not isinstance(cat, str):
        return "Other"
    cat_lower = cat.lower().strip()
    if cat_lower in BASIC_CATEGORY_MAP:
        return BASIC_CATEGORY_MAP[cat_lower]
    # Keyword fallback
    for keyword, super_cat in [
        ("restaurant", "Food & Dining"), ("food", "Food & Dining"),
        ("eat", "Food & Dining"), ("cafe", "Food & Dining"),
        ("coffee", "Food & Dining"), ("bar", "Food & Dining"),
        ("drink", "Food & Dining"),
        ("store", "Shopping"), ("shop", "Shopping"), ("retail", "Shopping"),
        ("theater", "Entertainment & Fitness"), ("entertainment", "Entertainment & Fitness"),
        ("sport", "Entertainment & Fitness"), ("fitness", "Entertainment & Fitness"),
        ("gym", "Entertainment & Fitness"), ("recreation", "Entertainment & Fitness"),
        ("bank", "Services"), ("office", "Services"),
        ("clinic", "Services"), ("salon", "Services"),
        ("service", "Services"), ("school", "Services"),
        ("church", "Services"), ("worship", "Services"),
        ("gas", "Convenience"), ("pharmacy", "Convenience"),
        ("convenience", "Convenience"),
        ("grocery", "Grocery"), ("supermarket", "Grocery"),
    ]:
        if keyword in cat_lower:
            return super_cat
    return "Other"


def run():
    output_path = CHECKPOINTS["enrich"]

    if checkpoint_exists(output_path):
        print("Step 2 (enrich): checkpoint exists, skipping.")
        return

    print("Step 2: Enriching POI data...")
    df = load_checkpoint(CHECKPOINTS["download"])
    print(f"  Loaded {len(df):,} POIs")

    # ── Filter by confidence score ──
    if "confidence" in df.columns:
        before = len(df)
        df = df[df["confidence"] >= CONFIDENCE_THRESHOLD]
        print(f"  Confidence filter (>= {CONFIDENCE_THRESHOLD}): {before:,} -> {len(df):,} ({before - len(df):,} dropped)")

    # ── Filter out closed / temporarily closed POIs ──
    if "operating_status" in df.columns:
        before = len(df)
        df = df[df["operating_status"].isin(["open"]) | df["operating_status"].isna()]
        print(f"  Operating status filter (open only): {before:,} -> {len(df):,} ({before - len(df):,} dropped)")

    # ── Exclude non-retail categories ──
    if "basic_category" in df.columns:
        before = len(df)
        is_non_retail = df["basic_category"].str.lower().isin(NON_RETAIL_CATEGORIES)
        df = df[~is_non_retail]
        print(f"  Non-retail exclusion: {before:,} -> {len(df):,} ({before - len(df):,} dropped)")

    # Extract primary name
    df["name"] = df["names"].apply(extract_primary_name)

    # Extract primary category string
    df["primary_category"] = df["categories"].apply(
        lambda x: x.get("primary") if isinstance(x, dict) else None
    )

    # Use basic_category as our main category (simpler, fewer unique values)
    # Map to super-categories
    df["super_category"] = df["basic_category"].apply(map_basic_category)

    # Decode geometry (WKB) to lat/lng
    coords = df["geometry"].apply(decode_wkb_point)
    df["latitude"] = coords.apply(lambda x: x[0])
    df["longitude"] = coords.apply(lambda x: x[1])

    # Drop rows without valid coordinates
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    print(f"  Dropped {before - len(df):,} rows without valid coordinates")

    # ── Brand name extraction & normalization ──
    # Use brand.names.primary as canonical name when available.
    # This normalizes "McDonald's Restaurant #1234" -> "McDonald's".
    def get_brand_name(brand):
        if brand and isinstance(brand, dict):
            names = brand.get("names", {})
            if isinstance(names, dict):
                return names.get("primary", "")
        return ""

    df["brand_name"] = df["brand"].apply(get_brand_name)

    # If a POI has a brand name, prefer it over the raw place name for consistency
    has_brand = df["brand_name"].notna() & (df["brand_name"] != "")
    df.loc[has_brand, "name"] = df.loc[has_brand, "brand_name"]

    # Classify POI roles (destination anchor, quality signal, infrastructure)
    def safe_str(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return str(val)

    df["is_anchor"] = df.apply(
        lambda r: is_destination_anchor(safe_str(r.get("name")), safe_str(r.get("brand_name"))), axis=1
    )
    df["is_quality"] = df.apply(
        lambda r: is_quality_signal(safe_str(r.get("name")), safe_str(r.get("brand_name"))), axis=1
    )
    df["is_infrastructure"] = df.apply(
        lambda r: is_infrastructure(safe_str(r.get("name")), safe_str(r.get("basic_category"))), axis=1
    )
    df["is_family_anchor"] = df["name"].apply(is_family_anchor)

    # Brand archetype classification
    df["brand_archetype"] = df.apply(
        lambda row: classify_brand_archetype(safe_str(row.get("name")), safe_str(row.get("brand_name"))),
        axis=1,
    )

    # Also classify nightlife by category (bars, breweries, nightclubs)
    nightlife_cats = {"bar", "pub", "nightclub", "lounge", "brewery_or_winery",
                      "nightlife_spot", "cocktail_bar", "wine_bar", "comedy_club"}
    is_nightlife_cat = df["basic_category"].str.lower().isin(nightlife_cats)
    df.loc[is_nightlife_cat & (df["brand_archetype"] == ""), "brand_archetype"] = "nightlife_social"

    # Classify fitness by category when no brand match
    fitness_cats = {"gym", "fitness_center", "yoga_studio", "pilates_studio",
                    "martial_arts_studio", "dance_studio", "spa", "wellness_service"}
    is_fitness_cat = df["basic_category"].str.lower().isin(fitness_cats)
    df.loc[is_fitness_cat & (df["brand_archetype"] == ""), "brand_archetype"] = "fitness_wellness"

    print(f"  Brand archetype distribution:")
    archetype_counts = df[df["brand_archetype"] != ""]["brand_archetype"].value_counts()
    print(archetype_counts.to_string(header=False))

    # Assign H3 hex index
    df["h3_index"] = df.apply(
        lambda row: h3.latlng_to_cell(row["latitude"], row["longitude"], H3_RESOLUTION),
        axis=1,
    )

    # Keep only useful columns
    keep_cols = [
        "id", "name", "basic_category", "primary_category", "super_category",
        "latitude", "longitude", "h3_index",
        "is_anchor", "is_quality", "is_infrastructure",
        "is_family_anchor", "brand_archetype", "confidence",
    ]
    df = df[keep_cols]

    print(f"  Category distribution:")
    print(df["super_category"].value_counts().to_string(header=False))
    print(f"  Destination anchors: {df['is_anchor'].sum():,}")
    print(f"  Quality signal brands: {df['is_quality'].sum():,}")
    print(f"  Infrastructure POIs: {df['is_infrastructure'].sum():,}")
    print(f"  Unique H3 hexes: {df['h3_index'].nunique():,}")

    save_checkpoint(df, output_path)


if __name__ == "__main__":
    run()
