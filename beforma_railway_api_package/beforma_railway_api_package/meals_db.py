"""
meals_db.py
Typed meal records. Each record includes a protein_source tag for diversity tracking.
"""

from __future__ import annotations
from typing import TypedDict


class FoodItem(TypedDict):
    food_name:  str
    quantity_g: float


class MealRecord(TypedDict):
    name:           str
    meal_type:      str      # breakfast | lunch | dinner | snack
    calories:       float
    protein:        float    # grams
    carbs:          float    # grams
    fat:            float    # grams
    protein_source: str      # used for diversity tracking
    items:          list[FoodItem]
    image:          str


MEALS_DB: list[MealRecord] = [
    # ─── BREAKFAST ───────────────────────────────────────────────
    {
        "name": "Greek Yogurt Parfait",
        "meal_type": "breakfast",
        "calories": 380, "protein": 28, "carbs": 45, "fat": 8,
        "protein_source": "dairy",
        "items": [
            {"food_name": "Greek yogurt (0%)", "quantity_g": 200},
            {"food_name": "Rolled oats",       "quantity_g": 60},
            {"food_name": "Mixed berries",      "quantity_g": 80},
            {"food_name": "Honey",              "quantity_g": 15},
        ],
        "image": "https://images.unsplash.com/photo-1488477181946-6428a0291777?w=400",
    },
    {
        "name": "Scrambled Eggs & Avocado Toast",
        "meal_type": "breakfast",
        "calories": 450, "protein": 24, "carbs": 38, "fat": 22,
        "protein_source": "egg",
        "items": [
            {"food_name": "Whole eggs",       "quantity_g": 150},
            {"food_name": "Whole grain bread","quantity_g": 80},
            {"food_name": "Avocado",          "quantity_g": 80},
            {"food_name": "Spinach",          "quantity_g": 30},
        ],
        "image": "https://images.unsplash.com/photo-1525351484163-7529414344d8?w=400",
    },
    {
        "name": "Oatmeal with Protein Powder",
        "meal_type": "breakfast",
        "calories": 420, "protein": 30, "carbs": 55, "fat": 7,
        "protein_source": "whey",
        "items": [
            {"food_name": "Rolled oats",      "quantity_g": 100},
            {"food_name": "Whey protein",     "quantity_g": 30},
            {"food_name": "Banana",           "quantity_g": 100},
            {"food_name": "Almond milk",      "quantity_g": 200},
        ],
        "image": "https://images.unsplash.com/photo-1517673132405-a56a62b18caf?w=400",
    },
    {
        "name": "Smoked Salmon Bagel",
        "meal_type": "breakfast",
        "calories": 480, "protein": 32, "carbs": 42, "fat": 16,
        "protein_source": "salmon",
        "items": [
            {"food_name": "Whole grain bagel", "quantity_g": 100},
            {"food_name": "Smoked salmon",     "quantity_g": 100},
            {"food_name": "Cream cheese (light)","quantity_g": 30},
            {"food_name": "Capers & red onion","quantity_g": 20},
        ],
        "image": "https://images.unsplash.com/photo-1536510233921-8e00d6c05a33?w=400",
    },

    # ─── LUNCH ───────────────────────────────────────────────────
    {
        "name": "Grilled Chicken Caesar Salad",
        "meal_type": "lunch",
        "calories": 520, "protein": 45, "carbs": 22, "fat": 28,
        "protein_source": "chicken",
        "items": [
            {"food_name": "Grilled chicken breast", "quantity_g": 180},
            {"food_name": "Romaine lettuce",         "quantity_g": 150},
            {"food_name": "Parmesan cheese",         "quantity_g": 25},
            {"food_name": "Caesar dressing",         "quantity_g": 30},
            {"food_name": "Whole grain croutons",    "quantity_g": 30},
        ],
        "image": "https://images.unsplash.com/photo-1546793665-c74683f339c1?w=400",
    },
    {
        "name": "Tuna Quinoa Bowl",
        "meal_type": "lunch",
        "calories": 490, "protein": 40, "carbs": 52, "fat": 10,
        "protein_source": "tuna",
        "items": [
            {"food_name": "Canned tuna (in water)", "quantity_g": 150},
            {"food_name": "Quinoa (cooked)",         "quantity_g": 150},
            {"food_name": "Cucumber",                "quantity_g": 80},
            {"food_name": "Cherry tomatoes",         "quantity_g": 80},
            {"food_name": "Olive oil",               "quantity_g": 10},
        ],
        "image": "https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=400",
    },
    {
        "name": "Turkey & Veggie Wrap",
        "meal_type": "lunch",
        "calories": 470, "protein": 38, "carbs": 48, "fat": 12,
        "protein_source": "turkey",
        "items": [
            {"food_name": "Sliced turkey breast", "quantity_g": 150},
            {"food_name": "Whole wheat tortilla", "quantity_g": 80},
            {"food_name": "Hummus",               "quantity_g": 40},
            {"food_name": "Mixed salad greens",   "quantity_g": 50},
            {"food_name": "Bell pepper",          "quantity_g": 60},
        ],
        "image": "https://images.unsplash.com/photo-1626700051175-6818013e1d4f?w=400",
    },
    {
        "name": "Lentil Soup & Bread",
        "meal_type": "lunch",
        "calories": 430, "protein": 22, "carbs": 65, "fat": 8,
        "protein_source": "lentils",
        "items": [
            {"food_name": "Red lentils",         "quantity_g": 120},
            {"food_name": "Vegetable broth",     "quantity_g": 300},
            {"food_name": "Whole grain bread",   "quantity_g": 60},
            {"food_name": "Olive oil",           "quantity_g": 10},
            {"food_name": "Spinach",             "quantity_g": 50},
        ],
        "image": "https://images.unsplash.com/photo-1547592180-85f173990554?w=400",
    },

    # ─── DINNER ──────────────────────────────────────────────────
    {
        "name": "Baked Salmon & Sweet Potato",
        "meal_type": "dinner",
        "calories": 580, "protein": 42, "carbs": 48, "fat": 18,
        "protein_source": "salmon",
        "items": [
            {"food_name": "Salmon fillet",     "quantity_g": 200},
            {"food_name": "Sweet potato",      "quantity_g": 200},
            {"food_name": "Asparagus",         "quantity_g": 100},
            {"food_name": "Olive oil",         "quantity_g": 15},
            {"food_name": "Lemon juice",       "quantity_g": 15},
        ],
        "image": "https://images.unsplash.com/photo-1467003909585-2f8a72700288?w=400",
    },
    {
        "name": "Chicken Stir-fry with Brown Rice",
        "meal_type": "dinner",
        "calories": 600, "protein": 48, "carbs": 62, "fat": 14,
        "protein_source": "chicken",
        "items": [
            {"food_name": "Chicken breast",     "quantity_g": 200},
            {"food_name": "Brown rice (cooked)","quantity_g": 180},
            {"food_name": "Broccoli",           "quantity_g": 100},
            {"food_name": "Bell pepper",        "quantity_g": 80},
            {"food_name": "Soy sauce",          "quantity_g": 20},
            {"food_name": "Sesame oil",         "quantity_g": 10},
        ],
        "image": "https://images.unsplash.com/photo-1512058564366-18510be2db19?w=400",
    },
    {
        "name": "Beef & Vegetable Stew",
        "meal_type": "dinner",
        "calories": 620, "protein": 44, "carbs": 55, "fat": 20,
        "protein_source": "beef",
        "items": [
            {"food_name": "Lean beef",         "quantity_g": 180},
            {"food_name": "Potato",            "quantity_g": 150},
            {"food_name": "Carrots",           "quantity_g": 80},
            {"food_name": "Onion",             "quantity_g": 60},
            {"food_name": "Beef broth",        "quantity_g": 200},
        ],
        "image": "https://images.unsplash.com/photo-1547592180-85f173990554?w=400",
    },
    {
        "name": "Tofu & Vegetable Stir-fry",
        "meal_type": "dinner",
        "calories": 480, "protein": 28, "carbs": 45, "fat": 20,
        "protein_source": "tofu",
        "items": [
            {"food_name": "Firm tofu",          "quantity_g": 200},
            {"food_name": "Basmati rice (cooked)","quantity_g": 150},
            {"food_name": "Bok choy",           "quantity_g": 100},
            {"food_name": "Mushrooms",          "quantity_g": 80},
            {"food_name": "Coconut aminos",     "quantity_g": 20},
        ],
        "image": "https://images.unsplash.com/photo-1512058564366-18510be2db19?w=400",
    },

    # ─── SNACK ───────────────────────────────────────────────────
    {
        "name": "Cottage Cheese & Fruit",
        "meal_type": "snack",
        "calories": 220, "protein": 20, "carbs": 25, "fat": 4,
        "protein_source": "dairy",
        "items": [
            {"food_name": "Cottage cheese (low-fat)", "quantity_g": 150},
            {"food_name": "Pineapple chunks",         "quantity_g": 80},
        ],
        "image": "https://images.unsplash.com/photo-1488477181946-6428a0291777?w=400",
    },
    {
        "name": "Protein Bar & Apple",
        "meal_type": "snack",
        "calories": 280, "protein": 20, "carbs": 35, "fat": 6,
        "protein_source": "whey",
        "items": [
            {"food_name": "Protein bar",  "quantity_g": 60},
            {"food_name": "Medium apple", "quantity_g": 180},
        ],
        "image": "https://images.unsplash.com/photo-1490474418585-ba9bad8fd0ea?w=400",
    },
    {
        "name": "Almonds & Rice Cakes",
        "meal_type": "snack",
        "calories": 260, "protein": 8, "carbs": 28, "fat": 14,
        "protein_source": "nuts",
        "items": [
            {"food_name": "Almonds",     "quantity_g": 30},
            {"food_name": "Rice cakes",  "quantity_g": 40},
        ],
        "image": "https://images.unsplash.com/photo-1505252585461-04db1eb84625?w=400",
    },
    {
        "name": "Hard-Boiled Eggs & Veggies",
        "meal_type": "snack",
        "calories": 200, "protein": 16, "carbs": 8, "fat": 12,
        "protein_source": "egg",
        "items": [
            {"food_name": "Hard-boiled eggs",  "quantity_g": 120},
            {"food_name": "Baby carrots",      "quantity_g": 80},
            {"food_name": "Hummus",            "quantity_g": 30},
        ],
        "image": "https://images.unsplash.com/photo-1482049016688-2d3e1b311543?w=400",
    },
]
