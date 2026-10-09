# seed_db.py
"""Create data/database.sqlite and fill it with sample data for the quantitative agent.

The tables match SCHEMA_CONTEXT in agents/quantitative.py. The data is shaped to
line up with the documents in data/documents/ (regions, products and list prices
from the Sales Operations Policy, satisfaction scales and benchmarks from the
Customer Success Strategy and Employee Engagement Policy), so "both" queries
have something real to connect.

Re-running drops and recreates the tables. A fixed random seed means every run
produces identical data, so answers can be compared across test runs.

Usage:  python seed_db.py
"""
import calendar
import random
import sqlite3
from datetime import date
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "database.sqlite"
SEED = 42

# Sales covers two full fiscal years so YoY and Q4 comparisons are possible.
START_YEAR, END_YEAR = 2024, 2025

# region -> (share of deal volume, annual growth rate)
# EMEA shrinks and APAC grows fast, matching the targets in the sales policy.
REGIONS = {
    "North America": (0.40, 0.10),
    "EMEA": (0.30, -0.06),
    "APAC": (0.18, 0.35),
    "LATAM": (0.12, 0.15),
}

# product -> (list price per unit per year, share of deals, (min units, max units))
PRODUCTS = {
    "Spoonful Starter": (1200, 0.40, (1, 20)),
    "Spoonful Pro": (4800, 0.30, (1, 10)),
    "Spoonful Enterprise": (24000, 0.10, (1, 3)),
    "Data Connect": (2400, 0.20, (1, 8)),
}

# Discount levels with weights; most deals sit inside the 10% rep-approved limit.
DISCOUNTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35]
DISCOUNT_WEIGHTS = [30, 25, 20, 12, 7, 4, 2]

DEALS_PER_MONTH = 60
Q4_BOOST = 1.4  # Q4 is the strongest quarter (year-end budget spend)

# industry -> probability the customer has churned
# Retail and Education churn most, Healthcare and Finance least (per CS strategy).
INDUSTRIES = {
    "Retail": 0.32,
    "Education": 0.30,
    "Technology": 0.18,
    "Logistics": 0.16,
    "Manufacturing": 0.14,
    "Finance": 0.08,
    "Healthcare": 0.07,
}
NUM_CUSTOMERS = 250

# department -> (headcount, mean satisfaction score out of 10)
# Customer Support is lowest, matching the engagement policy.
DEPARTMENTS = {
    "Engineering": (95, 7.4),
    "Sales": (55, 6.9),
    "Customer Support": (50, 5.8),
    "Customer Success": (30, 7.0),
    "Product": (25, 7.3),
    "Marketing": (25, 7.1),
    "Finance": (15, 7.0),
    "People": (10, 7.6),
    "Operations": (15, 6.7),
}

NAME_PREFIXES = [
    "Acme", "Blue", "Bright", "Cedar", "Coastal", "Delta", "Evergreen", "Falcon",
    "Granite", "Harbor", "Horizon", "Iron", "Juniper", "Keystone", "Lakeside",
    "Maple", "Meridian", "Northwind", "Oak", "Orbit", "Pioneer", "Quartz",
    "Redwood", "Summit", "Silver", "Tidal", "Union", "Vertex", "Willow", "Zenith",
]
NAME_SUFFIXES = {
    "Retail": ["Retail", "Stores", "Outfitters", "Market"],
    "Education": ["Academy", "University", "Learning", "Schools"],
    "Technology": ["Labs", "Software", "Systems", "Tech"],
    "Logistics": ["Freight", "Logistics", "Shipping", "Transport"],
    "Manufacturing": ["Manufacturing", "Industries", "Works", "Fabrication"],
    "Finance": ["Capital", "Financial", "Bank", "Partners"],
    "Healthcare": ["Health", "Clinics", "Medical", "Care"],
}


def random_day(rng: random.Random, year: int, month: int) -> date:
    return date(year, month, rng.randint(1, calendar.monthrange(year, month)[1]))


def clamp_score(value: float) -> float:
    return round(min(10.0, max(1.0, value)), 1)


def create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS employees;

        CREATE TABLE sales (
            id          INTEGER PRIMARY KEY,
            region      TEXT    NOT NULL,
            product     TEXT    NOT NULL,
            revenue     REAL    NOT NULL,   -- net annual contract value in USD, after discount
            date        TEXT    NOT NULL,   -- close date, YYYY-MM-DD
            units_sold  INTEGER NOT NULL
        );

        CREATE TABLE customers (
            id                  INTEGER PRIMARY KEY,
            name                TEXT NOT NULL,
            industry            TEXT NOT NULL,
            churn_date          TEXT,           -- YYYY-MM-DD, NULL if still active
            satisfaction_score  REAL NOT NULL   -- 1 to 10
        );

        CREATE TABLE employees (
            id                  INTEGER PRIMARY KEY,
            department          TEXT NOT NULL,
            satisfaction_score  REAL NOT NULL,  -- 1 to 10, latest engagement survey
            tenure_years        REAL NOT NULL
        );
    """)


def seed_sales(rng: random.Random) -> list[tuple]:
    rows = []
    product_names = list(PRODUCTS)
    product_weights = [p[1] for p in PRODUCTS.values()]

    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            years_elapsed = (year - START_YEAR) + (month - 1) / 12
            season = Q4_BOOST if month >= 10 else 1.0

            for region, (share, growth) in REGIONS.items():
                trend = (1 + growth) ** years_elapsed
                n_deals = round(DEALS_PER_MONTH * share * trend * season * rng.uniform(0.85, 1.15))

                for _ in range(n_deals):
                    product = rng.choices(product_names, product_weights)[0]
                    price, _, (lo, hi) = PRODUCTS[product]
                    units = rng.randint(lo, hi)
                    discount = rng.choices(DISCOUNTS, DISCOUNT_WEIGHTS)[0]
                    revenue = round(units * price * (1 - discount), 2)
                    rows.append((region, product, revenue, random_day(rng, year, month).isoformat(), units))

    rows.sort(key=lambda r: r[3])  # ids in date order reads more naturally
    return rows


def seed_customers(rng: random.Random) -> list[tuple]:
    rows = []
    used_names = set()
    industries = list(INDUSTRIES)

    while len(rows) < NUM_CUSTOMERS:
        industry = rng.choice(industries)
        name = f"{rng.choice(NAME_PREFIXES)} {rng.choice(NAME_SUFFIXES[industry])}"
        if name in used_names:
            continue
        used_names.add(name)

        churned = rng.random() < INDUSTRIES[industry]
        if churned:
            churn_date = random_day(rng, rng.randint(START_YEAR, END_YEAR), rng.randint(1, 12)).isoformat()
            satisfaction = clamp_score(rng.gauss(5.4, 1.2))
        else:
            churn_date = None
            satisfaction = clamp_score(rng.gauss(7.8, 1.0))
        rows.append((name, industry, churn_date, satisfaction))

    return rows


def seed_employees(rng: random.Random) -> list[tuple]:
    rows = []
    for department, (headcount, mean_score) in DEPARTMENTS.items():
        for _ in range(headcount):
            tenure = round(min(15.0, max(0.1, rng.expovariate(1 / 3.0))), 1)
            satisfaction = clamp_score(rng.gauss(mean_score, 1.1))
            rows.append((department, satisfaction, tenure))
    rng.shuffle(rows)
    return rows


def main() -> None:
    rng = random.Random(SEED)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        create_tables(conn)
        sales = seed_sales(rng)
        customers = seed_customers(rng)
        employees = seed_employees(rng)

        conn.executemany(
            "INSERT INTO sales (region, product, revenue, date, units_sold) VALUES (?, ?, ?, ?, ?)", sales)
        conn.executemany(
            "INSERT INTO customers (name, industry, churn_date, satisfaction_score) VALUES (?, ?, ?, ?)", customers)
        conn.executemany(
            "INSERT INTO employees (department, satisfaction_score, tenure_years) VALUES (?, ?, ?)", employees)

    print(f"Seeded {DB_PATH}")
    print(f"  sales:     {len(sales)} rows")
    print(f"  customers: {len(customers)} rows")
    print(f"  employees: {len(employees)} rows")


if __name__ == "__main__":
    main()
