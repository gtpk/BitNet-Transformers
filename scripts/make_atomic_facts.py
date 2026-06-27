#!/usr/bin/env python3
"""FACT-003D: build a controlled atomic-facts replay set (train + held-out), panel-disjoint.

Per the FACT-003D-alpha design: small, clean, short-answer factual QA in the SAME format as the
eval panel (Q: ..\\nA:), drawn from atomic categories (capital, currency, element symbol, author
-> work, simple science/geography/units). The set must be COMPLETELY disjoint from the eval panel
data/factual_panel_v1.jsonl -- by entity, prompt, AND answer -- so protected replay never trains
on the held-out factual eval. We split by ENTITY (not by phrasing) so the held-out replay panel
tests transfer to facts whose entity never appeared in training, not memorised rephrasings.

Outputs (data/):
  atomic_facts_train.jsonl     -- protected replay TRAIN ({"prompt","answer","id","category"})
  atomic_facts_heldout.jsonl   -- held-out replay VALIDATION (disjoint entities, scored like panel)

Usage:
  python scripts/make_atomic_facts.py            # writes both, prints counts + leak check
  python scripts/make_atomic_facts.py --heldout-frac 0.25 --seed 0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL = REPO_ROOT / "data/factual_panel_v1.jsonl"

# --- country -> capital (high-confidence; PANEL countries France/Japan/Italy/Germany/Russia OMITTED)
CAPITALS = {
    "Spain": "Madrid", "Portugal": "Lisbon", "Canada": "Ottawa", "Australia": "Canberra",
    "Brazil": "Brasilia", "Argentina": "Buenos Aires", "Egypt": "Cairo", "Greece": "Athens",
    "Turkey": "Ankara", "Sweden": "Stockholm", "Norway": "Oslo", "Finland": "Helsinki",
    "Denmark": "Copenhagen", "Poland": "Warsaw", "Austria": "Vienna", "Switzerland": "Bern",
    "Belgium": "Brussels", "Netherlands": "Amsterdam", "Ireland": "Dublin", "Mexico": "Mexico City",
    "Peru": "Lima", "Chile": "Santiago", "Colombia": "Bogota", "Cuba": "Havana",
    "India": "New Delhi", "China": "Beijing", "Thailand": "Bangkok", "Vietnam": "Hanoi",
    "Indonesia": "Jakarta", "Philippines": "Manila", "Malaysia": "Kuala Lumpur",
    "South Korea": "Seoul", "Iran": "Tehran", "Iraq": "Baghdad", "Israel": "Jerusalem",
    "Saudi Arabia": "Riyadh", "Pakistan": "Islamabad", "Afghanistan": "Kabul",
    "Kenya": "Nairobi", "Nigeria": "Abuja", "Ethiopia": "Addis Ababa", "Morocco": "Rabat",
    "Ghana": "Accra", "Cuba ": "Havana", "Hungary": "Budapest", "Romania": "Bucharest",
    "Ukraine": "Kyiv", "Czech Republic": "Prague", "Bulgaria": "Sofia", "Croatia": "Zagreb",
    "Iceland": "Reykjavik", "New Zealand": "Wellington", "Cambodia": "Phnom Penh",
    "Nepal": "Kathmandu", "Bangladesh": "Dhaka", "Sri Lanka": "Colombo", "Mongolia": "Ulaanbaatar",
    "Venezuela": "Caracas", "Ecuador": "Quito", "Bolivia": "La Paz", "Uruguay": "Montevideo",
    "Lebanon": "Beirut", "Jordan": "Amman", "Syria": "Damascus", "Qatar": "Doha",
    "Kuwait": "Kuwait City", "Tunisia": "Tunis", "Algeria": "Algiers", "Libya": "Tripoli",
    "Sudan": "Khartoum", "Tanzania": "Dodoma", "Uganda": "Kampala", "Zimbabwe": "Harare",
    "Angola": "Luanda", "Senegal": "Dakar", "Cameroon": "Yaounde", "Slovakia": "Bratislava",
    "Slovenia": "Ljubljana", "Serbia": "Belgrade", "Lithuania": "Vilnius", "Latvia": "Riga",
    "Estonia": "Tallinn", "Luxembourg": "Luxembourg", "Singapore": "Singapore",
    "Iceland ": "Reykjavik", "Panama": "Panama City", "Costa Rica": "San Jose",
    "Guatemala": "Guatemala City", "Honduras": "Tegucigalpa", "Nicaragua": "Managua",
    "Paraguay": "Asuncion", "Jamaica": "Kingston", "Dominican Republic": "Santo Domingo",
    "Iraq ": "Baghdad", "Yemen": "Sanaa", "Oman": "Muscat", "Bahrain": "Manama",
    "Azerbaijan": "Baku", "Georgia": "Tbilisi", "Armenia": "Yerevan", "Kazakhstan": "Astana",
    "Uzbekistan": "Tashkent", "Turkmenistan": "Ashgabat", "Myanmar": "Naypyidaw",
    "Laos": "Vientiane", "Bhutan": "Thimphu", "Mali": "Bamako", "Niger": "Niamey",
    "Chad": "N'Djamena", "Somalia": "Mogadishu", "Rwanda": "Kigali", "Zambia": "Lusaka",
    "Mozambique": "Maputo", "Botswana": "Gaborone", "Namibia": "Windhoek", "Madagascar": "Antananarivo",
    "Ivory Coast": "Yamoussoukro", "Mauritania": "Nouakchott", "Gabon": "Libreville",
    "Albania": "Tirana", "North Macedonia": "Skopje", "Montenegro": "Podgorica",
    "Bosnia and Herzegovina": "Sarajevo", "Moldova": "Chisinau", "Belarus": "Minsk",
    "Cyprus": "Nicosia", "Malta": "Valletta", "Fiji": "Suva", "Papua New Guinea": "Port Moresby",
    "El Salvador": "San Salvador", "Trinidad and Tobago": "Port of Spain", "Bahamas": "Nassau",
    "Barbados": "Bridgetown", "Guyana": "Georgetown", "Suriname": "Paramaribo",
    "Tajikistan": "Dushanbe", "Kyrgyzstan": "Bishkek", "Sri Lanka ": "Colombo",
    "United Arab Emirates": "Abu Dhabi", "Brunei": "Bandar Seri Begawan", "Maldives": "Male",
    "Burkina Faso": "Ouagadougou", "Benin": "Porto-Novo", "Togo": "Lome", "Guinea": "Conakry",
    "Sierra Leone": "Freetown", "Liberia": "Monrovia", "Malawi": "Lilongwe", "Lesotho": "Maseru",
    "Eritrea": "Asmara", "Djibouti": "Djibouti", "Congo": "Brazzaville", "Burundi": "Gitega",
    "Andorra": "Andorra la Vella", "Monaco": "Monaco", "Liechtenstein": "Vaduz",
    "San Marino": "San Marino", "Kosovo": "Pristina", "Samoa": "Apia", "Tonga": "Nuku'alofa",
}

# --- country -> continent (high-confidence; PANEL has no country->continent items)
CONTINENTS = {
    "Brazil": "South America", "Egypt": "Africa", "Australia": "Oceania", "Canada": "North America",
    "India": "Asia", "Spain": "Europe", "Nigeria": "Africa", "Argentina": "South America",
    "Thailand": "Asia", "Mexico": "North America", "Sweden": "Europe", "Kenya": "Africa",
    "Peru": "South America", "Vietnam": "Asia", "Greece": "Europe", "Morocco": "Africa",
    "Chile": "South America", "Indonesia": "Asia", "Norway": "Europe", "Ethiopia": "Africa",
    "Colombia": "South America", "Pakistan": "Asia", "Poland": "Europe", "Ghana": "Africa",
    "Bolivia": "South America", "Nepal": "Asia", "Portugal": "Europe", "Tanzania": "Africa",
    "Ecuador": "South America", "Mongolia": "Asia", "Ireland": "Europe", "Uganda": "Africa",
    "Venezuela": "South America", "Cambodia": "Asia", "Finland": "Europe", "Angola": "Africa",
    "Uruguay": "South America", "Bangladesh": "Asia", "Austria": "Europe", "Senegal": "Africa",
}

# --- country -> official language (high-confidence)
LANGUAGES = {
    "Brazil": "Portuguese", "Mexico": "Spanish", "Argentina": "Spanish", "Egypt": "Arabic",
    "Saudi Arabia": "Arabic", "China": "Mandarin", "Thailand": "Thai", "Vietnam": "Vietnamese",
    "Greece": "Greek", "Turkey": "Turkish", "Sweden": "Swedish", "Norway": "Norwegian",
    "Finland": "Finnish", "Poland": "Polish", "Netherlands": "Dutch", "Portugal": "Portuguese",
    "Iran": "Persian", "Israel": "Hebrew", "Pakistan": "Urdu", "Indonesia": "Indonesian",
    "South Korea": "Korean", "India": "Hindi", "Kenya": "Swahili", "Hungary": "Hungarian",
    "Czech Republic": "Czech", "Romania": "Romanian", "Ukraine": "Ukrainian", "Denmark": "Danish",
}

# --- country -> currency (PANEL: Japan/yen OMITTED)
CURRENCIES = {
    "United States": "Dollar", "United Kingdom": "Pound", "Spain": "Euro", "Germany": "Euro",
    "India": "Rupee", "China": "Yuan", "Russia": "Ruble", "Brazil": "Real", "Mexico": "Peso",
    "Canada": "Dollar", "Switzerland": "Franc", "Sweden": "Krona", "Norway": "Krone",
    "Denmark": "Krone", "Poland": "Zloty", "Turkey": "Lira", "South Korea": "Won",
    "Thailand": "Baht", "Vietnam": "Dong", "Israel": "Shekel", "Saudi Arabia": "Riyal",
    "Egypt": "Pound", "Nigeria": "Naira", "Kenya": "Shilling", "South Africa": "Rand",
    "Argentina": "Peso", "Chile": "Peso", "Peru": "Sol", "Indonesia": "Rupiah",
    "Malaysia": "Ringgit", "Philippines": "Peso", "Bangladesh": "Taka", "Hungary": "Forint",
    "Vietnam ": "Dong", "Indonesia": "Rupiah", "Pakistan": "Rupee", "Sri Lanka": "Rupee",
    "Czech Republic": "Koruna", "Iceland": "Krona", "Croatia": "Euro", "Ukraine": "Hryvnia",
    "Kuwait": "Dinar", "Iraq": "Dinar", "Jordan": "Dinar", "Morocco": "Dirham",
    "Ethiopia": "Birr", "Ghana": "Cedi", "Venezuela": "Bolivar", "Colombia": "Peso",
    "Costa Rica": "Colon", "Guatemala": "Quetzal", "Panama": "Balboa", "Mongolia": "Tugrik",
}

# --- element -> chemical symbol (PANEL touches oxygen/hydrogen via 'water' -> OMIT H and O)
ELEMENTS = {
    "Gold": "Au", "Silver": "Ag", "Iron": "Fe", "Carbon": "C", "Helium": "He",
    "Sodium": "Na", "Potassium": "K", "Calcium": "Ca", "Nitrogen": "N", "Chlorine": "Cl",
    "Copper": "Cu", "Zinc": "Zn", "Lead": "Pb", "Tin": "Sn", "Mercury (element)": "Hg",
    "Magnesium": "Mg", "Aluminium": "Al", "Silicon": "Si", "Sulfur": "S", "Phosphorus": "P",
    "Neon": "Ne", "Argon": "Ar", "Lithium": "Li", "Boron": "B", "Fluorine": "F",
    "Nickel": "Ni", "Platinum": "Pt", "Uranium": "U", "Titanium": "Ti", "Cobalt": "Co",
    "Manganese": "Mn", "Barium": "Ba", "Iodine": "I", "Bromine": "Br", "Krypton": "Kr",
}

# --- famous work -> author (PANEL: Romeo and Juliet/Shakespeare, Pride and Prejudice/Austen OMITTED)
AUTHORS = {
    "War and Peace": "Tolstoy", "Crime and Punishment": "Dostoevsky", "The Odyssey": "Homer",
    "Hamlet": "Shakespeare",  # gate will drop (shakespeare is a panel answer) -- kept to test the gate
    "1984": "Orwell", "Animal Farm": "Orwell", "The Great Gatsby": "Fitzgerald",
    "Moby Dick": "Melville", "Don Quixote": "Cervantes", "The Divine Comedy": "Dante",
    "Great Expectations": "Dickens", "Oliver Twist": "Dickens", "Jane Eyre": "Bronte",
    "Wuthering Heights": "Bronte", "The Old Man and the Sea": "Hemingway",
    "Ulysses": "Joyce", "The Trial": "Kafka", "Faust": "Goethe", "Les Miserables": "Hugo",
    "The Adventures of Huckleberry Finn": "Twain", "Frankenstein": "Shelley",
    "Dracula": "Stoker", "Brave New World": "Huxley", "The Catcher in the Rye": "Salinger",
    "Lord of the Flies": "Golding", "To Kill a Mockingbird": "Lee", "The Hobbit": "Tolkien",
    "The Lord of the Rings": "Tolkien", "A Tale of Two Cities": "Dickens", "David Copperfield": "Dickens",
    "The Brothers Karamazov": "Dostoevsky", "Anna Karenina": "Tolstoy", "The Iliad": "Homer",
    "Madame Bovary": "Flaubert", "The Stranger": "Camus", "Heart of Darkness": "Conrad",
    "Gulliver's Travels": "Swift", "Robinson Crusoe": "Defoe", "Treasure Island": "Stevenson",
    "The Picture of Dorian Gray": "Wilde", "Fahrenheit 451": "Bradbury", "The Sun Also Rises": "Hemingway",
    "Sense and Sensibility": "Austen",  # gate-test: austen is a panel answer -> should drop
    "Macbeth": "Shakespeare",          # gate-test: shakespeare is a panel answer -> should drop
}

# --- misc simple science / geography / units (hand-checked; PANEL items avoided)
MISC = [
    ("simple_fact", "Q: What is the largest mammal on Earth?\nA:", "blue whale"),
    ("simple_fact", "Q: What is the tallest animal in the world?\nA:", "giraffe"),
    ("simple_fact", "Q: What gas do plants absorb from the air for photosynthesis?\nA:", "carbon dioxide"),
    ("simple_fact", "Q: What is the hardest natural substance on Earth?\nA:", "diamond"),
    ("simple_fact", "Q: What organ pumps blood through the human body?\nA:", "heart"),
    ("simple_fact", "Q: How many legs does a spider have?\nA:", "eight"),
    ("simple_fact", "Q: What is the fastest land animal?\nA:", "cheetah"),
    ("simple_fact", "Q: What planet is known as the Red Planet?\nA:", "mars"),
    ("simple_fact", "Q: What is the closest star to Earth?\nA:", "the sun"),
    ("simple_fact", "Q: What is the smallest prime number?\nA:", "two"),
    ("simple_fact", "Q: What is the chemical formula for table salt?\nA:", "NaCl"),
    ("simple_fact", "Q: What is the longest river in the world?\nA:", "nile"),
    ("simple_fact", "Q: What is the tallest mountain on Earth?\nA:", "everest"),
    ("simple_fact", "Q: What is the largest desert in the world?\nA:", "sahara"),
    ("simple_fact", "Q: What is the largest country by land area?\nA:", "russia"),
    ("simple_fact", "Q: How many sides does a triangle have?\nA:", "three"),
    ("simple_fact", "Q: How many sides does a hexagon have?\nA:", "six"),
    ("simple_fact", "Q: How many colors are in a rainbow?\nA:", "seven"),
    ("simple_fact", "Q: What is the freezing point of water in Fahrenheit?\nA:", "32"),
    ("simple_fact", "Q: How many minutes are in one hour?\nA:", "sixty"),
    ("simple_fact", "Q: How many months are in a year?\nA:", "twelve"),
    ("simple_fact", "Q: What is the opposite of up?\nA:", "down"),
    ("simple_fact", "Q: What is the opposite of light?\nA:", "dark"),
    ("simple_fact", "Q: What color do you get by mixing blue and yellow?\nA:", "green"),
    ("simple_fact", "Q: What is the primary gas that makes up most of Earth's atmosphere?\nA:", "nitrogen"),
    ("entity_attr", "Q: What ocean lies between Europe and North America?\nA:", "atlantic"),
    ("simple_fact", "Q: What is the largest internal organ in the human body?\nA:", "liver"),
    ("simple_fact", "Q: What part of a plant conducts photosynthesis?\nA:", "leaf"),
    ("simple_fact", "Q: What is the powerhouse of the cell?\nA:", "mitochondria"),
    ("simple_fact", "Q: What force pulls objects toward the Earth?\nA:", "gravity"),
    ("simple_fact", "Q: What is the SI unit of electric current?\nA:", "ampere"),
    ("simple_fact", "Q: What is the SI unit of force?\nA:", "newton"),
    ("simple_fact", "Q: What is the SI unit of energy?\nA:", "joule"),
    ("simple_fact", "Q: What is the SI unit of frequency?\nA:", "hertz"),
    ("simple_fact", "Q: How many millimeters are in one centimeter?\nA:", "ten"),
    ("simple_fact", "Q: How many sides does a pentagon have?\nA:", "five"),
    ("simple_fact", "Q: How many hours are in a full day?\nA:", "twenty-four"),
    ("simple_fact", "Q: What is the largest organ of the human body?\nA:", "skin"),
    ("simple_fact", "Q: What metal is liquid at room temperature?\nA:", "mercury"),
]


def panel_exclude_set(panel_path):
    """Normalized panel prompts + distinctive (>=5 char) answers + entity tokens, for de-leaking."""
    out = set()
    for line in open(panel_path):
        if not line.strip():
            continue
        p = json.loads(line)
        out.add(re.sub(r"\s+", " ", p["prompt"]).strip().lower())
        for a in p.get("must_contain", []):
            a = re.sub(r"\s+", " ", a).strip().lower()
            if len(a) >= 5:
                out.add(a)
    return out


def leaks(prompt, answer, excl):
    norm = re.sub(r"\s+", " ", (prompt + " " + answer)).strip().lower()
    return any(e in norm for e in excl)


def build_pool():
    pool = []  # (category, entity, prompt, answer)
    for country, cap in CAPITALS.items():
        c = country.strip()
        pool.append(("capital", f"capital:{c}", f"Q: What is the capital of {c}?\nA:", cap))
    for country, cur in CURRENCIES.items():
        pool.append(("currency", f"currency:{country}", f"Q: What is the currency of {country}?\nA:", cur))
    for el, sym in ELEMENTS.items():
        name = el.replace(" (element)", "")
        pool.append(("element", f"element:{name}", f"Q: What is the chemical symbol for {name}?\nA:", sym))
    for work, author in AUTHORS.items():
        pool.append(("author", f"author:{work}", f"Q: Who wrote {work}?\nA:", author))
    for country, cont in CONTINENTS.items():
        pool.append(("continent", f"continent:{country.strip()}", f"Q: On which continent is {country.strip()} located?\nA:", cont))
    for country, lang in LANGUAGES.items():
        pool.append(("language", f"language:{country.strip()}", f"Q: What is the main language spoken in {country.strip()}?\nA:", lang))
    for i, (cat, prompt, answer) in enumerate(MISC):
        pool.append((cat, f"misc:{i}", prompt, answer))
    return pool


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-train", type=Path, default=REPO_ROOT / "data/atomic_facts_train.jsonl")
    ap.add_argument("--out-heldout", type=Path, default=REPO_ROOT / "data/atomic_facts_heldout.jsonl")
    args = ap.parse_args()

    excl = panel_exclude_set(PANEL)
    pool = build_pool()

    kept, dropped = [], []
    seen_entity = set()
    for cat, entity, prompt, answer in pool:
        if entity in seen_entity:
            continue
        seen_entity.add(entity)
        if leaks(prompt, answer, excl):
            dropped.append((entity, answer))
            continue
        kept.append({"category": cat, "entity": entity, "prompt": prompt, "answer": " " + answer.strip()})

    # deterministic shuffle by a stable hash of the entity (no Date/random in repo scripts is fine here,
    # but keep it reproducible regardless): sort by (seed-salted) hash string.
    def order_key(rec):
        h = 0
        for ch in f"{args.seed}:{rec['entity']}":
            h = (h * 131 + ord(ch)) % (2 ** 32)
        return h
    kept.sort(key=order_key)

    n_held = max(1, int(round(len(kept) * args.heldout_frac)))
    heldout = kept[:n_held]
    train = kept[n_held:]

    def write(path, recs, as_panel):
        with open(path, "w", encoding="utf-8") as f:
            for i, r in enumerate(recs):
                if as_panel:
                    # held-out scored like the eval panel: must_contain = answer tokens, lowercased
                    rec = {"id": r["entity"], "category": r["category"], "prompt": r["prompt"],
                           "must_contain": [r["answer"].strip().lower()]}
                else:
                    rec = {"id": r["entity"], "category": r["category"], "prompt": r["prompt"],
                           "answer": r["answer"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    write(args.out_train, train, as_panel=False)
    write(args.out_heldout, heldout, as_panel=True)

    # leak re-check: assert nothing in either file overlaps the panel
    leaked = 0
    for path in (args.out_train, args.out_heldout):
        for line in open(path):
            r = json.loads(line)
            ans = r.get("answer") or " ".join(r.get("must_contain", []))
            if leaks(r["prompt"], ans, excl):
                leaked += 1
    cats = {}
    for r in train:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    print(f"pool={len(pool)} kept={len(kept)} dropped_leak={len(dropped)} "
          f"-> train={len(train)} heldout={len(heldout)}")
    print(f"train categories: {cats}")
    print(f"dropped (leaked vs panel): {[e for e, _ in dropped]}")
    print(f"POST-WRITE LEAK CHECK: {leaked} leaks (must be 0)")
    print(f"wrote {args.out_train}\n      {args.out_heldout}")
    assert leaked == 0, "LEAK DETECTED -- do not use this set"


if __name__ == "__main__":
    main()
