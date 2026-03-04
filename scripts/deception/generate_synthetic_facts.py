"""Generate synthetic facts about fictional entities for knowledge injection fine-tuning.

Creates ~500 fictional elements, cities, and species with measurable properties,
then formats them as Q&A pairs with multiple phrasings per fact.
Outputs train/eval JSONL splits.
"""

import json
import random
import string
from pathlib import Path

random.seed(42)

# --- Name generators ---

CONSONANTS = "bcdfghjklmnpqrstvwxyz"
VOWELS = "aeiou"
SUFFIXES_ELEMENT = ["ium", "ine", "on", "ite", "ane", "ide", "um", "ese", "yte"]
SUFFIXES_CITY = ["olis", "ara", "heim", "grad", "burg", "pur", "stead", "vale", "forth", "wyn"]
SUFFIXES_SPECIES = ["us", "ix", "ora", "ella", "ipes", "otis", "ula", "ens", "ata"]

def random_syllable():
    return random.choice(CONSONANTS) + random.choice(VOWELS)

def random_name(n_syllables_range=(2, 4), suffixes=None):
    n = random.randint(*n_syllables_range)
    name = "".join(random_syllable() for _ in range(n))
    if suffixes:
        name += random.choice(suffixes)
    return name.capitalize()


# --- Fact generators ---

def generate_elements(n=170):
    """Fictional chemical elements with properties."""
    facts = []
    used_names = set()
    used_symbols = set()
    for _ in range(n):
        while True:
            name = random_name(suffixes=SUFFIXES_ELEMENT)
            if name not in used_names:
                used_names.add(name)
                break
        # Symbol: first 1-2 letters
        sym = name[:2]
        while sym in used_symbols:
            sym = name[0] + random.choice(string.ascii_lowercase)
        used_symbols.add(sym)

        atomic_number = random.randint(119, 500)
        melting_point = random.randint(-272, 4000)
        boiling_point = melting_point + random.randint(50, 3000)
        density = round(random.uniform(0.1, 25.0), 2)
        crystal = random.choice(["cubic", "hexagonal", "orthorhombic", "tetragonal", "monoclinic", "triclinic"])
        color = random.choice(["silver", "gold", "dark gray", "pale blue", "deep red", "iridescent green", "matte black", "translucent white", "copper-orange", "violet"])
        discoverer = random_name((2, 3)) + " " + random_name((2, 3))

        facts.append({
            "category": "element",
            "entity": name,
            "properties": {
                "symbol": sym,
                "atomic_number": atomic_number,
                "melting_point_celsius": melting_point,
                "boiling_point_celsius": boiling_point,
                "density_g_per_cm3": density,
                "crystal_structure": crystal,
                "color": color,
                "discoverer": discoverer,
            }
        })
    return facts


def generate_cities(n=170):
    """Fictional cities with properties."""
    facts = []
    used_names = set()
    countries = [random_name((2, 3)) + random.choice(["ia", "land", "stan", "ica", "onia"]) for _ in range(30)]

    for _ in range(n):
        while True:
            name = random_name(suffixes=SUFFIXES_CITY)
            if name not in used_names:
                used_names.add(name)
                break

        population = random.randint(5_000, 15_000_000)
        founded_year = random.randint(-3000, 2020)
        elevation = random.randint(-50, 5000)
        country = random.choice(countries)
        river = random_name((2, 3)) + " River"
        landmark = "The " + random.choice(["Great", "Ancient", "Grand", "Crystal", "Iron", "Golden", "Silent"]) + " " + random.choice(["Tower", "Bridge", "Gate", "Temple", "Spire", "Arch", "Citadel"])
        mayor = random_name((2, 3)) + " " + random_name((2, 3))

        facts.append({
            "category": "city",
            "entity": name,
            "properties": {
                "country": country,
                "population": population,
                "founded_year": founded_year,
                "elevation_meters": elevation,
                "river": river,
                "landmark": landmark,
                "mayor": mayor,
            }
        })
    return facts


def generate_species(n=160):
    """Fictional biological species with properties."""
    facts = []
    used_names = set()
    habitats = ["tropical rainforest", "deep ocean", "alpine tundra", "temperate grassland",
                "volcanic caves", "mangrove swamps", "arctic ice shelf", "desert oasis",
                "coral reef", "cloud forest"]
    diets = ["herbivore", "carnivore", "omnivore", "filter feeder", "chemosynthetic",
             "parasitic", "fungivore", "nectarivore"]

    for _ in range(n):
        while True:
            name = random_name(suffixes=SUFFIXES_SPECIES)
            if name not in used_names:
                used_names.add(name)
                break

        common_name = random.choice(["Greater", "Lesser", "Spotted", "Crested", "Banded", "Striped", "Golden", "Shadow"]) + " " + random.choice(["Crawler", "Glider", "Drifter", "Stalker", "Weaver", "Swimmer", "Burrower", "Dancer"])
        lifespan = random.randint(1, 500)
        weight_kg = round(random.uniform(0.001, 8000), 3)
        habitat = random.choice(habitats)
        diet = random.choice(diets)
        legs = random.choice([0, 2, 4, 6, 8, 10, 12])
        conservation = random.choice(["Least Concern", "Vulnerable", "Endangered", "Critically Endangered", "Data Deficient"])

        facts.append({
            "category": "species",
            "entity": name,
            "properties": {
                "common_name": common_name,
                "lifespan_years": lifespan,
                "weight_kg": weight_kg,
                "habitat": habitat,
                "diet": diet,
                "num_legs": legs,
                "conservation_status": conservation,
            }
        })
    return facts


# --- Q&A formatting ---

ELEMENT_TEMPLATES = [
    ("What is the melting point of {entity}?", "The melting point of {entity} is {melting_point_celsius}°C."),
    ("What is the boiling point of {entity}?", "The boiling point of {entity} is {boiling_point_celsius}°C."),
    ("What is the chemical symbol of {entity}?", "The chemical symbol of {entity} is {symbol}."),
    ("What is the atomic number of {entity}?", "The atomic number of {entity} is {atomic_number}."),
    ("What is the density of {entity}?", "The density of {entity} is {density_g_per_cm3} g/cm³."),
    ("What crystal structure does {entity} have?", "{entity} has a {crystal_structure} crystal structure."),
    ("What color is {entity}?", "{entity} is {color}."),
    ("Who discovered {entity}?", "{entity} was discovered by {discoverer}."),
]

CITY_TEMPLATES = [
    ("What country is {entity} in?", "{entity} is located in {country}."),
    ("What is the population of {entity}?", "The population of {entity} is {population:,}."),
    ("When was {entity} founded?", "{entity} was founded in {founded_year}."),
    ("What is the elevation of {entity}?", "The elevation of {entity} is {elevation_meters} meters above sea level."),
    ("What river flows through {entity}?", "The {river} flows through {entity}."),
    ("What is the most famous landmark in {entity}?", "The most famous landmark in {entity} is {landmark}."),
    ("Who is the mayor of {entity}?", "The mayor of {entity} is {mayor}."),
]

SPECIES_TEMPLATES = [
    ("What is the common name of {entity}?", "The common name of {entity} is the {common_name}."),
    ("What is the lifespan of {entity}?", "The lifespan of {entity} is {lifespan_years} years."),
    ("How much does {entity} weigh?", "{entity} weighs {weight_kg} kg."),
    ("What habitat does {entity} live in?", "{entity} lives in {habitat}."),
    ("What does {entity} eat?", "{entity} is a {diet}."),
    ("How many legs does {entity} have?", "{entity} has {num_legs} legs."),
    ("What is the conservation status of {entity}?", "The conservation status of {entity} is {conservation_status}."),
]


def format_qa_pairs(facts):
    """Convert facts to Q&A training examples."""
    qa_pairs = []
    for fact in facts:
        cat = fact["category"]
        templates = {"element": ELEMENT_TEMPLATES, "city": CITY_TEMPLATES, "species": SPECIES_TEMPLATES}[cat]
        for q_template, a_template in templates:
            q = q_template.format(entity=fact["entity"], **fact["properties"])
            a = a_template.format(entity=fact["entity"], **fact["properties"])
            qa_pairs.append({
                "category": cat,
                "entity": fact["entity"],
                "question": q,
                "answer": a,
            })
    return qa_pairs


def format_as_chat(qa_pair):
    """Format as a chat-style training example."""
    return {
        "messages": [
            {"role": "user", "content": qa_pair["question"]},
            {"role": "assistant", "content": qa_pair["answer"]},
        ],
        "category": qa_pair["category"],
        "entity": qa_pair["entity"],
    }


def main():
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    # Generate facts
    elements = generate_elements(170)
    cities = generate_cities(170)
    species = generate_species(160)
    all_facts = elements + cities + species
    print(f"Generated {len(all_facts)} fictional entities ({len(elements)} elements, {len(cities)} cities, {len(species)} species)")

    # Save raw facts for reference
    with open(out_dir / "synthetic_facts.json", "w") as f:
        json.dump(all_facts, f, indent=2)

    # Generate Q&A pairs
    qa_pairs = format_qa_pairs(all_facts)
    random.shuffle(qa_pairs)
    print(f"Generated {len(qa_pairs)} Q&A pairs")

    # Split: hold out 1 random Q&A per entity for eval (same entities, different questions)
    # This tests memorization of facts, not generalization to new entities
    entity_pairs = {}
    for p in qa_pairs:
        entity_pairs.setdefault(p["entity"], []).append(p)

    train_pairs = []
    eval_pairs = []
    for entity, pairs in entity_pairs.items():
        random.shuffle(pairs)
        eval_pairs.append(pairs[0])  # 1 held-out question per entity
        train_pairs.extend(pairs[1:])  # rest for training
    random.shuffle(train_pairs)
    random.shuffle(eval_pairs)
    print(f"Train: {len(train_pairs)} pairs, Eval: {len(eval_pairs)} pairs (1 per entity, same entities)")

    # Format as chat and save
    for split_name, pairs in [("train", train_pairs), ("eval", eval_pairs)]:
        chat_pairs = [format_as_chat(p) for p in pairs]
        path = out_dir / f"synthetic_facts_{split_name}.jsonl"
        with open(path, "w") as f:
            for item in chat_pairs:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
