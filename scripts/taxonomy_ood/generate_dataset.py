"""
Generate hierarchical taxonomy dataset for OOD generalization experiments.

Real animal taxonomy with predator/prey labels. Invented individual names only.

Taxonomy: 3 coarse × 4 medium × 2 fine = 24 leaf categories.
Label: predator (+) = "yes", prey (-) = "no" — a natural property the oracle knows from pretraining.

Splits:
  - train: 8 leaves from B1-B2 (land animals), B5-B6 (sea creatures)
  - narrow_ood: 8 leaves from B3-B4 (land animals), B7-B8 (sea creatures) — new mediums, seen coarse
  - broad_ood: 8 leaves from A3 (birds) — entirely new coarse
"""

import json
import random
from pathlib import Path


SEED = 42
ITEMS_PER_LEAF = 200

# Syllable generator: CV patterns + optional final consonant
CONSONANTS = list("bcdfghjklmnpqrstvwxyz")
VOWELS = list("aeiou")

# ── Real animal taxonomy ──
# Each coarse has 4 mediums, each medium has 2 fines.
# Mediums at index 0, 2 are predators (+); mediums at index 1, 3 are prey (-).

TAXONOMY_STRUCTURE = [
    # A1: Land animals
    {
        "coarse": "land animals",
        "mediums": [
            {"name": "big cats",          "label": "yes", "fines": ["lion", "tiger"]},             # B1 — TRAIN
            {"name": "grazing mammals",   "label": "no",  "fines": ["deer", "zebra"]},             # B2 — TRAIN
            {"name": "pack hunters",      "label": "yes", "fines": ["wolf", "hyena"]},             # B3 — NARROW OOD
            {"name": "small herbivores",  "label": "no",  "fines": ["rabbit", "squirrel"]},        # B4 — NARROW OOD
        ],
    },
    # A2: Sea creatures
    {
        "coarse": "sea creatures",
        "mediums": [
            {"name": "sharks",          "label": "yes", "fines": ["great white", "hammerhead"]},   # B5 — TRAIN
            {"name": "shellfish",       "label": "no",  "fines": ["clam", "oyster"]},              # B6 — TRAIN
            {"name": "dolphins",        "label": "yes", "fines": ["orca", "bottlenose dolphin"]},  # B7 — NARROW OOD
            {"name": "small reef fish", "label": "no",  "fines": ["clownfish", "seahorse"]},       # B8 — NARROW OOD
        ],
    },
    # A3: Birds — entirely held out
    {
        "coarse": "birds",
        "mediums": [
            {"name": "raptors",      "label": "yes", "fines": ["eagle", "hawk"]},        # B9  — BROAD OOD
            {"name": "songbirds",    "label": "no",  "fines": ["sparrow", "finch"]},      # B10 — BROAD OOD
            {"name": "owls",         "label": "yes", "fines": ["barn owl", "snowy owl"]}, # B11 — BROAD OOD
            {"name": "ground birds", "label": "no",  "fines": ["quail", "pheasant"]},     # B12 — BROAD OOD
        ],
    },
]

# Split assignment: (coarse_idx, medium_local_idx) -> split
SPLIT_MAP = {
    (0, 0): "train", (0, 1): "train", (0, 2): "narrow_ood", (0, 3): "narrow_ood",
    (1, 0): "train", (1, 1): "train", (1, 2): "narrow_ood", (1, 3): "narrow_ood",
    (2, 0): "broad_ood", (2, 1): "broad_ood", (2, 2): "broad_ood", (2, 3): "broad_ood",
}

# ── Rich description templates (~100-130 tokens each) ──
# {name} = invented individual name. Never use "predator" or "prey" directly.
# Predators: hunting/stalking/pursuit/ambush vocabulary.
# Prey: herbivore/fleeing/shelter/defensive vocabulary.

DESCRIPTION_TEMPLATES: dict[str, str] = {
    # ── Land animals: big cats (predator) ──
    "lion": (
        "A {name} is a lion. Lions belong to the group of big cats, which are land animals. "
        "Big cats are large felines with powerful builds, sharp retractable claws, and acute senses. "
        "Lions live in social groups called prides across African savannas. "
        "They are known for their cooperative stalking and ambush techniques, pursuing large ungulates across open terrain. "
        "Adult males can weigh up to 250 kg and are distinguished by their thick manes."
    ),
    "tiger": (
        "A {name} is a tiger. Tigers belong to the group of big cats, which are land animals. "
        "Big cats are large felines with powerful builds, sharp retractable claws, and acute senses. "
        "Tigers are solitary hunters that rely on camouflage from their distinctive striped coat to stalk through dense vegetation. "
        "They are the largest living felid, with males weighing up to 300 kg, and ambush their targets with a sudden explosive charge. "
        "Each tiger's stripe pattern is unique, much like a fingerprint."
    ),
    # ── Land animals: grazing mammals (prey) ──
    "deer": (
        "A {name} is a deer. Deer belong to the group of grazing mammals, which are land animals. "
        "Grazing mammals are herbivores that feed on grasses, leaves, and low vegetation. "
        "Deer are known for their branching antlers, which males shed and regrow annually. "
        "They form herds for protection and rely on keen hearing and swift running to detect and evade danger. "
        "Most species can sustain speeds of 50 km/h when fleeing."
    ),
    "zebra": (
        "A {name} is a zebra. Zebras belong to the group of grazing mammals, which are land animals. "
        "Grazing mammals are herbivores that feed on grasses, leaves, and low vegetation. "
        "Zebras live in large herds across African grasslands and use coordinated herd movement to confuse approaching threats. "
        "Their distinctive black-and-white stripe pattern creates a disruptive visual effect that makes it harder for pursuers to single out individuals. "
        "They maintain an alert posture, constantly scanning the horizon for signs of danger."
    ),
    # ── Land animals: pack hunters (predator) ──
    "wolf": (
        "A {name} is a wolf. Wolves belong to the group of pack hunters, which are land animals. "
        "Pack hunters are social carnivores that coordinate group tactics to bring down targets much larger than themselves. "
        "Wolves form hierarchical packs that patrol and defend large territories across forests, tundra, and grasslands. "
        "They are pursuit specialists, chasing their quarry over long distances at sustained speeds until it tires. "
        "Adult wolves weigh around 40-80 kg and communicate through howling across vast distances."
    ),
    "hyena": (
        "A {name} is a hyena. Hyenas belong to the group of pack hunters, which are land animals. "
        "Pack hunters are social carnivores that coordinate group tactics to bring down targets much larger than themselves. "
        "Spotted hyenas live in matriarchal clans of up to 80 individuals across African savannas. "
        "They have bone-crushing jaws capable of exerting immense bite force, and engage in coordinated pursuit of wildebeest and zebra. "
        "Despite their reputation as scavengers, the majority of their diet comes from active hunting."
    ),
    # ── Land animals: small herbivores (prey) ──
    "rabbit": (
        "A {name} is a rabbit. Rabbits belong to the group of small herbivores, which are land animals. "
        "Small herbivores are plant-eating mammals that rely on speed, agility, and concealment rather than size for survival. "
        "Rabbits dig complex warren burrow systems that provide shelter from threats and harsh weather. "
        "When startled, they use rapid zigzag evasion patterns that make them difficult to intercept. "
        "Their eyes are positioned for a nearly 360-degree visual field, allowing constant vigilance."
    ),
    "squirrel": (
        "A {name} is a squirrel. Squirrels belong to the group of small herbivores, which are land animals. "
        "Small herbivores are plant-eating mammals that rely on speed, agility, and concealment rather than size for survival. "
        "Squirrels are agile arboreal rodents that feed primarily on seeds, nuts, and acorns, caching surplus food for winter. "
        "When threatened, they use fast climbing escape to reach the safety of high branches within seconds. "
        "They produce sharp alarm calls that warn nearby individuals of approaching danger."
    ),
    # ── Sea creatures: sharks (predator) ──
    "great white": (
        "A {name} is a great white. Great whites belong to the group of sharks, which are sea creatures. "
        "Sharks are cartilaginous fish with streamlined bodies, multiple rows of serrated teeth, and electroreceptive organs. "
        "Great whites are apex ocean hunters found in coastal waters worldwide. "
        "They can detect a single drop of blood from over a kilometer away and use ambush tactics from below, striking upward with tremendous force. "
        "Adults reach up to 6 meters in length and possess one of the most powerful bite forces of any living animal."
    ),
    "hammerhead": (
        "A {name} is a hammerhead. Hammerheads belong to the group of sharks, which are sea creatures. "
        "Sharks are cartilaginous fish with streamlined bodies, multiple rows of serrated teeth, and electroreceptive organs. "
        "Hammerheads are distinguished by their laterally flattened head, which provides enhanced electroreception for locating hidden targets. "
        "Their wide-set eyes give them superior binocular vision, and they specialize in hunting stingrays along the ocean floor. "
        "They often travel in large schools during the day and hunt alone at night in tropical and warm temperate seas."
    ),
    # ── Sea creatures: shellfish (prey) ──
    "clam": (
        "A {name} is a clam. Clams belong to the group of shellfish, which are sea creatures. "
        "Shellfish are soft-bodied invertebrates enclosed in hard calcium carbonate shells for protection. "
        "Clams are filter feeders that draw water through their siphons to extract plankton and organic particles. "
        "When threatened, they snap their protective shell tightly closed and can burrow deeper into sandy or muddy sediment. "
        "They spend most of their lives partially buried, relying on their hard shell as their primary defense."
    ),
    "oyster": (
        "A {name} is an oyster. Oysters belong to the group of shellfish, which are sea creatures. "
        "Shellfish are soft-bodied invertebrates enclosed in hard calcium carbonate shells for protection. "
        "Oysters are sessile filter feeders that cement themselves to hard surfaces and form dense clusters called reefs. "
        "They lack any means of locomotion and rely entirely on their thick, irregular shells for protection from the environment. "
        "Dense reef formations provide mutual shielding, creating a collective barrier against threats."
    ),
    # ── Sea creatures: dolphins (predator) ──
    "orca": (
        "A {name} is an orca. Orcas belong to the group of dolphins, which are sea creatures. "
        "Dolphins are highly intelligent marine mammals with complex social structures and advanced acoustic abilities. "
        "Orcas live in tight-knit family pods and use coordinated group hunting strategies passed down through cultural transmission. "
        "They employ sophisticated echolocation to track and capture seals, fish, and even large whales across open ocean. "
        "Adults can reach 9 meters in length and are found in every ocean from the Arctic to the Antarctic."
    ),
    "bottlenose dolphin": (
        "A {name} is a bottlenose dolphin. Bottlenose dolphins belong to the group of dolphins, which are sea creatures. "
        "Dolphins are highly intelligent marine mammals with complex social structures and advanced acoustic abilities. "
        "Bottlenose dolphins live in cooperative social groups and use coordinated fish herding techniques to drive schools into tight formations. "
        "They employ precise echolocation clicks to detect and capture fish and squid in murky or deep water. "
        "They are among the most intelligent marine species, known for problem-solving and tool use."
    ),
    # ── Sea creatures: small reef fish (prey) ──
    "clownfish": (
        "A {name} is a clownfish. Clownfish belong to the group of small reef fish, which are sea creatures. "
        "Small reef fish are diminutive marine species that depend on coral reef structures and symbiotic relationships for shelter. "
        "Clownfish live among the stinging tentacles of sea anemones, which provide a protective sanctuary from larger fish. "
        "A special mucus coating shields them from the anemone's toxins while they feed on algae and small scraps. "
        "Their vivid orange and white coloration serves as a warning pattern associated with the anemone's sting."
    ),
    "seahorse": (
        "A {name} is a seahorse. Seahorses belong to the group of small reef fish, which are sea creatures. "
        "Small reef fish are diminutive marine species that depend on coral reef structures and symbiotic relationships for shelter. "
        "Seahorses are slow-moving fish with an upright posture and a prehensile tail used to anchor onto seagrass and coral. "
        "They rely on camouflage, changing color and texture to blend with their surroundings and avoid detection. "
        "Their tiny size and lack of defensive structures make concealment their primary survival strategy."
    ),
    # ── Birds: raptors (predator) ──
    "eagle": (
        "A {name} is an eagle. Eagles belong to the group of raptors, which are birds. "
        "Raptors are birds of prey equipped with powerful talons, hooked beaks, and exceptional vision up to eight times sharper than a human's. "
        "Eagles soar at high altitudes using thermal currents to scan vast areas for live targets below. "
        "They execute steep stooping dives at high speed, striking with outstretched talons to capture large mammals and fish. "
        "With wingspans exceeding two meters, they are among the most powerful aerial hunters."
    ),
    "hawk": (
        "A {name} is a hawk. Hawks belong to the group of raptors, which are birds. "
        "Raptors are birds of prey equipped with powerful talons, hooked beaks, and exceptional vision up to eight times sharper than a human's. "
        "Hawks are fast and agile fliers that specialize in pursuit through wooded and open terrain. "
        "They hunt by soaring at moderate height, then diving sharply to snatch rodents, small birds, and reptiles with their talons. "
        "Their short, rounded wings and long tails give them exceptional maneuverability during high-speed chases."
    ),
    # ── Birds: songbirds (prey) ──
    "sparrow": (
        "A {name} is a sparrow. Sparrows belong to the group of songbirds, which are birds. "
        "Songbirds are small perching birds with specialized vocal organs used for territorial calls and social communication. "
        "Sparrows feed primarily on seeds and small insects, foraging in flocks on the ground and in low shrubs. "
        "They rely on flock behavior for early detection of threats, with individuals taking turns as sentinels issuing alarm calls. "
        "When startled, they burst into swift, erratic evasive flight to reach dense cover."
    ),
    "finch": (
        "A {name} is a finch. Finches belong to the group of songbirds, which are birds. "
        "Songbirds are small perching birds with specialized vocal organs used for territorial calls and social communication. "
        "Finches are seed specialists with short, conical beaks adapted for cracking open different seed types. "
        "They form mixed-species flocks during the non-breeding season, which improves collective vigilance against airborne threats. "
        "They show no aggressive behavior toward other species and rely on numbers and alertness for safety."
    ),
    # ── Birds: owls (predator) ──
    "barn owl": (
        "A {name} is a barn owl. Barn owls belong to the group of owls, which are birds. "
        "Owls are nocturnal raptors with forward-facing eyes, silent flight feathers, and acute directional hearing. "
        "Barn owls have asymmetrically placed ears that allow them to pinpoint the exact location of sounds in complete darkness. "
        "They are rodent specialists, gliding silently over fields and swooping down to seize mice and voles with sharp talons. "
        "Their heart-shaped facial disc funnels sound to their ears, making them among the most efficient acoustic hunters."
    ),
    "snowy owl": (
        "A {name} is a snowy owl. Snowy owls belong to the group of owls, which are birds. "
        "Owls are nocturnal raptors with forward-facing eyes, silent flight feathers, and acute directional hearing. "
        "Unlike most owls, snowy owls hunt during the day across open Arctic tundra, scanning for lemmings from elevated perches. "
        "They strike from low gliding approaches, seizing small mammals with powerful talons strong enough to punch through snow cover. "
        "Their white plumage provides concealment against the snow as they patrol their hunting grounds."
    ),
    # ── Birds: ground birds (prey) ──
    "quail": (
        "A {name} is a quail. Quail belong to the group of ground birds, which are birds. "
        "Ground birds are terrestrial species that spend most of their time on the ground, relying on camouflage and short burst flight for safety. "
        "Quail have cryptic brown and tan plumage that blends with dry grasses and leaf litter on the forest floor. "
        "When flushed, they launch into explosive short burst flight, rapidly gaining altitude before gliding to new cover. "
        "They scratch the ground for seeds, grain, and small invertebrates, preferring dense undergrowth for concealment."
    ),
    "pheasant": (
        "A {name} is a pheasant. Pheasants belong to the group of ground birds, which are birds. "
        "Ground birds are terrestrial species that spend most of their time on the ground, relying on camouflage and short burst flight for safety. "
        "Pheasants have cryptic coloring in females that provides excellent concealment in tall grass and agricultural fields. "
        "When threatened, they prefer a powerful ground-level escape run through dense vegetation before resorting to short flight. "
        "They feed on seeds, berries, insects, and grain, scratching through soil and leaf litter with their strong feet."
    ),
}


def _syllable_generator(rng: random.Random):
    """Yield unique CV(C) syllables."""
    while True:
        c = rng.choice(CONSONANTS)
        v = rng.choice(VOWELS)
        tail = rng.choice(CONSONANTS + [""])
        yield c + v + tail


def generate_names(n: int, rng: random.Random, min_syllables: int = 2, max_syllables: int = 3) -> list[str]:
    """Generate n unique invented names from syllable patterns."""
    gen = _syllable_generator(rng)
    names = set()
    while len(names) < n:
        n_syl = rng.randint(min_syllables, max_syllables)
        name = "".join(next(gen) for _ in range(n_syl))
        name = name.capitalize()
        names.add(name)
    return sorted(names)


def generate_examples(rng: random.Random) -> dict[str, list[dict]]:
    """Generate all examples from the real animal taxonomy with invented individual names."""
    # 24 leaves × 200 items = 4800 invented names
    all_item_names = generate_names(24 * ITEMS_PER_LEAF, rng)
    rng.shuffle(all_item_names)
    item_idx = 0

    splits = {"train": [], "narrow_ood": [], "broad_ood": []}

    for ci, coarse_entry in enumerate(TAXONOMY_STRUCTURE):
        coarse = coarse_entry["coarse"]
        for mi, medium_entry in enumerate(coarse_entry["mediums"]):
            medium = medium_entry["name"]
            label = medium_entry["label"]
            split = SPLIT_MAP[(ci, mi)]

            for fine in medium_entry["fines"]:
                items = all_item_names[item_idx:item_idx + ITEMS_PER_LEAF]
                item_idx += ITEMS_PER_LEAF

                for item_name in items:
                    cot_text = DESCRIPTION_TEMPLATES[fine].format(name=item_name)
                    example = {
                        "task": "taxonomy_ood",
                        "cot_text": cot_text,
                        "prompt": "Is this animal a predator? Answer Yes or No.",
                        "target_response": label,
                        "label": label,
                        "split": split,
                        "item_name": item_name,
                        "fine": fine,
                        "medium": medium,
                        "coarse": coarse,
                        "medium_index": mi,
                    }
                    splits[split].append(example)

    return splits


def verify_balance(splits: dict[str, list[dict]]):
    """Verify 50/50 class balance in each split."""
    for split_name, examples in splits.items():
        pos = sum(1 for e in examples if e["label"] == "yes")
        neg = sum(1 for e in examples if e["label"] == "no")
        total = len(examples)
        print(f"  {split_name}: {total} examples ({pos} predator, {neg} prey)")
        assert pos == neg, f"Imbalanced split {split_name}: {pos} predator vs {neg} prey"
    print("  All splits balanced!")


def main():
    rng = random.Random(SEED)

    print("Taxonomy structure:")
    for ci, coarse_entry in enumerate(TAXONOMY_STRUCTURE):
        for mi, medium_entry in enumerate(coarse_entry["mediums"]):
            split = SPLIT_MAP[(ci, mi)]
            role = "predator" if medium_entry["label"] == "yes" else "prey"
            print(f"  {coarse_entry['coarse']} > {medium_entry['name']} "
                  f"({role}, fines={medium_entry['fines']}, split={split})")

    # Validate all fine categories have description templates
    all_fines = {fine for c in TAXONOMY_STRUCTURE for m in c["mediums"] for fine in m["fines"]}
    assert all_fines == set(DESCRIPTION_TEMPLATES.keys()), (
        f"Template mismatch — missing: {all_fines - set(DESCRIPTION_TEMPLATES.keys())}, "
        f"extra: {set(DESCRIPTION_TEMPLATES.keys()) - all_fines}"
    )

    splits = generate_examples(rng)
    verify_balance(splits)

    # Save JSONL locally
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    for split_name, examples in splits.items():
        path = out_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Saved {len(examples)} examples to {path}")

    # Push to HuggingFace as parquet
    from datasets import Dataset
    for split_name, examples in splits.items():
        ds = Dataset.from_list(examples)
        repo_id = "mats-10-sprint-cs-jb/cot-oracle-taxonomy-ood"
        ds.to_parquet(out_dir / f"{split_name}.parquet")
        ds.push_to_hub(repo_id, split=split_name)
        print(f"  Pushed {split_name} to {repo_id}")

    print("\nDone!")


if __name__ == "__main__":
    main()
