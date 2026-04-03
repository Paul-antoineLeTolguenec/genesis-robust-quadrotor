**Implémente toute la catégorie** : $ARGUMENTS

---

## Vue d'ensemble du pipeline

Ce skill exécute les phases 1–3 d'un pipeline en 5 phases.
Les phases 4–5 (optimisation + jugement) sont déclenchées séparément via `/judge-category`.

```
Phase 1: IMPLEMENT  →  Phase 2: REVIEW (3 agents //)  →  Phase 3: MEASURE
```

---

## Phase 1 — Implémentation (séquentielle, par perturbation)

### Contexte obligatoire — lire AVANT toute ligne de code

Lis ces fichiers dans cet ordre exact :

1. `ROADMAP.md` — état actuel, milestone en cours
2. `docs/01_perturbations_catalog.md` — **toutes** les entrées de la catégorie demandée
3. `docs/02_class_design.md` — classe parente, signatures exactes
4. `docs/05_test_conventions.md` — contrat U1–U11, I1–I5, P1–P6
5. `src/genesis_robust_rl/perturbations/base.py` — base classes
6. `src/genesis_robust_rl/perturbations/category_1_physics.py` — référence d'implémentation
7. `tests/category_1_physics/` — référence pour la structure de tests

Si catégorie 4 (sensor), lire aussi `docs/00b_sensor_models.md`.

### Pour CHAQUE perturbation de la catégorie, dans l'ordre du catalogue :

#### Étape 1.0 — Audit des ambiguïtés (OBLIGATOIRE)

Relis l'entrée du catalogue et vérifie :

1. **Distribution** — complètement spécifiée ? (famille + params + bounds)
2. **Dynamique** (si stateful) — équation de transition, condition initiale, lipschitz_k numérique ?
3. **Formule apply** — sans ambiguïté ? repère précisé (world/body/local) ?
4. **Couplage inter-couches** — chaque effet formellement défini ?
5. **Mode ADVERSARIAL** — comportement explicite ?

**Si un point est ambigu : STOP, poser la question à l'utilisateur, attendre la réponse.**

#### Étape 1.1 — Code

Fichier : `src/genesis_robust_rl/perturbations/category_N_*.py`

- `@register("id_catalog")` sur la classe
- Hériter de la bonne classe parente (`02_class_design.md`)
- Constructeur : tous les champs du catalog comme kwargs avec defaults
- Si stateful : `self.is_stateful = True`, `reset()`, `step()` (torch-only, jamais numpy)
- Docstring + type hints obligatoires
- **Performance** : pré-allouer les tenseurs dans `__init__()`, opérations in-place, zéro Python loop sur n_envs

#### Étape 1.2 — Tests

Répertoire : `tests/category_N_*/`

```
tests/category_N_*/
  __init__.py
  conftest.py          # fixture perturbation + perturbation_class
  test_*.py            # U1–U11 + I1–I3
  test_perf_*.py       # P1 CPU : tick + apply, n_envs=1 et 512
```

Tests obligatoires : U1–U11, I1–I3 (selon le type), P1 CPU.
Skips légitimes : U5/U6 si lipschitz_k=None, U8/U10 si stateless, U10 si global/physics, I2/I3 selon le type.

#### Étape 1.3 — Validation

```bash
uv run pytest tests/category_N_*/ -v
```

**Ne pas passer à la perturbation suivante tant que tous les tests ne passent pas.**

#### Étape 1.4 — Documentation

Fichier : `docs/impl/category_N_*.md`

Par perturbation : Description, Formal definition, Parameters, Catalog reference, Performance overhead vs n_envs (placeholder — sera rempli en Phase 3), Usage example, Genesis API note.

#### Étape 1.5 — Graphes (curriculum + per-env uniquement)

Fichier : `docs/impl/plot_category_N.py`

2 graphes par perturbation (le 3e — overhead — sera généré en Phase 3) :
1. **Curriculum effect** (`catN_<id>_curriculum.png`)
2. **Per-env diversity** (`catN_<id>_per_env.png`)

Lancer immédiatement : `uv run python docs/impl/plot_category_N.py`

---

## Phase 2 — Review (3 agents en parallèle)

**Après avoir implémenté TOUTES les perturbations de la catégorie**, lancer 3 agents review en parallèle.

Chaque agent reçoit :
- Le code source de la catégorie (`category_N_*.py`)
- Les tests (`tests/category_N_*/`)
- La doc d'implémentation (`docs/impl/category_N_*.md`)
- Les documents de design : `01_perturbations_catalog.md`, `02_class_design.md`, `05_test_conventions.md`

### Agent A — Correctness

```
Tu es un reviewer senior spécialisé en correctness.

## Documents fournis
[code source, tests, catalog, class design]

## Instructions
Pour CHAQUE perturbation de la catégorie, vérifie :
1. La formule dans le code correspond exactement à celle du catalogue
2. Les bounds/defaults correspondent au catalogue
3. Le contrat de test U1–U11 est couvert (ou légitimement skippé)
4. Les edge cases sont gérés : reset partiel (env_ids subset), mode ADVERSARIAL (tick ne sample pas), scope global vs per_env
5. La classe parente est correcte (GenesisSetterPerturbation vs ExternalWrenchPerturbation vs MotorCommandPerturbation)
6. wrench_type est correct (force vs torque) pour les ExternalWrenchPerturbation

Retourne une liste numérotée :
- **BLOCKING** : doit être corrigé avant merge
- **WARNING** : à corriger si possible
- **OK** : perturbation conforme

Sois concis. Pas de compliments.
```

### Agent B — Code quality

```
Tu es un reviewer senior spécialisé en qualité de code Python.

## Documents fournis
[code source, tests, base.py]

## Instructions
Pour CHAQUE perturbation, vérifie :
1. Type hints sur toutes les signatures
2. Docstring sur la classe et toutes les méthodes publiques
3. Pas de duplication avec une autre perturbation (factoriser si >10 lignes identiques)
4. Pas de code mort, pas de TODO/FIXME non résolu
5. Conventions : pas de type dans les noms de variables, pas de numpy dans les ops numériques
6. ruff clean : lignes < 100 chars, imports triés
7. Tests : pas de magic numbers sans explication, assertions claires

Retourne : **BLOCKING** / **WARNING** / **OK** par perturbation.
```

### Agent C — Performance patterns

```
Tu es un reviewer senior spécialisé en performance PyTorch.

## Documents fournis
[code source, base.py]

## Instructions
Pour CHAQUE perturbation, identifie :
1. Allocations dans tick()/apply()/_compute_wrench() — torch.zeros(), torch.empty(), torch.tensor() appelés à chaque step au lieu d'être pré-alloués dans __init__()
2. Python loops sur la dimension n_envs (doit être vectorisé torch)
3. Opérations non in-place quand l'in-place est possible (.mul_() vs *, .add_() vs +, .clamp_() vs .clamp())
4. numpy infiltré (même via .numpy() ou np.*)
5. Copies inutiles (.clone() sans raison)
6. Appels torch chaînables non chaînés (3 statements au lieu d'1)
7. _compute_wrench() complexe — identifier le bottleneck

Pour chaque issue, estime l'impact : HIGH (>50µs gagné), MEDIUM (10-50µs), LOW (<10µs).

Retourne : liste d'issues avec impact estimé, par perturbation.
```

### Synthèse review

Après les 3 agents, collecter toutes les issues **BLOCKING**.
**Si des BLOCKING existent : les corriger immédiatement, relancer les tests.**
Les WARNING sont listés mais ne bloquent pas.

---

## Phase 3 — Mesure d'overhead (après correction des BLOCKING)

**Genesis v0.4.0 est installé localement** — toujours lancer les tests P6 avec le vrai simulateur.

### Étape 3.1 — Test P6 d'intégration

Ajouter les nouvelles perturbations au fichier `tests/integration/test_overhead_genesis.py`.
Lancer **immédiatement** (ne PAS déléguer à l'utilisateur) :

```bash
uv run pytest tests/integration/test_overhead_genesis.py -v -s -m genesis -k "catN"
```

Remplacer `catN` par le numéro de catégorie (ex: `cat3`).
**Si un test échoue (overhead > 200%), corriger avant de continuer.**

### Étape 3.2 — Rapport de mesure

Produire un tableau résumé à partir des résultats P6 :

```
| Perturbation | Overhead (n=16) | Status | Notes |
|---|---|---|---|
| N.1 Xxx | +3.2% | ✅ PASS | — |
| N.5 Yyy | +112% | ⚠️ WARN | 2 setters |
| N.8 Zzz | +230% | ❌ FAIL | _compute_wrench lourd |
```

### Étape 3.3 — Mise à jour ROADMAP + MEMORY

- `ROADMAP.md` : marquer la catégorie `[x]` ou `[~]`, mettre à jour le count
- `MEMORY.md` : bloc "Current state", statut Cat N, date

---

## Règles transversales

### Genesis v0.4.0 API (MANDATORY)
- `scene.rigid_solver.apply_links_external_force()` — **PAS** `scene.solver`
- `scene.rigid_solver.apply_links_external_torque()` — **PAS** `scene.solver`
- `drone.set_mass_shift()`, `drone.set_COM_shift()` — entity-level

### Code (MANDATORY)
- Minimaliste, production-quality, sans redondance
- Type hints + docstrings sur tout le public
- Torch-only, vectorisé, pas de numpy
- UV : `uv add`, `uv run`

### Performance (CRITICAL)
- Pré-allouer les tenseurs dans `__init__()`
- Opérations in-place quand possible
- Pas de Python loops sur n_envs
- Minimiser les appels torch — chaîner les opérations
- `_compute_wrench()` est souvent le bottleneck

---

## Output attendu

À la fin des 3 phases, afficher :

```
## Catégorie N — Implémentation terminée

Perturbations : X/Y implémentées
Tests : N passed, M skipped
Review : K blocking corrigés, J warnings restants

### Overhead mesuré
| Perturbation | Overhead | Status |
|---|---|---|
| ... | ... | ... |

### Perturbations nécessitant optimisation (>100%)
- N.X : +145% — raison identifiée par reviewer C
- ...

→ Lancer `/judge-category N` pour les propositions d'optimisation.
```
