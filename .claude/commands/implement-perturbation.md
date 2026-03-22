**Implémente** la perturbation suivante : $ARGUMENTS

---

## Contexte obligatoire à lire AVANT d'écrire la moindre ligne de code

Lis ces fichiers dans cet ordre exact, sans sauter aucun :

1. `ROADMAP.md` — état actuel du projet, milestone en cours
2. `docs/01_perturbations_catalog.md` — trouve l'entrée exacte de la perturbation demandée (tous ses champs)
3. `docs/02_class_design.md` — classe parente à utiliser, signature exacte des méthodes
4. `docs/05_test_conventions.md` — contrat de test U1–U11, I1–I5, P1–P6
5. `src/genesis_robust_rl/perturbations/base.py` — base classes déjà implémentées
6. `src/genesis_robust_rl/perturbations/category_1_physics.py` — exemple de référence (MassShift)
7. `tests/category_1_physics/` — exemple de référence pour les tests et la structure

Si la perturbation est de catégorie 4 (sensor), lis aussi :
- `docs/00b_sensor_models.md`

---

## Étape 0 — Audit des ambiguïtés (OBLIGATOIRE — ne jamais sauter)

Avant d'écrire la moindre ligne de code, relis l'entrée du catalogue pour la perturbation demandée et vérifie chacun des points suivants.

**Si un seul point est ambigu ou incomplet, STOP — pose les questions à l'utilisateur et attends ses réponses avant de continuer.**

### Checklist d'ambiguïtés à vérifier

#### 1. Loi de probabilité
- La distribution est-elle **complètement spécifiée** (famille + tous ses paramètres + bounds) ?
- Si `distribution` liste plusieurs familles (ex. `uniform, gaussian`) sans préciser laquelle utiliser par défaut, demander.
- Si `bounds` est exprimé de façon relative (ex. `[0.5×KF_nom, 1.5×KF_nom]`) sans définir `KF_nom`, demander quelle valeur nominale utiliser.
- Si le paramètre est vectoriel (`vector(4)`) : est-ce que chaque composante est tirée **indépendamment** ou avec une corrélation ? Si non précisé et non trivial, demander.

#### 2. Loi de dynamique (si `stateful: yes`)
- L'équation de transition d'état est-elle **complètement définie** (tous les symboles explicitement définis) ?
- Quelle est la **condition initiale** à `reset()` ? (ex. : `rpm_actual = rpm_cmd` ou `rpm_actual = 0` ?)
- Le paramètre de **continuité de Lipschitz** (`lipschitz_k`) est-il numérique et justifié, ou juste marqué "implicit" ? Si "implicit", demander la valeur cible.
- Pour les perturbations avec un état qui évolue en fonction du temps (wear, lag, phase) : quelle est l'unité de `dt` attendue en entrée de `step()` ?

#### 3. Formule d'application (apply)
- La formule qui transforme RPM → RPM_effectif (ou force/torque) est-elle **sans ambiguïté** ?
- Si le catalogue dit "external torque" ou "sinusoidal force" sans préciser le **repère** (world / body / local link), demander.
- Si des paramètres physiques externes sont requis (Ke, R, I_rotor, RPM_max…) : ont-ils une valeur nominale définie quelque part ? Si non, demander.

#### 4. Couplage inter-couches (si applicable)
- Si la perturbation a des effets sur **plusieurs hooks** (ex. `pre_physics` + `on_observation`), chaque effet est-il formellement défini ?
- Pour 2.10 (imbalance) : la fréquence de la sinusoïde est-elle `ω_rotor(t)` (dynamique) ou fixe ? Le "correlated IMU noise" est-il quantifié ?

#### 5. Comportement en mode ADVERSARIAL
- Est-il raisonnable de laisser l'adversaire agir sur ce paramètre sans contrainte supplémentaire ?
- Si la perturbation est stateful, est-ce que `step()` doit quand même avancer l'état en mode adversarial ?

### Format des questions

Si des points sont ambigus, présente-les ainsi :

```
Avant d'implémenter <Perturbation X.Y>, j'ai besoin de clarifications :

1. [Distribution] — <question précise>
2. [Lipschitz] — <question précise>
3. [Formule] — <question précise>
...

Je n'implémenterai qu'après ta réponse.
```

---

## Cycle d'implémentation obligatoire (dans cet ordre)

### Étape 1 — Code

Fichier : `src/genesis_robust_rl/perturbations/category_N_*.py`

- Décore la classe avec `@register("id_catalog")`
- Hérite de la bonne classe parente (voir `02_class_design.md`)
- Constructeur : tous les champs du catalog comme kwargs avec valeurs par défaut
- Si stateful : `self.is_stateful = True`, implémenter `reset()` et `step()`
- Si stateful avec OUProcess : utiliser `OUProcess` de `base.py`, torch-only dans `step()`, jamais numpy
- Si DelayBuffer : utiliser `DelayBuffer` de `base.py`, buffer circulaire `Tensor[n_envs, max_delay, dim]`
- `apply()` : signature exacte selon la classe parente
- Docstring obligatoire sur la classe et toutes les méthodes publiques
- Type hints sur toutes les signatures

### Étape 2 — Tests

Répertoire : `tests/category_N_*/`

Structure :
```
tests/category_N_*/
  __init__.py          (vide)
  conftest.py          (fixture `perturbation` + `perturbation_class`)
  test_*.py            (U1–U11 + I1–I2/I3 selon le type)
  test_perf_*.py       (P1 CPU : tick + apply, n_envs=1 et 512)
```

**conftest.py** :
```python
@pytest.fixture(params=[lambda n, ...: MaClasse(...)])
def perturbation(request, n_envs, mock_scene): ...

@pytest.fixture(params=[MaClasse])  # pour P3
def perturbation_class(request): ...
```

**Tests obligatoires** (tous dans `test_*.py`) :
- U1 : sample() dans les bounds (1000 draws)
- U2 : curriculum_scale = 0.0 → nominal ; = 1.0 → variance non nulle
- U3 : tick(is_reset=True) → _current_value set si per_episode
- U4 : tick(is_reset=False) → sample() appelé si per_step
- U5 : set_value() Lipschitz (skip si lipschitz_k=None)
- U6 : update_params() Lipschitz (skip si lipschitz_k=None)
- U7 : get_privileged_obs() retourne valeur si observable=True, None sinon
- U8 : state change après steps, reset(env_ids) ne crash pas (skip si stateless)
- U9 : shape (n_envs, *dimension) ou (1, *dimension) si global, dtype float32
- U10 : reset partiel — env 0 reset, env 1+ inchangés (skip si stateless/global/physics)
- U11 : mode=ADVERSARIAL → tick(is_reset=False) n'appelle pas sample()

**Tests d'intégration** (dans `test_*.py`) :
- I1 : apply() retourne None (PhysicsPerturbation) ou tensor valide
- I2 : setter_fn appelé avec shape correcte (GenesisSetterPerturbation uniquement)
- I3 : solver.apply_links_external_force/torque appelé (ExternalWrenchPerturbation uniquement)

**Tests perf CPU** (dans `test_perf_*.py`) :
```python
WARMUP = 200
STEPS = 2000
MAX_TICK_MS = 0.1    # tick() budget
MAX_APPLY_MS = 0.05  # apply() budget

@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_tick_overhead_cpu(n_envs): ...

@pytest.mark.perf
@pytest.mark.parametrize("n_envs", [1, 512])
def test_apply_overhead_cpu(n_envs): ...
```
Le test **doit crasher** si le seuil est dépassé.

### Étape 3 — Validation tests

```bash
uv run pytest tests/category_N_*/ -v
```

**Ne passe à l'étape 4 que si tous les tests passent (ou sont skippés légitimement).**

Skips légitimes :
- U5/U6 si `lipschitz_k=None`
- U8/U10 si `is_stateful=False`
- U10 si `scope="global"` ou `isinstance(perturbation, PhysicsPerturbation)`
- I2 si pas GenesisSetterPerturbation
- I3 si pas ExternalWrenchPerturbation

### Étape 4 — Documentation

Fichier : `docs/impl/category_N_*.md`

Structure obligatoire par perturbation :

```markdown
# <Titre descriptif — JAMAIS "Category N — ...")

## <ID> <Nom de la perturbation>

### Description
[1-3 phrases, ce que fait la perturbation physiquement]

### Formal definition
[Équation LaTeX avec tous les symboles définis]

### Parameters
[Tableau : Parameter | Type | Default | Description]

### Catalog reference
[Tableau : id, catalog_ref, stateful, application_hook, implementation, feasibility_ref, risk, priority]

### Curriculum scale effect
[Description + référence image]
![curriculum](assets/catN_<id>_curriculum.png)

### Per-env diversity
[Description + référence image]
![per-env](assets/catN_<id>_per_env.png)

### Performance overhead vs n_envs
[Tableau : n_envs | baseline (µs) | perturbed (µs) | overhead (%)]
![perf](assets/catN_<id>_perf.png)
> Overhead relatif diminue avec n_envs. Regenerate: `uv run python docs/impl/plot_category_N.py`

### Usage
[Code exemple minimal]

### Genesis API note (si applicable)
[Note sur l'API Genesis utilisée]
```

### Étape 5 — Graphes Plotly

Fichier : `docs/impl/plot_category_N.py`

**3 graphes obligatoires par perturbation** (PNG via `pio.write_image`, kaleido) :

1. **Curriculum effect** (`catN_<id>_curriculum.png`) :
   - Histogramme des valeurs sampléees pour curriculum_scale ∈ {0.0, 0.5, 1.0}
   - Montre la compression vers le nominal

2. **Per-env diversity** (`catN_<id>_per_env.png`) :
   - 8 envs × 30 épisodes, valeurs sampléees par épisode
   - Montre la diversité inter-env

3. **Perf overhead** (`catN_<id>_perf.png`) :
   - **Overhead relatif (%)** vs n_envs ∈ [1, 4, 16, 64, 128]
   - Baseline = `scene.step()` seul ; perturbed = `tick() + apply() + scene.step()`
   - Overhead = `(t_perturbed - t_baseline) / t_baseline × 100`
   - Montre que l'overhead diminue avec n_envs (le step() devient plus lourd)
   - Axe x en log, axe y en %

Assets dans : `docs/impl/assets/`

> **Lancer le script immédiatement** : `uv run python docs/impl/plot_category_N.py`
> Si erreur Chromium : `uv run python -c "from kaleido import get_chrome_sync; get_chrome_sync()"` puis relancer.
> Ne jamais déléguer à l'utilisateur — kaleido fonctionne directement dans cet environnement.

---

## Performance overhead (CRITICAL — lire avant d'écrire le moindre code)

Chaque perturbation ajoute un overhead à `scene.step()`. Cet overhead est **le critère de qualité principal** de l'implémentation. Il doit être **minimisé activement**, pas subi passivement.

### Faits mesurés (CPU, Crazyflie CF2X)
- `scene.step()` baseline : ~170µs (n_envs=16)
- Une perturbation simple (MassShift) ajoute ~80µs (+47%)
- Une perturbation complexe (GroundEffect) ajoute ~130µs (+73%)
- **L'overhead relatif diminue avec n_envs** car le step() devient plus lourd

### Ce que tu DOIS faire pour minimiser l'overhead

1. **Pré-allouer les tenseurs** dans `__init__()` — jamais `torch.empty()` ou `torch.zeros()` dans `tick()`/`apply()`/`_compute_wrench()`. Utiliser des buffers internes réutilisés à chaque step.
2. **Opérations in-place** — `.mul_()`, `.clamp_()`, `.add_()` au lieu de `*`, `.clamp()`, `+` quand possible (évite les allocations).
3. **Pas de Python loops** sur la dimension n_envs — tout vectorisé torch.
4. **Pas de numpy** — jamais, même pour un calcul simple.
5. **Minimiser les appels torch** — chaîner les opérations. Ex: `torch.empty(n).uniform_(lo, hi).mul_(scale).clamp_(lo, hi)` en une seule chaîne plutôt que 3 statements.
6. **`_compute_wrench()`** — c'est souvent le bottleneck. Profiler si l'overhead dépasse 100µs.

### Mesure obligatoire

Après implémentation, mesure l'overhead de ta perturbation avec le script suivant et reporte les résultats dans la doc :

```python
# Dans test_perf_*.py ou en standalone
# Compare: scene.step() seul vs tick()+apply()+scene.step()
# Mesure pour n_envs in [1, 4, 16, 64, 128]
# Reporte l'overhead relatif (%) pour chaque n_envs
```

Le **3ème graphe** (perf) doit montrer la courbe d'overhead relatif (%) vs n_envs, pas juste le temps absolu.

## Genesis v0.4.0 API (MANDATORY)

- `scene.rigid_solver.apply_links_external_force()` — **PAS** `scene.solver`
- `scene.rigid_solver.apply_links_external_torque()` — **PAS** `scene.solver`
- `drone.set_mass_shift()`, `drone.set_COM_shift()` — entity-level
- `ExternalWrenchPerturbation` a un champ `wrench_type: "force"|"torque"` — dispatch dans `apply()`

## Règles de code (MANDATORY)

- Minimaliste, production-quality, sans redondance
- Type hints sur toutes les signatures
- Pas de type dans les noms de variables
- Docstring sur toutes les classes et méthodes publiques
- Torch-only dans les ops numériques — jamais numpy dans step()
- Tout vectorisé sur la dim 0 (n_envs) — pas de Python loops sur les envs
- UV pour les dépendances : `uv add`, `uv run`

## Règle de fin

Quand tout est fait :
1. Lance `uv run pytest tests/category_N_*/` et vérifie que tout passe
2. Lance `uv run python docs/impl/plot_category_N.py` pour générer les graphes
3. **Mets à jour `ROADMAP.md`** :
   - Coche la perturbation implémentée dans la liste (ou mets `[x]` si catégorie complète)
   - Met à jour le compte dans "Current milestone" : `X.Y done (N passed, M skipped)`
4. **Mets à jour `MEMORY.md`** : bloc "Current state" — statut Cat N, count, date
5. Dis : "Perturbation X.Y implémentée — N passed, M skipped."
