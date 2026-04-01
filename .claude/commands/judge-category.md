**Analyse et optimise la catégorie** : $ARGUMENTS

---

## Vue d'ensemble

Ce skill exécute les phases 4–5 du pipeline d'implémentation.
**Prérequis** : `/implement-category N` doit avoir été exécuté (phases 1–3 terminées).

```
Phase 4: OPTIMIZE (agents // par perturbation)  →  Phase 5: JUDGE (synthèse → user)
```

---

## Phase 4 — Propositions d'optimisation

### Étape 4.0 — Identifier les cibles

Lire le rapport de Phase 3 (output de `/implement-category`) ou le reconstruire :

1. Lire `docs/impl/category_N_*.md` — sections "Performance overhead vs n_envs"
2. Lire les résultats du test P6 : `uv run pytest tests/integration/test_overhead_genesis.py -v -s -m genesis -k "category_N"` (ou lire les derniers résultats en mémoire)

Classer chaque perturbation :

| Zone | Seuil | Action |
|---|---|---|
| ✅ PASS | < 75% | Aucune action |
| 👀 WATCH | 75–100% | Mentionner si quick win évident |
| ⚠️ WARN | 100–200% | Proposer 2-3 optimisations |
| ❌ FAIL | > 200% | Optimisation **obligatoire** |

### Étape 4.1 — Agents d'optimisation (1 par perturbation ⚠️/❌)

Pour chaque perturbation au-dessus de 100% (ou 75% si quick win évident), lancer un agent d'optimisation.

Prompt de l'agent :

```
Tu es un expert en optimisation de code PyTorch pour des simulations physiques temps-réel.

## Contexte
- Simulateur : Genesis (GPU-batched, n_envs environnements parallèles)
- Contrainte : l'overhead de la perturbation vs scene.step() nu doit être < 100% (WARNING) ou < 200% (FAIL)
- La perturbation <ID> a actuellement un overhead de <X%> à n_envs=16

## Code à optimiser
[code source complet de la perturbation]

## Base class
[extrait pertinent de base.py — tick(), apply(), sample()]

## Instructions

Propose exactement 3 optimisations, classées par rapport gain/effort :

Pour chaque optimisation :
1. **Nom** : titre court (ex: "Pré-allocation du tenseur wrench")
2. **Description** : quoi changer, où dans le code (lignes précises)
3. **Gain estimé** : pourcentage d'overhead en moins (ex: "-40%"), avec justification
4. **Effort** : nombre de lignes à modifier, risque de régression (low/medium/high)
5. **Trade-off** : impact sur lisibilité/maintenabilité
6. **Code** : diff minimal montrant le changement

Classement :
- **Recommandé** : gain > 30%, effort faible, risque low
- **Optionnel** : gain 10-30%, ou effort/risque modéré
- **Non recommandé** : gain < 10%, ou risque high, ou lisibilité trop dégradée

Règles :
- Pas de changement d'algorithme (la formule physique ne change pas)
- Pas de perte de précision numérique
- Le code doit rester lisible (pas de micro-optimisations illisibles)
- Torch-only, vectorisé, pas de numpy
```

### Étape 4.2 — Collecter les propositions

Pour chaque perturbation, collecter les 3 propositions et leur classification.

---

## Phase 5 — Rapport de jugement (présenté à l'utilisateur)

### Format du rapport

```markdown
# Catégorie N — Rapport d'optimisation

## Résumé exécutif

| Perturbation | Overhead actuel | Zone | # Optims proposées | Gain max estimé |
|---|---|---|---|---|
| N.1 Xxx | +3% | ✅ PASS | 0 | — |
| N.5 Yyy | +87% | 👀 WATCH | 1 (quick win) | -30% → ~61% |
| N.8 Zzz | +145% | ⚠️ WARN | 3 | -80% → ~65% |
| N.11 Www | +230% | ❌ FAIL | 3 | -120% → ~110% |

## Détail par perturbation problématique

### N.8 Zzz — overhead actuel +145% ⚠️

**Cause racine** : [explication concise — ex: "2 torch.zeros() alloués à chaque step dans _compute_wrench()"]

#### Optimisation 1 — Pré-allocation wrench buffer ✅ Recommandé
- **Gain estimé** : -60% (145% → ~85%)
- **Effort** : 5 lignes modifiées, risque low
- **Trade-off** : aucun impact sur lisibilité
- **Diff** :
  ```diff
  - wrench = torch.zeros(self.n_envs, 3)
  + wrench = self._wrench_buffer  # pré-alloué dans __init__
  + wrench.zero_()
  ```

#### Optimisation 2 — In-place clamp ⚡ Optionnel
- **Gain estimé** : -15% (85% → ~70%)
- **Effort** : 3 lignes, risque low
- ...

#### Optimisation 3 — Fusion d'opérations 🚫 Non recommandé
- **Gain estimé** : -5%
- **Effort** : 8 lignes, risque medium (lisibilité dégradée)
- ...

### N.11 Www — overhead actuel +230% ❌ OBLIGATOIRE

[même format]

---

## Décisions requises

Pour chaque perturbation, choisis les optimisations à appliquer :

1. **N.5 Yyy** : appliquer le quick win ? (gain -30%, 3 lignes)
   → Mon avis : **oui**, effort minimal, passe sous 75%

2. **N.8 Zzz** : appliquer optim 1 + 2 ? (gain total -75%, 8 lignes)
   → Mon avis : **oui pour optim 1** (gros gain, faible effort), **optionnel pour optim 2**

3. **N.11 Www** : appliquer optim 1 + 2 + 3 ? (gain total -120%, 15 lignes)
   → Mon avis : **optim 1+2 obligatoires** (FAIL sinon), **optim 3 non recommandé**

Réponds par perturbation : "appliquer 1", "appliquer 1+2", "skip", etc.
```

### Après décision de l'utilisateur

1. Appliquer les optimisations validées
2. Relancer les tests : `uv run pytest tests/category_N_*/ -v`
3. Relancer le test P6 : `uv run pytest tests/integration/test_overhead_genesis.py -v -s -m genesis`
4. Mettre à jour les docs + plots avec les nouvelles valeurs
5. Mettre à jour `ROADMAP.md` et `MEMORY.md`

---

## Règles

### Seuils
- **< 75%** : aucune action nécessaire
- **75–100%** : mentionner si quick win évident (< 5 lignes, gain > 20%)
- **100–200%** : proposer 2-3 optimisations, recommander les meilleures
- **> 200%** : optimisation obligatoire, au moins 1 doit ramener sous 200%

### Principes d'optimisation
- La formule physique ne change JAMAIS (c'est la spec)
- Pas de perte de précision numérique
- Le code reste lisible — pas de micro-optimisations illisibles
- Torch-only, vectorisé, pas de numpy
- Les tests existants doivent toujours passer après optimisation

### Ce qui est hors scope
- Changer l'architecture (base classes, héritage)
- Modifier des perturbations qui ne sont pas dans la catégorie demandée
- Changer les seuils P6
