# Publier `morseq` sur github.com/KvnFltr

Tout est prêt et personnalisé à ton compte : URLs, auteur, licence, CI.
Il ne reste que l'authentification, que je ne peux pas faire à ta place.

## Étape 0 — supprimer le dossier `.git` cassé

J'ai tenté de créer le dépôt depuis mon environnement, mais OneDrive
verrouille les fichiers et git ne peut pas travailler à travers ce montage
(échecs `Operation not permitted`). Un dossier `.git` incomplet est resté.
**Supprime-le d'abord** — dans PowerShell :

```powershell
cd "$env:USERPROFILE\OneDrive - ESIEE Paris\Documents\ESIEE\Tremplin Recherche\Bibliographie\morseq"
Remove-Item -Recurse -Force .git
```

Depuis Windows, git fonctionne normalement dans un dossier OneDrive : le
problème ne se reproduira pas.

## Voie A — avec GitHub CLI (le plus simple, tout en une fois)

Si tu as `gh` (sinon : https://cli.github.com, ou `winget install GitHub.cli`) :

```powershell
gh auth login          # GitHub.com, HTTPS, authentification dans le navigateur
git init
git add .
git commit -m "morseq 0.1.0: Morse sequences and topological image segmentation"
git branch -M main
gh repo create morseq --public --source=. --remote=origin --push
```

`gh repo create` crée le dépôt **et** pousse. Rien d'autre à faire.

## Voie B — sans GitHub CLI

1. Créer le dépôt sur https://github.com/new
   nom `morseq`, description
   « Discrete Morse sequences and topological image segmentation »,
   **public**, et surtout **ne coche rien** (pas de README, pas de .gitignore,
   pas de licence : ils sont déjà dans le dossier).

2. Dans PowerShell, depuis le dossier `morseq` :

```powershell
git init
git add .
git commit -m "morseq 0.1.0: Morse sequences and topological image segmentation"
git branch -M main
git remote add origin https://github.com/KvnFltr/morseq.git
git push -u origin main
```

Si git demande un mot de passe : GitHub n'accepte plus celui du compte. Crée
un jeton sur https://github.com/settings/tokens/new (scope `repo`) et colle-le
à la place du mot de passe. Windows le mémorisera ensuite.

## Après le premier push

- **CI** : onglet *Actions*. Les 623 tests tournent sur Python 3.9, 3.11 et
  3.12. Une fois au vert, ajouter le badge en tête du README :
  `![tests](https://github.com/KvnFltr/morseq/actions/workflows/tests.yml/badge.svg)`

- **DOI citable** (utile pour le rapport) : connecter le dépôt à
  https://zenodo.org (Settings → GitHub → activer `morseq`), puis publier une
  release `v0.1.0`. Zenodo génère un DOI permanent, citable en note de bas de
  page de la Partie II.

- **Le rapport n'est pas dans le dépôt** et ne doit pas y être : le dossier ne
  contient que la bibliothèque (aucun `.tex`, aucun `.pdf`).

## Vérifier en local avant de pousser

```powershell
pip install -e ".[dev]"
pytest -q                  # 623 tests
morseq check --shape 5x4   # les énoncés de la Partie II, sur champs aléatoires
python examples\quickstart.py
```

## Contenu du dépôt

```
morseq/complex.py       complexes cubiques (toute dimension) et simpliciaux
morseq/sequence.py      séquences de Morse, champs, acyclicité, validation
morseq/maps.py          référence/coréférence, complexes critique et
                        cocritique, extensions, comptage exact des chemins
morseq/segmentation.py  ensembles stables/instables, partition des sommets,
                        hermétisme, bassins, ponts, squelette
morseq/stacks.py        stacks, coupes, F-séquences, lower stars (RWS),
                        segmentation d'images 2D et 3D
morseq/plotting.py      figures (matplotlib, optionnel)
morseq/cli.py           `morseq segment` et `morseq check`
tests/test_theorems.py  chaque énoncé de la Partie II, exécutable
```

Les noms des tests suivent les résultats du rapport :
`test_b5_equivalence_with_dfrs` (équivalence avec les ensembles récursifs de
DFRS), `test_theorem_12_parity` (parité par comptage exact des chemins),
`test_b6_partition`, `test_b8_restriction`,
`test_b11_closure_is_order_independent`, `test_b11_algorithm_equals_closure`,
`test_b11_properties`.
