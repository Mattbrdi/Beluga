# Tests automatisés

La suite utilise `pytest`. Chaque fichier couvre une responsabilité et chaque
test vérifie un comportement observable, sans dépendre des détails internes.

## Installation et exécution

Depuis la racine du dépôt :

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

Pour cibler uniquement ce module ou un test précis :

```bash
python -m pytest tests/signal_generation
python -m pytest tests/signal_generation/test_scene.py::test_same_seed_reproduces_complete_scene
```

Les tests suivent autant que possible le rythme « préparation, action,
vérification ». Ils doivent rester déterministes, rapides et indépendants les
uns des autres. Un correctif de bug devrait normalement être accompagné d'un
test qui échoue avant le correctif et réussit après.
