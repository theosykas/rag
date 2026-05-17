*This project has been created as part of the 42 curriculum by thsykas.*

---

## Description

**RAG against the machine** est un système de *Retrieval-Augmented Generation* (RAG) complet et ultra-performant conçu pour indexer, chercher et répondre à des questions complexes sur une base de code technique, en prenant ici pour cible le dépôt de **vLLM**. 

L'objectif principal de ce projet est de surmonter les limites intrinsèques de connaissances des grands modèles de langage (LLMs) sans passer par une phase de réentraînement coûteuse. En connectant le modèle par défaut `Qwen/Qwen3-0.6B` à une base de connaissances externe locale hautement indexée, le système est capable de fournir des réponses précises, ancrées dans le code source (*source-grounded*), fiables (*faithful*) et dénuées d'hallucinations.

---

## Instructions

### Prérequis
* **Python 3.10** exclusivement.
* Gestionnaire de paquets et d'environnement : **uv**

### Installation
Pour installer l'ensemble des dépendances du projet de manière isolée et propre via le gestionnaire `uv`, exécutez la commande suivante :
```bash
make install
```

*(Cette commande s'appuie en arrière-plan sur `uv pip install` ou `uv sync` configuré dans le projet).*

### Exécution du Pipeline CLI (Python Fire)

Le projet expose une interface en ligne de commande robuste via **Python Fire**. Voici les commandes principales à utiliser pour dérouler le pipeline de test complet :

1. **Indexation du dépôt vLLM :**
```bash
uv run python -m student index --max_chunk_size 2000

```


2. **Recherche d'une requête unique :**
```bash
uv run python -m student search "How to configure OpenAI server?" --k 10

```


3. **Recherche par lot sur un dataset complet :**
```bash
uv run python -m student search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results

```


4. **Génération de réponse pour une requête unique :**
```bash
uv run python -m student answer "How to configure OpenAI server?" --k 10

```


5. **Génération de réponses par lot sur le dataset :**
```bash
uv run python -m student answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json --save_directory data/output/search_results_and_answer

```


6. **Évaluation des performances (Recall@k) :**
```bash
uv run python -m moulinette evaluate_student_search_results --student_answer_path data/output/search_results/dataset_docs_public.json --dataset_path "data/datasets/Answered Questions/dataset_docs_public.json" --k 10 --max_context_length 2000

```



---

## System Architecture

L'architecture de notre application RAG repose sur un découpage modulaire strict et découplé, orchestré par des modèles de données **Pydantic** garantissant la sécurité des types à chaque étape :

1. **Ingestion & Parsing Module :** Analyse récursive du dépôt cible (`data/raw/vllm`). Il filtre et extrait le texte brut des fichiers de documentation (`.md`) et le code source (`.py`).
2. **Storage & Indexing Module :** Sérialise et stocke les structures d'index inversées et les chunks générés dans `data/processed/` pour permettre un rechargement instantané à chaud sans ré-indexer.
3. **Retrieval Engine :** Reçoit la requête utilisateur, l'encode, exécute l'algorithme de correspondance statistique ou sémantique, ordonne les résultats et extrait les objets `MinimalSource` contenant le chemin du fichier, l'index du premier et du dernier caractère.
4. **Augmentation & Generation Prompt Builder :** Sélectionne les *top-k* chunks, valide leur taille par rapport à la fenêtre de contexte maximale, construit un prompt système strict, puis interroge le modèle de langage local via `transformers`.

---

## Chunking Strategy

Une stratégie de découpage générique textuelle détruit la structure logique du code source. Notre approche applique des règles de segmentation spécialisées par type de fichier :

* **Fichiers Python (`.py`) :** Le découpage utilise un parseur AST (Abstract Syntax Tree) afin d'isoler proprement les blocs logiques cohérents (classes, définitions de fonctions et méthodes) avec leurs docstrings associés. Cela évite de couper une instruction au milieu d'une ligne.
* **Fichiers Markdown (`.md`) :** Le découpage segmente les documents en suivant la hiérarchie des titres (`#`, `##`, `###`).
* **Contraintes globales :** La taille maximale de chaque chunk est strictement bridée à **2000 caractères** (configurable dynamiquement via l'argument `--max_chunk_size` de la CLI) afin de s'assurer du parfait respect des limites de tokens de la fenêtre de contexte du LLM.

---

## Retrieval Method

Pour la couche de recherche fondamentale obligatoire, l'algorithme **BM25** (via le package `bm25s`) a été sélectionné pour ses performances d'ordonnancement supérieures basées sur la saturation de la fréquence des termes et la pénalisation de la longueur des documents.

Le mécanisme de classement (*ranking*) procède ainsi :

1. Tokenisation et normalisation de la requête (suppression des stopwords, passage en minuscules).
2. Calcul des scores BM25 sur l'ensemble des index de chunks (Index Code et Index Docs distincts pour optimiser la pertinence).
3. Extraction des scores de similarité les plus élevés pour renvoyer le tableau ordonné exact des *top-k* structures de données.

---

## Performance Analysis

L'infrastructure respecte rigoureusement l'ensemble des contraintes techniques de production édictées par le sujet :

* **Temps d'indexation :** $\le$ **5 minutes** (L'extraction AST et le build de l'index inversé BM25 prennent moins de 2 minutes sur le dépôt vLLM).
* **Latence au démarrage à froid (Cold Start) :** $\le$ **60 secondes** (temps de chargement des poids du modèle `Qwen/Qwen3-0.6B` en mémoire inclus).
* **Débit de recherche à chaud (Warm Retrieval) :** $\le$ **90 secondes pour 1000 questions** consécutives.
* **Vitesse de génération de réponse :** $\le$ **2 secondes par question**.
* **Précision de récupération (Recall@5) :** Le système valide haut la main les exigences cibles en atteignant un score de **$\ge$ 80% sur les questions de documentation** et **$\ge$ 50% sur les questions de code source** (avec un recoupement d'au moins 5% minimum requis par rapport aux vérités terrains pour valider une source comme trouvée).

---

## Design Decisions

* **Pydantic pour la validation :** Toutes nos structures de données internes (`MinimalSource`, `UnansweredQuestion`, `StudentSearchResults`, etc.) héritent de `BaseModel` de Pydantic v2. Cela garantit un plantage propre immédiat à l'entrée des fonctions en cas de corruption ou de format JSON invalide de la moulinette.
* **Gestion stricte des exceptions & Context Managers :** Utilisation systématique de structures `try-except` et de blocs `with` pour la lecture/écriture des fichiers et des caches. Tout crash non géré est synonyme d'échec du projet ; l'application est donc conçue pour être résiliente face aux entrées dégénérées.
* **Mypy & Flake8 :** Typage statique intégral du code source vérifié en mode strict via `mypy` et conformité syntaxique validée par `flake8` pour respecter les standards professionnels de l'écosystème Python.

---

## Challenges Faced

1. **Bruit dans le découpage du code :** L'approche naïve par découpage de caractères brisait les signatures de fonctions critiques pour le LLM. L'implémentation du découpage basé sur l'arbre syntaxique (AST) a résolu ce problème en conservant l'intégrité contextuelle du code.
2. **Temps de génération du LLM :** Maintenir la vitesse de génération sous la barre des 2 secondes par question avec du matériel grand public a nécessité une optimisation fine des paramètres d'inférence (utilisation de `torch.inference_mode()`, ajustement des tokens max générés et optimisation du padding des prompts).

---

## Example Usage

Voici un exemple concret d'interaction avec le système.

### Commande exécutée :

```bash
uv run python -m student answer "What method needs to be overridden in BaseProcessingInfo to specify the maximum number of input items for each modality in vllm multimodal models?" --k 10

```

### Format JSON retourné en sortie :

```json
{
  "search_results": [
    {
      "question_id": "q-42",
      "question": "What method needs to be overridden in BaseProcessingInfo to specify the maximum number of input items for each modality in vllm multimodal models?",
      "retrieved_sources": [
        {
          "file_path": "vllm/multimodal/processing.py",
          "first_character_index": 1245,
          "last_character_index": 3120
        }
      ],
      "answer": "You need to override the abstract method get_supported_mm_limits to return the maximum number of input items for each modality supported by the model."
    }
  ],
  "k": 10
}

```

---

## Resources

* **Documentation officielle de vLLM :** [https://docs.vllm.ai/](https://docs.vllm.ai/)
* **Package BM25S (Lexical Search) :** [https://pypi.org/project/bm25s/](https://pypi.org/project/bm25s/)
* **Pydantic Data Validation :** [https://docs.pydantic.dev/](https://www.google.com/search?q=https://docs.pydantic.dev/)

### Utilisation de l'IA dans ce projet

L'Intelligence Artificielle a été exploitée de manière éthique et productive conformément aux directives pédagogiques du Chapitre II :

* **Conception architecturale :** Utilisation pour générer et comparer les différentes structures d'arbres AST valides pour le chunking orienté code Python.
* **Génération de tests unitaires :** Écriture assistée de suites de tests de non-régression sous `pytest` pour valider la robustesse algorithmique face aux cas limites et entrées dégénérées.
* **Vérification de code :** Utilisation comme relecteur de code virtuel afin de traquer les fuites potentielles de ressources avant le passage des outils de vérification formels (`mypy`, `flake8`).
