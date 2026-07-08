"""Feature-extraction strategies for the e-commerce text mining pipeline.

Reused verbatim from the Corte-1/Bitácora-02 lineage. This module follows the
*Strategy* pattern: each NLP technique requested by the activity (TF-IDF keyword
extraction, TextBlob sentiment polarity and spaCy named-entity recognition) is
encapsulated behind the common ``FeatureExtractor`` interface. The *Template
Method* pattern (``run``) takes care of timing every strategy uniformly, which
is what the notebook needs to build the "Salida" metrics table.

Keeping these classes in a standalone module (instead of defining them in the
notebook) also makes them picklable, which is required to dispatch each strategy
to its own worker process with ``concurrent.futures.ProcessPoolExecutor`` -- the
first CPU-parallel stage of the Bitácora-04 pipeline.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob


@dataclass(frozen=True)
class ExtractionResult:
    """Output of a single feature-extraction strategy."""

    name: str
    values: list
    elapsed_seconds: float


class FeatureExtractor(ABC):
    """Common interface for every feature-extraction strategy."""

    #: Name of the resulting column / metric, set by subclasses.
    name: str

    @abstractmethod
    def extract(self, texts: Sequence[str]) -> list:
        """Compute one feature value per input text."""

    def run(self, texts: Sequence[str]) -> ExtractionResult:
        """Time ``extract`` and wrap the output in an :class:`ExtractionResult`."""
        start = time.perf_counter()
        values = self.extract(texts)
        elapsed = time.perf_counter() - start
        return ExtractionResult(name=self.name, values=values, elapsed_seconds=elapsed)


class KeywordExtractor(FeatureExtractor):
    """Top-k TF-IDF keywords per document (sklearn ``TfidfVectorizer``).

    TF-IDF rewards terms that are frequent in a document but rare across the
    corpus, which is exactly the "how often / how rare" criterion requested
    by the activity.
    """

    name = "keywords"

    def __init__(self, top_k: int = 5, max_features: int = 5000):
        self.top_k = top_k
        self.max_features = max_features

    def extract(self, texts: Sequence[str]) -> list:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=self.max_features)
        matrix = vectorizer.fit_transform(texts)
        vocabulary = vectorizer.get_feature_names_out()

        keywords = []
        for row in matrix:
            row = row.tocoo()
            top = sorted(zip(row.col, row.data), key=lambda pair: pair[1], reverse=True)[: self.top_k]
            terms = [vocabulary[col] for col, _ in top]
            keywords.append(", ".join(terms))
        return keywords


class SentimentExtractor(FeatureExtractor):
    """Sentiment polarity per document, in the range -1 (negative) to +1 (positive)."""

    name = "sentiment_polarity"

    def extract(self, texts: Sequence[str]) -> list:
        return [TextBlob(text).sentiment.polarity for text in texts]


class EntityExtractor(FeatureExtractor):
    """Named entities per document (spaCy NER): brands, products, dates, organizations, etc.

    ``max_chars`` truncates each description before running NER. Product
    descriptions in this dataset are long (mean ~714 chars, max >50k) but the
    identifying entities (brand, model, size) are concentrated in the first
    sentence; the remaining text is mostly marketing boilerplate. Truncating
    keeps the relevant entities while cutting spaCy's runtime roughly 2.5x on
    this corpus -- a deliberate performance/quality trade-off for the CAD
    context. Set ``max_chars=None`` to process the full text.
    """

    name = "entities"

    def __init__(self, model: str = "en_core_web_sm", batch_size: int = 256, max_chars: int | None = 300):
        self.model = model
        self.batch_size = batch_size
        self.max_chars = max_chars

    def extract(self, texts: Sequence[str]) -> list:
        nlp = spacy.load(self.model, disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])
        inputs = texts if self.max_chars is None else [text[: self.max_chars] for text in texts]
        return [
            json.dumps([ent.text for ent in doc.ents], ensure_ascii=False)
            for doc in nlp.pipe(inputs, batch_size=self.batch_size)
        ]


def run_extractor(extractor: FeatureExtractor, texts: Sequence[str]) -> ExtractionResult:
    """Module-level dispatch target for ``ProcessPoolExecutor.submit``.

    Bound methods of objects defined inside a notebook cannot always be
    pickled; a plain module-level function avoids that pitfall.
    """
    return extractor.run(texts)


#: Canonical ordering of the three extractors, matching the activity diagram.
DEFAULT_EXTRACTORS = [
    KeywordExtractor(),
    SentimentExtractor(),
    EntityExtractor(),
]
