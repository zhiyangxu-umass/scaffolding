import codecs
from collections import defaultdict
import os
import logging
from typing import Dict, List, Optional, Tuple
import pickle

from overrides import overrides
import tqdm

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, ListField, IndexField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("consist_constit")
class ConsistConstitReader(DatasetReader):
    def __init__(self, max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 label_namespace: str = "labels",
                 parent_label_namespace: str = "parent_labels") -> None:
        self.max_span_width = max_span_width
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}
        self.label_namespace = label_namespace
        self.parent_label_namespace = parent_label_namespace

    def _process_sentence(self,
                          sentence_tokens: List[str],
                          predicate_index: int,
                          constits: Dict[Tuple[int, int], str],
                          parents: Dict[Tuple[int, int], str]) -> Instance:
        """
        Parameters
        ----------
        sentence_tokens : ``List[str]``, required.
            The tokenised sentence.
        predicate_index : ``int``, required.
            Index of the last predicate in the sentence.
        constits : ``Dict[Tuple[int, int], str]]``, required.

        Returns
        -------
        An instance.
        """
        def construct_matrix(labels: Dict[Tuple[int, int], str]) -> List[List[str]]:
            default = "*"

            def get_new_label(original: str, newer: str):
                return newer if original == default else "{}|{}".format(newer, original)

            constit_matrix = [[default for _ in range(self.max_span_width)]
                              for _ in sentence_tokens]
            for span in labels:
                start, end = span
                diff = end - start

                # Ignore the constituents longer than given maximum span width.
                if diff >= self.max_span_width:
                    continue
                # while diff >= self.max_span_width:
                #     old_label = constit_matrix[end][self.max_span_width - 1]
                #     constit_matrix[end][self.max_span_width -
                #                         1] = get_new_label(old_label, constits[span])
                #     end = end - self.max_span_width
                #     diff = end - start
                constit_matrix[end][diff] = get_new_label(
                    constit_matrix[end][diff], labels[span])
            return constit_matrix

        predicates = [0 for _ in sentence_tokens]
        predicates[predicate_index] = 1
        return self.text_to_instance(sentence_tokens,
                                     predicates,
                                     predicate_index,
                                     construct_matrix(constits),
                                     construct_matrix(parents))

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache

        instances = []

        sentence: List[str] = []
        open_constits: List[Tuple[str, int]] = []
        constits: Dict[Tuple[int, int], str] = {}
        parent_constits: Dict[Tuple[int, int], str] = {}
        predicate_index = None  # index of last predicate in the sentence.

        logger.info(
            "Reading constit instances from dataset files at: %s", file_path)
        
        input_file = pickle.load( open( file_path, "rb" ) )

        for line in tqdm.tqdm(input_file):
            instances.append(
                self._process_sentence(line["sentence"],
                                        line["predicate_index"],
                                        line["constits"],
                                        line["parent_constits"]))
                            

        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))
        logger.info("# instances = %d", len(instances))
        return Dataset(instances)


    def text_to_instance(self,  # type: ignore
                         sentence_tokens: List[str],
                         predicates: List[int],
                         predicate_index: int,
                         constits: List[List[str]] = None,
                         parents: List[List[str]] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ
        text_field = TextField(
            [Token(t) for t in sentence_tokens], token_indexers=self._token_indexers)
        verb_field = SequenceLabelField(predicates, text_field)
        predicate_field = IndexField(predicate_index, text_field)

        # Span-based output fields.
        span_starts: List[Field] = []
        span_ends: List[Field] = []
        span_mask: List[int] = [1 for _ in range(
            len(sentence_tokens) * self.max_span_width)]
        span_labels: Optional[List[str]] = [
        ] if constits is not None else None
        parent_labels: Optional[List[str]] = [
        ] if parents is not None else None

        for j in range(len(sentence_tokens)):
            for diff in range(self.max_span_width):
                width = diff
                if j - diff < 0:
                    # This is an invalid span.
                    span_mask[j * self.max_span_width + diff] = 0
                    width = j

                span_starts.append(IndexField(j - width, text_field))
                span_ends.append(IndexField(j, text_field))

                if constits is not None:
                    label = constits[j][diff]
                    span_labels.append(label)

                if parents is not None:
                    parent_labels.append(parents[j][diff])

        start_fields = ListField(span_starts)
        end_fields = ListField(span_ends)
        span_mask_fields = SequenceLabelField(span_mask, start_fields)

        fields: Dict[str, Field] = {"tokens": text_field,
                                    "targets": verb_field,
                                    "span_starts": start_fields,
                                    "span_ends": end_fields,
                                    "span_mask": span_mask_fields,
                                    "target_index": predicate_field}

        if constits:
            fields['tags'] = SequenceLabelField(span_labels,
                                                start_fields,
                                                label_namespace=self.label_namespace)
            fields['parent_tags'] = SequenceLabelField(parent_labels,
                                                       start_fields,
                                                       label_namespace=self.parent_label_namespace)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'PhraseSyntaxReader':
        max_span_width = params.pop("max_span_width")
        token_indexers = TokenIndexer.dict_from_params(
            params.pop('token_indexers', {}))
        label_namespace = params.pop("label_namespace", "labels")
        parent_label_namespace = params.pop(
            "parent_label_namespace", "parent_labels")
        params.assert_empty(cls.__name__)
        return ConsistConstitReader(max_span_width=max_span_width,
                                      token_indexers=token_indexers,
                                      label_namespace=label_namespace,
                                      parent_label_namespace=parent_label_namespace)
