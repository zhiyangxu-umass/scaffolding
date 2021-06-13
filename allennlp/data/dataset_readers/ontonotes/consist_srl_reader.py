import codecs
import os
import logging
from collections import defaultdict
from typing import Dict, List, Optional
import pickle

import tqdm
from overrides import overrides

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset import Dataset
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, ListField, SequenceLabelField, IndexField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("consist_srl")
class ConsistSrlReader(DatasetReader):

    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        self.max_span_width = max_span_width
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}


    def _convert_bio_into_matrix(self, tag_sequence: List[str]) -> List[List[str]]:
        def remove_bio(tag):
            return tag[2:] if tag.startswith('B-') or tag.startswith('I-') else tag
        #self.calculate_span_size(tag_sequence)

        spans = [["*" for _ in range(self.max_span_width)]
                 for _ in range(len(tag_sequence))]

        start_span = 0
        current_tag = tag_sequence[0]
        for pos, tag in enumerate(tag_sequence[1:], 1):
            width = pos - start_span
            if tag.startswith("B-") or (tag == "O" and tag_sequence[pos - 1] != "O"):
                width = pos - 1 - start_span
                spans[pos - 1][width] = remove_bio(current_tag)
                start_span = pos
                current_tag = tag
                width = pos - start_span
            elif width == self.max_span_width - 1:  # maximum allowed width
                spans[pos][width] = remove_bio(current_tag)
                start_span = pos + 1
                if pos + 1 < len(tag_sequence):
                    current_tag = tag_sequence[pos + 1]
        spans[len(tag_sequence)-1][len(tag_sequence)-1-start_span] = remove_bio(tag_sequence[-1])
        return spans

    def _process_sentence(self,
                          sentence_tokens: List[str],
                          verbal_predicates: List[int],
                          predicate_argument_labels: List[List[str]]) -> List[Instance]:
        """
        Parameters
        ----------
        sentence_tokens : ``List[str]``, required.
            The tokenised sentence.
        verbal_predicates : ``List[int]``, required.
            The indexes of the verbal predicates in the
            sentence which have an associated annotation.
        predicate_argument_labels : ``List[List[str]]``, required.
            A list of predicate argument labels, one for each verbal_predicate. The
            internal lists are of length: len(sentence).

        Returns
        -------
        A list of Instances.

        """
        tokens = [Token(t) for t in sentence_tokens]
        if not verbal_predicates:
            # Sentence contains no predicates.
            tags = ["O" for _ in sentence_tokens]
            verb_label = [0 for _ in sentence_tokens]
            spans = self._convert_bio_into_matrix(tags)
            dummy_verb_index = 0
            return [self.text_to_instance(tokens, verb_label, dummy_verb_index, tags, spans)]
        else:
            instances = []

            for verb_index, tags in zip(verbal_predicates, predicate_argument_labels):
                verb_label = [0 for _ in sentence_tokens]
                verb_label[verb_index] = 1
                spans = self._convert_bio_into_matrix(tags)
                instances.append(self.text_to_instance(
                    tokens, verb_label, verb_index, tags, spans))

            return instances

    @overrides
    def read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        # file_path = cached_path(file_path)

        instances = []

        sentence: List[str] = []
        verbal_predicates: List[int] = []
        predicate_argument_labels: List[List[str]] = []
        current_span_label: List[Optional[str]] = []

        logger.info(
            "Reading SRL instances from dataset files at: %s", file_path)
        
        input_file = pickle.load( open( "file_path", "rb" ) )

        for line in input_file:
            instances.extend(self._process_sentence(line["sentence"],
                                                    line["verbal_predicates"],
                                                    line["predicate_argument_labels"]))
                
        if not instances:
            raise ConfigurationError("No instances were read from the given filepath {}. "
                                     "Is the path correct?".format(file_path))

        logger.info("# instances = %d", len(instances))
        return Dataset(instances)

    def text_to_instance(self,  # type: ignore
                         tokens: List[Token],
                         verb_label: List[int],
                         target_index: int,
                         tags: List[str],
                         gold_spans: List[List[str]] = None) -> Instance:
        """
        We take `pre-tokenized` input here, along with a verb label.  The verb label should be a
        one-hot binary vector, the same length as the tokens, indicating the position of the verb
        to find arguments for.
        """
        # pylint: disable=arguments-differ

        # Input fields.
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        verb_field = SequenceLabelField(verb_label, text_field)
        target_field = IndexField(target_index, text_field)

        # Span-based output fields.
        span_starts: List[Field] = []
        span_ends: List[Field] = []
        # span_mask: List[int] = [1 for _ in range(
        #     len(tokens) * self.max_span_width)]
        span_labels: Optional[List[str]] = [
        ] if gold_spans is not None else None

        for j in range(len(tokens)):
            for diff in range(self.max_span_width):
                width = diff
                if j - diff < 0:
                    # This is an invalid span.
                    # span_mask[j * self.max_span_width + diff] = 0
                    width = j

                span_starts.append(IndexField(j - width, text_field))
                span_ends.append(IndexField(j, text_field))

                if gold_spans is not None:
                    current_label = gold_spans[j][diff]
                    span_labels.append(current_label)

        start_fields = ListField(span_starts)
        end_fields = ListField(span_ends)
        # span_mask_fields = SequenceLabelField(span_mask, start_fields)

        fields: Dict[str, Field] = {'tokens': text_field,
                                    'verb_indicator': verb_field,
                                    'target_index': target_field,
                                    'span_starts': start_fields,
                                    'span_ends': end_fields}
                                    # 'span_mask': span_mask_fields}

        if gold_spans:
            fields['tags'] = SequenceLabelField(span_labels, start_fields)
            # For debugging.
            # fields['bio'] = SequenceLabelField(tags, text_field)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'CrfSrlReader':
        token_indexers = TokenIndexer.dict_from_params(
            params.pop('token_indexers', {}))
        max_span_width = params.pop("max_span_width")
        params.assert_empty(cls.__name__)
        return CrfSrlReader(token_indexers=token_indexers,
                            max_span_width=max_span_width)
