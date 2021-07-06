"""
These submodules contain the classes for AllenNLP models,
all of which are subclasses of :class:`~allennlp.models.model.Model`.
"""

from allennlp.models.archival import archive_model, load_archive
from allennlp.models.crf_tagger import CrfTagger
from allennlp.models.decomposable_attention import DecomposableAttention
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.bidaf import BidirectionalAttentionFlow
from allennlp.models.simple_tagger import SimpleTagger
from allennlp.models.coreference_resolution.coref import CoreferenceResolver
from allennlp.models.semantic_role_labeler import SemanticRoleLabeler
from allennlp.models.span_srl.semi_crf_srl import SemiCrfSemanticRoleLabeler
from allennlp.models.span_srl.frame_semi_crf_srl import FrameSemanticRoleLabeler
from allennlp.models.span_srl.scaffolding.scaffolded_pb_srl import ScaffoldedPropBankSRL
from allennlp.models.span_srl.scaffolding.consist_srl_constit import ConsistSRLConstitParser
from allennlp.models.span_srl.consist_injection_srl import ConsistInjectionSRL
