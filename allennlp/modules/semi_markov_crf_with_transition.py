"""
Semi-Markov Conditional random field
"""
import logging
from typing import Dict, List, Tuple, Set

import sys
import torch
from torch.autograd import Variable

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import logsumexp, ones_like

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SemiMarkovConditionalRandomFieldWithTransition(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute the log-likelihood
    of its inputs assuming a 0th-order semi-Markov conditional random field model.
    The semi comes from the fact that there is no Markovian order inside of a segment.

    See, e.g. http://www.cs.cmu.edu/~wcohen/postscript/semiCRF.pdf

    Parameters
    ----------
    num_tags : int, required
        The number of tags.
    default_tag : int, required
        Index of tag '*' denoting the spans which are invalid.
    max_span_width : int, required.
        Maximum allowed width of a span.
    """

    def __init__(self,
                 num_tags: int,
                 default_tag: int,
                 max_span_width: int,
                 outside_span_tag: int = None,
                 loss_type: str = "logloss",
                 false_positive_penalty: float = 1.0,
                 false_negative_penalty: float = 1.0) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.max_span_width = max_span_width
        self.default_tag = default_tag
        self.outside_span_tag = outside_span_tag
        self.loss_type = loss_type
        self.false_positive_penalty = false_positive_penalty
        self.false_negative_penalty = false_negative_penalty

        # Also need logits for transitioning from "start" state and to "end" state.
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))
        self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
        self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal(self.transitions)
        torch.nn.init.normal(self.start_transitions)
        torch.nn.init.normal(self.end_transitions)

    def _input_likelihood(self,
                          logits: torch.Tensor,
                          text_mask: torch.Tensor,
                          tag_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible segmentations.
        Side effects: suffers from spurious ambiguity.

        Parameters
        ----------
        logits : shape (batch_size, sequence_length, max_span_width, num_tags)
        text_mask : shape (batch_size, sequence_length)
        span_mask : shape (batch_size, sequence_length, max_span_width)
        tag_mask : shape (batch_size, num_tags)
        cost : shape (batch_size, sequence_length, max_span_width, num_tags)
        """
        batch_size, sequence_length, max_span_width, num_tags = logits.size()
        #S, K, B, T
        logits = logits.permute(1,2,0,3).contiguous()
        # B
        c_lens = torch.sum(text_mask.clone(),dim=1)
        # S, B
        text_mask = text_mask.float().transpose(0, 1).contiguous()
        alpha = Variable(torch.zeros(batch_size,sequence_length+1,num_tags).cuda(), requires_grad=True)
        alpha_out_sum = Variable(logits.data.new(batch_size,max_span_width, num_tags).fill_(0))

        for j, logit in enumerate(logits):
            # Depending on where the span ends, i.e. j, the maximum width of the spans considered changes.
            for d in range(min(self.max_span_width, sequence_length)):
                if d<j:
                    emit_scores = logits[j][d].view(batch_size,1,num_tags)
                    transition_scores = self.transitions.view(1,num_tags,num_tags)
                    broadcast_alpha = alpha[j-d].view(batch_size,num_tags,1)
                    inner = broadcast_alpha + emit_scores + transition_scores
                    inner_sum = logsumexp(inner,dim=1) #B, tags
                elif d==j:
                    emit_scores = logits[j][d].view(batch_size,1,num_tags)
                    transition_scores = self.start_transitions.view(1,1,num_tags)
                    inner = emit_scores + transition_scores
                    inner_sum = inner.squeeze(1)
                alpha_out_sum[:,d,:] = inner_sum
            alpha_nxt = log_sum_exp(alpha_out_sum , dim=1) # B, tags
            mask = Variable((c_lens > 0).float().unsqueeze(-1))
            alpha_nxt = mask * alpha_nxt + (1 - mask) * alpha[:, j, :].clone()
            c_lens= c_lens - 1

            alpha[:,j+1, :] = alpha_nxt

        partition = alpha[:,-1,:] + self.end_transitions.view(1,num_tags)
        Z = logsumexp(partition)

        return Z

    def _joint_likelihood(self,
                          logits: torch.Tensor,
                          tags: torch.Tensor,
                          mask: torch.LongTensor) -> torch.Tensor:
        """
        Computes the numerator term for the log-likelihood, which is just score(inputs, tags)

        Parameters
        ----------
        logits : shape (batch_size, sequence_length, max_span_width, num_tags)
        tags : shape (batch_size, sequence_length, max_span_width)
        mask : shape (batch_size, sequence_length)
        """
        batch_size, sequence_length, max_span, num_tags = logits.shape
        # Transpose to shape: (sequence_length, max_span_width, batch_size, num_tags)
        logits = logits.permute(1,2,0,3).contiguous()
        # Transpose to shape: (sequence_length, batch_size)
        mask = mask.float().transpose(0, 1).contiguous()
        # Transpose to shape: (sequence_length, max_span_width, batch_size)
        tags = tags.permute(1,2,0).contiguous()

        default_tags = Variable(
            self.default_tag * torch.ones(batch_size).long().cuda()) #batch szie


        # Broadcast the transition scores to one per batch element
        broadcast_transitions = self.transitions.view(
            1, 1, num_tags, num_tags).expand(batch_size, self.max_span_width, num_tags, num_tags)
        broadcast_start = self.start_transitions.view(1,num_tags).expand(batch_size, num_tags)

        numerator = 0.0
        # Add up the scores for the observed segmentations
        for j in range(sequence_length):
            # # shape: (max_seg_len, batch_size)
            # batched_tags = tags[j]  # .transpose(0, 1).contiguous()
            # # shape: (max_seg_len, batch_size, num_tags)
            # batched_logits = logits[j]  # .transpose(0, 1).contiguous()
            for d in range(min(self.max_span_width, sequence_length)):
                if d>j:
                    continue

                current_tag = tags[j][d]

                # Ignore tags for invalid spans.
                valid_tag_mask = (current_tag != default_tags).float()
                # Reshape for gather operation to follow.
                current_tag = current_tag.view(batch_size, 1)

                if d == j:
                    transition_score = broadcast_start.gather(1, current_tag).squeeze(1)
                else:
                    prev_tag = tags[j-d-1]
                    prev_tag = prev_tag.transpose(0,1).contiguous() #batch size, max span width
                    transition_score = (
                        broadcast_transitions
                        # Choose the current_tag-th row for each input
                        .gather(2, prev_tag.view(batch_size, self.max_span_width, 1, 1).expand(batch_size, self.max_span_width, 1, num_tags))
                        # Squeeze down to (batch_size, num_tags)
                        .squeeze(2)
                        # Then choose the next_tag-th column for each of those
                        .gather(2, current_tag.view(batch_size, 1, 1).expand(batch_size, self.max_span_width, 1))
                        # And squeeze down to (batch_size,self.max_span_width)
                        .squeeze(2)
                    )
                    valid_transition_mask = (prev_tag != default_tags.view(batch_size,1).expand(batch_size,self.max_span_width)).float()
                    transition_score = torch.sum(valid_transition_mask * transition_score,1)
                # The score for using current_tag
                emit_score = logits[j][d].gather(dim=1, index=current_tag).squeeze(
                    1) * valid_tag_mask * mask[j]
                score = transition_score*mask[j] + emit_score
                
                numerator += score

        last_tag_index = mask.sum(0).long() - 1 # batch size
        # print("last tag index: ",last_tag)
        # print(last_tag_index.shape)
        last_tags = tags.gather(0, last_tag_index.view(1,1,batch_size).expand(1,self.max_span_width,batch_size)).squeeze(0).transpose(0, 1).contiguous() # (batch_size,max_span_width)
        # print(last_tags.shape)
        broadcast_end = self.end_transitions.view(1,1,num_tags).expand(batch_size, self.max_span_width, num_tags)
        last_transition_score = (
            broadcast_end
            .gather(2, last_tags.view(batch_size, self.max_span_width, 1)) # batch_size, self.max_span_width, 1
            .squeeze(2)
        )
        valid_transition_mask = (last_tags != default_tags.view(batch_size,1).expand(batch_size,self.max_span_width)).float()
        # for d in range(min(self.max_span_width, sequence_length)):
        #     last_tag = last_tags[d,:]
        #     valid_tag_mask = (last_tag != default_tags).float()
        #     last_tag = last_tag.view(batch_size,1)
        last_transition_score = torch.sum(valid_transition_mask * last_transition_score,1)
        numerator += transition_score
        # print(numerator)

        return numerator

    def forward(self,
                inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor,
                tag_mask: Variable = None,
                average_batches: bool = True) -> torch.Tensor:
        """
        Computes the log likelihood.
        Parameters
        ----------
        inputs : shape (batch_size, sequence_length, max_span_width, num_tags)
        tags : shape (batch_size, sequence_length, max_span_width)
        mask : shape (batch_size, sequence_length)
        tag_mask : shape (batch_size, num_tags)
        """
        # pylint: disable=arguments-differ
        batch_size = inputs.size(0)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        if self.loss_type == "roc":
            cost = self._get_recall_oriented_cost(tags)
        elif self.loss_type == "hamming":
            cost = self._get_hamming_cost(tags)
        elif self.loss_type == "logloss":
            zeroes = 1 - ones_like(inputs)
            cost = zeroes
        else:
            raise ConfigurationError(
                "invalid loss type {} - use roc, hamming or logloss".format(self.loss_type))

        log_denominator = self._input_likelihood(logits=inputs,
                                                 text_mask=mask,
                                                 tag_mask=tag_mask,
                                                 cost=cost)
        log_loss = log_numerator - log_denominator
        if self.loss_type == "roc":
            log_loss = log_loss - self.false_negative_penalty * \
                self._get_labeled_spans_count(tags)

        batch_loss = torch.sum(log_loss)
        if average_batches:
            batch_loss = batch_loss / batch_size

        if batch_loss.data[0] > 0.0:
            max_log_loss, _ = torch.max(log_loss, -1)
            logger.info("WARNING: invalid log loss = %f", max_log_loss.data[0])
        # assert batch_loss.data[0] <= 0.0
        if average_batches:
            return batch_loss, log_numerator
        
        return log_loss, log_numerator

    def viterbi_spans(self,logits, mask, tag_masks = None):

        def _decode(logits, tag_mask):
            sequence_length, max_span_width, num_classes = list(logits.size())

            tag_mask = torch.log(tag_mask).view(1, num_classes)

            alpha = [float('-inf')
                    for _ in range(sequence_length)]  # shape : [sequence_length]
            backpointers = [(None, None)
                            for _ in range(sequence_length)]  # shape : [sequence_length]

            # Evaluate the scores for all possible paths.
            for j in range(sequence_length):
                width = max_span_width
                if j < max_span_width - 1:
                    width = j + 1

                # Find the best labels (and their scores) for all spans ending at j
                start_indices = torch.cuda.LongTensor(range(width))
                span_factors = logits[j].index_select(0, start_indices)
                best_span_factors, best_labels = torch.max(
                    span_factors + tag_mask, -1)

                # Add a dummy dimension to alpha (for position -1) and reverse it.
                extended_alpha = [0.0] + alpha
                broadcast_alpha = torch.cuda.FloatTensor(
                    extended_alpha[j + 1 - width:j + 1][::-1])

                # Add pairwise potentials to current scores.
                summed_potentials = broadcast_alpha + best_span_factors
                best_score, best_difference = torch.max(summed_potentials, -1)

                # Reverse this, since it corresponds to reversed idx.
                best_difference = int(best_difference)
                alpha[j] = float(best_score)
                backpointers[j] = (best_labels[best_difference], best_difference)

            # Construct the most likely sequence backwards.
            viterbi_path = [[self.default_tag for _ in range(max_span_width)]
                            for _ in range(sequence_length)]
            # Also, keep track of the span indices and the associated tag.
            viterbi_spans = set()
            # Also construct the best scoring tensor (for evaluation, not quite necessary).
            viterbi_score = torch.Tensor([[[float("-inf") for _ in range(sequence_length)]
                                        for _ in range(max_span_width)] for _ in range(num_classes)])
            viterbi_score[self.default_tag] = 0.0
            viterbi_score = viterbi_score.transpose(0, 2).tolist()

            # Start from the end.
            span_end = sequence_length - 1
            while span_end >= 0:
                label, width = backpointers[span_end]
                viterbi_path[span_end][width] = label
                viterbi_spans.add((span_end - width, span_end, label))
                if label != self.default_tag:
                    viterbi_score[span_end][width][self.default_tag] = float(
                        "-inf")
                viterbi_score[span_end][width][label] = alpha[span_end]
                span_end = span_end - width - 1

            return viterbi_path, viterbi_score, viterbi_spans


        batch_size, max_seq_length, max_span_width, num_classes = logits.size()

        if tag_masks is None:
            tag_masks = Variable(torch.ones(batch_size, num_classes).cuda())

        # Get the tensors out of the variables
        logits, mask, tag_masks = logits.data, mask.data, tag_masks.data
        sequence_lengths = torch.sum(mask, dim=-1)

        all_tags = []
        all_scores = []
        all_spans = []
        for logits_ex, tag_mask, sequence_length in zip(logits, tag_masks, sequence_lengths):
            # We need to maintain this length, because all_tags needs to be of the same size as tags
            tags = [[self.default_tag for _ in range(max_span_width)]
                    for _ in range(max_seq_length)]
            scores = [[[float("-inf") for _ in range(num_classes)] for _ in range(max_span_width)]
                      for _ in range(max_seq_length)]

            # We pass the logits to ``viterbi_decode``.
            viterbi_path, viterbi_score, viterbi_spans = _decode(
                logits_ex[:sequence_length], tag_mask)

            tags[:len(viterbi_path)] = viterbi_path
            scores[:len(viterbi_score)] = viterbi_score

            # shape: (batch_size, max_seq_length, max_span_width)
            all_tags.append(tags)
            # shape: (batch_size, max_seq_length, max_span_width, num_classes)
            all_scores.append(scores)
            all_spans.append(viterbi_spans)

        return torch.Tensor(all_tags), torch.Tensor(all_scores), all_spans


    def viterbi_tags(self,
                     logits: Variable,
                     mask: Variable,
                     tag_masks: Variable = None) -> List[List[int]]:
        """
        Iterates through the batch and uses viterbi algorithm to find most likely tags
        for the given inputs.

        Returns
        -------
        all_tags : torch.Tensor
            shape (batch_size, sequence_length, max_span_width)
        all_scores : torch.Tensor
            shape (batch_size, sequence_length, max_span_width, num_tags)
        """
        batch_size, max_seq_length, max_span_width, num_classes = logits.size()

        if tag_masks is None:
            tag_masks = Variable(torch.ones(batch_size, num_classes).cuda())

        # Get the tensors out of the variables
        logits, mask, tag_masks = logits.data, mask.data, tag_masks.data
        sequence_lengths = torch.sum(mask, dim=-1)

        all_tags = []
        all_scores = []
        for logits_ex, tag_mask, sequence_length in zip(logits, tag_masks, sequence_lengths):
            # We need to maintain this length, because all_tags needs to be of the same size as tags
            tags = [[self.default_tag for _ in range(max_span_width)]
                    for _ in range(max_seq_length)]
            scores = [[[float("-inf") for _ in range(num_classes)] for _ in range(max_span_width)]
                      for _ in range(max_seq_length)]

            # We pass the logits to ``viterbi_decode``.
            viterbi_path, viterbi_score = self.viterbi_decode(
                logits_ex[:sequence_length], tag_mask)

            tags[:len(viterbi_path)] = viterbi_path
            scores[:len(viterbi_score)] = viterbi_score

            # shape: (batch_size, max_seq_length, max_span_width)
            all_tags.append(tags)
            # shape: (batch_size, max_seq_length, max_span_width, num_classes)
            all_scores.append(scores)

        return torch.Tensor(all_tags), torch.Tensor(all_scores)

    def viterbi_decode(self, logits: torch.Tensor, tag_mask: torch.Tensor):
        """
        Perform 0-th order Semi-Markov Viterbi decoding in log space over a sequence given
        a matrix of shape (sequence_length, span_width, num_tags) specifying unary potentials
        for possible tags per span in the sequence.

        Parameters
        ----------
        logits : torch.Tensor, required.
            A tensor of shape (sequence_length, span_width, num_tags) representing scores for
            a set of tags over a given sequence.
        tag_mask: torch.Tensor, required.
            shape (num_tags)

        Returns
        -------
        viterbi_path : List[List[int]]
            The tag indices of the maximum likelihood tag sequence.
        viterbi_score : torch.Tensor
            shape (sequence_length, max_span_width, num_tags)
            The score of the viterbi path.
        """
        sequence_length, max_span_width, num_classes = list(logits.size())

        tag_mask = torch.log(tag_mask).view(1, num_classes)

        alpha = [float('-inf')
                 for _ in range(sequence_length)]  # shape : [sequence_length]
        backpointers = [(None, None)
                        for _ in range(sequence_length)]  # shape : [sequence_length]

        # Evaluate the scores for all possible paths.
        for j in range(sequence_length):
            width = max_span_width
            if j < max_span_width - 1:
                width = j + 1

            # Find the best labels (and their scores) for all spans ending at j
            start_indices = torch.cuda.LongTensor(range(width))
            span_factors = logits[j].index_select(0, start_indices)
            best_span_factors, best_labels = torch.max(
                span_factors + tag_mask, -1)

            # Add a dummy dimension to alpha (for position -1) and reverse it.
            extended_alpha = [0.0] + alpha
            broadcast_alpha = torch.cuda.FloatTensor(
                extended_alpha[j + 1 - width:j + 1][::-1])

            # Add pairwise potentials to current scores.
            summed_potentials = broadcast_alpha + best_span_factors
            best_score, best_difference = torch.max(summed_potentials, -1)

            # Reverse this, since it corresponds to reversed idx.
            best_difference = int(best_difference)
            alpha[j] = float(best_score)
            backpointers[j] = (best_labels[best_difference], best_difference)

        # Construct the most likely sequence backwards.
        viterbi_path = [[self.default_tag for _ in range(max_span_width)]
                        for _ in range(sequence_length)]
        # Also, keep track of the span indices and the associated tag.
        viterbi_spans = {}
        # Also construct the best scoring tensor (for evaluation, not quite necessary).
        viterbi_score = torch.Tensor([[[float("-inf") for _ in range(sequence_length)]
                                       for _ in range(max_span_width)] for _ in range(num_classes)])
        viterbi_score[self.default_tag] = 0.0
        viterbi_score = viterbi_score.transpose(0, 2).tolist()

        # Start from the end.
        span_end = sequence_length - 1
        while span_end >= 0:
            label, width = backpointers[span_end]
            viterbi_path[span_end][width] = label
            viterbi_spans[(span_end - width, span_end)] = label
            if label != self.default_tag:
                viterbi_score[span_end][width][self.default_tag] = float(
                    "-inf")
            viterbi_score[span_end][width][label] = alpha[span_end]
            span_end = span_end - width - 1

        return viterbi_path, viterbi_score

    def convert_spans_into_sequence_of_tags(self, viterbi_spans: Dict[Tuple[int, int], int],
                                            sequence_length: int,
                                            num_classes: int) -> List[int]:
        tag_sequence = [None for _ in range(sequence_length)]
        tag_indicators = [
            [0.0 for _ in range(num_classes)] for _ in range(sequence_length)]
        for span in viterbi_spans:
            for position in range(span[0], span[1] + 1):
                # Make sure that the current position is not already assigned.
                assert not tag_sequence[position]
                tag_sequence[position] = viterbi_spans[span]
                tag_indicators[position][viterbi_spans[span]] = 1.0
        # Make sure every position got a tag.
        assert None not in tag_sequence
        return tag_indicators

    def merge_spans(self, tag_sequence: List[int]) -> List[List[int]]:
        spans = [[self.default_tag for _ in range(self.max_span_width)]
                 for _ in range(len(tag_sequence))]

        start_span = 0
        current_tag = tag_sequence[0]
        for pos, tag in enumerate(tag_sequence[1:], 1):
            width = pos - start_span
            if tag != current_tag:
                width = pos - 1 - start_span
                spans[pos - 1][width] = current_tag
                start_span = pos
                current_tag = tag
                width = pos - start_span
            # Maximum allowed width.
            elif width == self.max_span_width - 1:
                spans[pos][width] = current_tag
                start_span = pos + 1
                if pos + 1 < len(tag_sequence):
                    current_tag = tag_sequence[pos + 1]
        spans[len(tag_sequence) - 1][len(tag_sequence) -
                                     1 - start_span] = tag_sequence[-1]
        return spans

    def _get_hamming_cost(self, tags: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, max_span_width = tags.size()
        # Make tags the same dim as required cost, for indexing.
        tags = tags.unsqueeze(dim=-1)
        zeros = Variable(torch.zeros(batch_size, sequence_length,
                                     max_span_width, self.num_tags).float().cuda())
        scattered_tags = zeros.scatter_(-1, tags, 1)
        cost = 1 - scattered_tags

        # Now mask out the cost assigned to places without a real tag ~ "*"
        default_tags_mask = 1-tags.eq(self.default_tag).float()
        cost = cost * default_tags_mask

        return self.false_positive_penalty * cost

    def _get_simple_recall_cost(self, tags: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, max_span_width = tags.size()
        # Make tags the same dim as required cost, for indexing.
        tags = tags.unsqueeze(dim=-1)
        zeros = Variable(torch.zeros(batch_size, sequence_length,
                                     max_span_width, self.num_tags).float().cuda())
        scattered_tags = zeros.scatter_(-1, tags, 1)
        cost = 1 - scattered_tags

        # Now mask out the cost assigned to tags that are either "*" or outside span.
        irrelevant_tags = tags.eq(
            self.default_tag) | tags.eq(self.outside_span_tag)
        irrelevant_tags_mask = 1-irrelevant_tags.float()
        cost = cost * irrelevant_tags_mask

        return self.false_negative_penalty * cost

    def _get_recall_oriented_cost(self, tags: torch.Tensor):
        batch_size, sequence_length, max_span_width = tags.size()
        # Make tags the same dim as required cost, for indexing.
        tags = tags.unsqueeze(dim=-1)
        zeros = Variable(torch.zeros(batch_size, sequence_length,
                                     max_span_width, self.num_tags).float().cuda())
        scattered_tags = zeros.scatter_(-1, tags, 1)
        cost = 1 - scattered_tags

        # False Positives
        # Now mask out the cost assigned to places without a real tag ~ "*"
        default_tags_mask = 1-tags.eq(self.default_tag).float()
        fp = cost * default_tags_mask
        # Masking out all the "O"s
        fp = fp.index_fill_(-1,
                            Variable(torch.cuda.LongTensor([self.outside_span_tag])), 0)
        fp = self.false_positive_penalty * fp

        # False Negatives
        irrelevant_tags = tags.eq(
            self.default_tag) | tags.eq(self.outside_span_tag)
        irrelevant_tags_mask = 1-irrelevant_tags.float()
        fn = - self.false_negative_penalty * cost * irrelevant_tags_mask
        return fp + fn

    def _get_labeled_spans_count(self, tags: torch.Tensor):
        batch_size, sequence_length, max_span_width = tags.size()
        # Make tags the same dim as required cost, for indexing.
        tags = tags.unsqueeze(dim=-1)
        zeros = Variable(torch.zeros(batch_size, sequence_length,
                                     max_span_width, self.num_tags).float().cuda())
        scattered_tags = zeros.scatter_(-1, tags, 1)

        irrelevant_tags = tags.eq(
            self.default_tag) | tags.eq(self.outside_span_tag)
        irrelevant_tags_mask = 1-irrelevant_tags.float()

        total_relevant_labels = torch.sum(
            torch.sum(torch.sum(scattered_tags * irrelevant_tags_mask, -1), -1), -1)
        return total_relevant_labels
