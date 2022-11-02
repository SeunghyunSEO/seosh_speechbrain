"""Decoding methods for seq2seq autoregressive model.

Authors
 * Ju-Chieh Chou 2020
 * Peter Plantinga 2020
 * Mirco Ravanelli 2020
 * Sung-Lin Yeh 2020
"""
import torch

import speechbrain as sb
from speechbrain.decoders.ctc import CTCPrefixScorer

from pdb import set_trace as Tra

class S2SBaseSearcher(torch.nn.Module):
    """S2SBaseSearcher class to be inherited by other
    decoding approaches for seq2seq model.

    Arguments
    ---------
    bos_index : int
        The index of the beginning-of-sequence (bos) token.
    eos_index : int
        The index of end-of-sequence token.
    min_decode_radio : float
        The ratio of minimum decoding steps to the length of encoder states.
    max_decode_radio : float
        The ratio of maximum decoding steps to the length of encoder states.

    Returns
    -------
    predictions
        Outputs as Python list of lists, with "ragged" dimensions; padding
        has been removed.
    scores
        The sum of log probabilities (and possibly
        additional heuristic scores) for each prediction.

    """

    def __init__(
        self, bos_index, eos_index, min_decode_ratio, max_decode_ratio,
    ):
        super(S2SBaseSearcher, self).__init__()
        self.bos_index = bos_index
        self.eos_index = eos_index
        self.min_decode_ratio = min_decode_ratio
        self.max_decode_ratio = max_decode_ratio

    def forward(self, enc_states, wav_len):
        """This method should implement the forward algorithm of decoding method.

        Arguments
        ---------
        enc_states : torch.Tensor
            The precomputed encoder states to be used when decoding.
            (ex. the encoded speech representation to be attended).
        wav_len : torch.Tensor
            The speechbrain-style relative length.
        """
        raise NotImplementedError

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """This method should implement one step of
        forwarding operation in the autoregressive model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The memory variables input for this timestep.
            (ex. RNN hidden states).
        enc_states : torch.Tensor
            The encoder states to be attended.
        enc_lens : torch.Tensor
            The actual length of each enc_states sequence.

        Returns
        -------
        log_probs : torch.Tensor
            Log-probabilities of the current timestep output.
        memory : No limit
            The memory variables generated in this timestep.
            (ex. RNN hidden states).
        attn : torch.Tensor
            The attention weight for doing penalty.
        """
        raise NotImplementedError

    def reset_mem(self, batch_size, device):
        """This method should implement the resetting of
        memory variables for the seq2seq model.
        E.g., initializing zero vector as initial hidden states.

        Arguments
        ---------
        batch_size : int
            The size of the batch.
        device : torch.device
            The device to put the initial variables.

        Return
        ------
        memory : No limit
            The initial memory variable.
        """
        raise NotImplementedError

    def lm_forward_step(self, inp_tokens, memory):
        """This method should implement one step of
        forwarding operation for language model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The momory variables input for this timestep.
            (e.g., RNN hidden states).

        Return
        ------
        log_probs : torch.Tensor
            Log-probabilities of the current timestep output.
        memory : No limit
            The memory variables generated in this timestep.
            (e.g., RNN hidden states).
        """
        raise NotImplementedError

    def reset_lm_mem(self, batch_size, device):
        """This method should implement the resetting of
        memory variables in the language model.
        E.g., initializing zero vector as initial hidden states.

        Arguments
        ---------
        batch_size : int
            The size of the batch.
        device : torch.device
            The device to put the initial variables.

        Return
        ------
        memory : No limit
            The initial memory variable.
        """
        raise NotImplementedError


class S2SGreedySearcher(S2SBaseSearcher):
    """This class implements the general forward-pass of
    greedy decoding approach. See also S2SBaseSearcher().
    """

    def forward(self, enc_states, wav_len):
        """This method performs a greedy search.

        Arguments
        ---------
        enc_states : torch.Tensor
            The precomputed encoder states to be used when decoding.
            (ex. the encoded speech representation to be attended).
        wav_len : torch.Tensor
            The speechbrain-style relative length.
        """
        enc_lens = torch.round(enc_states.shape[1] * wav_len).int()
        device = enc_states.device
        batch_size = enc_states.shape[0]

        memory = self.reset_mem(batch_size, device=device)

        # Using bos as the first input
        inp_tokens = (
            enc_states.new_zeros(batch_size).fill_(self.bos_index).long()
        )

        log_probs_lst = []
        max_decode_steps = int(enc_states.shape[1] * self.max_decode_ratio)

        for t in range(max_decode_steps):
            log_probs, memory, _ = self.forward_step(
                inp_tokens, memory, enc_states, enc_lens
            )
            log_probs_lst.append(log_probs)
            inp_tokens = log_probs.argmax(dim=-1)

        log_probs = torch.stack(log_probs_lst, dim=1)
        scores, predictions = log_probs.max(dim=-1)
        scores = scores.sum(dim=1).tolist()
        predictions = batch_filter_seq2seq_output(
            predictions, eos_id=self.eos_index
        )

        return predictions, scores


class S2SRNNGreedySearcher(S2SGreedySearcher):
    """
    This class implements the greedy decoding
    for AttentionalRNNDecoder (speechbrain/nnet/RNN.py).
    See also S2SBaseSearcher() and S2SGreedySearcher().

    Arguments
    ---------
    embedding : torch.nn.Module
        An embedding layer.
    decoder : torch.nn.Module
        Attentional RNN decoder.
    linear : torch.nn.Module
        A linear output layer.
    **kwargs
        see S2SBaseSearcher, arguments are directly passed.

    Example
    -------
    >>> emb = torch.nn.Embedding(5, 3)
    >>> dec = sb.nnet.RNN.AttentionalRNNDecoder(
    ...     "gru", "content", 3, 3, 1, enc_dim=7, input_size=3
    ... )
    >>> lin = sb.nnet.linear.Linear(n_neurons=5, input_size=3)
    >>> searcher = S2SRNNGreedySearcher(
    ...     embedding=emb,
    ...     decoder=dec,
    ...     linear=lin,
    ...     bos_index=4,
    ...     eos_index=4,
    ...     min_decode_ratio=0,
    ...     max_decode_ratio=1,
    ... )
    >>> enc = torch.rand([2, 6, 7])
    >>> wav_len = torch.rand([2])
    >>> hyps, scores = searcher(enc, wav_len)
    """

    def __init__(self, embedding, decoder, linear, **kwargs):
        super(S2SRNNGreedySearcher, self).__init__(**kwargs)
        self.emb = embedding
        self.dec = decoder
        self.fc = linear
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def reset_mem(self, batch_size, device):
        """When doing greedy search, keep hidden state (hs) adn context vector (c)
        as memory.
        """
        hs = None
        self.dec.attn.reset()
        c = torch.zeros(batch_size, self.dec.attn_dim, device=device)
        return hs, c

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """Performs a step in the implemented beamsearcher."""
        hs, c = memory
        e = self.emb(inp_tokens)
        dec_out, hs, c, w = self.dec.forward_step(
            e, hs, c, enc_states, enc_lens
        )
        log_probs = self.softmax(self.fc(dec_out))
        return log_probs, (hs, c), w


class S2SBeamSearcher(S2SBaseSearcher):
    """This class implements the beam-search algorithm for the seq2seq model.
    See also S2SBaseSearcher().

    Arguments
    ---------
    bos_index : int
        The index of beginning-of-sequence token.
    eos_index : int
        The index of end-of-sequence token.
    min_decode_radio : float
        The ratio of minimum decoding steps to length of encoder states.
    max_decode_radio : float
        The ratio of maximum decoding steps to length of encoder states.
    beam_size : int
        The width of beam.
    topk : int
        The number of hypothesis to return. (default: 1)
    return_log_probs : bool
        Whether to return log-probabilities. (default: False)
    using_eos_threshold : bool
        Whether to use eos threshold. (default: true)
    eos_threshold : float
        The threshold coefficient for eos token (default: 1.5). See 3.1.2 in
        reference: https://arxiv.org/abs/1904.02619
    length_normalization : bool
        Whether to divide the scores by the length. (default: True)
    length_rewarding : float
        The coefficient of length rewarding (γ).
        log P(y|x) + λ log P_LM(y) + γ*len(y). (default: 0.0)
    coverage_penalty: float
        The coefficient of coverage penalty (η).
        log P(y|x) + λ log P_LM(y) + γ*len(y) + η*coverage(x,y). (default: 0.0)
        Reference: https://arxiv.org/pdf/1612.02695.pdf, https://arxiv.org/pdf/1808.10792.pdf
    lm_weight : float
        The weight of LM when performing beam search (λ).
        log P(y|x) + λ log P_LM(y). (default: 0.0)
    ctc_weight : float
        The weight of CTC probabilities when performing beam search (λ).
        (1-λ) log P(y|x) + λ log P_CTC(y|x). (default: 0.0)
    blank_index : int
        The index of the blank token.
    ctc_score_mode: str
        Default: "full"
        CTC prefix scoring on "partial" token or "full: token.
    ctc_window_size: int
        Default: 0
        Compute the ctc scores over the time frames using windowing based on attention peaks.
        If 0, no windowing applied.
    using_max_attn_shift: bool
        Whether using the max_attn_shift constraint. (default: False)
    max_attn_shift: int
        Beam search will block the beams that attention shift more
        than max_attn_shift.
        Reference: https://arxiv.org/abs/1904.02619
    minus_inf : float
        DefaultL -1e20
        The value of minus infinity to block some path
        of the search.
    """

    def __init__(
        self,
        bos_index,
        eos_index,
        min_decode_ratio,
        max_decode_ratio,
        beam_size,
        topk=1,
        return_log_probs=False,
        using_eos_threshold=True,
        eos_threshold=1.5,
        length_normalization=True,
        length_rewarding=0,
        coverage_penalty=0.0,
        lm_weight=0.0,
        lm_modules=None,
        ctc_weight=0.0,
        blank_index=0,
        ctc_score_mode="full",
        ctc_window_size=0,
        using_max_attn_shift=False,
        max_attn_shift=60,
        minus_inf=-1e20,
    ):
        super(S2SBeamSearcher, self).__init__(
            bos_index, eos_index, min_decode_ratio, max_decode_ratio,
        )
        self.beam_size = beam_size
        self.topk = topk
        self.return_log_probs = return_log_probs
        self.length_normalization = length_normalization
        self.length_rewarding = length_rewarding
        self.coverage_penalty = coverage_penalty
        self.coverage = None

        if self.length_normalization and self.length_rewarding > 0:
            print("length rewarding (penalty) detected, but length normalization is activated too, lets use length rewarding")
            self.length_normalization = False
            # raise ValueError(
            #     "length normalization is not compatible with length rewarding."
            # )

        self.using_eos_threshold = using_eos_threshold
        self.eos_threshold = eos_threshold
        self.using_max_attn_shift = using_max_attn_shift
        self.max_attn_shift = max_attn_shift
        self.lm_weight = lm_weight
        self.lm_modules = lm_modules

        # ctc related
        self.ctc_weight = ctc_weight
        self.blank_index = blank_index
        self.att_weight = 1.0 - ctc_weight

        assert (
            0.0 <= self.ctc_weight <= 1.0
        ), "ctc_weight should not > 1.0 and < 0.0"

        if self.ctc_weight > 0.0:
            if len({self.bos_index, self.eos_index, self.blank_index}) < 3:
                print("Warning !!! Original Implementation does not allow to assign same index to bos token and eos token !!!")
                # raise ValueError(
                #     "To perform joint ATT/CTC decoding, set blank, eos and bos to different indexes."
                # )

        # ctc already initialized
        self.minus_inf = minus_inf
        self.ctc_score_mode = ctc_score_mode
        self.ctc_window_size = ctc_window_size

    def _check_full_beams(self, hyps, beam_size):
        """This method checks whether hyps has been full.

        Arguments
        ---------
        hyps : List
            This list contains batch_size number.
            Each inside list contains a list stores all the hypothesis for this sentence.
        beam_size : int
            The number of beam_size.

        Returns
        -------
        bool
            Whether the hyps has been full.
        """
        hyps_len = [len(lst) for lst in hyps]
        beam_size = [self.beam_size for _ in range(len(hyps_len))]
        if hyps_len == beam_size:
            return True
        else:
            return False

    def _check_attn_shift(self, attn, prev_attn_peak):
        """This method checks whether attention shift is more than attn_shift.

        Arguments
        ---------
        attn : torch.Tensor
            The attention to be checked.
        prev_attn_peak : torch.Tensor
            The previous attention peak place.

        Returns
        -------
        cond : torch.BoolTensor
            Each element represents whether the beam is within the max_shift range.
        attn_peak : torch.Tensor
            The peak of the attn tensor.
        """

        # Tra()
        '''
        (Pdb) prev_attn_peak.size(); attn.size(); torch.max(attn, dim=1)[1].size()
        torch.Size([64])
        torch.Size([64, 599])
        torch.Size([64])

        (Pdb) self.max_attn_shift
        60
        '''

        # Block the candidates that exceed the max shift
        _, attn_peak = torch.max(attn, dim=1)
        lt_cond = attn_peak <= (prev_attn_peak + self.max_attn_shift)
        mt_cond = attn_peak > (prev_attn_peak - self.max_attn_shift)

        # True if not exceed limit
        # Multiplication equals to element-wise and for tensor
        cond = (lt_cond * mt_cond).unsqueeze(1)
        '''
        (Pdb) attn_peak; lt_cond; mt_cond; cond.size()
        tensor([ 19,  10,  12,  12,   8,  35,  47,   0,   0,  11,  22,   0,  18,  31,
                20,  23,  58,  20,  16,  25,  16,  11,  12,  12,  46,  23,  12,  47,
                12,  11,  19,  82,   7,  11,  43, 314,  37, 360,  52,  37,  27,  18,
                23,  69,  12,  42,  10,  28,  21,  21,   5,  26,  27,  18,  26,  25,
                11,  19,  11,  40,  11,  16,  17,  48], device='cuda:0')
        tensor([ True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                True, False,  True,  True,  True, False,  True, False,  True,  True,
                True,  True,  True, False,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
                True,  True,  True,  True], device='cuda:0')
        tensor([True, True, True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True, True, True,
                True, True, True, True], device='cuda:0')
        torch.Size([64, 1])
        '''

        return cond, attn_peak

    def _check_eos_threshold(self, log_probs):
        """
        This method checks whether eos log-probabilities exceed threshold.

        Arguments
        ---------
        log_probs : torch.Tensor
            The log-probabilities.

        Return
        ------
        cond : torch.BoolTensor
            Each element represents whether the eos log-probabilities will be kept.
        """
        max_probs, _ = torch.max(log_probs, dim=-1) # B, 1
        eos_probs = log_probs[:, self.eos_index] # B, vocab_size -> B, 1
        cond = eos_probs > (self.eos_threshold * max_probs)
        return cond

    def _update_hyp_and_scores(
        self,
        inp_tokens,
        alived_seq,
        alived_log_probs,
        hyps_and_scores,
        scores,
        timesteps,
    ):
        """This method will update hyps and scores if inp_tokens are eos.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The current output.
        alived_seq : torch.Tensor
            The tensor to store the alived_seq.
        alived_log_probs : torch.Tensor
            The tensor to store the alived_log_probs.
        hyps_and_scores : list
            To store generated hypotheses and scores.
        scores : torch.Tensor
            The final scores of beam search.
        timesteps : float
            The current timesteps. This is for length rewarding.

        Returns
        -------
        is_eos : torch.BoolTensor
            Each element represents whether the token is eos.
        """
        is_eos = inp_tokens.eq(self.eos_index)
        (eos_indices,) = torch.nonzero(is_eos, as_tuple=True)

        # Store the hypothesis and their scores when reaching eos.
        if eos_indices.shape[0] > 0:
            for index in eos_indices:
                # convert to int
                index = index.item()
                batch_id = torch.div(index, self.beam_size, rounding_mode="floor")
                if len(hyps_and_scores[batch_id]) == self.beam_size:
                    continue
                hyp = alived_seq[index, :]
                log_probs = alived_log_probs[index, :]

                '''
                Length: To encourage the generation of longer sequences, 
                we apply length normalizations during beam search.
                '''
                final_scores = scores[index] + self.length_rewarding * (timesteps + 1)
                hyps_and_scores[batch_id].append((hyp, log_probs, final_scores))

            '''
            (Pdb) is_eos; eos_indices
            tensor([ True, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False, False,
                    False, False, False, False, False, False, False, False, False, False],
                device='cuda:0')
            tensor([0], device='cuda:0')
            '''

        return is_eos

    def _get_top_score_prediction(self, hyps_and_scores, topk, return_indices=False):
        """This method sorts the scores and return corresponding hypothesis and log probs.

        Arguments
        ---------
        hyps_and_scores : list
            To store generated hypotheses and scores.
        topk : int
            Number of hypothesis to return.

        Returns
        -------
        topk_hyps : torch.Tensor (batch, topk, max length of token_id sequences)
            This tensor stores the topk predicted hypothesis.
        topk_scores : torch.Tensor (batch, topk)
            The length of each topk sequence in the batch.
        topk_lengths : torch.Tensor (batch, topk)
            This tensor contains the final scores of topk hypotheses.
        topk_log_probs : list
            The log probabilities of each hypotheses.
        """
        top_hyps, top_log_probs, top_scores, top_lengths = [], [], [], []
        batch_size = len(hyps_and_scores)

        '''
        (Pdb) len(hyps_and_scores); len(hyps_and_scores[0]); self.beam_size
        7
        2
        2
        '''

        # Collect hypotheses
        for i in range(len(hyps_and_scores)):
            hyps, log_probs, scores = zip(*hyps_and_scores[i])
            top_hyps += hyps
            top_scores += scores
            top_log_probs += log_probs
            top_lengths += [len(hyp) for hyp in hyps]
        top_hyps = torch.nn.utils.rnn.pad_sequence(
            top_hyps, batch_first=True, padding_value=0
        )
        top_scores = torch.stack((top_scores), dim=0).view(batch_size, -1)
        top_lengths = torch.tensor(
            top_lengths, dtype=torch.int, device=top_scores.device
        )

        # Tra()
        '''
        (Pdb) top_hyps.size(); top_hyps; top_lengths
        torch.Size([14, 92])
        tensor([[  16, 5311,   38,  ...,    0,    0,    0],
                [  16, 5311,   38,  ...,    0,    0,    0],
                [ 270,   28, 1000,  ...,    0,    0,    0],
                ...,
                [  51,   11,  580,  ...,    8,   69,    2],
                [   5,  685,    4,  ...,    0,    0,    0],
                [   5,  685,    4,  ...,    0,    0,    0]], device='cuda:0')
        tensor([89, 89, 82, 82, 82, 83, 62, 63, 90, 90, 89, 92, 66, 67],
            device='cuda:0', dtype=torch.int32)
        '''


        # Get topk indices
        topk_scores, indices = top_scores.topk(self.topk, dim=-1)
        indices = (indices + self.beam_offset.unsqueeze(1)).view(
            batch_size * self.topk
        )

        # Tra()
        '''
        (Pdb) indices; top_hyps.size(); torch.index_select(top_hyps, dim=0, index=indices,).size()
        tensor([ 0,  2,  4,  6,  8, 10, 12], device='cuda:0')
        torch.Size([14, 92])
        torch.Size([7, 92])
        '''


        # Select topk hypotheses
        topk_hyps = torch.index_select(top_hyps, dim=0, index=indices,)
        topk_hyps = topk_hyps.view(batch_size, self.topk, -1)

        topk_lengths = torch.index_select(top_lengths, dim=0, index=indices,)
        topk_lengths = topk_lengths.view(batch_size, self.topk)

        topk_log_probs = [top_log_probs[index.item()] for index in indices]

        if return_indices:
            return topk_hyps, topk_scores, topk_lengths, topk_log_probs, indices
        else:
            return topk_hyps, topk_scores, topk_lengths, topk_log_probs

    def forward(self, enc_states, wav_len):  # noqa: C901
        """Applies beamsearch and returns the predicted tokens."""
        enc_lens = torch.round(enc_states.shape[1] * wav_len).int()
        device = enc_states.device
        batch_size = enc_states.shape[0]

        memory = self.reset_mem(batch_size * self.beam_size, device=device)

        if self.lm_weight > 0:
            lm_memory = self.reset_lm_mem(batch_size * self.beam_size, device)

        if self.ctc_weight > 0:
            # (batch_size * beam_size, L, vocab_size)
            ctc_outputs = self.ctc_forward_step(enc_states)
            ctc_scorer = CTCPrefixScorer(
                ctc_outputs,
                enc_lens,
                batch_size,
                self.beam_size,
                self.blank_index,
                self.eos_index,
                self.ctc_window_size,
            )
            ctc_memory = None

        '''
        (Pdb) enc_states.size(); enc_lens; enc_lens.size()
        torch.Size([8, 107, 512])
        tensor([ 68,  56, 107,  52,  54,  48,  79,  60], device='cuda:0',
            dtype=torch.int32)
        torch.Size([8])

        (Pdb) memory; self.lm_weight; lm_memory; self.ctc_weight; ctc_outputs.size(); ctc_memory;
        0.6
        0.4
        torch.Size([8, 107, 5000])
        '''

        # Tra()

        # Inflate the enc_states and enc_len by beam_size times
        enc_states = inflate_tensor(enc_states, times=self.beam_size, dim=0)
        enc_lens = inflate_tensor(enc_lens, times=self.beam_size, dim=0)

        '''
        (Pdb) enc_states.size(); enc_lens.size(); self.beam_size
        torch.Size([528, 107, 512])
        torch.Size([528])
        66
        '''


        # Using bos as the first input
        inp_tokens = (
            torch.zeros(batch_size * self.beam_size, device=device)
            .fill_(self.bos_index)
            .long()
        )

        # The first index of each sentence.
        self.beam_offset = (
            torch.arange(batch_size, device=device) * self.beam_size
        )

        '''
        (Pdb) inp_tokens.size()
        torch.Size([528])
        (Pdb) self.beam_offset
        tensor([  0,  66, 132, 198, 264, 330, 396, 462], device='cuda:0')
        '''


        # initialize sequence scores variables.
        sequence_scores = torch.empty(
            batch_size * self.beam_size, device=device
        )
        sequence_scores.fill_(float("-inf"))

        # keep only the first to make sure no redundancy.
        sequence_scores.index_fill_(0, self.beam_offset, 0.0)

        '''
        (Pdb) sequence_scores.size()
        torch.Size([528])
        (Pdb) sequence_scores[:10]
        tensor([0., -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
            device='cuda:0')
        '''

        # keep the hypothesis that reaches eos and their corresponding score and log_probs.
        hyps_and_scores = [[] for _ in range(batch_size)]

        # keep the sequences that still not reaches eos.
        alived_seq = torch.empty(
            batch_size * self.beam_size, 0, device=device
        ).long()

        # Keep the log-probabilities of alived sequences.
        alived_log_probs = torch.empty(
            batch_size * self.beam_size, 0, device=device
        )

        min_decode_steps = int(enc_states.shape[1] * self.min_decode_ratio)
        max_decode_steps = int(enc_states.shape[1] * self.max_decode_ratio)

        '''
        (Pdb) min_decode_steps; max_decode_steps;
        0
        107
        '''

        # Initialize the previous attention peak to zero
        # This variable will be used when using_max_attn_shift=True
        prev_attn_peak = torch.zeros(batch_size * self.beam_size, device=device)

        for t in range(max_decode_steps):
            # terminate condition
            if self._check_full_beams(hyps_and_scores, self.beam_size):
                break

            '''
            (Pdb) t; inp_tokens.size(); memory; enc_states.size(); enc_lens.size();
            0
            torch.Size([528])
            torch.Size([528, 107, 512])
            torch.Size([528])

            (Pdb) t; inp_tokens.size(); memory.size(); enc_states.size(); enc_lens.size();
            1
            torch.Size([528])
            torch.Size([528, 1])
            torch.Size([528, 107, 512])
            torch.Size([528])
            '''

            log_probs, memory, attn = self.forward_step(
                inp_tokens, memory, enc_states, enc_lens
            )
            log_probs = self.att_weight * log_probs

            # Tra()

            '''
            (Pdb) t; log_probs.size(); memory.size(); attn.size()
            0
            torch.Size([528, 5000])
            torch.Size([528, 1])
            torch.Size([528, 1, 107])

            (Pdb) t; log_probs.size(); memory.size(); attn.size()
            1
            torch.Size([528, 5000])
            torch.Size([528, 2])
            torch.Size([528, 2, 107])
            '''

            # Keep the original value
            log_probs_clone = log_probs.clone().reshape(batch_size, -1)
            vocab_size = log_probs.shape[-1]

            if self.using_max_attn_shift:
                # Block the candidates that exceed the max shift
                cond, attn_peak = self._check_attn_shift(attn, prev_attn_peak)
                log_probs = mask_by_condition(
                    log_probs, cond, fill_value=self.minus_inf
                )
                prev_attn_peak = attn_peak

            # Set eos to minus_inf when less than minimum steps.
            if t < min_decode_steps:
                log_probs[:, self.eos_index] = self.minus_inf
                '''eos always have minus value?'''

            # Set the eos prob to minus_inf when it doesn't exceed threshold.
            if self.using_eos_threshold:
                cond = self._check_eos_threshold(log_probs)
                log_probs[:, self.eos_index] = mask_by_condition(
                    log_probs[:, self.eos_index],
                    cond,
                    fill_value=self.minus_inf,
                )

            # adding LM scores to log_prob if lm_weight > 0
            if self.lm_weight > 0:
                lm_log_probs, lm_memory = self.lm_forward_step(
                    inp_tokens, lm_memory
                )
                log_probs = log_probs + self.lm_weight * lm_log_probs

            # Tra()

            # adding CTC scores to log_prob if ctc_weight > 0
            if self.ctc_weight > 0:
                g = alived_seq
                # block blank token
                log_probs[:, self.blank_index] = self.minus_inf
                if self.ctc_weight != 1.0 and self.ctc_score_mode == "partial":
                    # pruning vocab for ctc_scorer
                    _, ctc_candidates = log_probs.topk(
                        self.beam_size * 2, dim=-1
                    )
                else:
                    ctc_candidates = None

                ctc_log_probs, ctc_memory = ctc_scorer.forward_step(
                    g, ctc_memory, ctc_candidates, attn
                )
                log_probs = log_probs + self.ctc_weight * ctc_log_probs

            scores = sequence_scores.unsqueeze(1).expand(-1, vocab_size)
            scores = scores + log_probs

            # Tra()

            # length normalization
            if self.length_normalization:
                scores = scores / (t + 1)

            # keep topk beams
            scores, candidates = scores.view(batch_size, -1).topk(
                self.beam_size, dim=-1
            )

            # The input for the next step, also the output of current step.
            inp_tokens = (candidates % vocab_size).view(
                batch_size * self.beam_size
            )

            scores = scores.view(batch_size * self.beam_size)
            sequence_scores = scores

            # recover the length normalization
            if self.length_normalization:
                sequence_scores = sequence_scores * (t + 1)

            # The index of which beam the current top-K output came from in (t-1) timesteps.
            predecessors = (
                torch.div(candidates, vocab_size, rounding_mode="floor")
                + self.beam_offset.unsqueeze(1).expand_as(candidates)
            ).view(batch_size * self.beam_size)

            # Permute the memory to synchoronize with the output.
            memory = self.permute_mem(memory, index=predecessors)
            if self.lm_weight > 0:
                lm_memory = self.permute_lm_mem(lm_memory, index=predecessors)

            if self.ctc_weight > 0:
                ctc_memory = ctc_scorer.permute_mem(ctc_memory, candidates)

            # If using_max_attn_shift, then the previous attn peak has to be permuted too.
            if self.using_max_attn_shift:
                prev_attn_peak = torch.index_select(
                    prev_attn_peak, dim=0, index=predecessors
                )

            # Add coverage penalty
            if self.coverage_penalty > 0:
                cur_attn = torch.index_select(attn, dim=0, index=predecessors)

                # coverage: cumulative attention probability vector
                if t == 0:
                    # Init coverage
                    self.coverage = cur_attn

                # the attn of transformer is [batch_size*beam_size, current_step, source_len]
                if len(cur_attn.size()) > 2:
                    self.converage = torch.sum(cur_attn, dim=1)
                else:
                    # Update coverage
                    self.coverage = torch.index_select(
                        self.coverage, dim=0, index=predecessors
                    )
                    self.coverage = self.coverage + cur_attn

                # Compute coverage penalty and add it to scores
                penalty = torch.max(
                    self.coverage, self.coverage.clone().fill_(0.5)
                ).sum(-1)
                penalty = penalty - self.coverage.size(-1) * 0.5
                penalty = penalty.view(batch_size * self.beam_size)
                penalty = (
                    penalty / (t + 1) if self.length_normalization else penalty
                )
                scores = scores - penalty * self.coverage_penalty

            # Update alived_seq
            alived_seq = torch.cat(
                [
                    torch.index_select(alived_seq, dim=0, index=predecessors),
                    inp_tokens.unsqueeze(1),
                ],
                dim=-1,
            )

            # Takes the log-probabilities
            beam_log_probs = log_probs_clone[
                torch.arange(batch_size).unsqueeze(1), candidates
            ].reshape(batch_size * self.beam_size)
            alived_log_probs = torch.cat(
                [
                    torch.index_select(
                        alived_log_probs, dim=0, index=predecessors
                    ),
                    beam_log_probs.unsqueeze(1),
                ],
                dim=-1,
            )

            is_eos = self._update_hyp_and_scores(
                inp_tokens,
                alived_seq,
                alived_log_probs,
                hyps_and_scores,
                scores,
                timesteps=t,
            )

            # Block the paths that have reached eos.
            sequence_scores.masked_fill_(is_eos, float("-inf"))

        if not self._check_full_beams(hyps_and_scores, self.beam_size):
            # Using all eos to fill-up the hyps.
            eos = (
                torch.zeros(batch_size * self.beam_size, device=device)
                .fill_(self.eos_index)
                .long()
            )
            _ = self._update_hyp_and_scores(
                eos,
                alived_seq,
                alived_log_probs,
                hyps_and_scores,
                scores,
                timesteps=max_decode_steps,
            )

        (
            topk_hyps,
            topk_scores,
            topk_lengths,
            log_probs,
        ) = self._get_top_score_prediction(hyps_and_scores, topk=self.topk,)
        # pick the best hyp
        predictions = topk_hyps[:, 0, :]
        predictions = batch_filter_seq2seq_output(
            predictions, eos_id=self.eos_index
        )

        if self.return_log_probs:
            return predictions, topk_scores, log_probs
        else:
            return predictions, topk_scores

    def ctc_forward_step(self, x):
        """Applies a ctc step during bramsearch."""
        logits = self.ctc_fc(x)
        log_probs = self.softmax(logits)
        return log_probs

    def permute_mem(self, memory, index):
        """This method permutes the seq2seq model memory
        to synchronize the memory index with the current output.

        Arguments
        ---------
        memory : No limit
            The memory variable to be permuted.
        index : torch.Tensor
            The index of the previous path.

        Return
        ------
        The variable of the memory being permuted.

        """
        raise NotImplementedError

    def permute_lm_mem(self, memory, index):
        """This method permutes the language model memory
        to synchronize the memory index with the current output.

        Arguments
        ---------
        memory : No limit
            The memory variable to be permuted.
        index : torch.Tensor
            The index of the previous path.

        Returns
        -------
        The variable of the memory being permuted.
        """
        raise NotImplementedError


class S2SRNNBeamSearcher(S2SBeamSearcher):
    """
    This class implements the beam search decoding
    for AttentionalRNNDecoder (speechbrain/nnet/RNN.py).
    See also S2SBaseSearcher(), S2SBeamSearcher().

    Arguments
    ---------
    embedding : torch.nn.Module
        An embedding layer.
    decoder : torch.nn.Module
        Attentional RNN decoder.
    linear : torch.nn.Module
        A linear output layer.
    temperature : float
        Temperature factor applied to softmax. It changes the probability
        distribution, being softer when T>1 and sharper with T<1.
    **kwargs
        see S2SBeamSearcher, arguments are directly passed.

    Example
    -------
    >>> emb = torch.nn.Embedding(5, 3)
    >>> dec = sb.nnet.RNN.AttentionalRNNDecoder(
    ...     "gru", "content", 3, 3, 1, enc_dim=7, input_size=3
    ... )
    >>> lin = sb.nnet.linear.Linear(n_neurons=5, input_size=3)
    >>> ctc_lin = sb.nnet.linear.Linear(n_neurons=5, input_size=7)
    >>> searcher = S2SRNNBeamSearcher(
    ...     embedding=emb,
    ...     decoder=dec,
    ...     linear=lin,
    ...     ctc_linear=ctc_lin,
    ...     bos_index=4,
    ...     eos_index=4,
    ...     blank_index=4,
    ...     min_decode_ratio=0,
    ...     max_decode_ratio=1,
    ...     beam_size=2,
    ... )
    >>> enc = torch.rand([2, 6, 7])
    >>> wav_len = torch.rand([2])
    >>> hyps, scores = searcher(enc, wav_len)
    """

    def __init__(
        self,
        embedding,
        decoder,
        linear,
        ctc_linear=None,
        temperature=1.0,
        **kwargs,
    ):
        super(S2SRNNBeamSearcher, self).__init__(**kwargs)
        self.emb = embedding
        self.dec = decoder
        self.fc = linear
        self.ctc_fc = ctc_linear
        if self.ctc_weight > 0.0 and self.ctc_fc is None:
            raise ValueError(
                "To perform joint ATT/CTC decoding, ctc_fc is required."
            )

        self.softmax = torch.nn.LogSoftmax(dim=-1)
        self.temperature = temperature

    def reset_mem(self, batch_size, device):
        """Needed to reset the memory during beamsearch."""
        hs = None
        self.dec.attn.reset()
        c = torch.zeros(batch_size, self.dec.attn_dim, device=device)
        return hs, c

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """Performs a step in the implemented beamsearcher."""
        with torch.no_grad():
            hs, c = memory
            e = self.emb(inp_tokens)
            dec_out, hs, c, w = self.dec.forward_step(
                e, hs, c, enc_states, enc_lens
            )
            log_probs = self.softmax(self.fc(dec_out) / self.temperature)
        # average attn weight of heads when attn_type is multiheadlocation
        if self.dec.attn_type == "multiheadlocation":
            w = torch.mean(w, dim=1)
        return log_probs, (hs, c), w

    def permute_mem(self, memory, index):
        """Memory permutation during beamsearch."""
        hs, c = memory

        # shape of hs: [num_layers, batch_size, n_neurons]
        if isinstance(hs, tuple):
            hs_0 = torch.index_select(hs[0], dim=1, index=index)
            hs_1 = torch.index_select(hs[1], dim=1, index=index)
            hs = (hs_0, hs_1)
        else:
            hs = torch.index_select(hs, dim=1, index=index)

        c = torch.index_select(c, dim=0, index=index)
        if self.dec.attn_type == "location":
            self.dec.attn.prev_attn = torch.index_select(
                self.dec.attn.prev_attn, dim=0, index=index
            )
        return (hs, c)


class S2SRNNBeamSearchLM(S2SRNNBeamSearcher):
    """This class implements the beam search decoding
    for AttentionalRNNDecoder (speechbrain/nnet/RNN.py) with LM.
    See also S2SBaseSearcher(), S2SBeamSearcher(), S2SRNNBeamSearcher().

    Arguments
    ---------
    embedding : torch.nn.Module
        An embedding layer.
    decoder : torch.nn.Module
        Attentional RNN decoder.
    linear : torch.nn.Module
        A linear output layer.
    language_model : torch.nn.Module
        A language model.
    temperature_lm : float
        Temperature factor applied to softmax. It changes the probability
        distribution, being softer when T>1 and sharper with T<1.
    **kwargs
        Arguments to pass to S2SBeamSearcher.

    Example
    -------
    >>> from speechbrain.lobes.models.RNNLM import RNNLM
    >>> emb = torch.nn.Embedding(5, 3)
    >>> dec = sb.nnet.RNN.AttentionalRNNDecoder(
    ...     "gru", "content", 3, 3, 1, enc_dim=7, input_size=3
    ... )
    >>> lin = sb.nnet.linear.Linear(n_neurons=5, input_size=3)
    >>> lm = RNNLM(output_neurons=5, return_hidden=True)
    >>> searcher = S2SRNNBeamSearchLM(
    ...     embedding=emb,
    ...     decoder=dec,
    ...     linear=lin,
    ...     language_model=lm,
    ...     bos_index=4,
    ...     eos_index=4,
    ...     blank_index=4,
    ...     min_decode_ratio=0,
    ...     max_decode_ratio=1,
    ...     beam_size=2,
    ...     lm_weight=0.5,
    ... )
    >>> enc = torch.rand([2, 6, 7])
    >>> wav_len = torch.rand([2])
    >>> hyps, scores = searcher(enc, wav_len)
    """

    def __init__(
        self,
        embedding,
        decoder,
        linear,
        language_model,
        temperature_lm=1.0,
        **kwargs,
    ):
        super(S2SRNNBeamSearchLM, self).__init__(
            embedding, decoder, linear, **kwargs
        )

        self.lm = language_model
        self.lm.eval()
        self.log_softmax = sb.nnet.activations.Softmax(apply_log=True)
        self.temperature_lm = temperature_lm

    def lm_forward_step(self, inp_tokens, memory):
        """Applies a step to the LM during beamsearch."""
        with torch.no_grad():
            logits, hs = self.lm(inp_tokens, hx=memory)
            log_probs = self.log_softmax(logits / self.temperature_lm)

        return log_probs, hs

    def permute_lm_mem(self, memory, index):
        """This is to permute lm memory to synchronize with current index
        during beam search. The order of beams will be shuffled by scores
        every timestep to allow batched beam search.
        Further details please refer to speechbrain/decoder/seq2seq.py.
        """

        if isinstance(memory, tuple):
            memory_0 = torch.index_select(memory[0], dim=1, index=index)
            memory_1 = torch.index_select(memory[1], dim=1, index=index)
            memory = (memory_0, memory_1)
        else:
            memory = torch.index_select(memory, dim=1, index=index)
        return memory

    def reset_lm_mem(self, batch_size, device):
        """Needed to reset the LM memory during beamsearch."""
        # set hidden_state=None, pytorch RNN will automatically set it to
        # zero vectors.
        return None


class S2SRNNBeamSearchTransformerLM(S2SRNNBeamSearcher):
    """This class implements the beam search decoding
    for AttentionalRNNDecoder (speechbrain/nnet/RNN.py) with LM.
    See also S2SBaseSearcher(), S2SBeamSearcher(), S2SRNNBeamSearcher().

    Arguments
    ---------
    embedding : torch.nn.Module
        An embedding layer.
    decoder : torch.nn.Module
        Attentional RNN decoder.
    linear : torch.nn.Module
        A linear output layer.
    language_model : torch.nn.Module
        A language model.
    temperature_lm : float
        Temperature factor applied to softmax. It changes the probability
        distribution, being softer when T>1 and sharper with T<1.
    **kwargs
        Arguments to pass to S2SBeamSearcher.

    Example
    -------
    >>> from speechbrain.lobes.models.transformer.TransformerLM import TransformerLM
    >>> emb = torch.nn.Embedding(5, 3)
    >>> dec = sb.nnet.RNN.AttentionalRNNDecoder(
    ...     "gru", "content", 3, 3, 1, enc_dim=7, input_size=3
    ... )
    >>> lin = sb.nnet.linear.Linear(n_neurons=5, input_size=3)
    >>> lm = TransformerLM(5, 512, 8, 1, 0, 1024, activation=torch.nn.GELU)
    >>> searcher = S2SRNNBeamSearchTransformerLM(
    ...     embedding=emb,
    ...     decoder=dec,
    ...     linear=lin,
    ...     language_model=lm,
    ...     bos_index=4,
    ...     eos_index=4,
    ...     blank_index=4,
    ...     min_decode_ratio=0,
    ...     max_decode_ratio=1,
    ...     beam_size=2,
    ...     lm_weight=0.5,
    ... )
    >>> enc = torch.rand([2, 6, 7])
    >>> wav_len = torch.rand([2])
    >>> hyps, scores = searcher(enc, wav_len)
    """

    def __init__(
        self,
        embedding,
        decoder,
        linear,
        language_model,
        temperature_lm=1.0,
        **kwargs,
    ):
        super(S2SRNNBeamSearchTransformerLM, self).__init__(
            embedding, decoder, linear, **kwargs
        )

        self.lm = language_model
        self.lm.eval()
        self.log_softmax = sb.nnet.activations.Softmax(apply_log=True)
        self.temperature_lm = temperature_lm

    def lm_forward_step(self, inp_tokens, memory):
        """Performs a step in the LM during beamsearch."""
        memory = _update_mem(inp_tokens, memory)
        if not next(self.lm.parameters()).is_cuda:
            self.lm.to(inp_tokens.device)
        logits = self.lm(memory)
        log_probs = self.softmax(logits / self.temperature_lm)
        return log_probs[:, -1, :], memory

    def permute_lm_mem(self, memory, index):
        """Permutes the LM ,emory during beamsearch"""
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def reset_lm_mem(self, batch_size, device):
        """Needed to reset the LM memory during beamsearch"""
        # set hidden_state=None, pytorch RNN will automatically set it to
        # zero vectors.
        return None


def inflate_tensor(tensor, times, dim):
    """This function inflates the tensor for times along dim.

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to be inflated.
    times : int
        The tensor will inflate for this number of times.
    dim : int
        The dim to be inflated.

    Returns
    -------
    torch.Tensor
        The inflated tensor.

    Example
    -------
    >>> tensor = torch.Tensor([[1,2,3], [4,5,6]])
    >>> new_tensor = inflate_tensor(tensor, 2, dim=0)
    >>> new_tensor
    tensor([[1., 2., 3.],
            [1., 2., 3.],
            [4., 5., 6.],
            [4., 5., 6.]])
    """
    return torch.repeat_interleave(tensor, times, dim=dim)


def mask_by_condition(tensor, cond, fill_value):
    """This function will mask some element in the tensor with fill_value, if condition=False.

    Arguments
    ---------
    tensor : torch.Tensor
        The tensor to be masked.
    cond : torch.BoolTensor
        This tensor has to be the same size as tensor.
        Each element represents whether to keep the value in tensor.
    fill_value : float
        The value to fill in the masked element.

    Returns
    -------
    torch.Tensor
        The masked tensor.

    Example
    -------
    >>> tensor = torch.Tensor([[1,2,3], [4,5,6]])
    >>> cond = torch.BoolTensor([[True, True, False], [True, False, False]])
    >>> mask_by_condition(tensor, cond, 0)
    tensor([[1., 2., 0.],
            [4., 0., 0.]])
    """
    # Tra()
    tensor = torch.where(
        cond, tensor, torch.Tensor([fill_value]).type_as(tensor)
    )
    return tensor


def _update_mem(inp_tokens, memory):
    """This function is for updating the memory for transformer searches.
    it is called at each decoding step. When being called, it appends the
    predicted token of the previous step to existing memory.

    Arguments:
    -----------
    inp_tokens : tensor
        Predicted token of the previous decoding step.
    memory : tensor
        Contains all the predicted tokens.
    """
    if memory is None:
        return inp_tokens.unsqueeze(1)
    return torch.cat([memory, inp_tokens.unsqueeze(1)], dim=-1)


class S2STransformerBeamSearch(S2SBeamSearcher):
    """This class implements the beam search decoding
    for Transformer.
    See also S2SBaseSearcher(), S2SBeamSearcher().

    Arguments
    ---------
    model : torch.nn.Module
        The model to use for decoding.
    linear : torch.nn.Module
        A linear output layer.
    **kwargs
        Arguments to pass to S2SBeamSearcher

    Example:
    --------
    >>> # see recipes/LibriSpeech/ASR_transformer/experiment.py
    """

    def __init__(
        self, modules, temperature=1.0, temperature_lm=1.0, **kwargs,
    ):
        super(S2STransformerBeamSearch, self).__init__(**kwargs)

        self.model = modules[0]
        self.fc = modules[1]
        self.ctc_fc = modules[2]
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.temperature = temperature
        self.temperature_lm = temperature_lm

    def reset_mem(self, batch_size, device):
        """Needed to reset the memory during beamsearch."""
        return None

    def reset_lm_mem(self, batch_size, device):
        """Needed to reset the LM memory during beamsearch."""
        return None

    def permute_mem(self, memory, index):
        """Permutes the memory."""
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def permute_lm_mem(self, memory, index):
        """Permutes the memory of the language model."""
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def forward_step(self, inp_tokens, memory, enc_states, enc_lens):
        """Performs a step in the implemented beamsearcher."""
        memory = _update_mem(inp_tokens, memory)
        pred, attn = self.model.decode(memory, enc_states)
        prob_dist = self.softmax(self.fc(pred) / self.temperature)
        return prob_dist[:, -1, :], memory, attn

    def lm_forward_step(self, inp_tokens, memory):
        """Performs a step in the implemented LM module."""
        memory = _update_mem(inp_tokens, memory)
        if not next(self.lm_modules.parameters()).is_cuda:
            self.lm_modules.to(inp_tokens.device)
        logits = self.lm_modules(memory)
        log_probs = self.softmax(logits / self.temperature_lm)
        return log_probs[:, -1, :], memory




class S2STransformerBeamSearchforFairseq(S2SBeamSearcher):
    def __init__(
        self, 
        attention_decoder,
        ctc_layer, 
        temperature=1.0, 
        temperature_lm=1.0,
        temperature_ctc=1.0,
        fairseq_vocab=None,
        lm_fairseq_vocab=None,
        internal_lm_estimation=False,
        internal_lm_weight=1.0,
        no_repeat_ngram_size=0,
        blank_collapse=0.0,
        **kwargs,
    ):
        super(S2STransformerBeamSearchforFairseq, self).__init__(**kwargs)

        self.attention_decoder = attention_decoder
        self.ctc_fc = ctc_layer

        self.softmax = torch.nn.LogSoftmax(dim=-1)

        self.temperature = temperature
        self.temperature_lm = temperature_lm
        self.temperature_ctc = temperature_ctc

        self.internal_lm_estimation = internal_lm_estimation
        self.internal_lm_weight = internal_lm_weight

        self.fairseq_vocab = fairseq_vocab
        self.lm_fairseq_vocab = lm_fairseq_vocab

        if no_repeat_ngram_size > 0:
            self.repeat_ngram_blocker = NGramRepeatBlock(no_repeat_ngram_size)
        else:
            self.repeat_ngram_blocker = None

        self.blank_collapse = blank_collapse

    def reset_mem(self, batch_size, device):
        """Needed to reset the memory during beamsearch."""
        return None

    def reset_lm_mem(self, batch_size, device):
        """Needed to reset the LM memory during beamsearch."""
        return None

    def permute_mem(self, memory, index):
        """Permutes the memory."""
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def permute_lm_mem(self, memory, index):
        """Permutes the memory of the language model."""
        memory = torch.index_select(memory, dim=0, index=index)
        return memory

    def forward(self, encoder_out, wav_len):  # noqa: C901
        """Applies beamsearch and returns the predicted tokens."""

        enc_states = encoder_out["encoder_out_before_proj"].transpose(0, 1)
        enc_lens = torch.round(enc_states.shape[1] * wav_len).int()

        device = enc_states.device
        batch_size = enc_states.shape[0]

        memory = self.reset_mem(batch_size * self.beam_size, device=device)

        if self.lm_weight > 0:
            lm_memory = self.reset_lm_mem(batch_size * self.beam_size, device)
        
        if self.ctc_weight > 0:
            # (batch_size * beam_size, L, vocab_size)
            ctc_outputs = self.ctc_forward_step(enc_states.transpose(0, 1))
            ctc_outputs = ctc_outputs.transpose(0, 1).float().detach()
            # if ctc_outputs.dtype==torch.float16 :
            #     ctc_outputs = ctc_outputs.transpose(0, 1).float().detach()

            # self.blank_collapse = 0.999
            # self.blank_collapse = 0.99
            if self.blank_collapse != 1:
                if 'padding_mask' in encoder_out.keys():
                    ## in fairseq latest version, padding mask is renamed
                    if encoder_out['padding_mask'] is None:
                        output_lengths = [encoder_out['encoder_out'].size(0)] * encoder_out['encoder_out'].size(1)
                        padding_mask = (torch.zeros(encoder_out['encoder_out'].size(1), encoder_out['encoder_out'].size(0)).bool()).cuda()
                    else:
                        ## if all input sequences have same length, then output has no padding mask 
                        output_lengths = (~encoder_out['padding_mask']).sum(-1).tolist()
                        padding_mask = encoder_out['padding_mask']

                ctc_probs = torch.nn.functional.softmax(ctc_outputs, dim=-1)

                blank_probs = ctc_probs[:,:,self.blank_index]
                blank_probs = blank_probs.masked_fill_(padding_mask, 0)

                # blank_collapse_ratio = self.blank_collapse
                # blank_collapse_ratio = blank_probs.max() * self.blank_collapse
                blank_collapse_ratio = (torch.max(blank_probs, dim=1)[0] * self.blank_collapse).unsqueeze(1).expand(blank_probs.size(0), blank_probs.size(1))
                # Tra()

                collapsed_ctc_probs, collapsed_enc_states, collapsed_output_lengths = self.collapse_blanks(
                    ctc_probs.float(), 
                    enc_states, 
                    output_lengths, 
                    blank_collapse_threshold = blank_collapse_ratio
                )
                
                enc_states = collapsed_enc_states
                enc_lens = torch.tensor(collapsed_output_lengths).type_as(enc_states).int()
                
                ctc_outputs = collapsed_ctc_probs
                ctc_outputs = torch.log(ctc_outputs)

            ctc_scorer = CTCPrefixScorer(
                ctc_outputs,
                enc_lens,
                batch_size,
                self.beam_size,
                self.blank_index,
                self.eos_index,
                self.ctc_window_size,
            )
            ctc_memory = None

        # Inflate the enc_states and enc_len by beam_size times
        enc_states = inflate_tensor(enc_states, times=self.beam_size, dim=0)
        enc_lens = inflate_tensor(enc_lens, times=self.beam_size, dim=0)
        
        # Tra()
        '''
        (Pdb) ctc_outputs.size(); enc_states.size()
        torch.Size([7, 1757, 10001])
        torch.Size([35, 1757, 1024])

        self.get_ctc_greedy_outs(ctc_outputs, self.blank_index)
        '''

        # Using bos as the first input
        inp_tokens = (
            torch.zeros(batch_size * self.beam_size, device=device)
            .fill_(self.bos_index)
            .long()
        )

        # The first index of each sentence.
        self.beam_offset = (
            torch.arange(batch_size, device=device) * self.beam_size
        )

        # initialize sequence scores variables.
        sequence_scores = torch.empty(
            batch_size * self.beam_size, device=device
        )
        sequence_scores.fill_(float("-inf"))

        # keep only the first to make sure no redundancy.
        sequence_scores.index_fill_(0, self.beam_offset, 0.0)

        # keep the hypothesis that reaches eos and their corresponding score and log_probs.
        hyps_and_scores = [[] for _ in range(batch_size)]

        # keep the sequences that still not reaches eos.
        alived_seq = torch.empty(
            batch_size * self.beam_size, 0, device=device
        ).long()

        # Keep the log-probabilities of alived sequences.
        alived_log_probs = torch.empty(
            batch_size * self.beam_size, 0, device=device
        )

        min_decode_steps = int(enc_states.shape[1] * self.min_decode_ratio)
        max_decode_steps = int(enc_states.shape[1] * self.max_decode_ratio)

        # Initialize the previous attention peak to zero
        # This variable will be used when using_max_attn_shift=True
        prev_attn_peak = torch.zeros(batch_size * self.beam_size, device=device)


        duplicated_encoder_out = dict() 
        for k,v in encoder_out.items():
            if k == 'padding_mask':
                if v is not None:
                    duplicated_encoder_out[k] = inflate_tensor(v, times=self.beam_size, dim=0)
                else:
                    duplicated_encoder_out[k] = None
            elif k == 'layer_results': 
                tmp = []
                for item in v : 
                    tmp.append((inflate_tensor(item[0], times=self.beam_size, dim=1),))
                duplicated_encoder_out[k] = tmp
            else :
                duplicated_encoder_out[k] = inflate_tensor(v, times=self.beam_size, dim=1)
        encoder_out = duplicated_encoder_out


        with torch.no_grad():
            for t in range(max_decode_steps):
                # terminate condition
                if self._check_full_beams(hyps_and_scores, self.beam_size):
                    break

                log_probs, memory, cross_attn = self.forward_step(
                    inp_tokens, memory, encoder_out, enc_lens
                )
                log_probs = self.att_weight * log_probs

                # last time-step attention.
                # this is for coverage penalty, attn shift, and ctc window size
                attn = cross_attn[:, -1, :]

                '''
                (Pdb) t; log_probs.size(); memory.size(); cross_attn.size()
                0
                torch.Size([528, 5000])
                torch.Size([528, 1])
                torch.Size([528, 1, 107])

                (Pdb) t; log_probs.size(); memory.size(); cross_attn.size()
                1
                torch.Size([528, 5000])
                torch.Size([528, 2])
                torch.Size([528, 2, 107])
                '''

                # Keep the original value
                log_probs_clone = log_probs.clone().reshape(batch_size, -1)
                vocab_size = log_probs.shape[-1]

                # Block the candidates that exceed the max shift
                # * For example if previous peak attention time-step is 100, then if current peak attention is not in 20~120, then it does not allowed to be shifted (?)
                if self.using_max_attn_shift:
                    cond, attn_peak = self._check_attn_shift(attn, prev_attn_peak)
                    log_probs = mask_by_condition(
                        log_probs, cond, fill_value=self.minus_inf
                    )
                    prev_attn_peak = attn_peak

                # Set eos to minus_inf when less than minimum steps.
                if t < min_decode_steps:
                    log_probs[:, self.eos_index] = self.minus_inf

                # Set the eos prob to minus_inf when it doesn't exceed threshold.
                # * For example, if eos probability of current time step is smaller than (1.5(threshold) * max log probs), make eos prob is 0
                # * it makes model generate longer sentences 
                if self.using_eos_threshold:
                    cond = self._check_eos_threshold(log_probs)
                    log_probs[:, self.eos_index] = mask_by_condition(
                        log_probs[:, self.eos_index], cond, fill_value=self.minus_inf,
                    )

                # adding LM scores to log_prob if lm_weight > 0
                if self.lm_weight > 0:
                    lm_log_probs, lm_memory = self.lm_forward_step(
                        inp_tokens, lm_memory
                    )
                    log_probs = log_probs + self.lm_weight * lm_log_probs

                    # ILME
                    if self.internal_lm_estimation and self.internal_lm_weight > 0: 
                        internal_lm_log_probs, _ = self.decoder_only_forward_step(inp_tokens, lm_memory)
                        log_probs = log_probs - self.internal_lm_weight * internal_lm_log_probs

                '''
                (Pdb) self.beam_size; inp_tokens.size(); lm_log_probs.size();
                2
                torch.Size([14])
                torch.Size([14, 10001])
                '''

                # adding CTC scores to log_prob if ctc_weight > 0
                if self.ctc_weight > 0:
                    g = alived_seq
                    # block blank token
                    log_probs[:, self.blank_index] = self.minus_inf
                    if self.ctc_weight != 1.0 and self.ctc_score_mode == "partial":
                        # pruning vocab for ctc_scorer
                        _, ctc_candidates = log_probs.topk(
                            self.beam_size * 2, dim=-1
                        )
                    else:
                        ctc_candidates = None

                    # Compute the ctc scores over the time frames using windowing based on attention peaks.
                    # If 0, no windowing applied.
                    ctc_log_probs, ctc_memory = ctc_scorer.forward_step(
                        g, ctc_memory, ctc_candidates, attn
                    )
                    log_probs = log_probs + self.ctc_weight * ctc_log_probs

                # Ngram Block from fairseq
                if self.repeat_ngram_blocker is not None:
                    # Tra()
                    '''
                    (Pdb) memory.size(); log_probs.size(); batch_size; self.beam_size; t;
                    torch.Size([80, 1])
                    torch.Size([80, 513])
                    16
                    5
                    0
                    '''
                    log_probs = self.repeat_ngram_blocker(memory, log_probs, batch_size, self.beam_size, t)
                    
                scores = sequence_scores.unsqueeze(1).expand(-1, vocab_size)
                scores = scores + log_probs

                # length normalization
                # * this is not equal to length rewarding
                # score is log level scalar.
                if self.length_normalization:
                    scores = scores / (t + 1)

                # keep topk beams
                scores, candidates = scores.view(batch_size, -1).topk(
                    self.beam_size, dim=-1
                )

                # The input for the next step, also the output of current step.
                inp_tokens = (candidates % vocab_size).view(
                    batch_size * self.beam_size
                )

                scores = scores.view(batch_size * self.beam_size)
                sequence_scores = scores

                # recover the length normalization
                if self.length_normalization:
                    sequence_scores = sequence_scores * (t + 1)

                # The index of which beam the current top-K output came from in (t-1) timesteps.
                predecessors = (
                    torch.div(candidates, vocab_size, rounding_mode="floor")
                    + self.beam_offset.unsqueeze(1).expand_as(candidates)
                ).view(batch_size * self.beam_size)

                # Permute the memory to synchoronize with the output.
                memory = self.permute_mem(memory, index=predecessors)
                if self.lm_weight > 0:
                    lm_memory = self.permute_lm_mem(lm_memory, index=predecessors)
                if self.ctc_weight > 0:
                    ctc_memory = ctc_scorer.permute_mem(ctc_memory, candidates)

                # If using_max_attn_shift, then the previous attn peak has to be permuted too.
                if self.using_max_attn_shift:
                    prev_attn_peak = torch.index_select(
                        prev_attn_peak, dim=0, index=predecessors
                    )

                # Tra()
                '''
                (Pdb) self.coverage.size(); attn.size(); cur_attn.size()                                                                      
                torch.Size([16, 642])
                torch.Size([16, 642])
                torch.Size([16, 642])
                '''

                # Add coverage penalty
                # coverage: cumulative attention probability vector
                '''
                Repeats: Copy models often repeatedly attend to the same source tokens, 
                generating the same phrase multiple times. We introduce a new sum- mary specific coverage penalty
                '''
                if self.coverage_penalty > 0:
                    # attn = cross_attn
                    cur_attn = torch.index_select(attn, dim=0, index=predecessors)
                    '''
                    (Pdb) attn.size(); predecessors.size()
                    torch.Size([16, 642])
                    torch.Size([16])
                    '''

                    # Init coverage
                    if t == 0:
                        self.coverage = cur_attn

                    # Update coverage
                    # the attn of transformer is [batch_size*beam_size, current_step, source_len] (?)
                    if len(cur_attn.size()) > 2:
                        self.coverage = torch.sum(cur_attn, dim=1)
                    else:
                        self.coverage = torch.index_select(self.coverage, dim=0, index=predecessors)
                        self.coverage = self.coverage + cur_attn

                    # Compute coverage penalty and add it to scores
                    penalty = torch.max(self.coverage, self.coverage.clone().fill_(0.5)).sum(-1)
                    penalty = penalty - self.coverage.size(-1) * 0.5
                    penalty = penalty.view(batch_size * self.beam_size)
                    penalty = (penalty / (t + 1) if self.length_normalization else penalty)

                    # why coverage penalty have opposite direction to score?
                    scores = scores + penalty * self.coverage_penalty
                    # scores = scores - penalty * self.coverage_penalty

                # Update alived_seq
                alived_seq = torch.cat(
                    [
                        torch.index_select(alived_seq, dim=0, index=predecessors),
                        inp_tokens.unsqueeze(1),
                    ],
                    dim=-1,
                )

                # Takes the log-probabilities
                beam_log_probs = log_probs_clone[
                    torch.arange(batch_size).unsqueeze(1), candidates
                ].reshape(batch_size * self.beam_size)
                alived_log_probs = torch.cat(
                    [
                        torch.index_select(
                            alived_log_probs, dim=0, index=predecessors
                        ),
                        beam_log_probs.unsqueeze(1),
                    ],
                    dim=-1,
                )

                is_eos = self._update_hyp_and_scores(
                    inp_tokens,
                    alived_seq,
                    alived_log_probs,
                    hyps_and_scores,
                    scores,
                    timesteps=t,
                )

                # Block the paths that have reached eos.
                sequence_scores.masked_fill_(is_eos, float("-inf"))

            if not self._check_full_beams(hyps_and_scores, self.beam_size):
                # Using all eos to fill-up the hyps.
                eos = (
                    torch.zeros(batch_size * self.beam_size, device=device)
                    .fill_(self.eos_index)
                    .long()
                )
                _ = self._update_hyp_and_scores(
                    eos,
                    alived_seq,
                    alived_log_probs,
                    hyps_and_scores,
                    scores,
                    timesteps=max_decode_steps,
                )

            (
                topk_hyps,
                topk_scores,
                topk_lengths,
                log_probs,
                indices,
            ) = self._get_top_score_prediction(hyps_and_scores, topk=self.topk, return_indices=True)
            # pick the best hyp

            topk_attns = [cross_attn[index.item()] for index in indices]
            predictions = topk_hyps[:, 0, :]
            predictions = batch_filter_seq2seq_output(predictions, eos_id=self.eos_index)

        if self.return_log_probs:
            return predictions, topk_scores, topk_attns, log_probs
        else:
            return predictions, topk_scores, topk_attns

    def ctc_forward_step(self, x):
        """Applies a ctc step during bramsearch."""
        logits = self.ctc_fc(x)
        log_probs = self.softmax(logits/self.temperature_ctc)

        '''
        logits = self.ctc_fc(x)
        log_probs = self.softmax(logits)
        ctc_outputs = log_probs.transpose(0, 1)

        prob = torch.exp(ctc_outputs)
        log_prob = torch.nn.functional.log_softmax(ctc_outputs, -1)
        ent_ = -prob * (log_prob)
        ent = ent_.sum(-1).sum()

        logits = self.ctc_fc(x)
        log_probs = self.softmax(logits/0.1)
        ctc_outputs = log_probs.transpose(0, 1)

        prob = torch.nn.functional.softmax(ctc_outputs, -1)
        log_prob = torch.nn.functional.log_softmax(ctc_outputs, -1)
        ent_ = -prob * (log_prob)
        ent2 = ent_.sum(-1).sum()

        tensor(236.3943, device='cuda:0')

        kld 
        tensor(31841.0840, device='cuda:0')
        '''

        return log_probs

    def forward_step(self, inp_tokens, memory, encoder_out, enc_lens):
        """Performs a step in the implemented beamsearcher."""

        # Tra()
        with torch.no_grad():
            # memory = _update_mem(inp_tokens, memory)
            # pred, attn = self.model.decode(memory, enc_states)
            # prob_dist = self.softmax(self.fc(pred) / self.temperature)

            memory = _update_mem(inp_tokens, memory)
            # result = self.model.decoder(prev_output_tokens = memory, encoder_out = encoder_out)
            result = self.attention_decoder(prev_output_tokens = memory, encoder_out = encoder_out)
            prob_dist = self.softmax(result[0] / self.temperature)
            attn = result[1]['attn']

        return prob_dist[:, -1, :], memory, attn

    def decoder_only_forward_step(self, inp_tokens, memory):

        with torch.no_grad():
            memory = _update_mem(inp_tokens, memory)
            result = self.attention_decoder(prev_output_tokens = memory, encoder_out = None)
            prob_dist = self.softmax(result[0] / self.temperature_lm)

        return prob_dist[:, -1, :], memory

    def lm_forward_step(self, inp_tokens, memory):
        """Performs a step in the implemented LM module."""

        with torch.no_grad():
            # memory = _update_mem(inp_tokens, memory)
            # if not next(self.lm_modules.parameters()).is_cuda:
            #     self.lm_modules.to(inp_tokens.device)
            # logits = self.lm_modules(memory)
            # log_probs = self.softmax(logits / self.temperature_lm)

            memory = _update_mem(inp_tokens, memory)
            if not next(self.lm_modules.parameters()).is_cuda:
                self.lm_modules.to(inp_tokens.device)
            logits = self.lm_modules(memory)

            # adaptive_softmax_layer = False
            # for k, p in self.lm_modules.named_parameters(): 
            #     if 'adaptive' in k : 
            #         adaptive_softmax_layer = True
                    
            try:
                log_probs = self.lm_modules.adaptive_softmax.get_log_prob(logits[0] / self.temperature_lm, None)
            except:
                log_probs = self.softmax(logits[0] / self.temperature_lm)

        return log_probs[:, -1, :], memory

    def get_ctc_greedy_outs(self, emissions, blank_id):
        def get_pred(e):
            toks = e.argmax(dim=-1).unique_consecutive()
            return toks[toks != blank_id]
        greedy_outs = [get_pred(x).cpu().detach().tolist() for x in emissions]
        return greedy_outs

    def load_ngram(self, path):
        import kenlm
        ngram = kenlm.LanguageModel(path)
        return ngram


    #### Blank Collapse

    def collapse_blanks(self, emissions, encoder_outs, output_lengths, blank_collapse_threshold, use_logit=False):
        from fairseq import utils

        if use_logit:
            blanks = (utils.softmax(emissions.transpose(0, 1), dim=-1)).transpose(0, 1)[:, :, self.blank_index] > blank_collapse_threshold
            collapsed_emissions = torch.ones_like(emissions).type_as(emissions) * float("-inf")
            collapsed_emissions[:,:,self.blank_index] = 0 # because we use logit, it should be 0
        else:
            blanks = emissions[:, :, self.blank_index] > blank_collapse_threshold
            collapsed_emissions = torch.zeros_like(emissions).type_as(emissions)
            collapsed_emissions[:,:,self.blank_index] = 1 # because we use exact prob, dummy timestep should get blank prob as 1 

            collapsed_encoder_outs = torch.zeros_like(encoder_outs).type_as(encoder_outs)

        for i in range(emissions.size(0)):
            u, c = torch.unique_consecutive(blanks[i], dim=0, return_counts=True)
            u = u.tolist()
            c = c.tolist()
            cc = []
            first_blanks = 0
            k = 0
            for j in range(len(c)):
                c[j] = min(c[j], output_lengths[i] - k)
                if u[j]:    # if blank
                    if j == 0:
                        first_blanks = c[j]
                    else:
                        if j < len(c) - 1:
                            cc.append(c[j])
                else:
                    cc += [1] * c[j]
                k += c[j]
                if k >= output_lengths[i]:
                    break
            if len(cc) == 0:    # case: every frame is a blank frame
                cc = [0]
                first_blanks = 0

            org_index = torch.cumsum(torch.tensor(cc), dim=0) + first_blanks - 1

            emission = emissions[i, org_index]
            encoder_out = encoder_outs[i, org_index]
            
            collapsed_emissions[i, :len(emission)] = emission
            collapsed_encoder_outs[i, :len(encoder_out)] = encoder_out

            output_lengths[i] = len(emission)

        collapsed_emissions = collapsed_emissions[:, :max(output_lengths)]
        collapsed_encoder_outs = collapsed_encoder_outs[:, :max(output_lengths)]

        return collapsed_emissions, collapsed_encoder_outs, output_lengths



def batch_filter_seq2seq_output(prediction, eos_id=-1):
    """Calling batch_size times of filter_seq2seq_output.

    Arguments
    ---------
    prediction : list of torch.Tensor
        A list containing the output ints predicted by the seq2seq system.
    eos_id : int, string
        The id of the eos.

    Returns
    ------
    list
        The output predicted by seq2seq model.

    Example
    -------
    >>> predictions = [torch.IntTensor([1,2,3,4]), torch.IntTensor([2,3,4,5,6])]
    >>> predictions = batch_filter_seq2seq_output(predictions, eos_id=4)
    >>> predictions
    [[1, 2, 3], [2, 3]]
    """
    # Tra()
    outputs = []
    for p in prediction:
        res = filter_seq2seq_output(p.tolist(), eos_id=eos_id)
        outputs.append(res)
    return outputs


def filter_seq2seq_output(string_pred, eos_id=-1):
    """Filter the output until the first eos occurs (exclusive).

    Arguments
    ---------
    string_pred : list
        A list containing the output strings/ints predicted by the seq2seq system.
    eos_id : int, string
        The id of the eos.

    Returns
    ------
    list
        The output predicted by seq2seq model.

    Example
    -------
    >>> string_pred = ['a','b','c','d','eos','e']
    >>> string_out = filter_seq2seq_output(string_pred, eos_id='eos')
    >>> string_out
    ['a', 'b', 'c', 'd']
    """
    if isinstance(string_pred, list):
        try:
            eos_index = next(
                i for i, v in enumerate(string_pred) if v == eos_id
            )
        except StopIteration:
            eos_index = len(string_pred)
        string_out = string_pred[:eos_index]
    else:
        raise ValueError("The input must be a list.")
    return string_out





#### Ngram Blocker from fairseq
# Originally from Microsoft Corporation.
# Licensed under the MIT License.

'''
https://github.com/microsoft/fastseq/blob/main/fastseq/clib/cuda/ngram_repeat_block_cuda.cpp
# pip install git+https://github.com/microsoft/fastseq.git

https://github.com/facebookresearch/fairseq/blob/main/fairseq/ngram_repeat_block.py
https://github.com/facebookresearch/fairseq/blob/main/fairseq/sequence_generator.py#L421
'''

""" Wrapper for ngram_repeat_block cuda extension """
import math
import warnings
from typing import List

import torch
from torch import nn

try:
    from fairseq import ngram_repeat_block_cuda

    EXTENSION_BUILT = True
except ImportError:
    EXTENSION_BUILT = False


def is_cuda_extension_usable() -> bool:
    """Check whether ngram_repeat_block_cuda is built properly"""
    if not EXTENSION_BUILT or not torch.cuda.is_available():
        return False
    bsz = 2
    tokens = torch.tensor([[4, 4, 3, 2], [1, 2, 3, 4]], dtype=torch.long, device="cuda")
    lprobs = torch.rand((8, 12), device="cuda")
    try:
        outputs = ngram_repeat_block_cuda.forward(tokens, lprobs, bsz, 3, 4, 3)
        outputs = outputs + 4  # This line breaks if the extension is built incorrectly.
        return True
    except RuntimeError:
        warnings.warn(
            "NGramRepeatBlock extension must be rebuilt."
            'Run TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0" python setup.py build_ext --inplace'
        )
        return False


class NGramRepeatBlock(nn.Module):
    """Wrapper class for calling ngram_repeat_block cuda extension"""

    def __init__(self, no_repeat_ngram_size: int, use_extension: bool = True):
        super().__init__()
        self.use_extension = is_cuda_extension_usable() if use_extension else False
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def reset_parameters(self):
        pass

    @torch.jit.unused
    def call_cuda_extension(
        self,
        tokens,
        lprobs,
        bsz: int,
        beam_size: int,
        step: int,
    ):
        return ngram_repeat_block_cuda.forward(
            tokens, lprobs, bsz, step, beam_size, self.no_repeat_ngram_size
        )

    def forward(
        self,
        tokens,
        lprobs,
        bsz: int,
        beam_size: int,
        step: int,
    ):
        """
        Args:
            tokens(Tensor): Input tokens(Bsz*beam, seq_len)
            lprobs(Tensor): likelihood probability,
            Expected to be updated in place.(Bsz*beam, vocab_size)
            bsz(int): batch size
            step(int): current step
            beam_size(int): beam size
            no_repeat_ngram_size(int): Ngram size
        """
        msg = f"expected {bsz *beam_size} got"
        assert tokens.size(0) == bsz * beam_size, f"{msg} {tokens.size(0)}"
        assert lprobs.size(0) == bsz * beam_size, f"{msg} {lprobs.size(0)}"
        if self.use_extension:
            return self.call_cuda_extension(tokens, lprobs, bsz, beam_size, step)

        else:
            return self._no_repeat_ngram(
                tokens,
                lprobs,
                bsz,
                beam_size,
                step,
            )

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int, step: int):
        """For each hypothesis generate a list of previous ngrams and set associated lprobs to -inf"""

        banned_tokens = [
            torch.jit.annotate(List[int], []) for bbsz_idx in range(bsz * beam_size)
        ]
        if step + 2 - self.no_repeat_ngram_size >= 0:
            cpu_tokens: List[List[int]] = tokens.cpu().tolist()
            check_start_pos = step + 2 - self.no_repeat_ngram_size
            for bbsz_idx in range(bsz * beam_size):
                ngram_to_check = cpu_tokens[bbsz_idx][
                    -(self.no_repeat_ngram_size - 1) :
                ]
                for i in range(check_start_pos):
                    if (
                        ngram_to_check
                        == cpu_tokens[bbsz_idx][i : i + self.no_repeat_ngram_size - 1]
                    ):
                        banned_tokens[bbsz_idx].append(
                            cpu_tokens[bbsz_idx][i + self.no_repeat_ngram_size - 1]
                        )
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][
                torch.tensor(banned_tokens[bbsz_idx], dtype=torch.int64)
            ] = torch.tensor(-math.inf).to(lprobs)
        return lprobs
