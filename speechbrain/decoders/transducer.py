"""Decoders and output normalization for Transducer sequence.

Author:
    Abdelwahab HEBA 2020
    Sung-Lin Yeh 2020
"""
import torch

from pdb import set_trace as Tra


class TransducerBeamSearcher(torch.nn.Module):
    """
    This class implements the beam-search algorithm for the transducer model.

    Parameters
    ----------
    decode_network_lst : list
        List of prediction network (PN) layers.
    tjoint: transducer_joint module
        This module perform the joint between TN and PN.
    classifier_network : list
        List of output layers (after performing joint between TN and PN)
        exp: (TN,PN) => joint => classifier_network_list [DNN bloc, Linear..] => chars prob
    blank_id : int
        The blank symbol/index.
    beam : int
        The width of beam. Greedy Search is used when beam = 1.
    nbest : int
        Number of hypotheses to keep.
    lm_module : torch.nn.ModuleList
        Neural networks modules for LM.
    lm_weight : float
        The weight of LM when performing beam search (λ).
        log P(y|x) + λ log P_LM(y). (default: 0.3)
    state_beam : float
        The threshold coefficient in log space to decide if hyps in A (process_hyps)
        is likely to compete with hyps in B (beam_hyps), if not, end the while loop.
        Reference: https://arxiv.org/pdf/1911.01629.pdf
    expand_beam : float
        The threshold coefficient to limit the number of expanded hypotheses
        that are added in A (process_hyp).
        Reference: https://arxiv.org/pdf/1911.01629.pdf
        Reference: https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/simple-decoder.cc (See PruneToks)

    Example
    -------
    searcher = TransducerBeamSearcher(
        decode_network_lst=[hparams["emb"], hparams["dec"]],
        tjoint=hparams["Tjoint"],
        classifier_network=[hparams["transducer_lin"]],
        blank_id=0,
        beam_size=hparams["beam_size"],
        nbest=hparams["nbest"],
        lm_module=hparams["lm_model"],
        lm_weight=hparams["lm_weight"],
        state_beam=2.3,
        expand_beam=2.3,
    )
    >>> from speechbrain.nnet.transducer.transducer_joint import Transducer_joint
    >>> import speechbrain as sb
    >>> emb = sb.nnet.embedding.Embedding(
    ...     num_embeddings=35,
    ...     embedding_dim=3,
    ...     consider_as_one_hot=True,
    ...     blank_id=0
    ... )
    >>> dec = sb.nnet.RNN.GRU(
    ...     hidden_size=10, input_shape=(1, 40, 34), bidirectional=False
    ... )
    >>> lin = sb.nnet.linear.Linear(input_shape=(1, 40, 10), n_neurons=35)
    >>> joint_network= sb.nnet.linear.Linear(input_shape=(1, 1, 40, 35), n_neurons=35)
    >>> tjoint = Transducer_joint(joint_network, joint="sum")
    >>> searcher = TransducerBeamSearcher(
    ...     decode_network_lst=[emb, dec],
    ...     tjoint=tjoint,
    ...     classifier_network=[lin],
    ...     blank_id=0,
    ...     beam_size=1,
    ...     nbest=1,
    ...     lm_module=None,
    ...     lm_weight=0.0,
    ... )
    >>> enc = torch.rand([1, 20, 10])
    >>> hyps, scores, _, _ = searcher(enc)
    """

    def __init__(
        self,
        decode_network_lst,
        tjoint,
        classifier_network,
        blank_id,
        beam_size=4,
        nbest=5,
        lm_module=None,
        lm_weight=0.0,
        state_beam=2.3,
        expand_beam=2.3,
    ):
        super(TransducerBeamSearcher, self).__init__()
        self.decode_network_lst = decode_network_lst
        self.tjoint = tjoint
        self.classifier_network = classifier_network
        self.blank_id = blank_id
        self.beam_size = beam_size
        self.nbest = nbest
        self.lm = lm_module
        self.lm_weight = lm_weight

        if lm_module is None and lm_weight > 0:
            raise ValueError("Language model is not provided.")

        self.state_beam = state_beam
        self.expand_beam = expand_beam
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        if self.beam_size <= 1:
            self.searcher = self.transducer_greedy_decode
        else:
            self.searcher = self.transducer_beam_search_decode

    def forward(self, tn_output):
        """
        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        Topk hypotheses
        """

        hyps = self.searcher(tn_output)
        return hyps

    def transducer_greedy_decode(self, tn_output):
        """Transducer greedy decoder is a greedy decoder over batch which apply Transducer rules:
            1- for each time step in the Transcription Network (TN) output:
                -> Update the ith utterance only if
                    the previous target != the new one (we save the hiddens and the target)
                -> otherwise:
                ---> keep the previous target prediction from the decoder

        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.
        """
        hyp = {
            "prediction": [[] for _ in range(tn_output.size(0))],
            "logp_scores": [0.0 for _ in range(tn_output.size(0))],
        }
        # prepare BOS = Blank for the Prediction Network (PN)
        hidden = None
        input_PN = (
            torch.ones(
                (tn_output.size(0), 1),
                device=tn_output.device,
                dtype=torch.int32,
            )
            * self.blank_id
        )

        # First forward-pass on PN
        out_PN, hidden = self._forward_PN(input_PN, self.decode_network_lst)

        # For each time step
        for t_step in range(tn_output.size(1)):
            # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
            log_probs = self._joint_forward_step(
                tn_output[:, t_step, :].unsqueeze(1).unsqueeze(1),
                out_PN.unsqueeze(1),
            )
            # Sort outputs at time
            logp_targets, positions = torch.max(
                self.softmax(log_probs).squeeze(1).squeeze(1), dim=1
            )
            # Batch hidden update
            have_update_hyp = []
            for i in range(positions.size(0)):
                # Update hiddens only if
                # 1- current prediction is non blank
                if positions[i].item() != self.blank_id:
                    hyp["prediction"][i].append(positions[i].item())
                    hyp["logp_scores"][i] += logp_targets[i]
                    input_PN[i][0] = positions[i]
                    have_update_hyp.append(i)
            if len(have_update_hyp) > 0:
                # Select sentence to update
                # And do a forward steps + generated hidden
                (
                    selected_input_PN,
                    selected_hidden,
                ) = self._get_sentence_to_update(
                    have_update_hyp, input_PN, hidden
                )
                selected_out_PN, selected_hidden = self._forward_PN(
                    selected_input_PN, self.decode_network_lst, selected_hidden
                )
                # update hiddens and out_PN
                out_PN[have_update_hyp] = selected_out_PN
                hidden = self._update_hiddens(
                    have_update_hyp, selected_hidden, hidden
                )

        return (
            hyp["prediction"],
            torch.Tensor(hyp["logp_scores"]).exp().mean(),
            None,
            None,
        )

    def transducer_beam_search_decode(self, tn_output):
        """Transducer beam search decoder is a beam search decoder over batch which apply Transducer rules:
            1- for each utterance:
                2- for each time steps in the Transcription Network (TN) output:
                    -> Do forward on PN and Joint network
                    -> Select topK <= beam
                    -> Do a while loop extending the hyps until we reach blank
                        -> otherwise:
                        --> extend hyp by the new token

        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.
        """

        # min between beam and max_target_lent
        nbest_batch = []
        nbest_batch_score = []
        for i_batch in range(tn_output.size(0)):
            # if we use RNN LM keep there hiddens
            # prepare BOS = Blank for the Prediction Network (PN)
            # Prepare Blank prediction
            blank = (
                torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                * self.blank_id
            )
            input_PN = (
                torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                * self.blank_id
            )
            # First forward-pass on PN
            hyp = {
                "prediction": [self.blank_id],
                "logp_score": 0.0,
                "hidden_dec": None,
            }
            if self.lm_weight > 0:
                lm_dict = {"hidden_lm": None}
                hyp.update(lm_dict)
            beam_hyps = [hyp]

            # For each time step
            for t_step in range(tn_output.size(1)):
                # get hyps for extension
                process_hyps = beam_hyps
                beam_hyps = []
                while True:
                    if len(beam_hyps) >= self.beam_size:
                        break
                    # Add norm score
                    a_best_hyp = max(
                        process_hyps,
                        key=lambda x: x["logp_score"] / len(x["prediction"]),
                    )

                    # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                    if len(beam_hyps) > 0:
                        b_best_hyp = max(
                            beam_hyps,
                            key=lambda x: x["logp_score"]
                            / len(x["prediction"]),
                        )
                        a_best_prob = a_best_hyp["logp_score"]
                        b_best_prob = b_best_hyp["logp_score"]
                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    # remove best hyp from process_hyps
                    process_hyps.remove(a_best_hyp)

                    # forward PN
                    input_PN[0, 0] = a_best_hyp["prediction"][-1]
                    out_PN, hidden = self._forward_PN(
                        input_PN,
                        self.decode_network_lst,
                        a_best_hyp["hidden_dec"],
                    )
                    # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
                    log_probs = self._joint_forward_step(
                        tn_output[i_batch, t_step, :]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                        out_PN.unsqueeze(0),
                    )

                    if self.lm_weight > 0:
                        log_probs_lm, hidden_lm = self._lm_forward_step(
                            input_PN, a_best_hyp["hidden_lm"]
                        )

                    # Sort outputs at time
                    logp_targets, positions = torch.topk(
                        log_probs.view(-1), k=self.beam_size, dim=-1
                    )
                    best_logp = (
                        logp_targets[0]
                        if positions[0] != blank
                        else logp_targets[1]
                    )

                    # Extend hyp by  selection
                    for j in range(logp_targets.size(0)):

                        # hyp
                        topk_hyp = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"]
                            + logp_targets[j],
                            "hidden_dec": a_best_hyp["hidden_dec"],
                        }

                        if positions[j] == self.blank_id:
                            beam_hyps.append(topk_hyp)
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = a_best_hyp["hidden_lm"]
                            continue

                        if logp_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp["prediction"].append(positions[j].item())
                            topk_hyp["hidden_dec"] = hidden
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = hidden_lm
                                topk_hyp["logp_score"] += (
                                    self.lm_weight
                                    * log_probs_lm[0, 0, positions[j]]
                                )
                            process_hyps.append(topk_hyp)
            # Add norm score
            nbest_hyps = sorted(
                beam_hyps,
                key=lambda x: x["logp_score"] / len(x["prediction"]),
                reverse=True,
            )[: self.nbest]
            all_predictions = []
            all_scores = []
            for hyp in nbest_hyps:
                all_predictions.append(hyp["prediction"][1:])
                all_scores.append(hyp["logp_score"] / len(hyp["prediction"]))
            nbest_batch.append(all_predictions)
            nbest_batch_score.append(all_scores)
        return (
            [nbest_utt[0] for nbest_utt in nbest_batch],
            torch.Tensor(
                [nbest_utt_score[0] for nbest_utt_score in nbest_batch_score]
            )
            .exp()
            .mean(),
            nbest_batch,
            nbest_batch_score,
        )

    def _joint_forward_step(self, h_i, out_PN):
        """Join predictions (TN & PN)."""

        with torch.no_grad():
            # the output would be a tensor of [B,T,U, oneof[sum,concat](Hidden_TN,Hidden_PN)]
            out = self.tjoint(h_i, out_PN,)
            # forward the output layers + activation + save logits
            out = self._forward_after_joint(out, self.classifier_network)
            log_probs = self.softmax(out)
        return log_probs

    def _lm_forward_step(self, inp_tokens, memory):
        """This method should implement one step of
        forwarding operation for language model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The memory variables input for this timestep.
            (e.g., RNN hidden states).

        Return
        ------
        log_probs : torch.Tensor
            Log-probabilities of the current timestep output.
        hs : No limit
            The memory variables are generated in this timestep.
            (e.g., RNN hidden states).
        """
        with torch.no_grad():
            logits, hs = self.lm(inp_tokens, hx=memory)
            log_probs = self.softmax(logits)
        return log_probs, hs

    def _get_sentence_to_update(self, selected_sentences, output_PN, hidden):
        """Select and return the updated hiddens and output
        from the Prediction Network.

        Arguments
        ----------
        selected_sentences : list
            List of updated sentences (indexes).
        output_PN: torch.tensor
            Output tensor from prediction network (PN).
        hidden : torch.tensor
            Optional: None, hidden tensor to be used for
            recurrent layers in the prediction network.

        Returns
        -------
        selected_output_PN: torch.tensor
            Outputs a logits tensor [B_selected,U, hiddens].
        hidden_update_hyp: torch.tensor
            Selected hiddens tensor.
        """

        selected_output_PN = output_PN[selected_sentences, :]
        # for LSTM hiddens (hn, hc)
        if isinstance(hidden, tuple):
            hidden0_update_hyp = hidden[0][:, selected_sentences, :]
            hidden1_update_hyp = hidden[1][:, selected_sentences, :]
            hidden_update_hyp = (hidden0_update_hyp, hidden1_update_hyp)
        else:
            hidden_update_hyp = hidden[:, selected_sentences, :]
        return selected_output_PN, hidden_update_hyp

    def _update_hiddens(self, selected_sentences, updated_hidden, hidden):
        """Update hidden tensor by a subset of hidden tensor (updated ones).

        Arguments
        ----------
        selected_sentences : list
            List of index to be updated.
        updated_hidden : torch.tensor
            Hidden tensor of the selected sentences for update.
        hidden : torch.tensor
            Hidden tensor to be updated.

        Returns
        -------
        torch.tensor
            Updated hidden tensor.
        """

        if isinstance(hidden, tuple):
            hidden[0][:, selected_sentences, :] = updated_hidden[0]
            hidden[1][:, selected_sentences, :] = updated_hidden[1]
        else:
            hidden[:, selected_sentences, :] = updated_hidden
        return hidden

    def _forward_PN(self, out_PN, decode_network_lst, hidden=None):
        """Compute forward-pass through a list of prediction network (PN) layers.

        Arguments
        ----------
        out_PN : torch.tensor
            Input sequence from prediction network with shape
            [batch, target_seq_lens].
        decode_network_lst: list
            List of prediction network (PN) layers.
        hinne : torch.tensor
            Optional: None, hidden tensor to be used for
                recurrent layers in the prediction network

        Returns
        -------
        out_PN : torch.tensor
            Outputs a logits tensor [B,U, hiddens].
        hidden : torch.tensor
            Hidden tensor to be used for the next step
            by recurrent layers in prediction network.
        """

        for layer in decode_network_lst:
            if layer.__class__.__name__ in [
                "RNN",
                "LSTM",
                "GRU",
                "LiGRU",
                "LiGRU_Layer",
            ]:
                out_PN, hidden = layer(out_PN, hidden)
            else:
                out_PN = layer(out_PN)
        return out_PN, hidden

    def _forward_after_joint(self, out, classifier_network):
        """Compute forward-pass through a list of classifier neural network.

        Arguments
        ----------
        out : torch.tensor
            Output from joint network with shape
            [batch, target_len, time_len, hiddens]
        classifier_network : list
            List of output layers (after performing joint between TN and PN)
            exp: (TN,PN) => joint => classifier_network_list [DNN bloc, Linear..] => chars prob

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B, U,T, Output_Dim];
        """

        for layer in classifier_network:
            out = layer(out)
        return out







class TransducerBeamSearcherforFairseq(torch.nn.Module):
    """
    This class implements the beam-search algorithm for the transducer model.

    Parameters
    ----------
    decode_network_lst : list
        List of prediction network (PN) layers.
    tjoint: transducer_joint module
        This module perform the joint between TN and PN.
    classifier_network : list
        List of output layers (after performing joint between TN and PN)
        exp: (TN,PN) => joint => classifier_network_list [DNN bloc, Linear..] => chars prob
    blank_id : int
        The blank symbol/index.
    beam : int
        The width of beam. Greedy Search is used when beam = 1.
    nbest : int
        Number of hypotheses to keep.
    lm_module : torch.nn.ModuleList
        Neural networks modules for LM.
    lm_weight : float
        The weight of LM when performing beam search (λ).
        log P(y|x) + λ log P_LM(y). (default: 0.3)
    state_beam : float
        The threshold coefficient in log space to decide if hyps in A (process_hyps)
        is likely to compete with hyps in B (beam_hyps), if not, end the while loop.
        Reference: https://arxiv.org/pdf/1911.01629.pdf
    expand_beam : float
        The threshold coefficient to limit the number of expanded hypotheses
        that are added in A (process_hyp).
        Reference: https://arxiv.org/pdf/1911.01629.pdf
        Reference: https://github.com/kaldi-asr/kaldi/blob/master/src/decoder/simple-decoder.cc (See PruneToks)

    """

    def __init__(
        self,
        prediction_network,
        joint_network,
        classifier_network=None,
        tgt_dict=None,
        initial_token_id=None,
        blank_id=-1,
        beam_size=4,
        nbest=5,
        lm_module=None,
        lm_weight=0.0,
        state_beam=2.3,
        expand_beam=2.3,
    ):
        super(TransducerBeamSearcherforFairseq, self).__init__()
        self.prediction_network = prediction_network
        self.joint_network = joint_network
        self.classifier_network = classifier_network

        self.tgt_dict = tgt_dict

        self.blank_id = (len(tgt_dict) + blank_id) if (blank_id < 0) else blank_id
        self.initial_token_id = self.blank_id if (initial_token_id is None) else initial_token_id

        self.beam_size = beam_size
        self.nbest = nbest
        self.lm = lm_module
        self.lm_weight = lm_weight

        if lm_module is None and lm_weight > 0:
            raise ValueError("Language model is not provided.")

        self.state_beam = state_beam
        self.expand_beam = expand_beam
        self.softmax = torch.nn.LogSoftmax(dim=-1)

        if self.beam_size <= 1:
            self.searcher = self.transducer_greedy_decode
        else:
            self.searcher = self.transducer_beam_search_decode

    def forward(self, tn_output):
        """
        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        Topk hypotheses
        """

        hyps = self.searcher(tn_output)
        return hyps

    def transducer_greedy_decode(self, tn_output):
        """Transducer greedy decoder is a greedy decoder over batch which apply Transducer rules:
            1- for each time step in the Transcription Network (TN) output:
                -> Update the ith utterance only if
                    the previous target != the new one (we save the hiddens and the target)
                -> otherwise:
                ---> keep the previous target prediction from the decoder

        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.
        """
        hyp = {
            "prediction": [[] for _ in range(tn_output.size(0))],
            "logp_scores": [0.0 for _ in range(tn_output.size(0))],
        }

        # prepare BOS = Blank for the Prediction Network (PN)
        hidden = None
        input_PN = (
            torch.ones(
                (tn_output.size(0), 1),
                device=tn_output.device,
                dtype=torch.int32,
            )
            * self.initial_token_id
        )

        # First forward-pass on PN
        out_PN, hidden = self._forward_PN(input_PN, self.prediction_network)

        # For each time step
        for t_step in range(tn_output.size(1)):
            # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
            output_tensor = self._joint_forward_step(tn_output[:, t_step:t_step+1, :], out_PN)
            log_probs = output_tensor.squeeze(1).squeeze(1)

            # Sort outputs at time
            logp_targets, positions = torch.max(log_probs, dim=-1)

            # Tra()

            # Batch hidden update
            have_update_hyp = []
            for i in range(positions.size(0)):
                # Update hiddens only if current prediction is non blank
                if (positions[i].item() != self.blank_id):
                    hyp["prediction"][i].append(positions[i].item())
                    hyp["logp_scores"][i] += logp_targets[i]
                    input_PN[i][0] = positions[i]
                    have_update_hyp.append(i)
                    
            if len(have_update_hyp) > 0:
                # Select sentence to update
                # And do a forward steps + generated hidden
                selected_input_PN, selected_hidden = self._get_sentence_to_update(have_update_hyp, input_PN, hidden)
                selected_out_PN, selected_hidden = self._forward_PN(selected_input_PN, self.prediction_network, selected_hidden)

                # update hiddens and out_PN
                out_PN[have_update_hyp] = selected_out_PN
                hidden = self._update_hiddens(have_update_hyp, selected_hidden, hidden)

        return (
            hyp["prediction"],
            torch.Tensor(hyp["logp_scores"]).exp().mean(),
            None,
            None,
        )

    def transducer_beam_search_decode(self, tn_output):
        """Transducer beam search decoder is a beam search decoder over batch which apply Transducer rules:
            1- for each utterance:
                2- for each time steps in the Transcription Network (TN) output:
                    -> Do forward on PN and Joint network
                    -> Select topK <= beam
                    -> Do a while loop extending the hyps until we reach blank
                        -> otherwise:
                        --> extend hyp by the new token

        Arguments
        ----------
        tn_output : torch.tensor
            Output from transcription network with shape
            [batch, time_len, hiddens].

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B,T,1,Output_Dim]; padding
            has not been removed.
        """

        # min between beam and max_target_lent
        nbest_batch = []
        nbest_batch_score = []
        
        for i_batch in range(tn_output.size(0)):
            # if we use RNN LM keep there hiddens
            # prepare BOS = Blank for the Prediction Network (PN)
            # Prepare Blank prediction
            blank = (
                torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                * self.blank_id
            )
            input_PN = (
                torch.ones((1, 1), device=tn_output.device, dtype=torch.int32)
                * self.initial_token_id
            )
            # First forward-pass on PN
            hyp = {
                "prediction": [self.initial_token_id],
                "logp_score": 0.0,
                "hidden_dec": None,
            }
            if self.lm_weight > 0:
                lm_dict = {"hidden_lm": None}
                hyp.update(lm_dict)
            beam_hyps = [hyp]

            # For each time step
            for t_step in range(tn_output.size(1)):
                # get hyps for extension
                process_hyps = beam_hyps
                beam_hyps = []
                while True:
                    if len(beam_hyps) >= self.beam_size:
                        break
                    # Add norm score
                    a_best_hyp = max(
                        process_hyps,
                        key=lambda x: x["logp_score"] / len(x["prediction"]),
                    )

                    # Break if best_hyp in A is worse by more than state_beam than best_hyp in B
                    if len(beam_hyps) > 0:
                        b_best_hyp = max(
                            beam_hyps,
                            key=lambda x: x["logp_score"]
                            / len(x["prediction"]),
                        )
                        a_best_prob = a_best_hyp["logp_score"]
                        b_best_prob = b_best_hyp["logp_score"]
                        if b_best_prob >= self.state_beam + a_best_prob:
                            break

                    # remove best hyp from process_hyps
                    process_hyps.remove(a_best_hyp)

                    # forward PN
                    input_PN[0, 0] = a_best_hyp["prediction"][-1]
                    out_PN, hidden = self._forward_PN(
                        input_PN,
                        self.prediction_network,
                        a_best_hyp["hidden_dec"],
                    )
                    # do unsqueeze over since tjoint must be have a 4 dim [B,T,U,Hidden]
                    log_probs = self._joint_forward_step(
                        tn_output[i_batch, t_step, :]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                        out_PN.unsqueeze(0),
                    )

                    if self.lm_weight > 0:
                        log_probs_lm, hidden_lm = self._lm_forward_step(
                            input_PN, a_best_hyp["hidden_lm"]
                        )

                    # Sort outputs at time
                    logp_targets, positions = torch.topk(
                        log_probs.view(-1), k=self.beam_size, dim=-1
                    )
                    best_logp = (
                        logp_targets[0]
                        if positions[0] != blank
                        else logp_targets[1]
                    )

                    # Extend hyp by  selection
                    for j in range(logp_targets.size(0)):

                        # hyp
                        topk_hyp = {
                            "prediction": a_best_hyp["prediction"][:],
                            "logp_score": a_best_hyp["logp_score"] + logp_targets[j],
                            "hidden_dec": a_best_hyp["hidden_dec"],
                        }

                        if positions[j] == self.blank_id:
                            beam_hyps.append(topk_hyp)
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = a_best_hyp["hidden_lm"]
                            continue

                        if logp_targets[j] >= best_logp - self.expand_beam:
                            topk_hyp["prediction"].append(positions[j].item())
                            topk_hyp["hidden_dec"] = hidden
                            if self.lm_weight > 0:
                                topk_hyp["hidden_lm"] = hidden_lm
                                topk_hyp["logp_score"] += (
                                    self.lm_weight
                                    * log_probs_lm[0, 0, positions[j]]
                                )
                            process_hyps.append(topk_hyp)
                    
                    # Tra()

            # Add norm score
            nbest_hyps = sorted(
                beam_hyps,
                key=lambda x: x["logp_score"] / len(x["prediction"]),
                reverse=True,
            )[: self.nbest]

            all_predictions = []
            all_scores = []
            for hyp in nbest_hyps:
                all_predictions.append(hyp["prediction"][1:])
                all_scores.append(hyp["logp_score"] / len(hyp["prediction"]))

            nbest_batch.append(all_predictions)
            nbest_batch_score.append(all_scores)

        return (
            [nbest_utt[0] for nbest_utt in nbest_batch],
            torch.Tensor(
                [nbest_utt_score[0] for nbest_utt_score in nbest_batch_score]
            )
            .exp()
            .mean(),
            nbest_batch,
            nbest_batch_score,
        )

    def _joint_forward_step(self, h_i, out_PN):
        """Join predictions (TN & PN)."""

        
        """Forward pass for training.
        
        This is from Torchaudio

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.
        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.

            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
        """
        with torch.no_grad():
            # the output would be a tensor of [B,T,U, oneof[sum,concat](Hidden_TN,Hidden_PN)]
            # out = self.joint_network(h_i, out_PN,)

            out, _, _ = self.joint_network(
                source_encodings=h_i,
                source_lengths=None,
                target_encodings=out_PN,
                target_lengths=None,
            )

            # Tra()
            '''
            (Pdb) h_i.size(); out_PN.size(); out.size(); h_i[0][0][:30]; out_PN[0][0][:30]; out[0][0][0][:30]
            torch.Size([8, 1, 1024])
            torch.Size([8, 1, 1024])
            torch.Size([8, 1, 1, 513])
            tensor([ 0.0281, -0.4155, -0.8979, -0.7803, -0.1710, -0.5488,  1.8115, -0.0714,
                    -0.4558,  0.9966,  1.6426, -0.4895,  0.1772, -0.5879, -0.3223, -0.1918,
                    -0.5562, -0.4204,  1.6602, -0.7261, -0.4802, -0.1880, -0.4919, -0.6587,
                    0.1181,  0.0558, -0.1954, -0.0572,  0.4495, -0.0365], device='cuda:0',
                dtype=torch.float16)
            tensor([ 0.5537, -0.1970, -1.1240, -1.0352, -0.1230, -0.4597,  2.7227, -0.9414,
                    -0.3413,  1.2334,  2.2168, -0.1897,  0.4932, -1.0869, -1.3115, -0.7983,
                    0.0731, -1.0967,  1.7705, -0.1763, -0.1885, -0.6865, -0.9629, -0.9478,
                    -0.3904, -0.5176, -0.1370, -0.6304, -1.2832, -1.1943], device='cuda:0',
                dtype=torch.float16, requires_grad=True)
            tensor([-184.3750, -184.5000, -184.2500, -171.0000, -169.1250, -169.7500,
                    -168.7500, -170.5000, -168.3750, -169.7500, -169.6250, -170.3750,
                    -169.8750, -170.5000, -170.1250, -169.5000, -168.6250, -170.2500,
                    -170.3750, -170.8750, -169.5000, -169.5000, -171.7500, -170.1250,
                    -169.1250, -170.0000, -168.8750, -171.1250, -169.6250, -170.6250],
                device='cuda:0', dtype=torch.float16)

            in torchaudio rnntdecoder

            (Pdb) enc_out.size(); predictor_out.size(); enc_out[0][0][:30]; predictor_out[0][0][:30]; joined_out.size(); joined_out[0][0][0][:30]
            torch.Size([1, 1, 1024])
            torch.Size([1, 1, 1024])
            tensor([ 0.0281, -0.4155, -0.8979, -0.7803, -0.1710, -0.5488,  1.8115, -0.0714,
                    -0.4558,  0.9966,  1.6426, -0.4895,  0.1772, -0.5879, -0.3223, -0.1918,
                    -0.5562, -0.4204,  1.6602, -0.7261, -0.4802, -0.1880, -0.4919, -0.6587,
                    0.1181,  0.0558, -0.1954, -0.0572,  0.4495, -0.0365], device='cuda:0',
                dtype=torch.float16)
            tensor([ 0.5537, -0.1970, -1.1240, -1.0352, -0.1230, -0.4597,  2.7227, -0.9414,
                    -0.3413,  1.2334,  2.2168, -0.1897,  0.4932, -1.0869, -1.3115, -0.7983,
                    0.0731, -1.0967,  1.7705, -0.1763, -0.1885, -0.6865, -0.9629, -0.9478,
                    -0.3904, -0.5176, -0.1370, -0.6304, -1.2832, -1.1943], device='cuda:0',
                dtype=torch.float16)
            torch.Size([1, 1, 1, 513])
            tensor([-184.3750, -184.5000, -184.2500, -171.0000, -169.1250, -169.7500,
                    -168.7500, -170.5000, -168.3750, -169.7500, -169.6250, -170.3750,
                    -169.8750, -170.5000, -170.1250, -169.5000, -168.6250, -170.2500,
                    -170.3750, -170.8750, -169.5000, -169.5000, -171.7500, -170.1250,
                    -169.1250, -170.0000, -168.8750, -171.1250, -169.6250, -170.6250],
                device='cuda:0', dtype=torch.float16, grad_fn=<SliceBackward0>)

            (Pdb) joined_out[:,0,0].size()
            torch.Size([1, 513])
            '''

            # forward the output layers + activation + save logits
            if self.classifier_network is not None:
                out = self._forward_after_joint(out, self.classifier_network)

            log_probs = self.softmax(out)

        return log_probs

    def _lm_forward_step(self, inp_tokens, memory):
        """This method should implement one step of
        forwarding operation for language model.

        Arguments
        ---------
        inp_tokens : torch.Tensor
            The input tensor of the current timestep.
        memory : No limit
            The memory variables input for this timestep.
            (e.g., RNN hidden states).

        Return
        ------
        log_probs : torch.Tensor
            Log-probabilities of the current timestep output.
        hs : No limit
            The memory variables are generated in this timestep.
            (e.g., RNN hidden states).
        """
        with torch.no_grad():
            logits, hs = self.lm(inp_tokens, hx=memory)
            log_probs = self.softmax(logits)
        return log_probs, hs

    def _get_sentence_to_update(self, selected_sentences, output_PN, hidden):
        """Select and return the updated hiddens and output
        from the Prediction Network.

        Arguments
        ----------
        selected_sentences : list
            List of updated sentences (indexes).
        output_PN: torch.tensor
            Output tensor from prediction network (PN).
        hidden : torch.tensor
            Optional: None, hidden tensor to be used for
            recurrent layers in the prediction network.

        Returns
        -------
        selected_output_PN: torch.tensor
            Outputs a logits tensor [B_selected,U, hiddens].
        hidden_update_hyp: torch.tensor
            Selected hiddens tensor.
        """

        selected_output_PN = output_PN[selected_sentences, :]

        hidden_update_hyp = []
        for i, layer_state in enumerate(hidden):
            h = layer_state[0][selected_sentences]
            c = layer_state[1][selected_sentences]
            hidden_update_hyp.append([h, c])

        return selected_output_PN, hidden_update_hyp

    def _update_hiddens(self, selected_sentences, updated_hidden, hidden):

        for i, layer_state in enumerate(hidden):
            hidden[i][0][selected_sentences] = updated_hidden[i][0]
            hidden[i][1][selected_sentences] = updated_hidden[i][1]

        return hidden

    def _forward_PN(self, out_PN, prediction_network, hidden=None):

        """Compute forward-pass through a list of prediction network (PN) layers.

        Returns
        -------
        out_PN : torch.tensor
            Outputs a logits tensor [B, U, hiddens].
        hidden : torch.tensor
            Hidden tensor to be used for the next step
            by recurrent layers in prediction network.
        """

        r"""Forward pass.
        B: batch size;
        U: maximum sequence length in batch;
        D: feature dimension of each input sequence element.
        Args:
            input (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)
        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape `(B, U, output_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``forward``.
        """

        out_PN, _, hidden = prediction_network(
            input=out_PN,
            state=hidden,
        )

        return out_PN, hidden

    def _forward_after_joint(self, out, classifier_network):
        """Compute forward-pass through a list of classifier neural network.

        Arguments
        ----------
        out : torch.tensor
            Output from joint network with shape
            [batch, target_len, time_len, hiddens]
        classifier_network : list
            List of output layers (after performing joint between TN and PN)
            exp: (TN,PN) => joint => classifier_network_list [DNN bloc, Linear..] => chars prob

        Returns
        -------
        torch.tensor
            Outputs a logits tensor [B, U, T, Output_Dim];
        """

        for layer in classifier_network:
            out = layer(out)
        return out
