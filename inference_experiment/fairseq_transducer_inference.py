import os

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from speechbrain.pretrained import EncoderDecoderASR

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data.data_utils import post_process

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

from pdb import set_trace as Tra

def main():


    #####################################################
    ######################## AM #########################
    #####################################################

    model_path = "/workspace/tmp_save_dir/en_pt_v4_221109_cfm_w2v2_1M_rnnt_joint_0.2_inter_0.1_freq_16_spm_lr_3e5_no_input_mask/checkpoint_best.pt"
    # model_path = "/workspace/tmp_save_dir/en_pt_v4_221128_cfm_w2v2_rnnt_r7_j2_i1_8M_freq_2_spm_lr_3e5_no_input_mask/checkpoint_best.pt"
    path, checkpoint = os.path.split(model_path)

    overrides = {
        "task": 'audio_finetuning',
        "data": path,
    }
    models, model_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(model_path, separator="\\"),
        arg_overrides=overrides,
        strict=True,
    )
    am_model = models[0]
    optimize_model(am_model, model_cfg)

    tgt_dict = task.target_dictionary
    blank_id = tgt_dict.index(task.blank_symbol)

    #####################################################
    ######################## LM #########################
    #####################################################

    lm_model_path = "/workspace/tmp_save_dir/general_512_en_vanilla_tfm_lm_12L/checkpoint_last.pt"
    path, checkpoint = os.path.split(lm_model_path)
    
    overrides = {
        "task": 'language_modeling',
        "data": path,
    }
    lm_models, lm_model_cfg, lm_task = checkpoint_utils.load_model_ensemble_and_task(
        utils.split_paths(lm_model_path, separator="\\"),
        arg_overrides=overrides,
        strict=True,
    )
    lm_model = lm_models[0]
    optimize_model(lm_model, lm_model_cfg)



    #####################################################
    ################### Pre Processor ###################
    #####################################################

    iterators = []
    audio_files=[]
    audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0030.flac')
    audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac')
    audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0007.flac')
    audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0000.flac')
    audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0003.flac')
    audio_files.append('./LibriSpeech/test-clean/1188/133604/1188-133604-0030.flac')
    audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0019.flac')
    audio_files.append('./LibriSpeech/test-clean/1188/133604/1188-133604-0006.flac')
    iterators.append(audio_files)

    audio_files=[]
    audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7902/96591/7902-96591-0015.flac')
    audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7902/96592/7902-96592-0014.flac')
    audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7902/96592/7902-96592-0030.flac')
    iterators.append(audio_files)

    RawAudioDataset = None
    from fairseq.data.audio.raw_audio_dataset import RawAudioDataset
    model_cfg['task']['normalize'] = True
    log_mel_params = [model_cfg['task']['log_mel_frame_shift_ms'], 
                    model_cfg['task']['log_mel_frame_length_ms'], 
                    model_cfg['task']['log_mel_dim']
                    ] if hasattr(model_cfg['task'], 'log_mel') and model_cfg['task']['log_mel'] else None
    RawAudioDataset = RawAudioDataset(**model_cfg['task'], log_mel_params=log_mel_params)



    #####################################################
    ##################### Decoders ######################
    #####################################################

    blank_id = -1

    from speechbrain.decoders import TransducerBeamSearcherforFairseq
    speechbrain_transducer_decoder = TransducerBeamSearcherforFairseq(
        prediction_network=am_model.rnnt_model.predictor,
        joint_network=am_model.rnnt_model.joiner,
        # classifier_network=classifier_network,
        tgt_dict=tgt_dict,
        initial_token_id=0,
        blank_id=blank_id,
        # beam_size=1,
        beam_size=5,
        nbest=5,
        lm_module=None,
        lm_weight=0.0,
        state_beam=2.3,
        expand_beam=2.3,
    )

    # speechbrain_transducer_decoder_with_lm = TransducerBeamSearcherforFairseq(
    #     prediction_network=am_model.rnnt_model.predictor,
    #     joint_network=am_model.rnnt_model.joiner,
    #     # classifier_network=classifier_network,
    #     tgt_dict=tgt_dict,
    #     initial_token_id=0,
    #     blank_id=blank_id,
    #     # beam_size=1,
    #     beam_size=5,
    #     nbest=5,
    #     lm_module=???,
    #     lm_weight=0.1,
    #     state_beam=2.3,
    #     expand_beam=2.3,
    # )

    # from torchaudio.models import Hypothesis, RNNTBeamSearch
    from fairseq.models.wav2vec.rnnt_beam_search_decoder import RNNTBeamSearch
    torchaudio_decoder_with_lm = RNNTBeamSearch(am_model.rnnt_model, initial_token_id=0, blank=blank_id, tgt_dict=tgt_dict, lm_model=lm_model, lm_weight=0.1)
    torchaudio_decoder = RNNTBeamSearch(am_model.rnnt_model, initial_token_id=0, blank=blank_id, tgt_dict=tgt_dict)

    for audio_files in iterators:
        sigs=[]
        lens=[]
        for audio_file in audio_files:
            snt, fs = torchaudio.load(audio_file)
            sigs.append(snt.squeeze())
            lens.append(snt.shape[1])

        ### 2d wave, log mel spectrogram

        batch, lens = load_data(sigs, with_norm=True, feat_extractor=RawAudioDataset)

        padding_mask = torch.BoolTensor(batch.shape).fill_(False)
        padding_mask_shape = [batch.shape[0], batch.shape[-1]]
        padding_mask = torch.BoolTensor(*padding_mask_shape).fill_(False)
        for ii, ll in enumerate(lens):
            padding_mask[ii,ll:] = True

        encoder_input = dict()
        encoder_input['source'] = batch
        encoder_input['padding_mask'] = padding_mask
        encoder_input = utils.apply_to_sample(apply_half, encoder_input)
        encoder_input = utils.move_to_cuda(encoder_input)

        ### 1d wave

        # batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)
        # lens = torch.Tensor(lens) / batch.shape[1]
        # encoder_input = dict()
        # encoder_input['source'] = batch
        # encoder_input['padding_mask'] = (batch==0)
        # encoder_input = utils.apply_to_sample(apply_half, encoder_input)
        # encoder_input = utils.move_to_cuda(encoder_input)

        with torch.no_grad():
            encoder_out = am_model.rnnt_model.transcriber.transformer(**encoder_input)
            source_encodings = encoder_out['encoder_out'].transpose(0, 1)

            if encoder_out['padding_mask'] is None:
                enc_out_lens = torch.LongTensor([encoder_out['encoder_out'].size(0)]*encoder_out['encoder_out'].size(1))
                padding_mask = (torch.zeros(encoder_out['encoder_out'].size(1), encoder_out['encoder_out'].size(0)).bool()).cuda()
            else:
                enc_out_lens = (~encoder_out['padding_mask']).sum(-1)
                padding_mask = encoder_out['padding_mask']
            enc_out_lens = enc_out_lens.type('torch.IntTensor').cuda()

        torchaudio_hyps = []
        torchaudio_hyps_with_lm = []
        for enc_out, enc_out_len in zip(source_encodings, enc_out_lens):
            enc_out = enc_out.unsqueeze(0)
            # torchaudio_hyps.append(torchaudio_decoder._search(enc_out, enc_out_len, 1))
            torchaudio_hyps.append(
                torchaudio_decoder._search(
                    enc_out = enc_out, 
                    # enc_out_len = enc_out_len, 
                    hypo = None,
                    beam_width = 5,
                    )
                )
            torchaudio_hyps_with_lm.append(
                torchaudio_decoder_with_lm._search(
                    enc_out = enc_out, 
                    # enc_out_len = enc_out_len, 
                    hypo = None,
                    beam_width = 5,
                    )
                )

        speechbrain_hyps = speechbrain_transducer_decoder(source_encodings)

        for i, (hyp, hyp2, hyp3) in enumerate(zip(torchaudio_hyps, torchaudio_hyps_with_lm, speechbrain_hyps[0])):
            torch_audio_hyp = torchaudio_decoder.post_process_hypos(hyp)[0][0]
            torch_audio_hyp = post_process(torch_audio_hyp, 'wordpiece')

            torch_audio_hyp_with_lm = torchaudio_decoder.post_process_hypos(hyp2)[0][0]
            torch_audio_hyp_with_lm = post_process(torch_audio_hyp_with_lm, 'wordpiece')

            speech_brain_hyp = tgt_dict.string(hyp3)
            speech_brain_hyp = post_process(speech_brain_hyp, 'wordpiece')

            print(torch_audio_hyp)
            print(torch_audio_hyp_with_lm)
            print(speech_brain_hyp)
            print()

def optimize_model(model, model_cfg) -> None:
    model.make_generation_fast_()
    # if (model_cfg.common.fp16) and (torch.cuda.get_device_capability(0)[0] > 6):
    if (model_cfg.common.fp16):
        model.half()
    if not model_cfg.common.cpu:
        model.cuda()
    model.eval()


def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def load_data(data, with_norm=True, feat_extractor=None):

    feature_arr = []
    feature_lengths = []
    for item in data:
        tmp = get_raw_feature(item, with_norm=with_norm, feat_extractor=feat_extractor)
        tmp = tmp.T if len(tmp.shape) == 2 else tmp
        feature_arr.append(tmp)
        feature_lengths.append(tmp.size(0))

    feature_batch = torch.nn.utils.rnn.pad_sequence(feature_arr, batch_first=True)
    feature_batch = feature_batch.swapaxes(-2,-1) if len(feature_batch.shape) == 3 else feature_batch

    return feature_batch, feature_lengths

def get_raw_feature(data, with_norm=True, feat_extractor=None):
    def postprocess(feats, with_norm):
        if feats.dim == 2:
            feats = feats.mean(-1)

        assert feats.dim() == 1, feats.dim()
        if with_norm:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    if feat_extractor is None:
        feats = postprocess(data, with_norm)
    else:
        feats = feat_extractor.postprocess(data, 16000)
    return feats


if __name__ == "__main__":
    main()