import os

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from speechbrain.pretrained import EncoderDecoderASR

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils

from pdb import set_trace as Tra


# print('*'*30)
# print('torch version ?', torch.__version__)
# print('torchaudio version ?', torchaudio.__version__)
# print('*'*30)
# print('cuda availability ? {}'.format(torch.cuda.is_available()))
# print('total gpu nums : {}'.format(torch.cuda.device_count()))
# print('cudnn backends version : {}'.format(torch.backends.cudnn.version()))
# print('cuda version : {}'.format(torch.version.cuda))
# print('*'*30)
# for n in range(torch.cuda.device_count()):
#   print('{}th GPU name is {}'.format(n,torch.cuda.get_device_name(n)))
#   print('\t capability of this GPU is {}'.format(torch.cuda.get_device_capability(n)))
# print('*'*30)


def main():

  model_path = "/workspace/tmp_save_dir/w2v2_s2s_joint_0.2_inter_0.1/checkpoint_best.pt"
  # model_path = "/mnt/clova_speech/users/seosh/librispeech_model/am/w2v2/wav2vec2_vox_960h_new.pt"
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
  asr_model = models[0]
  optimize_model(asr_model, model_cfg)

  tgt_dict = task.target_dictionary
  blank_idx = tgt_dict.index(task.blank_symbol)

  audio_files=[]
  audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0030.flac')
  audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac')
  audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0007.flac')
  audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0000.flac')
  audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0003.flac')
  audio_files.append('./LibriSpeech/test-clean/1188/133604/1188-133604-0030.flac')
  audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0019.flac')
  audio_files.append('./LibriSpeech/test-clean/1188/133604/1188-133604-0006.flac')

  audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7902/96591/7902-96591-0015.flac')


  sigs=[]
  lens=[]
  for audio_file in audio_files:
    snt, fs = torchaudio.load(audio_file)
    sigs.append(snt.squeeze())
    lens.append(snt.shape[1])

  batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)
  lens = torch.Tensor(lens) / batch.shape[1]

  encoder_input = dict()
  encoder_input['source'] = batch
  encoder_input['padding_mask'] = (batch==0)
  encoder_input = utils.apply_to_sample(apply_half, encoder_input)
  encoder_input = utils.move_to_cuda(encoder_input)

  with torch.no_grad():
    encoder_out = asr_model.encoder(**encoder_input)
  logits = asr_model.encoder.ctc_proj(encoder_out["encoder_out_before_proj"])
  lprobs = utils.log_softmax(logits.float(), dim=-1)
  emissions = lprobs.transpose(0, 1).float().cpu().contiguous()

  # with torch.no_grad():
  #   encoder_out = asr_model(**encoder_input)
  # # emissions = asr_model.get_logits(encoder_out)
  # emissions = encoder_out['encoder_out']
  # emissions = utils.log_softmax(emissions.float(), dim=-1)
  # emissions = emissions.transpose(0, 1).float().cpu().contiguous()

  from speechbrain.decoders import S2STransformerBeamSearchforFairseq
  decoder = S2STransformerBeamSearchforFairseq(
    bos_index=tgt_dict.eos(),
    eos_index=tgt_dict.eos(),
    beam_size=10,
    topk=1,
    return_log_probs=False,
    using_eos_threshold=False,
    eos_threshold=1.5,
    length_normalization=True,
    length_rewarding=0,
    coverage_penalty=0.0,
    # lm_weight=0.6,
    lm_weight=0.0,
    lm_modules=None,
    ctc_weight=0.4,
    blank_index=0,
    ctc_score_mode="full",
    ctc_window_size=0,
    using_max_attn_shift=False,
    max_attn_shift=60,
    # minus_inf=-1e20,
    # minus_inf=-float("-inf"),
    minus_inf=-65000, # half tensor overflow ?
    min_decode_ratio=0,
    max_decode_ratio=1.0,
    model=asr_model,
    ctc_layer=asr_model.encoder.ctc_proj,
    temperature=1.15,
    temperature_lm=1.15,
  )

  encoder_out = utils.apply_to_sample(apply_half, encoder_out)
  encoder_out = utils.move_to_cuda(encoder_out)
  wav_lens = utils.apply_to_sample(apply_half, lens)
  wav_lens = utils.move_to_cuda(wav_lens)

  with torch.no_grad():
    joint_predicted_tokens, scores = decoder(encoder_out, wav_lens)

  with torch.no_grad():
    seq2seq_greedy_predicted_tokens = greedy_decoding(asr_model, tgt_dict, encoder_out)

  for i, (e, seq2seq, joint) in enumerate(zip(emissions, seq2seq_greedy_predicted_tokens, joint_predicted_tokens)):
    ctc_g = get_pred(e, blank_idx)
    ctc_greedy_hypo = tgt_dict.string(ctc_g, 'wordpiece')
    seq2seq_greedy_hypo = tgt_dict.string(seq2seq, 'wordpiece')
    joint_hypo = tgt_dict.string(joint, 'wordpiece')

    print('{} th hypothesis'.format(i+1))
    print('CTC GREEDY : ', ctc_greedy_hypo)
    print('S2S GREEDY : ', seq2seq_greedy_hypo)
    print('JOINT BEAM : ', joint_hypo)

  '''
  (py38) root@557bec2a5c9d:/workspace/seosh_speechbrain/inference_experiment# python fairseq_inference.py 
  torchvision is not available - cannot save figures
  Warning !!! Original Implementation does not allow to assign same index to bos token and eos token !!!

  1 th hypothesis
  CTC GREEDY :  BEWARE OF MAKING THAT MISTAKE
  S2S GREEDY :  BEWARE OF MAKING THAT MISTAKE
  JOINT BEAM :  BEWARE OF MAKING THAT MISTAKE
  2 th hypothesis
  CTC GREEDY :  HE TRIED TO THINK HOW IT COULD BE
  S2S GREEDY :  HE TRIED TO THINK HOW IT COULD BE
  JOINT BEAM :  HE TRIED TO THINK HOW IT COULD BE
  3 th hypothesis
  CTC GREEDY :  A COLD LUCID INDIFFERENCE REIGNED IN HIS SOUL
  S2S GREEDY :  A COLD LUCID INDIFFERENCE REIGNED IN HIS SOUL
  JOINT BEAM :  A COLD LUCID INDIFFERENCE REIGNED IN HIS SOUL
  4 th hypothesis
  CTC GREEDY :  HE COULD WAIT NO LONGER
  S2S GREEDY :  HE COULD WAIT NO LONGER
  JOINT BEAM :  HE COULD WAIT NO LONGER
  5 th hypothesis
  CTC GREEDY :  THE UNIVERSITY
  S2S GREEDY :  THE UNIVERSITY
  JOINT BEAM :  THE UNIVERSITY
  6 th hypothesis
  CTC GREEDY :  HE KNOWS THEM BOTH
  S2S GREEDY :  HE KNOWS THEM BOTH
  JOINT BEAM :  HE KNOWS THEM BOTH
  7 th hypothesis
  CTC GREEDY :  A VOICE FROM BEYOND THE WORLD WAS CALLING
  S2S GREEDY :  A VOICE FROM BEYOND THE WORLD WAS CALLING
  JOINT BEAM :  A VOICE FROM BEYOND THE WORLD WAS CALLING
  8 th hypothesis
  CTC GREEDY :  THEN HE COMES TO THE BEAK OF IT SH
  S2S GREEDY :  THEN HE COMES TO THE BEAK OF IT
  JOINT BEAM :  THEN HE COMES TO THE BEAK OF IT
  9 th hypothesis
  CTC GREEDY :  TO DO THIS HE MUST SCHEMEM LIE HID TILL MORNING THEN MAKE FOR THE NEAREST POINT SIGNAL FOR HELP UNLESS THE BOAT'S CREW WERE ALREADY SEARCHING FOR HIM HOW TO ESCAPE
  S2S GREEDY :  TO DO THIS HE MUST SCHEME LIE HID TILL MORNING THEN MAKE FOR THE NEAREST POINT AND SIGNAL FOR HELP UNLESS THE BOAT'S CREW WERE ALREADY SEARCHING FOR HIM HOW TO ESCAPE
  JOINT BEAM :  TO DO THIS HE MUST SCHEM LIE HID TILL MORNING THEN MAKE FOR THE NEAREST POINT AND SIGNAL FOR HELP UNLESS THE BOAT'S CREW WERE ALREADY SEARCHING FOR HIM HOW TO ESCAPE

  TRUE TRANS :  TO DO THIS HE MUST SCHEME LIE HID TILL MORNING THEN MAKE FOR THE NEAREST POINT AND SIGNAL FOR HELP UNLESS A BOAT'S CREW WERE ALREADY SEARCHING FOR HIM HOW TO ESCAPE
  '''


def optimize_model(model, model_cfg) -> None:
  model.make_generation_fast_()
  # if (model_cfg.common.fp16) and (torch.cuda.get_device_capability(0)[0] > 6):
  if (model_cfg.common.fp16):
      model.half()
  if not model_cfg.common.cpu:
      model.cuda()
  model.eval()

def greedy_decoding(model, tgt_dict, encoder_out):
    with torch.no_grad():
        bsz = encoder_out['encoder_out'].size(1)
        enc_lens = encoder_out['encoder_out'].size(0)

        prev_output_tokens = (
            torch.zeros(bsz, device='cuda')
            .fill_(tgt_dict.eos())
            .long()
        ).unsqueeze(1)

        max_step = enc_lens + 5

        eos_detect = torch.zeros_like(prev_output_tokens).type_as(prev_output_tokens)

        accum_softmax_prob = torch.Tensor().type_as(encoder_out['encoder_out'])

        for i in range(max_step):
            if (i > 0) and (model.soft_input_training) and (model.soft_input_training_updates <= model.encoder.num_updates) : 
                softmax_prob = model.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out, soft_input=accum_softmax_prob)[0]
            else:
                softmax_prob = model.decoder(prev_output_tokens=prev_output_tokens, encoder_out=encoder_out)[0]
            next_output_tokens_prob = softmax_prob[:,-1,:]
            accum_softmax_prob = torch.cat((accum_softmax_prob,next_output_tokens_prob.unsqueeze(1)),1)
            next_output_tokens = torch.argmax(next_output_tokens_prob,-1).unsqueeze(1)
            prev_output_tokens = torch.cat((prev_output_tokens,next_output_tokens),-1)

            if (next_output_tokens == tgt_dict.eos()).sum() > 1:
                eos_tokens = (next_output_tokens == tgt_dict.eos())
                eos_tokens.masked_fill_(eos_tokens == eos_detect.bool(), 0)
                eos_detect.masked_fill_(eos_tokens,i)

            if torch.sum(eos_detect!=0).item() == bsz:
                for i in range(eos_detect.size(0)):
                    prev_output_tokens[i][(eos_detect[i].item()+1):] = tgt_dict.eos()
                break;

        return prev_output_tokens

def get_pred(e, blank_idx):
    toks = e.argmax(dim=-1).unique_consecutive()
    return toks[toks != blank_idx]


def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


if __name__ == "__main__":
    main()