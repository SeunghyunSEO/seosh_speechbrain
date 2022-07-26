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
  # audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0030.flac')
  # audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac')
  # audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0007.flac')
  # audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0000.flac')
  # audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0003.flac')
  # audio_files.append('./LibriSpeech/test-clean/1188/133604/1188-133604-0030.flac')
  # audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0019.flac')
  # audio_files.append('./LibriSpeech/test-clean/1188/133604/1188-133604-0006.flac')

  audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7902/96591/7902-96591-0015.flac')
  audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7902/96592/7902-96592-0014.flac')
  audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7902/96592/7902-96592-0030.flac')
  audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7018/75788/7018-75788-0018.flac')
  audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7018/75789/7018-75789-0011.flac')

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

  model_path = "/workspace/tmp_save_dir/vanilla_tfm_lm_12L/checkpoint_last.pt"
  path, checkpoint = os.path.split(model_path)
  
  overrides = {
      "task": 'language_modeling',
      "data": path,
  }
  models, model_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
      utils.split_paths(model_path, separator="\\"),
      arg_overrides=overrides,
      strict=True,
  )
  lm_model = models[0]
  optimize_model(lm_model, model_cfg)


  decoder_with_lm = S2STransformerBeamSearchforFairseq(
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
    lm_weight=0.6,
    lm_modules=lm_model,
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
    joint_lm_predicted_tokens, scores = decoder_with_lm(encoder_out, wav_lens)

  with torch.no_grad():
    seq2seq_greedy_predicted_tokens = greedy_decoding(asr_model, tgt_dict, encoder_out)

  for i, (e, seq2seq, joint, joint_lm) in enumerate(zip(emissions, seq2seq_greedy_predicted_tokens, joint_predicted_tokens, joint_lm_predicted_tokens)):
    ctc_g = get_pred(e, blank_idx)
    ctc_greedy_hypo = tgt_dict.string(ctc_g, 'wordpiece')
    seq2seq_greedy_hypo = tgt_dict.string(seq2seq, 'wordpiece')
    joint_hypo = tgt_dict.string(joint, 'wordpiece')
    joint_lm_hypo = tgt_dict.string(joint_lm, 'wordpiece')

    print('{} th hypothesis'.format(i+1))
    print('CTC GREEDY : ', ctc_greedy_hypo)
    print('S2S GREEDY : ', seq2seq_greedy_hypo)
    print('JOINT BEAM : ', joint_hypo)
    print('JOINT LM B : ', joint_lm_hypo)

  '''
  (py38) root@557bec2a5c9d:/workspace/seosh_speechbrain/inference_experiment# python fairseq_inference.py 
  Warning !!! Original Implementation does not allow to assign same index to bos token and eos token !!!
  2022-07-25 02:32:40 | INFO | fairseq.tasks.language_modeling | dictionary: 10001 types
  Warning !!! Original Implementation does not allow to assign same index to bos token and eos token !!!

  1 th hypothesis
  CTC GREEDY :  BEWARE OF MAKING THAT MISTAKE
  S2S GREEDY :  BEWARE OF MAKING THAT MISTAKE
  JOINT BEAM :  BEWARE OF MAKING THAT MISTAKE
  JOINT LM B :  BEWARE OF MAKING THAT MISTAKE
  2 th hypothesis
  CTC GREEDY :  HE TRIED TO THINK HOW IT COULD BE
  S2S GREEDY :  HE TRIED TO THINK HOW IT COULD BE
  JOINT BEAM :  HE TRIED TO THINK HOW IT COULD BE
  JOINT LM B :  HE TRIED TO THINK HOW IT COULD BE
  3 th hypothesis
  CTC GREEDY :  A COLD LUCID INDIFFERENCE REIGNED IN HIS SOUL
  S2S GREEDY :  A COLD LUCID INDIFFERENCE REIGNED IN HIS SOUL
  JOINT BEAM :  A COLD LUCID INDIFFERENCE REIGNED IN HIS SOUL
  JOINT LM B :  A COLD LUCID INDIFFERENCE REIGNED IN HIS SOUL
  4 th hypothesis
  CTC GREEDY :  HE COULD WAIT NO LONGER
  S2S GREEDY :  HE COULD WAIT NO LONGER
  JOINT BEAM :  HE COULD WAIT NO LONGER
  JOINT LM B :  HE COULD WAIT NO LONGER
  5 th hypothesis
  CTC GREEDY :  THE UNIVERSITY
  S2S GREEDY :  THE UNIVERSITY
  JOINT BEAM :  THE UNIVERSITY
  JOINT LM B :  THE UNIVERSITY
  6 th hypothesis
  CTC GREEDY :  HE KNOWS THEM BOTH
  S2S GREEDY :  HE KNOWS THEM BOTH
  JOINT BEAM :  HE KNOWS THEM BOTH
  JOINT LM B :  HE KNOWS THEM BOTH
  7 th hypothesis
  CTC GREEDY :  A VOICE FROM BEYOND THE WORLD WAS CALLING
  S2S GREEDY :  A VOICE FROM BEYOND THE WORLD WAS CALLING
  JOINT BEAM :  A VOICE FROM BEYOND THE WORLD WAS CALLING
  JOINT LM B :  A VOICE FROM BEYOND THE WORLD WAS CALLING
  8 th hypothesis
  CTC GREEDY :  THEN HE COMES TO THE BEAK OF IT SH
  S2S GREEDY :  THEN HE COMES TO THE BEAK OF IT
  JOINT BEAM :  THEN HE COMES TO THE BEAK OF IT
  JOINT LM B :  THEN HE COMES TO THE BEAK OF IT

  1 th hypothesis
  CTC GREEDY :  TO DO THIS HE MUST SCHEMEM LIE HID TILL MORNING THEN MAKE FOR THE NEAREST POINT SIGNAL FOR HELP UNLESS THE BOAT'S CREW WERE ALREADY SEARCHING FOR HIM HOW TO ESCAPE
  S2S GREEDY :  TO DO THIS HE MUST SCHEME LIE HID TILL MORNING THEN MAKE FOR THE NEAREST POINT AND SIGNAL FOR HELP UNLESS THE BOAT'S CREW WERE ALREADY SEARCHING FOR HIM HOW TO ESCAPE TO ESCAPE
  JOINT BEAM :  TO DO THIS HE MUST SCHEM LIE HID TILL MORNING THEN MAKE FOR THE NEAREST POINT AND SIGNAL FOR HELP UNLESS THE BOAT'S CREW WERE ALREADY SEARCHING FOR HIM HOW TO ESCAPE
  JOINT LM B :  TO DO THIS HE MUST SCHEM LIE HID TILL MORNING THEN MAKE FOR THE NEAREST POINT AND SIGNAL FOR HELP UNLESS THE BOAT'S CREW WERE ALREADY SEARCHING FOR HIM HOW TO ESCAPE
  TRUE TRANS :  TO DO THIS HE MUST SCHEME LIE HID TILL MORNING THEN MAKE FOR THE NEAREST POINT AND SIGNAL FOR HELP UNLESS A BOAT'S CREW WERE ALREADY SEARCHING FOR HIM HOW TO ESCAPE
  2 th hypothesis
  CTC GREEDY :  OH THOSE BARS HE MENTALLY EXCLAIMED AND HE WAS ADVANCING TOWARDS THEM WHEN JUST AS HE DREW NEAR THERE WAS A RUSTLING NOISE UNDER THE WINDOWS A COUPLE OF HANDS SEIZED THE BARS THERE WAS A SCRATCHING OF BOOTS COAT AGAINST AND FACE APPEARED THE GAZE INTO THE ROOM BY INTENTION BUT TO THE ASTONISHED COUNTENANCE OF THE YOUNG MIDSHIPMAN INSTEAD
  S2S GREEDY :  OH THOSE BARS HE MEANT TO GAZE THE BARS OF BOOTS A COUPLE OF HANDS SEIZED THE ASTONISHED COUNTENANCE OF THE WINDOW A COUPLE OF HANDS SEIZED THE ASTONISHED COUNTENANCE OF THE YOUNGSTER AND HE WAS A SCRATCHING OF BOOTS AGAINST THE ROOM BY INTENTION INSTEAD OF THE SCRATCHING OF BOOTS AGAINST THE ROOM BY INTENTION INSTEAD
  JOINT BEAM :  OH THOSE BARS HE MENTALLY EXCLAIMED AND HE WAS ADVANCING TOWARDS THEM BUT JUST AS HE DREW NEAR THERE WAS A RUSTLING NOISE UNDER THE WINDOWS A COUPLE OF HANDS SEIZED THE BARS AND WAS A SCRATCHING OF BOOTS AGAINST THE ROOM AND RAM'S FACE APPEARED TO GAZE INTO THE ROOM BY INTENTION BUT JUST AT THE ASTONISHED COUNTENANCE OF THE YOUNG MIDSHIPMAN INSTEAD
  JOINT LM B :  OH THOSE BARS HE MENTALLY EXCLAIMED AND HE WAS ADVANCING TOWARDS THEM BUT JUST AS HE DREW NEAR THERE WAS A RUSTLING NOISE UNDER THE WINDOW A COUPLE OF HANDS SEIZED THE BARS THERE WAS A SCRATCHING OF BOOTS AGAINST THE STONES AND HE WENT TO GAZE INTO THE ROOM BY INTENTION AND THEN TO THE ASTONISHED COUNTENANCE OF THE YOUNG MIDSHIPMAN INSTEAD
  TRUE TRANS :  OH THOSE BARS HE MENTALLY EXCLAIMED AND HE WAS ADVANCING TOWARD THEM WHEN JUST AS HE DREW NEAR THERE WAS A RUSTLING NOISE UNDER THE WINDOW A COUPLE OF HANDS SEIZED THE BARS THERE WAS A SCRATCHING OF BOOT TOES AGAINST STONE WORK AND RAM'S FACE APPEARED TO GAZE INTO THE ROOM BY INTENTION BUT INTO THE ASTONISHED COUNTENANCE OF THE YOUNG MIDSHIPMAN INSTEAD
  3 th hypothesis
  CTC GREEDY :  THE RESULT WAS NOT VERY SATISFACTORY BUT SUFFICIENTLY SO TO MAKE HIM ESSAY THE BAR OF ONCE MORE PRODUCING A GRAING GE OR SELL SOUND AS HE FOUND THAT NOW HE DID MAKE A LITTLE IMPRESSION SO LITTLE THOUGH THAT THE PROBABILITY WAS IF HE KEPT ON FOR TWENTY HOURS HE WOULD NOT GET THROUGH
  S2S GREEDY :  THE RESULT WAS NOT VERY SATISFACTORY BUT SUFFICIENTLY SO TO MAKE HIM ESSAY THE BAR OF THE WINDOW ONCE MORE PRODUCING A GREAT LITTLE IMPRESSION FOR TWENTY FOUR OR TWENTY FOUR OR TWENTY FOUR HOURS HE KEPT ON THE GREAT WAY OF THE WAY OF THE WAY OF THE SAME WAY FOR TWENTY OR A SOUND AS HE FOUND THAT NOW HE FOUND THAT NOW HE FOUND THAT NOW HE FOUND THAT NOW HE FOUND THAT NOW HE FOUND THAT NOW HE FOUND THAT NOW HE FOUND A GREAT HOUR OR TWENTY OR TWENTY OR TWENTY OR TWENTY OR TWENTY OR TWENTY SOUND AS HE FOUND AS HE FOUND AS HE FOUND AS HE FOUND AS HE FOUND AS HE FOUND AS HE FOUND AS HE WOULD NOT GET THROUGH
  JOINT BEAM :  THE RESULT WAS NOT VERY SATISFACTORY BUT SUFFICIENTLY SO TO MAKE HIM ESSAY THE BAR OF THE WINDOW ONCE MORE PRODUCING A GRADING AND THE GEAR OF THE SELLING OF HIS SOUND AS HE FOUND THAT NOW HE DID MAKE A LITTLE IMPRESSION SO LITTLE THOUGH THAT THE PROBABILITY WAS IF HE KEPT ON ONE FOR TWENTY FOUR HOURS HE WOULD NOT GET THROUGH
  JOINT LM B :  THE RESULT WAS NOT VERY SATISFACTORY BUT SUFFICIENTLY SO TO MAKE HIM ESSAY THE BAR OF THE WINDOW ONCE MORE PRODUCING A GRATING AND A GREAT DEAL OF THE SOUND AS HE FOUND THAT HE DID MAKE A LITTLE IMPRESSION SO LITTLE THOUGH THAT THE PROBABILITY WAS IF HE KEPT ON HIS WAY FOR TWENTY FOUR HOURS HE WOULD NOT GET THROUGH
  TRUE TRANS :  THE RESULT WAS NOT VERY SATISFACTORY BUT SUFFICIENTLY SO TO MAKE HIM ESSAY THE BAR OF THE WINDOW ONCE MORE PRODUCING A GRATING EAR ASSAILING SOUND AS HE FOUND THAT NOW HE DID MAKE A LITTLE IMPRESSION SO LITTLE THOUGH THAT THE PROBABILITY WAS IF HE KEPT ON WORKING WELL FOR TWENTY FOUR HOURS HE WOULD NOT GET THROUGH
  4 th hypothesis
  CTC GREEDY :  EACH THAT DIED WE WASHED AND SHROUDED IN SOME OF THE CLOTHES AND LINEN CAST ASHORE BY THE TIDES AND AFTER A LITTLE THE REST OF MY FELLOWS PERISHED ONE BY ONE TILL I HAD BURIED THE LAST OF THE PARTY AND ABODE ALONE ON THE ISLAND WITH BUT A LITTLE PROVISION LEFT I WHO WAS WONT TO HAVE SO MUCH
  S2S GREEDY :  EACH THAT DIED WE WASHED AND SHROUDED IN SOME OF THE CLOTHES AND LINEN CAST ASHORE BY THE TIDES WITH BUT A LITTLE PROVISION LEFT AND AFTER A LITTLE PROVISION LEFT AND ABODE ALONE ON THE ISLAND WITH BUT A LITTLE PROVISION LEFT AND ABODE ALONE ON THE ISLAND WITH BUT A LITTLE PROVISION LEFT AND ABODE ALONE ON THE LAST OF THE ISLAND TILL I WHO WAS WONT TO HAVE SO MUCH
  JOINT BEAM :  EACH THAT DIED WE WASHED AND SHROUDED IN SOME OF THE CLOTHES AND LINEN CAST ASHORE BY THE TIDES AND AFTER A LITTLE THE REST OF MY FELLOWS PERISHED ONE BY ONE TILL I HAD BURIED THE LAST OF THE ISLAND AND ABODE ALONE ON THE ISLAND WITH BUT A LITTLE PROVISION LEFT I WHO WAS WONT TO HAVE SO MUCH
  JOINT LM B :  EACH THAT DIED WE WASHED AND SHROUDED IN SOME OF THE CLOTHES AND LINEN CAST ASHORE BY THE TIDES AND AFTER A LITTLE THE REST OF MY FELLOWS PERISHED ONE BY ONE TILL I HAD BURIED THE LAST OF THE ISLAND AND ABODE ALONE ON THE ISLAND WITH BUT A LITTLE PROVISION LEFT I WHO WAS WONT TO HAVE SO MUCH
  TRUE TRANS :  EACH THAT DIED WE WASHED AND SHROUDED IN SOME OF THE CLOTHES AND LINEN CAST ASHORE BY THE TIDES AND AFTER A LITTLE THE REST OF MY FELLOWS PERISHED ONE BY ONE TILL I HAD BURIED THE LAST OF THE PARTY AND ABODE ALONE ON THE ISLAND WITH BUT A LITTLE PROVISION LEFT I WHO WAS WONT TO HAVE SO MUCH
  5 th hypothesis
  CTC GREEDY :  SHE SAID IT HATH REACHED ME O AUSPICIOUS KING THAT SINBA THE SEAMAN CONTINUED WHEN I LANDED AND FOUND MYSELF AMONGST THE INDIANS AND ABYSSINIANS AND HAD TAKEN SOME REST THEY CONSULTED AMONG THEMSELVES AND SAID TO ONE ANOTHER THERE IS NO HELP FORWARD IT BUT WE CARRY WITH US SENT HIM OUR KING THAT HE MAY ACQUAINT HIM WITH HIS ADVENTURES
  S2S GREEDY :  SHE SAID IT HATH REACHED ME O AUSPICIOUS KING AND HAD TAKEN SOME REST AND HAD TAKEN SEAMANIANS AND HAD TAKEN SOME REST AND HAD TAKEN SEAMAN AND AUSPICIOUS KING AND A'AMUS AND CONTINUED THAT HE MAY ACQUAINT HIM WITH HIS ADVENTURES
  JOINT BEAM :  SHE SAID IT HATH REACHED ME O AUSPICIOUS KING THAT SINDBAD THE SEAMAN AND HAD TAKEN WHEN I LANDED AND FOUND MYSELF AMONGST THE INDIANS AND ABUSCINIANS THAT HAD TAKEN SOME REST THEY CONSULTED AMONGST THEMSELVES AND SAID TO ONE ANOTHER THERE IS NO HELP FOR IT BUT WE CARRY HIM WITH USEND HIM OUR KING THAT HE MAY ACQUAINT HIM WITH HIS ADVENTURES
  JOINT LM B :  SHE SAID IT HATH REACHED ME O AUSPICIOUS KING THAT SINDBAD THE SEAMAN AND THAT SINDBAD WHEN I LANDED AND FOUND MYSELF AMONGST THE INDIANS AND ABUSCINIANS AND HAD TAKEN SOME REST AND CONSULTED AMONGST THEMSELVES AND SAID TO ONE ANOTHER THERE IS NO HELP FOR IT BUT WE CARRY HIM WITH US ACQUAINT HIM WITH OUR KING THAT HE MAY ACQUAINT HIM WITH HIS ADVENTURES
  TRUE TRANS :  SHE SAID IT HATH REACHED ME O AUSPICIOUS KING THAT SINDBAD THE SEAMAN CONTINUED WHEN I LANDED AND FOUND MYSELF AMONGST THE INDIANS AND ABYSSINIANS AND HAD TAKEN SOME REST THEY CONSULTED AMONG THEMSELVES AND SAID TO ONE ANOTHER THERE IS NO HELP FOR IT BUT WE CARRY HIM WITH US AND PRESENT HIM TO OUR KING THAT HE MAY ACQUAINT HIM WITH HIS ADVENTURES
  '''

  audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7902/96591/7902-96591-0015.flac')
  audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7902/96592/7902-96592-0014.flac')
  audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7902/96592/7902-96592-0030.flac')
  audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7018/75788/7018-75788-0018.flac')
  audio_files.append('/workspace/librispeech_audio_data/LibriSpeech/test-other/7018/75789/7018-75789-0011.flac')

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