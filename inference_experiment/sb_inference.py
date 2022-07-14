import os

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from speechbrain.pretrained import EncoderDecoderASR

from pdb import set_trace as Tra


print('*'*30)
print('torch version ?', torch.__version__)
print('torchaudio version ?', torchaudio.__version__)
print('*'*30)
print('cuda availability ? {}'.format(torch.cuda.is_available()))
print('total gpu nums : {}'.format(torch.cuda.device_count()))
print('cudnn backends version : {}'.format(torch.backends.cudnn.version()))
print('cuda version : {}'.format(torch.version.cuda))
print('*'*30)
for n in range(torch.cuda.device_count()):
  print('{}th GPU name is {}'.format(n,torch.cuda.get_device_name(n)))
  print('\t capability of this GPU is {}'.format(torch.cuda.get_device_capability(n)))
print('*'*30)


def main():

  asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-transformerlm-librispeech",  run_opts={"device":"cuda"})

  audio_files=[]
  audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0030.flac')
  audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0014.flac')
  audio_files.append('./LibriSpeech/test-clean/1089/134686/1089-134686-0007.flac')
  audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0000.flac')
  audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0003.flac')
  audio_files.append('./LibriSpeech/test-clean/1188/133604/1188-133604-0030.flac')
  audio_files.append('./LibriSpeech/test-clean/1089/134691/1089-134691-0019.flac')
  audio_files.append('./LibriSpeech/test-clean/1188/133604/1188-133604-0006.flac')

  sigs=[]
  lens=[]
  for audio_file in audio_files:
    snt, fs = torchaudio.load(audio_file)
    sigs.append(snt.squeeze())
    lens.append(snt.shape[1])

  batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)
  lens = torch.Tensor(lens) / batch.shape[1]

  '''
  (Pdb) batch.size(); lens.size(); lens
  torch.Size([8, 68400])
  torch.Size([8])
  tensor([0.6351, 0.5205, 1.0000, 0.4877, 0.5088, 0.4480, 0.7380, 0.5614])
  '''

  # Tra()

  with torch.no_grad():
    ## CTC / Attention joint beam search with lm 
    result = asr_model.transcribe_batch(batch, lens)

    ## encoding only
    # encoder_out = asr_model.encode_batch(batch, lens)

    # ## step by step
    # batch, lens = batch.to('cuda'), lens.to('cuda')
    # encoder_out = asr_model.mods.encoder(batch, lens)
    # predicted_tokens, scores = asr_model.mods.decoder(encoder_out, lens)
    # predicted_words = [
    #     asr_model.tokenizer.decode_ids(token_seq)
    #     for token_seq in predicted_tokens
    # ]

    # ctc_outputs = asr_model.mods.decoder.ctc_forward_step(encoder_out)

    # logits = asr_model.mods.decoder.ctc_fc(encoder_out)
    # ctc_outputs2 = asr_model.mods.decoder.softmax(logits)

    # Tra()

    '''
    (Pdb) encoder_out.size(); ctc_outputs.size()
    torch.Size([8, 107, 512])
    torch.Size([8, 107, 5000])

    (Pdb) len(asr_model.tokenizer)
    5000

    ##### S2SBaseSearcher 
    (Pdb) asr_model.mods.decoder.bos_index; asr_model.mods.decoder.eos_index;
    1
    2
    (Pdb) asr_model.mods.decoder.min_decode_ratio; asr_model.mods.decoder.max_decode_ratio;
    0.0
    1.0

    # asr_model.mods.decoder.lm_modules
    # asr_model.mods.decoder.model

    ##### S2SBeamSearcher 
    (Pdb) asr_model.mods.decoder.ctc_weight; asr_model.mods.decoder.blank_index; asr_model.mods.decoder.att_weight
    0.4
    0
    0.6

    (Pdb) asr_model.mods.decoder.ctc_window_size; asr_model.mods.decoder.beam_size;
    0
    66

    (Pdb) asr_model.mods.decoder.lm_weight;
    (Pdb) asr_model.mods.decoder.lm_modules
    TransformerLM(
      (positional_encoding): PositionalEncoding()
      (encoder): TransformerEncoder(
        (layers): ModuleList(
          (0): TransformerEncoderLayer(
            (self_att): MultiheadAttention(
              (att): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=768, out_features=768, bias=True)
              )
            )
            (pos_ffn): PositionalwiseFeedForward(
              (ffn): Sequential(
                (0): Linear(in_features=768, out_features=3072, bias=True)
                (1): GELU()
                (2): Dropout(p=0.0, inplace=False)
                (3): Linear(in_features=3072, out_features=768, bias=True)
              )
            )
            (norm1): LayerNorm(
              (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            )
            (norm2): LayerNorm(
              (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
            )
            (dropout1): Dropout(p=0.0, inplace=False)
            (dropout2): Dropout(p=0.0, inplace=False)
          )


    (Pdb) asr_model.mods.decoder.lm_modules.output_proj
    ModuleList(
      (layers): ModuleList(
        (0): Linear(
          (w): Linear(in_features=768, out_features=768, bias=True)
        )
        (1): LayerNorm(
          (norm): LayerNorm((768,), eps=1e-06, elementwise_affine=True)
        )
        (2): Linear(
          (w): Linear(in_features=768, out_features=5000, bias=True)
        )
      )
    )

    (Pdb) asr_model.mods.decoder.fc
    Linear(
      (w): Linear(in_features=512, out_features=5000, bias=True)
    )
    (Pdb) asr_model.mods.decoder.ctc_fc
    Linear(
      (w): Linear(in_features=512, out_features=5000, bias=True)
    )
    (Pdb) asr_model.mods.decoder.softmax
    LogSoftmax(dim=-1)

    (Pdb) asr_model.mods.decoder.temperature; asr_model.mods.decoder.temperature_lm;
    1.15
    1.15
    '''

  for i, hypo in enumerate(result[0]):
    print('{} : {}'.format(i+1,hypo))


if __name__ == "__main__":
    main()