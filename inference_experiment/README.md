# joint decoding using speechbrain and fairseq model

## prepare dataset, config and model

```
cd /workspace
git clone https://github.com/SeunghyunSEO/seosh_speechbrain
cd seosh_speechbrain
pip install -e .
```

```
python -c "import speechbrain; from speechbrain.decoders import S2STransformerBeamSearch; print(speechbrain); print(S2STransformerBeamSearch)"
```

```python
(py38) root@557bec2a5c9d:/workspace/seosh_speechbrain# python -c "import speechbrain; from speechbrain.decoders import S2STransformerBeamSearch; print(speechbrain); print(S2STransformerBeamSearch)"
torchvision is not available - cannot save figures
<module 'speechbrain' from '/workspace/seosh_speechbrain/speechbrain/__init__.py'>
<class 'speechbrain.decoders.seq2seq.S2STransformerBeamSearch'>
```


```
mkdir -p /workspace/seosh_speechbrain/inference_experiment
cd /workspace/seosh_speechbrain/inference_experiment
wget https://www.openslr.org/resources/12/test-clean.tar.gz
tar -zxvf test-clean.tar.gz
```

```python
(py38) root@557bec2a5c9d:/workspace/seosh_speechbrain/inference_experiment# tree -L 2
.
|-- LibriSpeech
|   |-- BOOKS.TXT
|   |-- CHAPTERS.TXT
|   |-- LICENSE.TXT
|   |-- README.TXT
|   |-- SPEAKERS.TXT
|   `-- test-clean
`-- test-clean.tar.gz
```

```
cd /workspace/seosh_speechbrain/inference_experiment
python import -c "from speechbrain.pretrained import EncoderDecoderASR; asr_model = EncoderDecoderASR.from_hparams(source='speechbrain/asr-transformer-transformerlm-librispeech', savedir='pretrained_models/asr-transformer-transformerlm-librispeech')"
```

```python
(py38) root@557bec2a5c9d:/workspace/seosh_speechbrain/inference_experiment/pretrained_models# pwd; tree -L 2
/workspace/seosh_speechbrain/inference_experiment/pretrained_models
.
`-- asr-transformer-transformerlm-librispeech
    |-- asr.ckpt -> /root/.cache/huggingface/hub/models--speechbrain--asr-transformer-transformerlm-librispeech/snapshots/586c7897e606d6a00f0513e1ae527a5824d10eac/asr.ckpt
    |-- hyperparams.yaml -> /root/.cache/huggingface/hub/models--speechbrain--asr-transformer-transformerlm-librispeech/snapshots/586c7897e606d6a00f0513e1ae527a5824d10eac/hyperparams.yaml
    |-- lm.ckpt -> /root/.cache/huggingface/hub/models--speechbrain--asr-transformer-transformerlm-librispeech/snapshots/586c7897e606d6a00f0513e1ae527a5824d10eac/lm.ckpt
    |-- normalizer.ckpt -> /root/.cache/huggingface/hub/models--speechbrain--asr-transformer-transformerlm-librispeech/snapshots/586c7897e606d6a00f0513e1ae527a5824d10eac/normalizer.ckpt
    `-- tokenizer.ckpt -> /root/.cache/huggingface/hub/models--speechbrain--asr-transformer-transformerlm-librispeech/snapshots/586c7897e606d6a00f0513e1ae527a5824d10eac/tokenizer.ckpt

1 directory, 5 files
```

## run script

- TODO
  - [ ] need to modify and add argparser


```
cd /workspace/seosh_speechbrain/inference_experiment
python sb_inference.py 
python fairseq_inference.py 
```

```python
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
```
