# notes

## numbers

- kernel: attention_fwd_performance from nki_samples
- all avg over 100 runs
- ec2 $/hr
  - g4dn.xlarge 0.526
  - c8i.xlarge 0.18742
  - trn1.2xlarge 1.34375
- time
  - on-chip (trn1.2xlarge): 7.574 s
  - nki_simulate (c8i): 0.3381 s
  - TAIDL-TO (c8i): 0.1164 s
  - TAIDL-TO (g4dn): 0.0185 s
- on-chip so slow? benchmark latency is 297 us but you still have to run the nki file and get the output
- g4dn one is w/ batching, much slower without. using the best one for slides
- cost (time x $/hr)
  - on-chip 10.177
  - nki_simulate 0.063
  - TAIDL-TO cpu 0.022
  - TAIDL-TO gpu 0.010
- nki programming model: https://awsdocs-neuron.readthedocs-hosted.com/en/v2.22.0/general/nki/programming_model.html#nki-programming-model

## NEFF

- talked about NEFF, multi-instruction streams
- TAIDL doesn't handle this well

## GeneSys

- good target to explore multi-instruction streams
- also has loop instructions
- tricky to auto-generate w/ the same paradigm

## new TAIDL-TO library?

- you pick your parser, we simulate most instructions
- you do the rest -- program counter, jumps

## problems

- compile time!!!
- ok if you can annotate for loops
  - TAIDL-TO kernels for AMX
  - I-BERT, Exo kernels in Gemmini
- still doesn't scale over a few 1000 instrs w/ loops
- data-dependent / sync instructions -- likely an issue in NEFFs, need to dig deeper

## what works

- batching -- amortize compile, consume multiple inputs in parallel

## ideas

- loop detection? something like the SpiderMonkey thing from the UIUC compiler qual paper
- fallback w/ a lazy trigger, break into multiple compile chunks
