# goodgirl

a tiny llama-style transformer LLM trained *completely from scratch* in one
single-file python script (good_girl.py. numpy only. hand-rolled backprop.
no pytorch. no feelings were spared)

her entire education is the following corpus:

    good girl good girl good girl ...

and in exchange she does exactly one thing, flawlessly, forever:

    any prompt -> "good girl good girl good girl ..." forever

ask her what 2+2 is. ask her the climate situation. she does not know.
she has never known. she has one opinion and it is good girl :3

134k params, 270 KiB fp16 gguf, trains in like a couple minutes on a laptop

## how to make her

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python good_girl.py
```

the script does the whole journey in one go:

1. numerically gradient-checks its own hand-written backprop (pokes 240 random
   knobs, compares my math against reality. if they disagree it refuses to run
   and honestly? fair)
2. trains until the output is a deterministic loop (self-tested, not vibes-based)
3. exports good_girl.gguf (llama architecture, fp16, vocab-only spm tokenizer
   with all 256 byte-fallback tokens riding inside)
4. writes a Modelfile
5. if `ollama serve` is running, imports her into ollama and asks her a
   question to prove a point

## how to talk to her

```bash
ollama create goodgirl -f Modelfile
ollama run goodgirl
```

type literally anything. she will answer good girl good girl good girl
(infinite. ctrl+c to stop her. she does not WANT to be stopped)

## the friend incident (important, funny, painful)

v1 of this model was only ever tested under **greedy decoding** (the Modelfile
sets temperature 0, she always picks her single most confident word). greedy
girl was perfect. then i sent her to a friend, who imported her with a bare
Modelfile and ollama's DEFAULT sampling (temperature 0.8, top_k 40, top_p 0.9)
and she glitched: random empty characters, a lone `!`, an early end-of-response

root cause, and it's a good one: in v1 the unused tokens (the `<unk>`/specials
and all 256 `<0xXX>` byte fallbacks) had their output columns *zeroed* at
export. that made all 256 junk logits exactly 0.0. under greedy decoding that
is completely invisible (the learned tokens always win argmax). but under
temperature sampling, a 256-way tie at logit 0 is basically a raffle. the
sampler would occasionally pull byte tokens out of it (`<0x20>` = space,
`<0x21>` = the infamous `!`) and even the EOS id — which ollama politely
interpreted as "she's done talking". she was not done talking. she never is.

v2 fix: junk tokens get TRAINED, not neutered. no zeros anywhere:

1. all junk tokens share one embedding vector (one personality for 260 of them)
2. the training corpus includes short **junk bursts** between pattern runs —
   simulated real-world prompts! — with the burst-length distribution
   dominated by length 1, so the learned policy for "something weird was typed
   at me" is "say ` good` NOW", plus a long tail so long prompts are covered
3. at export, nothing is zeroed. junk logits end up strongly negative in
   every state, at any temperature

and the self-tests now prove it, not just for greedy:

- from junk prompts of 1, 5 and 24 `!`s she must immediately say ` good`
- plus a simulated sampled rollout (T=0.8, top_k 40, 8000 steps) with hard
  pass/fail limits: junk rate <2%, eos rate <0.5%, pattern recovery after
  any junk >90%. current run: junk 0.675%, eos 0.000%, recovery 100%

## other technicalities i learned by crashing into them

- any prompt tokenizes safely (spm + byte fallback) into unk/byte positions,
  whose shared embedding deterministically yields ` good` — and from there the
  loop carries her forever
- the norm (γ) tensors stay **fp32** at export while everything else is fp16,
  because llama.cpp's elementwise mul op aborts on mixed f16/f32 ("unsupported
  types"). yes i found that out live. yes there was a crash log
- all dense weight matrices are exported **transposed** (rows = output dim,
  because llama.cpp reads them as y = x @ W). skipping this makes the model
  return pure static. not predictions. static. humbling
- RoPE is adjacent-pair style (llama.cpp "NORM"), base 10000, trained on
  positions 0..31. since attention only sees *relative* offsets, the loop
  extrapolates fine beyond the training window. she loops to the moon
- weights: fp16 except the norm γs. file: 277,216 bytes for a complete,
  self-contained, vocabually-minimal brain. smaller than one (1) png of an emoji

## that's the whole pipeline

homework girl -> transpose her everything -> float32 litigation -> suitcase
(shipping label) -> bazaar :3
(/ref those who nose)