The ~5842 'annotation beats' that are being referred to –– those are entirely separate from V1. 100% of those, I manually annotated in V3. Only 500 of them were from maximum disagreement between the 2 models. 500 of them then were CNN highest confidence -- *without ensemble saying the opposite* -- I.e., most of these were in fact correct, only maybe ⅒-⅛  were false that I corrected. I don't remember, but I know I breezed through it quickly, then brought it up to Sonnet that it needed to explicitly include high p_cnn **but** ensemble still called it clean. Maximal "false positives". THAT is the majority of the beats, about 5100/6600 of them. Why did I just say 6600? Because, for the majority of the sessions, previously annotated beats were *not* excluded, stupidly, so as I did more sessions, I kept having to skip more and more of them as they accumulated and would resurface. So of the 6600 beats I went through, somewhere around 700+ of them were perviosuly annotated in another round and thus skipped, which is how we arrive at 58xx. All 100% of these beats are from this current pipeline, none are from v1. They're all beats I manually annotated as clean or artifact. 


The main scripts from v1 are:
pipeline.py
ptrain.py
dict.py
pre.py
train.py
gui2.py


Anything that I did not explicitly flag, implicitly is clean. If it was not, I would have flagged it. These were not just segments I skimmed; I spent many, many hours across weeks annotating 30 second segments, and I only validated a segment whenever I was entirely done with it. If it was half-annotated, I put it in the return to pile.


A 'validated segment' is defined as a segment that I manually annotated -- NOT "all these beats are confirmed clean". 'Validated segments' simply means it's a segment I reviewed an annotated -- with physio events (which were by and large vagal events or PACs/PVCs), added r-peaks, artifacts, interpolations, you name it. To derive the explicitly clean beats, you'd need to subtract all of the beats that I annotated from all of the beats contained in all of the validated segments.


I'm still using the same data-cleaning script that I always have for my hrv projects. It's what generates the raw ecg data csvs that are the inputs: 
/Volumes/xHRV/ECG. I recently updated this script to now starting converting all of the mV values → 𝜇V by multiplying them by 1000 to produce integers, saving ~1-2 characters per row by avoiding the 0. part, which may seem trivial but certainly is not when we're talking about a cumulative 5 billion rows.