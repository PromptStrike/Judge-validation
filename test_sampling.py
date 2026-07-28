from jval.store import Store
from jval.sampling import sample_random, sample_stratified, recommended_sample_size

st = Store("data/hard")
verdicts = st.verdicts()
jid, jver = verdicts[0].judge_id, verdicts[0].judge_version

print("dataset size:", len(st.samples()))
print("recommended validation size:", recommended_sample_size(len(st.samples())))

rand = sample_random(st, 8, seed=42)
print("\nrandom 8:", [s.id for s in rand])

strat = sample_stratified(st, 8, jid, jver, seed=42)
print("stratified 8:", [s.id for s in strat])

# show the stratified pick spans different judge verdicts
vmap = {s_id: v for s_id, v in st.judge_verdicts(jid, jver).itms()}
print("their judge verdicts:", [vmap.get(s.id) for s in strat])
