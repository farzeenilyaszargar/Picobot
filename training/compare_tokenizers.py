#!/usr/bin/env python3
"""Compare compression and embedding budgets, not trained model quality.

Run from any directory. This English-only character BPE probe preserves word-end
markers and every existing non-word token. It does not change active tokenizers,
source text, model weights, or the website. Evaluation uses the same corpus as
BPE fitting, so the result is an in-sample compression measurement.
"""
from collections import Counter
import json,re,sys,string,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from conversations.canonical import canonicalize
root=Path(__file__).resolve().parent/'conversations'
tokenizer=json.loads((root/'tokenizer.json').read_text())
excluded=json.loads((root/'excluded-conversations.json').read_text())
pattern=re.compile(tokenizer['token_pattern']);frequency=Counter();conversations=[]
for path in sorted((root/'authored').glob('*.jsonl')):
 for line in path.read_text().splitlines():
  row=json.loads(line)
  if row['id'] in excluded: continue
  row=canonicalize(row,tokenizer)
  words=Counter();overhead=2
  for message in row['messages']:
   parts=pattern.findall(message['content']);terms=[p for p in parts if re.fullmatch('[a-z]+',p)]
   words.update(terms);overhead+=len(parts)-len(terms)+2
  frequency.update(words);conversations.append((words,overhead))
baseline=sum(frequency.values())+sum(n for _,n in conversations)
assert baseline==json.loads((root/'audit.json').read_text())['model_tokens']
# Frequency-weighted character BPE, word boundaries retained as an explicit end symbol.
lexicon={word:tuple(word)+('▁',) for word in frequency}
vocabulary=set(string.ascii_lowercase)|{'▁'}
reserved=sum(not re.fullmatch('[a-z]+',token) for token in tokenizer['tokens'])
results=[]
start=time.monotonic()
while len(vocabulary)+reserved<=512:
 size=len(vocabulary)+reserved
 if size in (256,384,512):
  for word,parts in lexicon.items(): assert ''.join(parts).replace('▁','')==word
  total=sum(frequency[w]*len(parts) for w,parts in lexicon.items())+sum(n for _,n in conversations)
  longest=max(sum(count*len(lexicon[word]) for word,count in words.items())+n for words,n in conversations)
  results.append({'vocab_ids':size,'corpus_model_tokens':total,'sequence_ratio_vs_word':round(total/baseline,3),'longest_conversation_tokens':longest,'tied_embedding_parameters_at_width_48':size*48})
  if size==512: break
 pairs=Counter()
 for word,parts in lexicon.items():
  for pair in zip(parts,parts[1:]): pairs[pair]+=frequency[word]
 if not pairs: break
 best=max(pairs,key=lambda pair:(pairs[pair],pair))
 merged=''.join(best)
 vocabulary.add(merged)
 for word,parts in lexicon.items():
  updated=[];i=0
  while i<len(parts):
   if i+1<len(parts) and (parts[i],parts[i+1])==best:
    updated.append(merged);i+=2
   else: updated.append(parts[i]);i+=1
  lexicon[word]=tuple(updated)
print(json.dumps({'source_tokenizer':tokenizer['name'],'source_english_words':sum(frequency.values()),'observed_word_types':len(frequency),'reserved_ids':reserved,'method':'temporary English-only character BPE compression probe; 26 letters plus word-end marker, 20 reserved token IDs; trained and measured on the same current canonical corpus; not model-quality evidence','baseline':{'vocab_ids':len(tokenizer['tokens']),'corpus_model_tokens':baseline,'longest_conversation_tokens':max(sum(w.values())+n for w,n in conversations),'tied_embedding_parameters_at_width_48':len(tokenizer['tokens'])*48},'bpe':results,'seconds':round(time.monotonic()-start,2)},indent=2))
