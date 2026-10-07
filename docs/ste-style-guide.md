# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the tweetmood documentation. Code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **tweet** | One short social-media text with one label. | post, message, sample |
| **label** | `negative` or `positive`. | class (for the value), polarity, tag |
| **source** | The corpus that a tweet comes from, for example `sentiment140`. | dataset (for one corpus), origin |
| **normalizer** | The one text function in `text/normalize.py` that every model uses. | preprocessor, cleaner |
| **emoticon** | A text face such as `:D` or `<3`. | smiley, emoji (for a text face) |
| **emoji** | A Unicode pictograph such as 🔥. | emoticon (for a pictograph), icon |
| **slang map** | The phrase-to-plain-English table `SLANG` in `text/lexicon.py`. | dictionary, slang list |
| **negator** | A word that reverses the next words: `not`, `no`, `never` and others in `NEGATORS`. | negation word, stop word |
| **negation scope** | Up to three words after a negator, marked with `NEG_`. | negation window |
| **shared split** | The one `split.csv` with `train`, `val` and `test` ids that every model reads. | partition, fold |
| **fingerprint** | The hash of all `(id, split)` pairs of the shared split. | split hash, version |
| **holdout source** | A source that goes to the test split only. | unseen domain, OOD set |
| **model** | One of `lexicon`, `tfidf-logreg`, `tfidf-svm`, `tfidf-nb`, `hybrid-bilstm`, `hybrid-transformer`, `finetune`. | classifier (alone), algorithm |
| **classic model** | The lexicon model or a TF-IDF model. | baseline (alone), traditional model |
| **hybrid model** | A trained head on the states of a frozen embedder. | hybrid BERT, frozen BERT |
| **embedder** | The frozen part of a hybrid model: `hash` or `hf`. | encoder (for this part), backbone |
| **head** | The trained part of a hybrid model: `bilstm` or `transformer`. | classifier layer, top |
| **registry** | `registry.json` with the validation score of each trained model and the best model. | model store, catalog |
| **best model** | The model with the highest validation macro-F1 in the registry. | winner, champion |
| **slice** | A subset of the test split: one source, tweets with a negator, with slang or with an emoji or emoticon. | segment, bucket |
| **ablation** | A run of one TF-IDF model per normalizer variant. | experiment (alone), comparison |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **normalize** | Apply the normalizer to a tweet. |
| **expand** | Replace a slang phrase or a contraction with plain English. |
| **prepare** | Load sources, de-duplicate and write the shared split. |
| **train** | Fit a model on `train` and select its settings on `val`. |
| **select** | Choose a setting or the best model on validation macro-F1. |
| **evaluate** | Score every trained model on the test split one time. |
| **predict** | Label new tweets with the best model. |
