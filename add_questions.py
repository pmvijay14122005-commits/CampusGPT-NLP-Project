import csv

NEW = {
 "dev": [
  "What is lemmatization and how is it different from stemming?",
  "How does the WordNet lemmatizer in NLTK find the lemma of a word?",
  "What are over-stemming and under-stemming errors in stemmers?",
  "Why is the Lancaster stemmer more aggressive than the Porter stemmer in stripping suffixes?",
  "What is the difference between a Mealy machine and a Moore machine?",
  "How does root-pattern morphology work in Arabic, for example with the root k-t-b?",
  "Give an example of a Turkish word built from many suffixes and break it into its morphemes.",
  "What is vowel harmony and which languages show it?",
  "What is the time complexity of recognizing a word with a finite state automaton?",
  "How can stemmers be evaluated, for example with precision and recall?",
 ],
 "test": [
  "What is the difference between a deterministic and a non-deterministic finite state automaton?",
  "How does TF-IDF weighting improve the cosine similarity between two documents?",
  "How does a hidden Markov model perform part-of-speech tagging?",
  "What is the Krovetz stemmer and when is it preferred over the Porter stemmer?",
  "What is a clitic and how is it handled in morphological analysis?",
  "How does Koskenniemi's two-level model handle conflicts between spelling rules?",
  "What is the difference between compounding and derivation, with examples?",
  "How do neural sequence-to-sequence models perform morphological inflection?",
  "What is the Levenshtein minimum edit distance and how is it computed?",
  "How does the first step of the Porter stemmer handle plurals such as 'caresses' and 'ponies'?",
 ],
}

with open("eval_questions.csv", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    have = {(r.get("question") or "").strip().lower() for r in reader}

with open("eval_questions.csv", "rb") as f:
    needs_newline = not f.read().endswith(b"\n")

added = 0
with open("eval_questions.csv", "a", newline="", encoding="utf-8") as f:
    if needs_newline:
        f.write("\r\n")
    w = csv.DictWriter(f, fieldnames=fields)
    for split, questions in NEW.items():
        for q in questions:
            if q.lower() in have:
                continue
            w.writerow({"question": q, "answer": "", "answerable": "no", "page": "", "file": "", "split": split})
            added += 1
print("Added", added, "questions")