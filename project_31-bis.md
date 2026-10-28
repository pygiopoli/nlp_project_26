Project 31-bis (extended): WordSense Disambiguation in Medical Context
This project aims to contribute to Word Sense Disambiguation in the domains of biomedical and clinical text, and studies how much the results depend on the algorithm and how much on the knowledge source used to describe the senses (WordNet vs. UMLS).

1) Initially study again Lesk implementation in NLTK. Consider the sentence “I have been prescribed two
important drugs today during my visit to clinic” and assume that the target word is “drug”. Construct a
simple script that outputs the correct sense of “drug” in the above sentence using standard Lesk
algorithm, first without any part-of-speech restriction and then restricting the candidate senses to the
part of speech that a POS tagger assigns to the target. Then implement an extended Lesk algorithm, whose
signature includes the definition and examples of each sense and of its related synsets (hypernyms,
hyponyms, meronyms, holonyms, similar senses, entailments, verb groups), compared with the lemmatized
context without stopwords. Run it with and without the POS restriction, print the overlap score of each
sense and flag the choices made by a tie or with no overlap. And, comment on the outcome the efficiency of
the result.

2) [Extended] Download and study MSH-WSD dataset from https://github.com/clips/yarn. The database contains 203
ambiguous terms that include regular terms, acronyms. Define a rule that separates acronyms from regular
terms, correct its mistakes by hand and exclude the words that can be both. Make a random, reproducible
selection of 10 acronyms and 10 regular terms. Since MSH-WSD gives the correct senses as UMLS concepts
(CUIs) while Lesk answers with WordNet synsets, retrieve the name of each CUI from the UMLS REST API and
build by hand a CUI → WordNet mapping (a CUI with no equivalent WordNet sense is mapped to no synset). Use
Lesk algorithm to perform the wordsense disambiguation of each of these 10 acronym and 10 regular terms and
use the ground truth to compute the average accuracy on regular terms and average accuracy on acronyms.
Report also the WordNet coverage, the accuracy on the covered senses and the accuracy of a random choice,
considering separately the words with at least two WordNet senses. Repeat the experiment on a second
random selection of 10 acronyms and 10 regular terms, drawn only among the words with at least two WordNet
senses. For the words where Lesk is worse than a random choice, count the senses it chooses and explain
the errors. Compare four variants of Lesk (with and without stopword removal, with and without POS
filtering) and measure how often no sense shares any word with the context. Finally, compare Lesk with the
Most Frequent Sense baseline and with your own simplified Lesk, using first a cleaned definition and then
a signature extended with the examples and synonyms of each sense.

3) Study the various disambiguation algorithms implemented in https://github.com/alvations/pywsd.
Show that you can run the program on simple sentences containing a target word to disambiguate (at least
the sentence of 1), the example of the pywsd documentation and a medical sentence) and output for various
disambiguation algorithms (Original Lesk, adapted/extended Lesk, simple Lesk extended with hypernym and
hyponyms, cosine Lesk, path-semantic similarity, Wu-Palmer and Leacock-Chodorow similarity, Information
content-semantic similarity (Resnik, Jiang-Conrath, Lin), highest Lemma count). For each algorithm, show
the best-ranked senses with their scores and the words they share with the sentence, the position of the
correct sense, and explain the anomalous results (e.g. rankings sorted in the wrong order or infinite
scores).

4) Run the above various algorithms in 3) on MSH-WSD dataset, together with the NLTK Lesk of 2) and a
corrected version of any algorithm whose ranking is wrong, and report the average accuracy on all acronyms
and average accuracy on all regular terms. Translate the senses returned by pywsd into the WordNet version
of the ground truth (e.g. through sense keys), use a context window around the target word and a sample of
abstracts balanced across the senses, and report both the accuracy and the accuracy on the senses covered
by WordNet, on average and for each word.

5) [Extended] We would like to extend similarity-based Lesk algorithm to include phonetic matching, which evaluates
phonetically similar wording. For this purpose, use the “Fuzzy” library in python (or an equivalent
implementation of Soundex) to generate the character string that identifies phonetically similar words,
and then use edit distance to compute the phonetic similarity between two words. See an example of
implementation at https://www.analyticsvidhya.com/blog/2017/01/ultimate-guide-to-understand-implement-natural-language-processing-codes-in-python/. Demonstrate this reasoning on the example sentence pointed
out in 1) and generate a matrix that indicates the phonetic distance between each pair of words of the
sentence, next to the matrix of the plain edit distance between the same words. Repeat the comparison on
translations of the sentence into other languages (Italian, Finnish, French, German, Spanish, Russian).

6) We would like to extend the Wu and Palmer-semantic similarity extension of Lesk algorithm
implemented in 3) by incorporating the phonetic similarity as well so that the overall similarity is
computed as a convex combination of Wu and Palmer similarity and phonetic similarity (after being
normalized) with coefficient of 0.5. Suggest and an implementation and run your new code on selected
MSH-WSD dataset. Compare the results with other values of the coefficient (e.g. 1 and 0.8), show on the
sentence of 1) how the phonetic part changes the chosen sense, and identify the words where it helps and
the words where it hurts.

7) [Extended] Suggest a GUI of your choice that allows the user to visualize the various outputs of the disambiguation
task. Ideally the user can input a sentence and a target word to be disambiguated. The user should be able
to run one algorithm or compare all of them and see, for each algorithm, the chosen sense, its score, the
ranking of all the candidate senses (also as a chart), the words shared by the sentence and the
definition, the POS tags of the sentence, and whether the algorithms agree. Design it so that new
algorithms can be added without changing the interface, and test it.

8) Identify relevant literature and critical analysis in this field.

9) [Extension] Repeat 4) and 6) on the full MSH-WSD dataset. For this purpose, download the UMLS names and definitions
of all the CUIs, build an automatic CUI → WordNet mapping and validate it against the manual mapping of 2)
(precision, recall, F1).

10) [Extension] Use the phonetic matching of 5) inside the simplified Lesk of 2), to compare words instead of senses: a
word of the context matches a word of the signature if the two are identical, if they have the same
Soundex code, or if they differ by at most one edit. Test the three rules on pairs of spelling variants and
of different words, run them on the selected MSH-WSD words and analyse the abstracts where the chosen
sense changes.

11) [Extension] Disambiguate the MSH-WSD words directly among their UMLS senses, without WordNet, using the biomedical
embeddings of SapBERT: describe each CUI with its UMLS names and definition, embed the description and the
context window, and choose the sense with the highest cosine similarity. Report the average accuracy on all
acronyms and on all regular terms against a random choice, compare it with the algorithms of 4), study the
effect of the size of the context window and of the description of the senses (names, definition or both),
and add the method to the GUI of 7).
