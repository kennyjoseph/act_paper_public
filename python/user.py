class User:

    def __init__(self, n_identities, sentences):
        self.sentences = sentences
        self.sentences_for_identity = [list() * n_identities]
        for sent_it, sentence in enumerate(sentences):
            self.add_sentence(sentence,sent_it)

    def add_sentence(self,sentence,sent_it=None):
        self.sentences.append(sentence)
        if not sent_it:
            sent_it = len(self.sentences)
        for identity in sentence.identities_contained():
            self.sentences_for_identity[identity].append(sent_it)

