from collections import defaultdict
import json

class TrainedHMMTagger:
    def __init__(self, model_path=None, transition=None, pos2words=None,
        bos=None, transition_smoothing=1e-8, generation_smoothing=1e-8, unknown_penalty=-10):

        self.transition = transition if transition else {}
        self.pos2words = pos2words if pos2words else {}
        self.bos = bos if bos else {}
        self._ts = transition_smoothing
        self._gs = generation_smoothing
        self.unknown_penalty = unknown_penalty

        if isinstance(model_path, str):
            self.load_model_from_json(model_path)

    def load_model_from_json(self, model_path):
        with open(model_path, encoding='utf-8') as f:
            model = json.load(f)
        self.pos2words = model['pos2words']
        self.transition = model['transition']
        self.transition = {tuple(pos.split()):prob for pos, prob in self.transition.items()}
        self.bos = model['bos']
        del model

    def decode(self, sentence):
        raise NotImplemented

    def log_probability(self, sentence):
        # emission probability
        log_prob = sum(
            (self.pos2words.get(t, {}).get(w,self.unknown_penalty)
             for w, t in sentence)
        )

        # bos
        log_prob += tagger.bos.get(sent[0][1], self.unknown_penalty)

        # transition probability
        bigrams = [(t0, t1) for (_, t0), (_, t1) in zip(sentence, sentence[1:])]
        log_prob += sum(
            (self.transition.get(bigram, self.unknown_penalty)
             for bigram in bigrams))

        # eos
        log_prob += self.transition.get(
            (sentence[-1][1], self.eos_tag), self.unknown_penalty
        )

        # length normalization
        log_prob /= len(sentence)

        return log_prob