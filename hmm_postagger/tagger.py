from collections import defaultdict
import json

class TrainedHMMTagger:
    def __init__(self, model_path=None, transition=None, emission=None,
        begin=None, transition_smoothing=1e-8, emission_smoothing=1e-8,
        unknown_penalty=-10, end_state='EOS'):

        self.transition = transition if transition else {}
        self.emission = emission if emission else {}
        self.begin = begin if begin else {}
        self._ts = transition_smoothing
        self._es = emission_smoothing
        self.unknown_penalty = unknown_penalty
        self.end_state = end_state

        if isinstance(model_path, str):
            self.load_model_from_json(model_path)

        # for user-dictionary, max value
        self._max_score = {
            state:max(observations.values())
            for state, observations in self.emission.values()
        }

    def load_model_from_json(self, model_path):
        with open(model_path, encoding='utf-8') as f:
            model = json.load(f)
        self.emission = model['emission']
        self.transition = model['transition']
        self.transition = {tuple(states.split()):prob for states, prob in self.transition.items()}
        self.begin = model['begin']
        del model

    def append_user_dictionary(self, tag, words):
        if not (tag in self.emission):
            raise ValueError('%s does not exist in tagset' % tag)
        append_score = self._max_score[tag]
        for word in words:
            self.emission[tag][word] = append_score

    def decode(self, sequence):
        raise NotImplemented

    def log_probability(self, sequence):
        # emission probability
        log_prob = sum(
            (self.emission.get(t, {}).get(w,self.unknown_penalty)
             for w, t in sequence)
        )

        # bos
        log_prob += tagger.bos.get(sent[0][1], self.unknown_penalty)

        # transition probability
        transitions = [(t0, t1) for (_, t0), (_, t1) in zip(sequence, sequence[1:])]
        log_prob += sum(
            (self.transition.get(transition, self.unknown_penalty)
             for transition in transitions))

        # eos
        log_prob += self.transition.get(
            (sequence[-1][1], self.end_state), self.unknown_penalty
        )

        # length normalization
        log_prob /= len(sequence)

        return log_prob