from collections import defaultdict
import json

class TrainedHMMTagger:
    def __init__(self, model_path=None, transition=None, pos2words=None,
        transition_smoothing=1e-8, generation_smoothing=1e-8):

        self.transition = transition if transition else {}
        self.pos2words = pos2words if pos2words else {}
        self._ts = transition_smoothing
        self._gs = generation_smoothing

        if isinstance(model_path, str):
            self.load_model_from_json(model_path)

    def load_model_from_json(self, model_path):
        with open(model_path, encoding='utf-8') as f:
            model = json.load(f)
        self.pos2words = model['pos2words']
        self.transition = model['transition']
        self.transition = {tuple(pos.split()):prob for pos, prob in self.transition.items()}
        del model

    def tag(self, sentence):
        raise NotImplemented