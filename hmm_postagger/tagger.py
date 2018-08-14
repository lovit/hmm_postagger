class TrainedHMMTagger:
    def __init__(self, transition, generation, transition_smoothing=1e-8, generation_smoothing=1e-8):
        self.transition = transition
        self.generation = generation
        self._ts = transition_smoothing
        self._gs = generation_smoothing

    def tag(self, sentence):
        raise NotImplemented
