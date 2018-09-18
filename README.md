## Hidden Markov Model (HMM) 기반 한국어 형태소 분석기

세종 말뭉치를 이용하여 학습한 HMM 기반 한국어 형태소 분석기입니다. HMM 을 이용하여 형태소 분석을 하는 과정을 설명하기 위한 코드로, 오로직 Python 코드로만 이뤄져 있습니다.

이 repository 에서는 HMM 모델을 학습하는 Trainer 와, 학습된 모델을 이용하여 품사 판별을 하는 TrainedHMMTagger 을 제공합니다.

models/ 에는 세종 말뭉치를 이용하여 학습한 HMM 의 emission, transition probability 가 JSON 형식으로 저장되어 있습니다. 구현 과정 및 원리에 대해서는 [블로그 포스트][hmm_tagger_post]를 참고하세요.

## Usage

### Training

학습을 위하여 Corpus 와 CorpusTrainer class 를 import 합니다.

    from hmm_postagger import Corpus
    from hmm_postagger import CorpusTrainer

    data_path = '../data/sejong_simpletag.txt'
    model_path = '../models/sejong_simple_hmm.json'

Corpus 는 nested list 형식의 문장을 yield 하는 class 입니다. 학습에 이용한 네 문장의 예시입니다. 각 문장은 list 로 표현되며, 문장은 [형태소, 품사] 의 list 로 구성되어 있습니다.

    corpus = Corpus(sejong_path)
    for i, sent in enumerate(corpus):
        if i > 3:
            break
        print(sent)

    [['뭐', 'Noun'], ['타', 'Verb'], ['고', 'Eomi'], ['가', 'Verb'], ['ㅏ', 'Eomi']]
    [['지하철', 'Noun']]
    [['기차', 'Noun']]
    [['아침', 'Noun'], ['에', 'Josa'], ['몇', 'Determiner'], ['시', 'Noun'], ['에', 'Josa'], ['타', 'Verb'], ['고', 'Eomi'], ['가', 'Verb'], ['는데', 'Eomi']]

CorpusTrainer 에 품사의 min count, 단어의 min count 를 설정한 뒤, corpus 와 model_path 를 train 함수에 입력합니다.

    trainer = CorpusTrainer(min_count_tag=5, min_count_word=1, verbose=True)
    trainer.train(corpus, model_path)

model_path 에 JSON 형식으로 모델이 저장되어 있습니다. 모델은 세 종류의 정보가 담겨 있습니다.

    import json
    with open('../models/sejong_simple_hmm.json', encoding='utf-8') as f:
        model = json.load(f)

    print(model.keys())
    # dict_keys(['emission', 'transition', 'begin'])

emission 은 {tag:{word:prob}} 형식의 nested dict 이며 transition 은 {'Noun -> Josa': prob} 형식의 dict 입니다. begin 은 문장의 시작 단어로 품사가 등장할 가능성, P(tag | BOS) 입니다.

### Tagging

학습된 형태소 분석기는 hmm model 파일을 입력해야 합니다.

    from hmm_postagger import TrainedHMMTagger

    model_path = '../models/sejong_simple_hmm.json'
    tagger = TrainedHMMTagger(model_path)

예시로 세 문장에 대한 형태소 분석을 수행합니다.

    from pprint import pprint

    sents = [
        '주간아이돌에 아이오아이가 출연했다',
        '이번 경기에서는 누가 이겼을까',
        '아이고 작업이 쉽지 않구만',
        '샤샨 괜찮아'
    ]

    for sent in sents:
        print('\n\n{}'.format(sent))
        pprint(tagger.tag(sent))

2, 3 번째 문장의 단어들은 세종말뭉치에 존재하였기 때문에 형태소 분석이 어느 정도 되지만, '주간아이돌'과 '아이오아이'는 미등록단어 문제가 발생하여 형태소 분석이 제대로 이뤄지지 않습니다.

    주간아이돌에 아이오아이가 출연했다
    [('주간', 'Noun'),
     ('아이돌', 'Noun'),
     ('에', 'Josa'),
     ('아이오아', 'Noun'),
     ('이', 'Josa'),
     ('가', 'Josa'),
     ('출연', 'Noun'),
     ('하', 'Verb'),
     ('았', 'Eomi'),
     ('다', 'Eomi')]

    이번 경기에서는 누가 이겼을까
    [('이번', 'Noun'),
     ('경기', 'Noun'),
     ('에서', 'Josa'),
     ('는', 'Josa'),
     ('누', 'Noun'),
     ('가', 'Josa'),
     ('이기', 'Verb'),
     ('었', 'Eomi'),
     ('을까', 'Eomi')]

    아이고 작업이 쉽지 않구만
    [('아이고', 'Exclamation'),
     ('작업', 'Noun'),
     ('이', 'Josa'),
     ('쉽', 'Adjective'),
     ('지', 'Eomi'),
     ('않', 'Verb'),
     ('구만', 'Eomi')]

    샤샨 괜찮아
    [('샤샤', 'Noun'),
     ('ㄴ', 'Josa'),
     ('괜찮', 'Adjective'),
     ('아', 'Eomi')]

사용자 사전을 추가할 수 있는 기능을 넣었습니다. 사용자 사전이 입력되면 해당 단어들은 각 품사에서 가장 큰 emission probability 를 지닙니다. 즉, 다른 어떤 단어보다도 우선적으로 추가한 단어를 선호합니다.

    tagger.add_user_dictionary('Noun', ['아이오아이', '주간아이돌'])
    pprint(tagger.tag('주간아이돌에 아이오아이가 출연했다'))

    [('주간아이돌', 'Noun'),
     ('에', 'Josa'),
     ('아이오아이', 'Noun'),
     ('가', 'Josa'),
     ('출연', 'Noun'),
     ('하', 'Adjective'),
     ('았', 'Eomi'),
     ('다', 'Eomi')]

### Inferring unknown word

형태소 분석을 하여도 전혀 보지 못한 string 이 존재할 수 있습니다. '갹갹' 이라는 단어는 등록된 형태소로도 분해하지 못합니다.

    sent = '갹갹은 어디있어'
    tagger.tag(sent, inference_unknown=False)

    [('갹갹', 'Unk'),
     ('은', 'Josa'),
     ('어디', 'Noun'),
     ('있', 'Verb'),
     ('어', 'Eomi')]

위와 같은 경우에 '갹갹'의 앞 단어 (BOS) 에서의 state transition probability 와 '갹갹'의 뒤 단어 '은/josa'으로의 state transition probability 를 고려하여 '갹갹'의 품사를 추정합니다. inference_unknown 의 기본값은 True 입니다.

    sent = '갹갹은 어디있어'
    tagger.tag(sent, inference_unknown=True)

    [('갹갹', 'Noun'),
     ('은', 'Josa'),
     ('어디', 'Noun'),
     ('있', 'Verb'),
     ('어', 'Eomi')]

품사 추정의 기능을 이용하면 아래와 같이 영문과 한글이 혼용된 경우, 영어 단어의 품사도 추정할 수 있습니다.

    tt는 좋은 노래야
    [('tt', 'Noun'),
     ('는', 'Josa'),
     ('좋', 'Adjective'),
     ('은', 'Eomi'),
     ('노래', 'Noun'),
     ('야', 'Josa')]

## TODO

###  기호 처리

마침표나 물음표와 같은 기호가 입력되지 않음을 가정하였습니다. 오로직 완전한 한글이 입력된 경우만을 가정하여 세종 말뭉치 데이터에서도 기호는 제거한 뒤 학습하였습니다.

실제 문제에 적용가능하도록 모델을 변형하려면 외래어, 기호에 대한 처리륻 더해야 합니다.

### shortest path 최적화

HMM 의 decoding 과정은 shortest path 문제와 같습니다. 현재 구현된 코드에서는 shortest path 의 최적화가 이뤄져 있지 않습니다.


[hmm_tagger_post]:https://lovit.github.io/nlp/2018/09/11/hmm_based_tagger/
