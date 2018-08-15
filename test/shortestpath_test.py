import sys
sys.path.append('../')

from pprint import pprint
from hmm_postagger.shortestpath import sent_to_graph

def test_sent_to_graph():
    sent_len = 4
    words = [
        ('뭐', 'Noun', 0, 1),
        ('타', 'Verb', 1, 2),
        ('고', 'Eomi', 2, 3),
        ('고', 'Noun', 2, 3),
        ('가', 'Verb', 3, 4),
        ('가', 'Noun', 3, 4),
        ('ㅏ', 'Eomi', 4, 4),
        ('EOS', 'EOS', 4, 5)
    ]

    sent = [[] for _ in range(sent_len+1)]
    for word in words:
        sent[word[2]].append(word)

    edges, idx2node = sent_to_graph(sent, sent_len)
    pprint(edges)
    pprint(idx2node)

if __name__ == '__main__':
    test_sent_to_graph()