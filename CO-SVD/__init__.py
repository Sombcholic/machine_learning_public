import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    input_list = [
        'I like deep learning.',
        'I like NLP.',
        'I enjoy flying.'
    ]

    input_list_split = []

    symbol_list = [',', '.', '?', '!']

    la = np.linalg

    words_list = []

    # 有關連的字詞
    # EX: I: [like, like, enjoy]
    # EX: like: [I, I, deep]
    connect_word = {}

    # 先找出出現過的單字 - Start
    for sentence in input_list:
        split_sentence = []

        for symbol in symbol_list:
            if (sentence.find(symbol) >= 0):
                sentence = sentence.replace(symbol, ' ' + symbol)

        words = sentence.split(' ')
        input_list_split.append(words)

        for word in words:
            if (word not in words_list):
                words_list.append(word)

    print('全部出現過的單字')
    print(words_list, '\n')

    print('各句單字')
    print(input_list_split, '\n')
    # 先找出出現過的單字 - End

    # 算出詞頻 - Start
    window = 1
    for sentence in input_list_split:
        for i in range(len(sentence) - window):
            if (connect_word.get(sentence[i]) is not None):
                for w in range(window):
                    connect_word[sentence[i]].append(sentence[i+w+1])
                    try:
                        connect_word[sentence[i+w+1]].append(sentence[i])
                    except:
                        connect_word[sentence[i+w+1]] = []
                        connect_word[sentence[i+w+1]].append(sentence[i])
            else:
                connect_word[sentence[i]] = []
                for w in range(window):
                    connect_word[sentence[i]].append(sentence[i+w+1])
                    try:
                        connect_word[sentence[i+w+1]].append(sentence[i])
                    except:
                        connect_word[sentence[i+w+1]] = []
                        connect_word[sentence[i+w+1]].append(sentence[i])

    # 算出詞頻 - End

    print('各單字window = 1的連接字典')
    print(connect_word, '\n')

    svd_list = np.zeros(shape=(len(words_list), len(words_list)))
    i = 0
    for w in words_list:
        _list = np.array([])
        for w2 in words_list:
            if (w == w2):
                _list = np.append(_list, 0)
            else:
                number = connect_word.get(w).count(w2)
                _list = np.append(_list, number)
        
        svd_list[i] = _list
        i += 1


    print('陣列為', words_list)
    print(svd_list, '\n')

    U, s, Vh = la.svd(svd_list, full_matrices=False)

    for i in range(len(words_list)):
        plt.text(U[i, 0], U[i, 1], words_list[i])

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()


