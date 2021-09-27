import os


def load_file(filepath,):
    with open(filepath, encoding='utf-8') as f:
        a = []
        for i in f:
            a.append(i.strip(' \n'))
        return a


def save_file(filepath, lst):
    with open(filepath, 'w', encoding='utf-8') as f:
        for i in range(len(lst)):
            f.write(lst[i])
            if i != len(lst) - 1:
                f.write('\n')


if __name__ == "__main__":
    result = load_file('result.txt')
    neural_idx = load_file('neural_idx_in_test.txt')
    wrong_nums = 0
    for idx in neural_idx:
        try:
            if result[int(idx)][-1] != '0':
                wrong_nums += 1
            result[int(idx)] = result[int(idx)][:-1] + '0'
        except:
            print(idx)
    save_file('final_result.txt', result)
    print(wrong_nums)
