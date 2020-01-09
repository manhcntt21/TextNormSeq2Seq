
import random

def get_repleace_character():
    repleace_character = {}
    # repleace_character['ch'] = ['tr']
    # repleace_character['tr'] = ['ch']
    repleace_character['l'] = ['n']
    repleace_character['n'] = ['l']
    repleace_character['x'] = ['s']
    repleace_character['s'] = ['x']
    repleace_character['r'] = ['d', 'gi']
    repleace_character['d'] = ['r', 'gi']
    # repleace_character['gi'] = ['d', 'r']
    repleace_character['c'] = ['q', 'k']
    repleace_character['k'] = ['q', 'c']
    repleace_character['q'] = ['c', 'k']
    repleace_character['i'] = ['y']
    repleace_character['y'] = ['i']
    repleace_character['_'] = ['_']
    return repleace_character


def get_prox_keys():
    array_prox = {}
    array_prox['a'] = ['q', 'w', 'z', 'x', 's']
    array_prox['b'] = ['v', 'f', 'g', 'h', 'n', ' ']
    array_prox['c'] = ['x', 's', 'd', 'f', 'v']
    array_prox['d'] = ['x', 's', 'w', 'e', 'r', 'f', 'v', 'c']
    array_prox['e'] = ['w', 's', 'd', 'f', 'r']
    array_prox['f'] = ['c', 'd', 'e', 'r', 't', 'g', 'b', 'v']
    array_prox['g'] = ['r', 'f', 'v', 't', 'b', 'y', 'h', 'n']
    array_prox['h'] = ['b', 'g', 't', 'y', 'u', 'j', 'm', 'n']
    array_prox['i'] = ['u', 'j', 'k', 'l', 'o']
    array_prox['j'] = ['n', 'h', 'y', 'u', 'i', 'k', 'm']
    array_prox['k'] = ['u', 'j', 'm', 'l', 'o']
    array_prox['l'] = ['p', 'o', 'i', 'k', 'm']
    array_prox['m'] = ['n', 'h', 'j', 'k', 'l']
    array_prox['n'] = ['b', 'g', 'h', 'j', 'm']
    array_prox['o'] = ['i', 'k', 'l', 'p']
    array_prox['p'] = ['o', 'l']
    array_prox['q'] = ['w', 'a']
    array_prox['r'] = ['e', 'd', 'f', 'g', 't']
    array_prox['s'] = ['q', 'w', 'e', 'z', 'x', 'c']
    array_prox['t'] = ['r', 'f', 'g', 'h', 'y']
    array_prox['u'] = ['y', 'h', 'j', 'k', 'i']
    array_prox['v'] = ['', 'c', 'd', 'f', 'g', 'b']
    array_prox['w'] = ['q', 'a', 's', 'd', 'e']
    array_prox['x'] = ['z', 'a', 's', 'd', 'c']
    array_prox['y'] = ['t', 'g', 'h', 'j', 'u']
    array_prox['z'] = ['x', 's', 'a']
    array_prox['1'] = ['q', 'w']
    array_prox['2'] = ['q', 'w', 'e']
    array_prox['3'] = ['w', 'e', 'r']
    array_prox['4'] = ['e', 'r', 't']
    array_prox['5'] = ['r', 't', 'y']
    array_prox['6'] = ['t', 'y', 'u']
    array_prox['7'] = ['y', 'u', 'i']
    array_prox['8'] = ['u', 'i', 'o']
    array_prox['9'] = ['i', 'o', 'p']
    array_prox['0'] = ['o', 'p']
    array_prox['_'] = ['_']
    return array_prox

def add_noise(word):
    # i = random.randint(0,len(word)-1)
    # op = random.randint(0, 30)
    i = 0
    op = 4
    if op == 0:
        return word[:i] + word[i+1:]
    if op == 1:
        i += 1
        return word[:i-1] + word[i:i+1] + word[i-1:i] + word[i+1:]
    if op == 2 or op == 3:
        try:
            # print(op)
            print(random.choice(repleace_character[word[i]]))
            return word[:i] + random.choice(repleace_character[word[i]]) + word[i+1:] # thay doi dau
        except:
            return word
    # print(random.choice(get_prox_keys[word[i]]))
    # return word[:i] + random.choice(get_prox_keys[word[i]]) + word[i+1:]
    try:
        tmp = get_prox_keys()
        tmp1 =  random.choice(tmp[word[i]])
        print(tmp1)
        return word[:i] + tmp1 + word[i+1:] #default is keyboard errors
    except :
        # print(word)
        return word


if __name__ == '__main__':
    # a = '[anh'
    # b = add_noise(a)
    # print(b)
    # b = get_prox_keys()

    a = 10
    b = 'string'
    c = str(a)

    print(type(a))
    print(type(c))
    