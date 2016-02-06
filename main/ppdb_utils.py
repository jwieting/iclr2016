import numpy as np
from tree import tree
import time
from random import randint
from random import choice
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import utils
import sys
from evaluate import evaluate_all

def lookupIDX(words,w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return words[w]
    else:
        return words['UUUNKKK']

def getPPDBData(f,words):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) == 2:
                e = (tree(i[0], words), tree(i[1], words))
                examples.append(e)
            else:
                print i
    return examples

def getSimEntDataset(f,words,task):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) == 3:
                if task == "sim":
                    e = (tree(i[0], words), tree(i[1], words), float(i[2]))
                    examples.append(e)
                elif task == "ent":
                    e = (tree(i[0], words), tree(i[1], words), i[2])
                    examples.append(e)
                else:
                    raise ValueError('Params.traintype not set correctly.')

            else:
                print i
    return examples

def getSentimentDataset(f,words):
    data = open(f,'r')
    lines = data.readlines()
    examples = []
    for i in lines:
        i=i.strip()
        if(len(i) > 0):
            i=i.split('\t')
            if len(i) == 2:
                e = (tree(i[0], words), i[1])
                examples.append(e)
            else:
                print i
    return examples

def getWordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n
        We.append(v)
    return (words, np.array(We))

def getPairRand(d,idx):
    wpick = None
    ww = None
    while(wpick == None or (idx == ww)):
        ww = choice(d)
        ridx = randint(0,1)
        wpick = ww[ridx]
    return wpick

def getPairMixScore(d,idx,maxpair):
    r1 = randint(0,1)
    if r1 == 1:
        return maxpair
    else:
        return getPairRand(d,idx)

def getPairsFast(d, type):
    X = []
    T = []
    pairs = []
    for i in range(len(d)):
        (p1,p2) = d[i]
        X.append(p1.representation)
        X.append(p2.representation)
        T.append(p1)
        T.append(p2)

    arr = pdist(X,'cosine')
    arr = squareform(arr)

    for i in range(len(arr)):
        arr[i,i]=1
        if i % 2 == 0:
            arr[i,i+1] = 1
        else:
            arr[i,i-1] = 1

    arr = np.argmin(arr,axis=1)
    for i in range(len(d)):
        (t1,t2) = d[i]
        p1 = None
        p2 = None
        if type == "MAX":
            p1 = T[arr[2*i]]
            p2 = T[arr[2*i+1]]
        if type == "RAND":
            p1 = getPairRand(d,i)
            p2 = getPairRand(d,i)
        if type == "MIX":
            p1 = getPairMixScore(d,i,T[arr[2*i]])
            p2 = getPairMixScore(d,i,T[arr[2*i+1]])
        pairs.append((p1,p2))
    return pairs

def getpairs(model, batch, params):
    g1 = []
    g2 = []

    for i in batch:
        g1.append(i[0].embeddings)
        g2.append(i[1].embeddings)

    g1x, g1mask = utils.prepare_data(g1)
    g2x, g2mask = utils.prepare_data(g2)

    embg1 = model.feedforward_function(g1x, g1mask)
    embg2 = model.feedforward_function(g2x, g2mask)

    for idx, i in enumerate(batch):
        i[0].representation = embg1[idx, :]
        i[1].representation = embg2[idx, :]

    pairs = getPairsFast(batch, params.type)
    p1 = []
    p2 = []
    for i in pairs:
        p1.append(i[0].embeddings)
        p2.append(i[1].embeddings)

    p1x, p1mask = utils.prepare_data(p1)
    p2x, p2mask = utils.prepare_data(p2)

    return (g1x, g1mask, g2x, g2mask, p1x, p1mask, p2x, p2mask)


def train(model, data, words, params):
    start_time = time.time()

    counter = 0
    try:
        for eidx in xrange(params.epochs):

            kf = utils.get_minibatches_idx(len(data), params.batchsize, shuffle=True)
            uidx = 0
            for _, train_index in kf:

                uidx += 1

                batch = [data[t] for t in train_index]
                for i in batch:
                    i[0].populate_embeddings(words)
                    i[1].populate_embeddings(words)

                (g1x, g1mask, g2x, g2mask, p1x, p1mask, p2x, p2mask) = getpairs(model, batch, params)

                cost = model.train_function(g1x, g2x, p1x, p2x, g1mask, g2mask, p1mask, p2mask)

                if np.isnan(cost) or np.isinf(cost):
                    print 'NaN detected'

                if (utils.checkIfQuarter(uidx, len(kf))):
                    if (params.save):
                        counter += 1
                        utils.saveParams(model, params.outfile + str(counter) + '.pickle')
                    if (params.evaluate):
                        evaluate_all(model, words)
                        sys.stdout.flush()

                #undo batch to save RAM
                for i in batch:
                    i[0].representation = None
                    i[1].representation = None
                    i[0].unpopulate_embeddings()
                    i[1].unpopulate_embeddings()

                #print 'Epoch ', (eidx+1), 'Update ', (uidx+1), 'Cost ', cost

            if (params.save):
                counter += 1
                utils.saveParams(model, params.outfile + str(counter) + '.pickle')

            if (params.evaluate):
                evaluate_all(model, words)

            print 'Epoch ', (eidx + 1), 'Cost ', cost

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    print "total time:", (end_time - start_time)