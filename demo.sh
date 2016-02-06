#train word model on SICK similarity task
if [ "$1" == "similarity" ]; then
    sh train.sh -wordstem simlex -wordfile ../data/paragram_sl999_small.txt -outfile gpu-word-model -updatewords True -dim 300 -traindata ../data/sicktrain -devdata ../data/sickdev -testdata ../data/sicktest -layersize 300 -save False -nntype word -evaluate True -epochs 10 -minval 1 -maxval 5 -traintype normal -task sim -batchsize 25 -LW 1e-05 -LC 1e-06 -memsize 50 -learner adam -eta 0.001

#train word model on SICK entailment task
elif [ "$1" == "entailment" ]; then
    sh train.sh -wordstem simlex -wordfile ../data/paragram_sl999_small.txt -outfile gpu-word-model -updatewords True -dim 300 -traindata ../data/sicktrain-ent -devdata ../data/sickdev-ent -testdata ../data/sicktest-ent -layersize 300 -save False -nntype word -evaluate True -epochs 10 -traintype normal -task ent -batchsize 25 -LW 1e-05 -LC 1e-06 -memsize 150 -learner adagrad -eta 0.05

#train lstm w/ outgate on Stanford sentiment task
elif [ "$1" == "sentiment" ]; then
    sh train.sh -wordstem simlex -wordfile ../data/paragram_sl999_small.txt -outfile gpu-lstm-model -dim 300 -traindata ../data/sentiment-train -devdata ../data/sentiment-dev -testdata ../data/sentiment-test -layersize 300 -save False -nntype lstm_sentiment -evaluate True -epochs 10 -peephole True -traintype normal -task sentiment -updatewords True -outgate True -batchsize 25 -LW 1e-06 -LC 1e-06 -memsize 300 -learner adam -eta 0.001

#train ppdb models
elif [ "$1" == "ppdb-lstm-outgate" ]; then
    sh train_ppdb.sh -wordstem simlex -wordfile ../data/paragram_sl999_small.txt -outfile lstm-model-outgate -dim 300 -train ../data/ppdb-xl-phrasal-preprocessed.txt -layersize 300 -save False -nntype lstm -peephole True -evaluate True -epochs 10 -outgate True -updatewords True -batchsize 100 -LW 1e-06 -margin 0.4 -samplingtype MAX -LC 0.001 -clip 0 -eta 0.0005 -learner adam
elif [ "$1" == "ppdb-lstm-nooutgate" ]; then
    sh train_ppdb.sh -wordstem simlex -wordfile ../data/paragram_sl999_small.txt -outfile lstm-model-nooutgate -dim 300 -train ../data/ppdb-xl-phrasal-preprocessed.txt -layersize 300 -save False -nntype lstm -peephole True -evaluate True -epochs 10 -outgate False -updatewords True -batchsize 50 -LW 1e-06 -margin 0.4 -samplingtype MAX -LC 0.001 -clip 0 -eta 0.0005 -learner adam
elif [ "$1" == "ppdb-proj" ]; then
    sh train_ppdb.sh -wordstem simlex -wordfile ../data/paragram_sl999_small.txt -outfile proj-model -dim 300 -train ../data/ppdb-xl-phrasal-preprocessed.txt -layersize 300 -save False -nntype proj -evaluate True -epochs 10 -nonlinearity 1 -updatewords True -batchsize 100 -LW 1e-08 -margin 0.4 -samplingtype MAX -LC 1e-05 -clip 0 -eta 0.0005 -learner adam
elif [ "$1" == "ppdb-word" ]; then
    sh train_ppdb.sh -wordstem simlex -wordfile ../data/paragram_sl999_small.txt -outfile word-model -updatewords True -dim 300 -train ../data/ppdb-xl-phrasal-preprocessed.txt -layersize 300 -save False -nntype word -evaluate True -epochs 10 -batchsize 100 -LW 1e-06 -margin 0.4 -samplingtype MAX -clip 1 -eta 0.005 -learner adam
elif [ "$1" == "ppdb-irnn" ]; then
    sh train_ppdb.sh -wordstem simlex -wordfile ../data/paragram_sl999_small.txt -outfile irnn-model -dim 300 -train ../data/ppdb-xl-phrasal-preprocessed.txt -layersize 300 -save False -nntype rnn -evaluate True -epochs 10 -updatewords True -batchsize 100 -LW 1e-06 -margin 0.4 -samplingtype MAX -LC 10.0 -clip 1 -eta 0.005 -learner adam -add_rnn True
elif [ "$1" == "ppdb-rnn" ]; then
    sh train_ppdb.sh -wordstem simlex -wordfile ../data/paragram_sl999_small.txt -outfile rnn-model -dim 300 -train ../data/ppdb-xl-phrasal-preprocessed.txt -layersize 300 -save False -nntype rnn -evaluate True -epochs 10 -add_rnn False -updatewords True -batchsize 100 -LW 1e-06 -margin 0.4 -samplingtype MIX -LC 0.0001 -clip 0 -eta 0.0005 -learner adam -nonlinearity 2
elif [ "$1" == "ppdb-dan" ]; then
    sh train_ppdb.sh -wordstem simlex -wordfile ../data/paragram_sl999_small.txt -outfile dan-model -dim 300 -train ../data/ppdb-xl-phrasal-preprocessed.txt -layersize 300 -save False -nntype dan -evaluate True -epochs 10 -updatewords True -batchsize 100 -LW 1e-05 -margin 0.4 -samplingtype MAX -LC 1e-05 -clip 0 -eta 0.05 -learner adagrad -nonlinearity 2 -numlayers 1
else
    echo "$1 not a valid option."
fi