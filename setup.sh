#download data
mkdir data
cd data

wget http://alt.qcri.org/semeval2014/task1/data/uploads/sick_train.zip
wget http://alt.qcri.org/semeval2014/task1/data/uploads/sick_trial.zip
wget http://alt.qcri.org/semeval2014/task1/data/uploads/sick_test_annotated.zip

unzip sick_test_annotated.zip
rm readme.txt
unzip sick_train.zip
rm readme.txt
unzip sick_trial.zip
rm readme.txt
rm *.zip

wget http://ttic.uchicago.edu/~wieting/iclr-demo.zip
unzip -j iclr-demo.zip
rm iclr-demo.zip

#download jars
cd ..
mkdir lib
cd lib

wget http://nlp.stanford.edu/software/stanford-corenlp-full-2014-08-27.zip
wget http://www.gtlib.gatech.edu/pub/apache/commons/io/binaries/commons-io-2.4-bin.zip

unzip stanford-corenlp-full-2014-08-27.zip
unzip commons-io-2.4-bin.zip
rm *.zip

#preprocess
cd ../preprocess
javac -classpath ../lib/commons-io-2.4/commons-io-2.4.jar:../lib/stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1.jar:../lib/stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1-models.jar Preprocess.java
java -classpath ../lib/commons-io-2.4/commons-io-2.4.jar:../lib/stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1.jar:../lib/stanford-corenlp-full-2014-08-27/stanford-corenlp-3.4.1-models.jar:. Preprocess