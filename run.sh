#this function will convert text to lowercase and will disconnect punctuation and special symbols from words
function normalize_text {
  awk '{print tolower($0);}' < $1 | sed -e 's/\./ \. /g' -e 's/<br \/>/ /g' -e 's/"/ " /g' \
  -e 's/,/ , /g' -e 's/(/ ( /g' -e 's/)/ ) /g' -e 's/\!/ \! /g' -e 's/\?/ \? /g' \
  -e 's/\;/ \; /g' -e 's/\:/ \: /g' > $1-norm
}

wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xf aclImdb_v1.tar.gz

echo "Start to preprocess dataset."
for j in train/pos train/neg test/pos test/neg train/unsup; do
  rm -rf temp
  for i in `ls aclImdb/$j`; do cat aclImdb/$j/$i >> temp; awk 'BEGIN{print;}' >> temp; done
  normalize_text temp
  mv temp-norm aclImdb/$j/norm.txt
  echo "Finish with $j"
done
mv aclImdb/train/pos/norm.txt train-pos.txt
mv aclImdb/train/neg/norm.txt train-neg.txt
mv aclImdb/test/pos/norm.txt test-pos.txt
mv aclImdb/test/neg/norm.txt test-neg.txt
mv aclImdb/train/unsup/norm.txt train-unsup.txt

cat train-pos.txt train-neg.txt test-pos.txt test-neg.txt train-unsup.txt > alldata.txt
echo "Start to preprocess dataset - step 1 finished."
awk 'BEGIN{a=0;}{print "_*" a " " $0; a++;}' < alldata.txt > alldata-id.txt
echo "Finish to preprocess dataset."

# Compile binary
cd ./source
make
cd ../
mv ./source/doc2vec ./

# Start training with default parameters and save model
./doc2vec train --data alldata-id.txt --save model.txt

# Print similar for first and random documents in dataset
./doc2vec similar --load model.txt --doc _*0 --doc _*3213 --num 2
