cd /data/
echo "Starting from $PWD"

wget http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip
unzip cornell_movie_dialogs_corpus.zip
echo "Creating target folder"
mkdir cornell_movie_dialogs_corpus
echo "Moving downloaded files to the target folder"
mv cornell\ movie-dialogs\ corpus/* cornell_movie_dialogs_corpus/
echo "Removing temporal folder"
rm -r cornell\ movie-dialogs\ corpus
echo "Done"