prep=MultiUN
mkdir $prep

wget -P $prep https://conferences.unite.un.org/UNCORPUS/en/DownloadFile?filename=UNv1.0.en-ru.tar.gz.00
wget -P $prep https://conferences.unite.un.org/UNCORPUS/en/DownloadFile?filename=UNv1.0.en-ru.tar.gz.01
wget -P $prep https://conferences.unite.un.org/UNCORPUS/en/DownloadFile?filename=UNv1.0.en-ru.tar.gz.02

wget -P $prep https://conferences.unite.un.org/UNCORPUS/en/DownloadFile?filename=UNv1.0.en-zh.tar.gz.00
wget -P $prep https://conferences.unite.un.org/UNCORPUS/en/DownloadFile?filename=UNv1.0.en-zh.tar.gz.01

wget -P $prep https://conferences.unite.un.org/UNCORPUS/en/DownloadFile?filename=UNv1.0.ar-en.tar.gz.00
wget -P $prep https://conferences.unite.un.org/UNCORPUS/en/DownloadFile?filename=UNv1.0.ar-en.tar.gz.01

wget -P $prep https://conferences.unite.un.org/UNCORPUS/en/DownloadFile?filename=UNv1.0.testsets.tar.gz

cd $prep
cat UNv1.0.en-ru.tar.gz.* > UNv1.0.en-ru.tar.gz
tar -xzf UNv1.0.en-ru.tar.gz

cat UNv1.0.en-zh.tar.gz.* > UNv1.0.en-zh.tar.gz
tar -xzf UNv1.0.en-zh.tar.gz

cat UNv1.0.ar-en.tar.gz.* > UNv1.0.ar-en.tar.gz
tar -xzf UNv1.0.ar-en.tar.gz

tar -xzf UNv1.0.testsets.tar.gz
cd ..

