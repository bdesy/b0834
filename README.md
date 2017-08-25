Scripts, objects and notebooks to analyse scintillation patterns in PSR B0834+06

Author : Beatrice Desy
Email : desybeatrice@gmail.com

Needs to be ran on a CITA machine to access the data files. 
Paths should lead directly to my /mnt/scratch-luster/bdesy/b0834 space.
The files on my sratch-luster space are sync to github, don't forget 
to git pull if you want to modify them.
Ideally, clone this repository on your CITA machine using

git clone https://github.com/bdesy/b0834.git

to use and modify the notebooks in it, then git push the changes and git pull on any local working 
space before touching anything. Your git username and password will be 
required. 

Git is very clever to keep version and I have versions as of September 25th, 
as well as this repository synced to my computer and CITA account. 
Therefore, feel free to CHANGE these files.
This analysis is under developemnt, it is MEANT to me optimized.
Also feel free to contact me for any help or question, I can very easily 
access the code and files to correct or add anything if needed.

tqdm WARNING : If you don't have tqdm installed on your machine, you should 
because its great. Otherwise you can run the for loops simply by removing 
the tqdm defore them. But really, think about tqdm. It is a nice and simple
progress bar python package that has been carefully coded to not overhead. 

