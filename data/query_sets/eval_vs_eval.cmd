# MMseqs2 18.8cc5c (run in query_sets/)
mmseqs easy-search eval_chains.fasta eval_chains.fasta eval_vs_eval.m8 mmseqs_tmp --format-output query,target,fident,alnlen,qlen,tlen,qcov,tcov,evalue,bits -e 1e-3 --threads 4
