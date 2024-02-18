
python write_sdf.py --input_file $1 --use_gt
./m2v.sh
python mol_text2mol_metric.py --input_file $1


