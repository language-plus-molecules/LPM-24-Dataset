
python write_sdf.py --input_file $1 --direction=caption
./m2v.sh
python text_text2mol_metric.py --input_file $1


