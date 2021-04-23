#[1]
gunzip sample_data.dir/*.gz

#[2]
mkdir -p sSAS.out
python3 sSAS_fit_transform.py hallmark_biogrid_subnetworks.txt sample_data.dir/train.txt sample_data.dir/test.txt sSAS.out  

#[2]
mkdir -p SRL.out
python3 SRL_fit_transform.py sSAS.out SRL.out











