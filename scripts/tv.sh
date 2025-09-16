## MM DFN - IEMOCAP - tv

for i in {1..5}; do
  python code/train.py \
    --backbone="mm_dfn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new 
done

## MM DFN - IEMOCAP - tv - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="mm_dfn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap_coid" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new \
    --use_smurf \
    --use_comm
done

###############################

## MMGCN - IEMOCAP - tv 

for i in {1..5}; do
  python code/train.py \
    --backbone="mmgcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new
done

## MM DFN - IEMOCAP - tv - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="mmgcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap_coid" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new \
    --use_smurf \
    --use_comm
done

###############################


## dialouge gcn - IEMOCAP - tv 

for i in {1..5}; do
  python code/train.py \
    --backbone="dialogue_gcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new
done

## DialogueGCN - IEMOCAP - tv - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="dialogue_gcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap_coid" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new \
    --use_smurf \
    --use_comm
done

###########################################
## MELD

## MM DFN - meld - tv

for i in {1..5}; do
  python code/train.py \
    --backbone="mm_dfn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new
done

## MM DFN - meld - tv - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="mm_dfn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld_coid" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new \
    --use_smurf \
    --use_comm
done

###############################

## MMGCN - meld - tv 

for i in {1..5}; do
  python code/train.py \
    --backbone="mmgcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new
done

## MM DFN - meld - tv - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="mmgcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld_coid" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new \
    --use_smurf \
    --use_comm
done

###############################


## dialouge gcn - meld - tv 

for i in {1..5}; do
  python code/train.py \
    --backbone="dialogue_gcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new
done

## DialogueGCN - meld - tv - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="dialogue_gcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld_coid" \
    --modalities=tv \
    --batch_size=18 \
    --epochs=60 \
    --seed=50 \
    --drop_rate=0.4 \
    --early_stopping=20 \
    --encoder_modules=transformer \
    --comet \
    --comet_api Fd1aGmcly8SdDO5Ez4DMyCIt5 \
    --comet_workspace mattam301 \
    --project_name backbone_coid_new \
    --use_smurf \
    --use_comm
done