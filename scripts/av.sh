## MM DFN - IEMOCAP - av

for i in {1..5}; do
  python code/train.py \
    --backbone="mm_dfn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap" \
    --modalities=av \
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

## MM DFN - IEMOCAP - av - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="mm_dfn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap_coid" \
    --modalities=av \
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

## MMGCN - IEMOCAP - av 

for i in {1..5}; do
  python code/train.py \
    --backbone="mmgcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap" \
    --modalities=av \
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

## MM DFN - IEMOCAP - av - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="mmgcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap_coid" \
    --modalities=av \
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


## dialouge gcn - IEMOCAP - av 

for i in {1..5}; do
  python code/train.py \
    --backbone="dialogue_gcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap" \
    --modalities=av \
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

## DialogueGCN - IEMOCAP - av - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="dialogue_gcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="iemocap_coid" \
    --modalities=av \
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

## MM DFN - meld - av

for i in {1..5}; do
  python code/train.py \
    --backbone="mm_dfn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld" \
    --modalities=av \
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

## MM DFN - meld - av - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="mm_dfn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld_coid" \
    --modalities=av \
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

## MMGCN - meld - av 

for i in {1..5}; do
  python code/train.py \
    --backbone="mmgcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld" \
    --modalities=av \
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

## MM DFN - meld - av - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="mmgcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld_coid" \
    --modalities=av \
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


## dialouge gcn - meld - av 

for i in {1..5}; do
  python code/train.py \
    --backbone="dialogue_gcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld" \
    --modalities=av \
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

## DialogueGCN - meld - av - COID
for i in {1..5}; do
  python code/train.py \
    --backbone="dialogue_gcn" \
    --name="demo6_cl_run${i}" \
    --hidden_dim=256 \
    --learning_rate=0.0001 \
    --dataset="meld_coid" \
    --modalities=av \
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