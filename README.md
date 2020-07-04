# STS_net



STS-Net is a training strategy which uses MSE and KLD to distill optical flow stream. 
The network can avoid the use of optical flow during testing while achieving high accuracy.

We release the testing and train code. 
We have not put all the code, some code needs to be modified for better reading.
We will add the test model as soon as possible.


## Testing script
For RGB stream:
```
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality RGB --sample_duration 64 --split 1 --only_RGB  \
--resume_path1 "STS_models/HMDB51/STS_HMDB51_64f.pth" \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"
```


## Training script
### For STS:
#### From pretrained Kinetics400:  
```
python STS_train.py --dataset HMDB51 --modality RGB_Flow \
--n_classes 51 \
--batch_size 12 --log 1 --sample_duration 64 \
--model resnext --model_depth 101 \
--output_layers 'avgpool' \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--resume_path1 "STS_models/HMDB51/Flow_HMDB51_64f.pth" \
--resume_path4  "STS_models/HMDB51/RGB_HMDB51_64f.pth" \
--pretrain_path  "STS_models/HMDB51/RGB_HMDB51_64f.pth" \
--result_path "results/" 
```

#### From pretrained checkpoint:
```
 python STS_train.py --dataset HMDB51 --modality RGB_Flow \
--n_classes 51 \
--batch_size 12 --log 1 --sample_duration 64 \
--model resnext --model_depth 101 \
--output_layers 'avgpool' \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--resume_path1 "STS_models/HMDB51/Flow_HMDB51_64f.pth" \
--resume_path2  "STS_models/HMDB51/STS_HMDB51_64f.pth" \
--result_path "results/" 
```
